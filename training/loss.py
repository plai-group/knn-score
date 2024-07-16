"""Reimplementation of Consistency Loss as described in 'Improved Techniques for Consistency Training.'"""

import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch_utils.misc import local_seed
from piq import LPIPS
import numpy as np


class ConsistencyLoss:

    def __init__(self, solver, sigma_min, sigma_max, rho=7, sigma_data=0.5, metric='lpips',
                 huber_const=0.03, schedule='constant', s0=10, s1=1280, K=400000, weight='uniform',
                 sigma_dist='uniform', P_mean=-1.1, P_std=2.):

        if solver.lower() == 'euler':
            self.solver = self.euler
        elif solver.lower() == 'heun':
            self.solver = self.heun
        else:
            raise ValueError(f'Unknown solver {solver}.')

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigma_dist = sigma_dist
        self.sigma_data = sigma_data
        self.metric = metric
        if self.metric == 'lpips':
            self.lpips_loss = LPIPS(replace_pooling=True, reduction="none")
        self.huber_const = huber_const
        self.schedule = schedule
        self.s0 = s0
        self.s1 = s1
        self.K = K
        self.step = 0
        self.P_mean = P_mean
        self.P_std = P_std
        self.weight = weight
        self._set_n_steps(s0)

    def update_train_step(self, step):
        self.step = step
        N = self.n_steps
        if self.schedule == 'exp':
            k_prime = np.floor(self.K / (np.log2(np.floor(self.s1 / self.s0)) + 1))
            proposed_N = self.s0 * 2 ** np.floor(step / k_prime)
            N = min(self.s1, int(proposed_N)) + 1
        elif self.schedule == 'linear':
            duration = self.step / self.K
            N = int(self.s0 + duration * (self.s1 - self.s0))
        elif self.schedule == 'base':
            N = np.ceil(np.sqrt((step/self.K) * ((self.s1 + 1) ** 2 - self.s0**2) + self.s0**2) - 1) + 1
        else:
            raise ValueError(f'Unknown schedule {self.schedule}')
        if N != self.n_steps:
            self._set_n_steps(N)

    def _set_n_steps(self, n_steps):
        self.n_steps = int(n_steps)
        self.sigmas = self._get_sigmas()
        if self.sigma_dist == 'uniform':
            self.sigma_probs = torch.ones_like(self.sigmas)
        elif self.sigma_dist == 'lognormal':
            self.sigma_probs = Normal(self.P_mean,  self.P_std).log_prob(self.sigmas.log()).exp()
        elif self.sigma_dist == 'disc':
            erf1 = torch.erf((self.sigmas[1:].log() - self.P_mean) / (np.sqrt(2.) * self.P_std))
            erf2 = torch.erf((self.sigmas[:-1].log() - self.P_mean) / (np.sqrt(2.) * self.P_std))
            self.sigma_probs = torch.zeros_like(self.sigmas)
            self.sigma_probs[:-1] = erf1 - erf2
        else:
            raise ValueError(f'Unknown sigma distribution type {self.sigma_dist}')

        # Set sigma_max to have probability 1 since we sample pairs of (i, i+1).
        self.sigma_probs[-1] = 0.
        self.sigma_probs /= self.sigma_probs.sum()  # Normalize.

    def _get_sigmas(self):
        indices = torch.arange(self.n_steps)
        start = self.sigma_min ** (1. / self.rho)
        end = self.sigma_max ** (1. / self.rho)
        step_size = (end - start) / (self.n_steps - 1)
        return (start + indices * step_size) ** self.rho

    @staticmethod
    def euler(x, t, t_next, D, denoiser=None, x0=None, labels=None):
        return x + (t_next - t) * -(D - x) / t

    @staticmethod
    def heun(z, t, t_next, D, denoiser, x=None):
        dbl_z = z.to(torch.float64)
        dbl_t = t.to(torch.float64)
        dbl_tnext = t_next.to(torch.float64)
        dbl_D = D.to(torch.float64)

        d = -(dbl_D - dbl_z) / dbl_t
        z_next = dbl_z + (dbl_tnext - dbl_t) * d

        D_next = denoiser(z_next.to(z.dtype), t_next, x)
        D_next = D_next.to(torch.float64)
        d_next = -(D_next - z_next) / dbl_tnext

        out = dbl_z + (dbl_tnext - dbl_t) * 0.5 * (d_next + d)
        # return x + (t_next - t) * 0.5 * (d_next + d)
        return out.to(x.dtype)

    def get_weight(self, sigma, sigma_next):
        B = sigma.shape[0]
        weights = torch.ones_like(sigma)
        if self.weight == 'uniform':
            pass  # Weights are already ones.
        elif self.weight == 'inverse':
            weights = 1. / (sigma - sigma_next)
        else:
            raise ValueError(f'Uknown weight scheme {self.weight}')
        return weights

    def __call__(self, net, ema, denoiser, x):
        B = x.shape[0]
        indices = torch.multinomial(self.sigma_probs, B, replacement=True)
        sigma_next = self.sigmas[indices].view(B, 1, 1, 1).to(x.device)
        sigma = self.sigmas[indices + 1].view(B, 1, 1, 1).to(x.device)

        n = torch.randn_like(x) * sigma
        z = x + n

        D = denoiser(z, sigma, x)
        seed = np.random.randint(1 << 31)

        z_next = self.solver(z, sigma, sigma_next, D, denoiser, x)
        with torch.no_grad():
            with local_seed(seed):
                target = ema(z_next, sigma_next).detach()

        with local_seed(seed):
            x_est = net(x, sigma)

        if self.metric == 'lpips':
            if x.shape[-1] < 256:
                x_est = F.interpolate(x_est, size=224, mode='bilinear')
                target = F.interpolate(target, size=224, mode='bilinear')

            loss = self.lpips_loss((x_est + 1) / 2., (target + 1) / 2.)
        elif self.metric == 'l2':
            loss = (x_est - target) ** 2
        elif self.metric == 'l1':
            loss = (x_est - target).abs()
        elif self.metric == 'huber':
            loss = ((x_est - target) ** 2 + self.huber_const ** 2).sqrt() - self.huber_const
        else:
            raise ValueError(f'Unknown distance metric {self.metric}')

        weight = self.get_weight(sigma, sigma_next)
        return weight * loss
