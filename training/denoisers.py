from dataclasses import dataclass

import torch
import numpy as np
import faiss


class Denoiser:
    
    def __call__(self, z, sigma, x=None):
        """Estimate the posterior expectation of x given z and sigma.
        
        Args:
             z: A [N, ...] FloatTensor noisy latent with overall dimension D.
             sigma: A [N, ...] FloatTensor giving the noise level.
             x: Optional [N, ...] Float tensor of the clean element which generated z

        Returns:
             A [N, ...] Estimate of the posterior mean of x given z.
        
        """
        raise NotImplementedError()


class NetworkDenoiser(Denoiser):
    """Estimate the posterior expectation of x given z with a trained neural network"""
    
    def __init__(self, network):
        self.network = network

    def __call__(self, z, sigma, x=None):
        return self.network(z, sigma)


class SNISDenoiser(Denoiser):
    """Estimate the posterior expectation using self-normalizing importance sampling"""
    
    def __init__(self, data):
        # Compatibility for image based training.
        if isinstance(data, torch.utils.data.Dataset):
            self.D = int(np.prod(data.image_shape))
            self.N = len(data)
            self.data = torch.empty(
                (self.N, self.D),
                dtype=torch.uint8
            )
            for i, elem in enumerate(data):
                self.data[..., i, :] = torch.tensor(elem[0]).view(-1)
        elif isinstance(data, torch.tensor):
            # General purpose denoiser.
            self.N = data.shape[0]
            self.data = data.reshape(self.N, -1)
            self.D = data.shape[1]
        else:
            raise ValueError(f'Expected data to be a dataset or tensor, found {type(data)}.')

    @staticmethod
    def _posterior_mean_flat(z, sigma, samples, batch_idx=None, multipliers=None):
        """ Computes the posterior mean from flattened samples.

        Args:
            z: A [B, ...] batch of noisy latents with overall dimensionality D.
            sigma: A [B, 1] batch of noise levels.
            samples: A [L, D] batch of proposal samples.
            batch_idx: A [L] tensor where each element indicates the batch
                index for the corresponding sample.
            multipliers: An optional [L] batch of multipliers for each sample.
                multipliers are the count / the proposal probability


        Returns:
            A [B, ...] batch of posterior mean estimates in the same shape as z
        """
        B = z.shape[0]
        L = samples.shape[0]

        # Build a [B, L] matrix where each column is a one-hot vector
        # indicating that sample's batch membership.
        one_hot_indices = torch.stack([
            batch_idx,
            torch.arange(L, device=z.device)
        ])
        batch_one_hot = torch.sparse_coo_tensor(
            one_hot_indices,
            torch.ones(L, device=z.device),
            size=(B, L)
        )
        _, batch_counts = torch.unique(
            batch_idx, return_counts=True)

        # Compute the un-normalized forward likelihood
        z_scores = z.reshape(B, -1).repeat_interleave(batch_counts, dim=0)
        z_scores -= samples.view(L, -1)
        z_scores = (z_scores ** 2).sum(dim=-1)
        z_scores /= -2. * sigma.view(B).repeat_interleave(batch_counts) ** 2

        # Multiply the likelihood by the multipliers in log space.
        if multipliers is None:
            multipliers = torch.ones_like(z_scores)
        z_scores += torch.log(multipliers)

        # Construct a sparse COO tensor with each row being the z_scores
        # for a given batch.
        sparse_z_scores = torch.sparse_coo_tensor(
            one_hot_indices,
            z_scores,
            size=(B, L)
        )

        # Softmax normalizes the weights for SNIS.
        weights = torch.sparse.softmax(sparse_z_scores, dim=1)
        weights = weights.values().unsqueeze(1)

        # The matrix multiplication sums weighted proposal samples from the same
        # batch to estimate the posterior mean.
        return torch.mm(batch_one_hot, weights * samples.view(L, -1)).view_as(z)

    @staticmethod
    def _posterior_mean(z, sigma, samples, multipliers=None):
        """Estimate the posterior mean from proposal samples.

        Args:
            z: A [B, ...] batch of noisy latents of overall dimensionality D.
            sigma: A [B, 1, 1, 1] batch of noise levels.
            samples: A [B, R, D] batch of proposal samples.
            multipliers: A [B, R] batch of multipliers for each image based on
                proposal probability and sample frequency.

        Returns:
            A [B, ...] tensor of posterior mean estimates of the same shape as z.
    """
        B, R = samples.shape[:2]
        
        if multipliers is None:
            multipliers = torch.ones((B, R), device=z.device)

        targets = samples.view(B, R, -1)
        distance = (z.reshape(B, 1, -1) - targets) ** 2
        z_scores = distance.sum(dim=-1) / (-2. * sigma.view(B, 1) ** 2)

        # adding a constant to the log-weights to prevent numerical issue
        z_scores -= z_scores.max(dim=1, keepdim=True)[0]
        weights = torch.exp(z_scores.to(torch.float64))
        # self-normalize the per-sample weight of reference batch
        weights = weights * multipliers
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1E-6)
        weights = weights.to(targets.dtype)
        # calculate the stable targets with reference batch
        weighted_targets = weights.view(B, R, 1) * targets  # [B, R, -1]
        stable_target = weighted_targets.sum(dim=1)  # [B, -1]
        stable_target = torch.clamp(stable_target, min=-1, max=1)
        return stable_target.reshape(z.shape)


@dataclass
class KNNProposal:
    neighbour_probs: torch.FloatTensor
    neighbour_indices: torch.LongTensor
    uniform_probs: torch.FloatTensor


class KNNDenoiser(SNISDenoiser):

    def __init__(self, data, ref_size, k, z_thresh=1E3, gpu=True,
                 quantize=False, index=None, fp16=False):
        super().__init__(data)
        self.ref_size = int(ref_size)
        self.k = k
        self.z_thresh = z_thresh
        self.quantize = quantize


        index_type_str = "SQ8" if quantize else "Flat"

        # Data is quantized based on min/max. which is [-1, 1] with our preprocessing.
        x_train = torch.ones(2, self.D)
        x_train[0, :] *= -1

        # Build a KNN Index.
        if index is not None:
            self.index = index
        else:
            self.index = faiss.index_factory(self.D, index_type_str)
            if not self.index.is_trained:
                self.index.train(x_train)
            float_data = self.data.to(torch.float32) / 127.5 - 1
            self.index.add(float_data)

        if gpu:
            res = faiss.StandardGpuResources()
            device_index = torch.cuda.current_device()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = fp16
            self.index = faiss.index_cpu_to_gpu(res, device_index, self.index, co)

    def _get_neighbours(self, z):
        B = z.shape[0]
        device = z.device
        query = z.view(B, -1)
        D, I = self.index.search(query.cpu(), k=self.k)
        D = torch.tensor(D, device=device)  # [B, k] - The squared L2 distances.
        I = torch.tensor(I, device=device)  # [B, k] - The indices of the nearest neighbours.
        return D, I

    def _get_proposal(self, z, sigma):
        """Build the categorical proposal q_t(x | z)."""
        B = z.shape[0]
        distances, indices = self._get_neighbours(z)

        # Compute the log probabilities of the neighbours.
        log_probs = distances / (-2 * sigma.view(B, 1) ** 2)  # [B, k]
        log_probs -= log_probs.max(dim=1, keepdim=True)[0]  # [B, k]
        probs = log_probs.exp()  # [B, k]

        # We know that the lowest probability neighbour is more likely than
        # all other non-neighbours. We set the probability of all non-neighbour
        # elements to be equal to the minimum neighbour probability and
        # normalize.
        min_prob = probs.min(dim=1, keepdim=True)[0]
        tail_mass = min_prob * (self.N - self.k)
        normalizer = probs.sum(dim=1, keepdim=True) + tail_mass
        probs /= normalizer
        uniform_prob = min_prob / normalizer
        return KNNProposal(
            neighbour_probs=probs,
            neighbour_indices=indices,
            uniform_probs=uniform_prob,
        )

    def _sample_proposal(self, proposal):
        """Draw self.ref_size samples from the proposal, combining duplicate
            samples.

        Args:
            proposal: A KNNProposal object

        Returns:
            elems: The samples drawn from the proposal. Shape [L, -1]
            batch_indices: A tensor of shape [L] indicating batch membership
            multipliers: A tensor of shape [L] giving count/proposal prob.
        """
        B = proposal.neighbour_probs.shape[0]
        device = proposal.neighbour_probs.device

        # Sample indices uniformly.
        uniform_indices = torch.randint(self.N, (B, self.ref_size), device=device)

        # Sample elements from the neighbour set.
        adjusted_neighbour_probs = proposal.neighbour_probs - proposal.uniform_probs
        neighbour_indices = torch.multinomial(
            adjusted_neighbour_probs.to(torch.float64),
            self.ref_size, replacement=True)
        neighbour_indices = torch.gather(proposal.neighbour_indices, dim=-1, index=neighbour_indices)

        # Randomly select uniform or neighbour per sample.
        uniform_mask = torch.rand((B, self.ref_size), device=device)
        uniform_mask = uniform_mask < (proposal.uniform_probs * self.N)
        sampled_indices = torch.where(uniform_mask, uniform_indices, neighbour_indices).view(-1)

        # Produce a tensor of unique (batch_idx, dataset_idx) tuples.
        batch_indices = torch.arange(B, dtype=torch.int64, device=device)
        batch_indices = batch_indices.repeat_interleave(self.ref_size)
        stacked_indices = torch.stack(
            [batch_indices, sampled_indices.view(-1)], dim=-1).view(-1, 2)
        unique_stacked_indices, counts, = torch.unique(
            stacked_indices, return_counts=True, dim=0)

        # Get the number of unique samples for each batch index.
        _, batch_counts = torch.unique(
            unique_stacked_indices[:, 0], return_counts=True)
        L = unique_stacked_indices.shape[0]

        # Fetch the sampled elements.
        elems = self.data[unique_stacked_indices[:, 1].cpu()].to(device) / 127.5 - 1

        # Compute the proposal probabilities for the sampled elements.
        # This is a sparse mask tensor. 1 if the element was sampled at least
        # once.
        has_sample = torch.sparse_coo_tensor(
            unique_stacked_indices.T,
            torch.ones(L, device=device, dtype=torch.bool),
            size=(B, self.N),
            device=device
        ).coalesce()

        # The neighbour only proposal tensor. Zero for non-neighbours,
        # neighbour_prob - uniform_prob for neighbour tensors.
        neighbour_proposal_probs = torch.sparse_coo_tensor(
            torch.stack([
                torch.arange(B, device=device).repeat_interleave(self.k),
                proposal.neighbour_indices.view(-1)
            ]),
            adjusted_neighbour_probs.view(-1),
            size=(B, self.N),
            device=device
        ).coalesce()

        # Select elements from proposal which have samples.
        sampled_q = neighbour_proposal_probs.sparse_mask(has_sample).values()
        # Add back the uniform portion of the proposal probs.
        sampled_q += proposal.uniform_probs.repeat_interleave(batch_counts)
        multipliers = counts / sampled_q
        return elems, unique_stacked_indices[:, 0], multipliers

    def __call__(self, z, sigma, x=None):
        proposal = self._get_proposal(z, sigma)
        samples, batch_indices, multipliers = self._sample_proposal(proposal)
        return self._posterior_mean_flat(
            z, sigma, samples, batch_indices, multipliers)


class STFDenoiser(SNISDenoiser):

    def __init__(self, data, ref_size, use_batch_as_ref=True):
        super().__init__(data)
        self.ref_size = int(ref_size)
        self.use_batch_as_ref = use_batch_as_ref

    def __call__(self, z, sigma, x=None):
        B = z.shape[0]
        if self.use_batch_as_ref:
            R = self.ref_size - B
            ref_indices = torch.randint(self.N, (R,), device=self.data.device)
            ref_imgs = self.data[ref_indices].to(torch.float32).to(z.device).view(1, R, self.D).expand(B, R, self.D)
            ref_imgs = ref_imgs / 127.5 - 1.
        else:
            R = self.ref_size - 1
            n_ref_imgs = int(B * R)
            ref_indices = torch.randint(self.N, (n_ref_imgs,), device=self.data.device)
            ref_imgs = self.data[ref_indices].to(z.device).view(B, R, self.D) / 127.5 - 1
        if x is not None:
            ref_imgs = torch.cat([ref_imgs, x.unsqueeze(1).view(B, 1, self.D)], dim=1)
        return self._posterior_mean(z, sigma, ref_imgs)


class SingleSampleDenoiser(Denoiser):

    def __init__(self):
        super().__init__()

    def __call__(self, z, sigma, x=None):
        return x
