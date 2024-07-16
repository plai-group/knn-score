## Nearest Neighbour Score Estimators for Diffusion Generative Models <br><sub>Official PyTorch implementation of the ICML 2024 paper</sub>

**Nearest Neighbour Score Estimators for Diffusion Generative Models**<br>
Matthew Niedoba, Dylan Green, Saeid Naderiparizi, Vasileios Lioutas, Jonathan Wilder Lavington, Xiaoxuan Liang, Yunpeng Liu, Ke Zhang, Setareh Dabiri, Adam Ścibior, Berend Zwartsenberg, Frank Wood <br>
https://arxiv.org/abs/2402.08018 <br>


Abstract: *Score function estimation is the cornerstone of both training and sampling from diffusion generative models. Despite this fact, the most commonly used estimators are either biased neural network approximations or high variance Monte Carlo estimators based on the conditional score. We introduce a novel nearest neighbour score function estimator which utilizes multiple samples from the training set to dramatically decrease estimator variance. We leverage our low variance estimator in two compelling applications. Training consistency models with our estimator, we report a significant increase in both convergence speed and sample quality. In diffusion models, we show that our estimator can replace a learned network for probability-flow ODE integration, opening promising new avenues of future research.*

Environment details and code examples coming soon.
