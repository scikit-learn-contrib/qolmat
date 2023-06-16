from typing import Tuple
import torch


class DDPM:
    def __init__(self, num_noise_steps, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.loss_func = torch.nn.MSELoss(reduction="none")

        self.num_noise_steps = num_noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Section 2, equation 4 and near explation for alpha, alpha hat, beta.
        self.beta = torch.linspace(
            start=self.beta_start,
            end=self.beta_end,
            steps=self.num_noise_steps,
            device=self.device,
        )  # Linear noise schedule
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        # Section 3.2, algorithm 1 formula implementation. Generate values early reuse later.
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat)

        # Section 3.2, equation 2 precalculation values.
        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.std_beta = torch.sqrt(self.beta)

    def q_sample(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Section 3.2, algorithm 1 formula implementation. Forward process, defined by `q`.
        Found in section 2. `q` gradually adds gaussian noise according to variance schedule. Also,
        can be seen on figure 2.
        """

        sqrt_alpha_hat = self.sqrt_alpha_hat[t].view(-1, 1)
        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_hat[t].view(-1, 1)
        if x.dim() == 3:  # in the case of time series
            sqrt_alpha_hat = self.sqrt_alpha_hat[t].view(-1, 1, 1)
            sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_hat[t].view(-1, 1, 1)

        epsilon = torch.randn_like(x, device=self.device)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon
