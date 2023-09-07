from typing import Tuple
import torch
import math

import gc


class DDPM:
    """Diffusion model based on the works of
    Ho et al., 2020 (https://arxiv.org/abs/2006.11239)
    This implementation follows the implementation found in
    https://github.com/quickgrid/pytorch-diffusion/tree/main
    """

    def __init__(self, num_noise_steps: int, beta_start: float = 1e-4, beta_end: float = 0.02):
        """Diffusion model based on the works of
        Ho et al., 2020 (https://arxiv.org/abs/2006.11239)
        This implementation follows the implementation found in
        https://github.com/quickgrid/pytorch-diffusion/tree/main

        Parameters
        ----------
        num_noise_steps : int
            Number of steps in forward/reverse processes
        beta_start : float, optional
            Range of beta (noise scale value), by default 1e-4
        beta_end : float, optional
            Range of beta (noise scale value), by default 0.02
        """
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

        Parameters
        ----------
        x : torch.Tensor
            Data input
        t : torch.Tensor
            Noise step

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Noised data at noise step t
        """

        sqrt_alpha_hat = self.sqrt_alpha_hat[t].view(-1, 1)
        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_hat[t].view(-1, 1)
        if x.dim() == 3:  # in the case of time series
            sqrt_alpha_hat = self.sqrt_alpha_hat[t].view(-1, 1, 1)
            sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_hat[t].view(-1, 1, 1)

        epsilon = torch.randn_like(x, device=self.device)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon


class ResidualBlock(torch.nn.Module):
    """Residual block based on the work of Gorishniy et al., 2023
    (https://arxiv.org/abs/2106.11959).
    We follow the implementation found in
    https://github.com/Yura52/rtdl/blob/main/rtdl/nn/_backbones.py"""

    def __init__(self, dim_input: int, dim_embedding: int = 128, p_dropout: float = 0.1):
        """Residual block based on the work of Gorishniy et al., 2023
        (https://arxiv.org/abs/2106.11959).
        We follow the implementation found in
        https://github.com/Yura52/rtdl/blob/main/rtdl/nn/_backbones.py

        Parameters
        ----------
        dim_input : int
            Input dimension
        dim_embedding : int, optional
            Embedding dimension, by default 128
        p_dropout : float, optional
            Dropout probability, by default 0.1
        """

        super().__init__()

        self.layer_norm = torch.nn.LayerNorm(dim_input)
        self.linear_in = torch.nn.Linear(dim_input, dim_embedding)
        self.linear_out = torch.nn.Linear(dim_embedding, dim_input)
        self.dropout = torch.nn.Dropout(p_dropout)

        self.linear_out = torch.nn.Linear(dim_embedding, dim_input)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return an output of a residual block

        Parameters
        ----------
        x : torch.Tensor
            Data input
        t : torch.Tensor
            Noise step

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Output data at noise step t
        """

        x_t = x + t
        x_t = self.layer_norm(x_t)
        x_t_emb = torch.nn.functional.relu(self.linear_in(x_t))
        x_t_emb = self.dropout(x_t_emb)
        x_t_emb = self.linear_out(x_t_emb)

        return x + x_t_emb, x_t_emb


class AutoEncoder(torch.nn.Module):
    """Epsilon_theta model of the Algorithm 1 in
    Ho et al., 2020 (https://arxiv.org/abs/2006.11239).
    This implementation is based on the work of
    Tashiro et al., 2021 (https://arxiv.org/abs/2107.03502).
    Their code: https://github.com/ermongroup/CSDI/blob/main/diff_models.py"""

    def __init__(
        self,
        num_noise_steps: int,
        dim_input: int,
        dim_embedding: int = 128,
        num_blocks: int = 1,
        p_dropout: float = 0.0,
    ):
        """Epsilon_theta model in Algorithm 1 in
        Ho et al., 2020 (https://arxiv.org/abs/2006.11239)

        Parameters
        ----------
        num_noise_steps : int
            Number of steps in forward/reverse processes
        dim_input : int
            Input dimension
        dim_embedding : int, optional
            Embedding dimension, by default 128
        num_blocks : int, optional
            Number of residual blocks, by default 1
        p_dropout : float, optional
            Dropout probability, by default 0.0
        """
        super().__init__()

        self.layer_x = torch.nn.Linear(dim_input, dim_embedding)

        self.register_buffer(
            "embedding_noise_step",
            self._build_embedding(num_noise_steps, int(dim_embedding / 2)),
            persistent=False,
        )
        self.layer_t_1 = torch.nn.Linear(dim_embedding, dim_embedding)
        self.layer_t_2 = torch.nn.Linear(dim_embedding, dim_embedding)

        self.layer_out_1 = torch.nn.Linear(dim_embedding, dim_embedding)
        self.layer_out_2 = torch.nn.Linear(dim_embedding, dim_input)
        self.dropout_out = torch.nn.Dropout(p_dropout)

        self.residual_layers = torch.nn.ModuleList(
            [ResidualBlock(dim_embedding, dim_embedding, p_dropout) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor, t: torch.LongTensor) -> torch.Tensor:
        """Predict a noise

        Parameters
        ----------
        x : torch.Tensor
            Data input
        t : torch.LongTensor
            Noise step

        Returns
        -------
        torch.Tensor
            Data output, noise predicted
        """
        # Noise step embedding
        t_emb = torch.as_tensor(self.embedding_noise_step)[t].squeeze()
        t_emb = self.layer_t_1(t_emb)
        t_emb = torch.nn.functional.silu(t_emb)
        t_emb = self.layer_t_2(t_emb)
        t_emb = torch.nn.functional.silu(t_emb)

        x_emb = torch.nn.functional.relu(self.layer_x(x))

        skip = []
        for layer in self.residual_layers:
            x_emb, skip_connection = layer(x_emb, t_emb)
            skip.append(skip_connection)

        out = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        out = torch.nn.functional.relu(self.layer_out_1(out))
        out = self.dropout_out(out)
        out = self.layer_out_2(out)

        return out

    def _build_embedding(self, num_noise_steps: int, dim: int = 64) -> torch.Tensor:
        """Build an embedding for noise step.
        More details in section E.1 of Tashiro et al., 2021
        (https://arxiv.org/abs/2107.03502)

        Parameters
        ----------
        num_noise_steps : int
            Number of noise steps
        dim : int, optional
            output dimension, by default 64

        Returns
        -------
        torch.Tensor
            _description_
        """
        steps = torch.arange(num_noise_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class ResidualBlockTS(torch.nn.Module):
    """Residual block based on the work of Gorishniy et al., 2023
    (https://arxiv.org/abs/2106.11959).
    We follow the implementation found in
    https://github.com/Yura52/rtdl/blob/main/rtdl/nn/_backbones.py
    This class is for Time-Series data where we add Tranformers to
    encode time-based/feature-based context."""

    def __init__(
        self,
        dim_input: int,
        size_window: int = 10,
        dim_embedding: int = 128,
        dim_feedforward: int = 64,
        nheads_feature: int = 5,
        nheads_time: int = 8,
        num_layers_transformer: int = 1,
    ):
        """Residual block based on the work of Gorishniy et al., 2023
        (https://arxiv.org/abs/2106.11959).
        We follow the implementation found in
        https://github.com/Yura52/rtdl/blob/main/rtdl/nn/_backbones.py
        This class is for Time-Series data where we add Tranformers to
        encode time-based/feature-based context.

        Parameters
        ----------
        dim_input : int
            Input dimension
        size_window : int, optional
            Size of window, by default 10
        dim_embedding : int, optional
            Embedding dimension, by default 128
        dim_feedforward : int, optional
            Feedforward layer dimension, by default 64
        nheads_feature : int, optional
            Number of heads to encode feature-based context, by default 5
        nheads_time : int, optional
            Number of heads to encode time-based context, by default 8
        num_layers_transformer : int, optional
            Number of transformer layer, by default 1
        """
        super().__init__()

        self.layer_norm = torch.nn.LayerNorm(dim_input)

        # encoder_layer_feature = torch.nn.TransformerEncoderLayer(
        #     d_model=size_window,
        #     nhead=nheads_feature,
        #     dim_feedforward=dim_feedforward,
        #     activation="gelu",
        #     batch_first=True,
        #     dropout=0.1,
        # )
        # self.feature_layer = torch.nn.TransformerEncoder(
        #     encoder_layer_feature, num_layers=num_layers_transformer
        # )

        encoder_layer_time = torch.nn.TransformerEncoderLayer(
            d_model=dim_embedding,
            nhead=nheads_time,
            dim_feedforward=dim_feedforward,
            activation="gelu",
            batch_first=True,
            dropout=0.1,
        )
        self.time_layer = torch.nn.TransformerEncoder(
            encoder_layer_time, num_layers=num_layers_transformer
        )

        self.linear_out = torch.nn.Linear(dim_embedding, dim_input)

    def forward(self, x: torch.Tensor, t: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return an output of a residual block

        Parameters
        ----------
        x : torch.Tensor
            Data input
        t : torch.LongTensor
            Noise step

        Returns
        -------
        torch.Tensor
            Data output, noise predicted
        """
        batch_size, size_window, dim_emb = x.shape

        x_emb = self.layer_norm(x)

        # x_emb_feat = x_emb.permute(0, 2, 1)
        # x_emb_feat = self.feature_layer(x_emb_feat)
        # x_emb_feat = x_emb_feat.permute(0, 2, 1)

        x_emb_time = self.time_layer(x_emb)
        t_emb = t.repeat(1, size_window).reshape(batch_size, size_window, dim_emb)

        x_t = x + x_emb_time + t_emb
        x_t = self.linear_out(x_t)

        return x + x_t, x_t


class AutoEncoderTS(AutoEncoder):
    """Epsilon_theta model of the Algorithm 1 in
    Ho et al., 2020 (https://arxiv.org/abs/2006.11239).
    This is for Time-series data.
    This implementation is based on the work of
    Tashiro et al., 2021 (https://arxiv.org/abs/2107.03502).
    Their code: https://github.com/ermongroup/CSDI/blob/main/diff_models.py"""

    def __init__(
        self,
        num_noise_steps: int,
        dim_input: int,
        size_window: int = 10,
        dim_embedding: int = 128,
        dim_feedforward: int = 64,
        num_blocks: int = 1,
        nheads_feature: int = 5,
        nheads_time: int = 8,
        num_layers_transformer: int = 1,
        p_dropout: float = 0.0,
    ):
        """Epsilon_theta model of the Algorithm 1 in
        Ho et al., 2020 (https://arxiv.org/abs/2006.11239).
        This is for Time-series data.
        This implementation is based on the work of
        Tashiro et al., 2021 (https://arxiv.org/abs/2107.03502).
        Their code: https://github.com/ermongroup/CSDI/blob/main/diff_models.py

        Parameters
        ----------
        num_noise_steps : int
            Number of noise steps
        dim_input : int
            Input dimension
        size_window : int, optional
            Size of window, by default 10
        dim_embedding : int, optional
            Embedding dimension, by default 128
        dim_feedforward : int, optional
            Feedforward layer dimension, by default 64
        num_blocks : int, optional
            Number of residual blocks, by default 1
        nheads_feature : int, optional
            Number of heads to encode feature-based context, by default 5
        nheads_time : int, optional
            Number of heads to encode time-based context, by default 8
        num_layers_transformer : int, optional
            Number of transformer layer, by default 1
        p_dropout : float, optional
            Dropout probability, by default 0.0
        """
        super().__init__(num_noise_steps, dim_input, dim_embedding, num_blocks, p_dropout)

        self.residual_layers = torch.nn.ModuleList(
            [
                ResidualBlockTS(
                    dim_embedding,
                    size_window,
                    dim_embedding,
                    dim_feedforward,
                    nheads_feature,
                    nheads_time,
                    num_layers_transformer,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor, t: torch.LongTensor) -> torch.Tensor:
        """Predict a noise

        Parameters
        ----------
        x : torch.Tensor
            Data input
        t : torch.LongTensor
            Noise step

        Returns
        -------
        torch.Tensor
            Data output, noise predicted
        """
        # Noise step embedding
        t_emb = torch.as_tensor(self.embedding_noise_step)[t].squeeze()
        t_emb = self.layer_t_1(t_emb)
        t_emb = torch.nn.functional.silu(t_emb)
        t_emb = self.layer_t_2(t_emb)
        t_emb = torch.nn.functional.silu(t_emb)

        x_emb = torch.nn.functional.relu(self.layer_x(x))

        skip = []
        for layer in self.residual_layers:
            x_emb, skip_connection = layer(x_emb, t_emb)
            skip.append(skip_connection)

        out = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        out = torch.nn.functional.relu(self.layer_out_1(out))
        out = self.dropout_out(out)
        out = self.layer_out_2(out)

        return out
