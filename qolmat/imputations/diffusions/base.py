from typing import Tuple
import torch
import math


class ResidualBlock(torch.nn.Module):
    """Residual block based on the work of Gorishniy et al., 2023
    (https://arxiv.org/abs/2106.11959).
    We follow the implementation found in
    https://github.com/Yura52/rtdl/blob/main/rtdl/nn/_backbones.py"""

    def __init__(self, dim_input: int, dim_embedding: int = 128, p_dropout: float = 0.0):
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

        x_t = self.layer_norm(x + t)
        x_t_emb = torch.nn.functional.relu(self.linear_in(x_t))
        x_t_emb = self.dropout(x_t_emb)
        x_t_emb = self.linear_out(x_t_emb)

        return x + x_t_emb, x_t_emb


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

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        x_emb_time = self.time_layer(x_emb)
        t_emb = t.repeat(1, size_window).reshape(batch_size, size_window, dim_emb)

        x_t = x + x_emb_time + t_emb
        x_t = self.linear_out(x_t)

        return x + x_t, x_t


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
        residual_block: torch.nn.Module,
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

        self.residual_layers = torch.nn.ModuleList([residual_block for _ in range(num_blocks)])

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
            List of embeddings for noise steps
        """
        steps = torch.arange(num_noise_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table
