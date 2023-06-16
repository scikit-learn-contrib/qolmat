from typing import Tuple
import torch
import math


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


class ResidualBlock(torch.nn.Module):
    def __init__(self, dim_input, dim_embedding=128, p_dropout=0.1):
        super().__init__()

        self.linear_in = torch.nn.Linear(dim_input, dim_embedding)
        self.linear_out = torch.nn.Linear(dim_embedding, dim_input)
        self.dropout = torch.nn.Dropout(p_dropout)

        self.linear_out = torch.nn.Linear(dim_embedding, dim_input)

    def forward(self, x, t):

        x_t = x + t
        x_t_emb = torch.nn.functional.relu(self.linear_in(x_t))
        x_t_emb = self.dropout(x_t_emb)
        x_t_emb = self.linear_out(x_t_emb)

        return x + x_t_emb, x_t_emb


class AutoEncoder(torch.nn.Module):
    def __init__(self, num_noise_steps, dim_input, dim_embedding=128, num_blocks=1, p_dropout=0.0):
        super().__init__()

        self.layer_x = torch.nn.Linear(dim_input, dim_embedding)

        self.register_buffer(
            "embedding_noise_step",
            self._build_embedding(num_noise_steps, dim_embedding / 2),
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
        # Noise step embedding
        t_emb = self.embedding_noise_step[t].squeeze()
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

    def _build_embedding(self, num_noise_steps, dim=64) -> torch.Tensor:
        steps = torch.arange(num_noise_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class ResidualBlockTS(torch.nn.Module):
    def __init__(
        self,
        dim_input,
        size_window=10,
        dim_embedding=128,
        dim_feedforward=64,
        nheads_feature=5,
        nheads_time=8,
        num_layers_transformer=1,
    ):
        super().__init__()

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

    def forward(self, x, t):
        batch_size, size_window, dim_emb = x.shape

        # x_emb = x.permute(0, 2, 1)
        # x_emb = self.feature_layer(x_emb)
        # x_emb = x_emb.permute(0, 2, 1)
        x_emb = x
        x_emb = self.time_layer(x_emb)
        t_emb = t.repeat(1, size_window).reshape(batch_size, size_window, dim_emb)

        x_t = x + x_emb + t_emb
        x_t = self.linear_out(x_t)

        return x + x_t, x_t


class AutoEncoderTS(AutoEncoder):
    def __init__(
        self,
        num_noise_steps,
        dim_input,
        size_window=10,
        dim_embedding=128,
        dim_feedforward=64,
        num_blocks=1,
        nheads_feature=5,
        nheads_time=8,
        num_layers_transformer=1,
        p_dropout=0.0,
    ):
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
        # Noise step embedding
        t_emb = self.embedding_noise_step[t].squeeze()
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
