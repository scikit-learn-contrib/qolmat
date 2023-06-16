from typing import Dict, List
import math
import numpy as np
import pandas as pd
import time
import gc

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing

from qolmat.imputations.diffusions.diffusion_base import DDPM
from qolmat.imputations.diffusions.utils import get_num_params
from qolmat.benchmark import missing_patterns, metrics


class ResidualBlock(torch.nn.Module):
    def __init__(self, dim_input, dim_embedding=128, p_dropout=0.1):
        super(ResidualBlock, self).__init__()

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
        super(AutoEncoder, self).__init__()

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


class TabDDPM(DDPM):
    def __init__(
        self,
        dim_input: int,
        num_noise_steps: int = 50,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        lr: float = 0.01,
        ratio_masked: float = 0.1,
        dim_embedding: int = 128,
        num_blocks: int = 1,
        p_dropout: float = 0.1,
    ):
        super(TabDDPM, self).__init__(num_noise_steps, beta_start, beta_end)

        self.lr = lr
        self.ratio_masked = ratio_masked
        self.num_noise_steps = num_noise_steps
        self.dim_input = dim_input
        self.dim_embedding = dim_embedding
        self.num_blocks = num_blocks
        self.p_dropout = p_dropout

        self.normalizer_x = preprocessing.StandardScaler()

        self.summary: Dict[str, List] = {
            "epoch_loss": [],
            "num_params": [get_num_params(self.eps_model)],
        }

    def fit(
        self,
        x: pd.DataFrame,
        epochs=10,
        batch_size=100,
        x_valid=None,
        x_valid_mask=None,
        print_valid=True,
    ):
        self.eps_model = AutoEncoder(
            self.num_noise_steps,
            self.dim_input,
            self.dim_embedding,
            self.num_blocks,
            self.p_dropout,
        ).to(self.device)
        self.optimiser = torch.optim.Adam(self.eps_model.parameters(), lr=self.lr)

        self.batch_size = batch_size
        self.columns = x.columns.tolist()
        self.normalizer_x.fit(x.values)

        x_processed, x_mask = self._process_data(x)

        if x_valid is not None:
            if x_valid_mask is None:
                x_valid_mask = missing_patterns.UniformHoleGenerator(
                    n_splits=1, ratio_masked=self.ratio_masked
                ).generate_mask(x_valid)
            x_processed_valid, x_mask_valid = self._process_data(x_valid, x_valid_mask)

        self.eps_model.train()
        x_tensor = torch.from_numpy(x_processed).float().to(self.device)
        x_mask_tensor = torch.from_numpy(x_mask).float().to(self.device)
        dataloader = DataLoader(
            TensorDataset(x_tensor, x_mask_tensor),
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
        )
        for epoch in range(epochs):
            loss_epoch = 0.0
            time_start = time.time()
            for id_batch, (x_batch, mask_x_batch) in enumerate(dataloader):
                mask_rand = (
                    torch.cuda.FloatTensor(mask_x_batch.size()).uniform_() > self.ratio_masked
                )
                mask_x_batch = mask_x_batch * mask_rand

                self.optimiser.zero_grad()
                t = torch.randint(
                    low=1,
                    high=self.num_noise_steps,
                    size=(x_batch.size(dim=0), 1),
                    device=self.device,
                )
                x_batch_t, noise = self.q_sample(x=x_batch, t=t)
                predicted_noise = self.eps_model(x=x_batch_t, t=t)
                loss = (self.loss_func(predicted_noise, noise) * mask_x_batch).mean()
                loss.backward()
                self.optimiser.step()
                loss_epoch += loss.item()

            time_duration = time.time() - time_start
            self.summary["epoch_loss"].append(loss.item())
            if x_valid is not None:
                dict_loss = self._eval(x_processed_valid, x_mask_valid, x_valid, x_valid_mask)
                for name_loss, value_loss in dict_loss.items():
                    if name_loss not in self.summary:
                        self.summary[name_loss] = [value_loss]
                    else:
                        self.summary[name_loss].append(value_loss)

            print_step = 1 if int(epochs / 10) == 0 else int(epochs / 10)
            if print_valid and (epoch) % print_step == 0:
                print(
                    f"""Epoch {epoch}: epoch_loss={self.summary['epoch_loss'][epoch]}
                    in {round(time_duration, 3)}"""
                )

    def _impute(self, x: np.ndarray, x_mask_obs: np.ndarray) -> np.ndarray:
        x_tensor = torch.from_numpy(x).float().to(self.device)
        x_mask_tensor = torch.from_numpy(x_mask_obs).float().to(self.device)
        dataloader = DataLoader(
            TensorDataset(x_tensor, x_mask_tensor),
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
        )
        with torch.no_grad():
            outputs = []
            for id_batch, (x_batch, mask_x_batch) in enumerate(dataloader):
                noise = torch.randn(x_batch.size(), device=self.device)

                for i in reversed(range(1, self.num_noise_steps)):
                    t = (
                        torch.ones((x_batch.size(dim=0), 1), dtype=torch.long, device=self.device)
                        * i
                    )

                    sqrt_alpha_t = self.sqrt_alpha[t].view(-1, 1)
                    beta_t = self.beta[t].view(-1, 1)
                    sqrt_one_minus_alpha_hat_t = self.sqrt_one_minus_alpha_hat[t].view(-1, 1)
                    epsilon_t = self.std_beta[t].view(-1, 1)

                    random_noise = torch.randn_like(noise) if i > 1 else torch.zeros_like(noise)

                    noise = (
                        (1 / sqrt_alpha_t)
                        * (
                            noise
                            - ((beta_t / sqrt_one_minus_alpha_hat_t) * self.eps_model(noise, t))
                        )
                    ) + (epsilon_t * random_noise)
                    noise = mask_x_batch * x_batch + (1.0 - mask_x_batch) * noise

                outputs.append(noise.detach().cpu().numpy())

        outputs = np.concatenate(outputs)
        return np.array(outputs)

    def _eval(
        self,
        x: np.ndarray,
        x_mask_obs: np.ndarray,
        x_df: pd.DataFrame,
        x_mask_obs_df: pd.DataFrame,
    ):
        x_imputed = self._impute(x, x_mask_obs)
        x_normalized = pd.DataFrame(
            self.normalizer_x.inverse_transform(x_imputed), columns=self.columns, index=x_df.index
        )

        return {
            "MAE": metrics.mean_absolute_error(x_df, x_normalized, ~x_mask_obs_df).mean(),
            "KL": metrics.kl_divergence(
                x_df, x_normalized, ~x_mask_obs_df, method="gaussian"
            ).mean(),
        }

    def predict(self, x: pd.DataFrame):
        self.eps_model.eval()

        x_processed, x_mask = self._process_data(x)
        x_imputed = self._impute(x_processed, x_mask)

        x_out = self.normalizer_x.inverse_transform(x_imputed)
        x_out = pd.DataFrame(x_out, columns=x.columns, index=x.index)
        x_out = x.fillna(x_out)
        return x_out

    def _process_data(self, x: pd.DataFrame, mask: pd.DataFrame = None):
        x_windows_processed = self.normalizer_x.transform(x.fillna(x.mean()).values)
        x_windows_mask_processed = ~x.isna().to_numpy()
        if mask is not None:
            x_windows_mask_processed = mask.to_numpy()
        return x_windows_processed, x_windows_mask_processed

    def cuda_empty_cache(self):
        del self.eps_model
        del self.optimiser
        gc.collect()
        torch.cuda.empty_cache()
