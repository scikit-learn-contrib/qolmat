from typing import Dict, List, Callable
import math
import numpy as np
import pandas as pd
import time
import gc
from datetime import timedelta

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing

from qolmat.imputations.diffusions.base import DDPM, AutoEncoder, AutoEncoderTS
from qolmat.imputations.diffusions.utils import get_num_params
from qolmat.benchmark import missing_patterns, metrics


class TabDDPM(DDPM):
    def __init__(
        self,
        dim_input: int,
        num_noise_steps: int = 50,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        lr: float = 0.001,
        ratio_masked: float = 0.1,
        dim_embedding: int = 128,
        num_blocks: int = 1,
        p_dropout: float = 0.0,
    ):
        super().__init__(num_noise_steps, beta_start, beta_end)

        self.lr = lr
        self.ratio_masked = ratio_masked
        self.num_noise_steps = num_noise_steps
        self.dim_input = dim_input
        self.dim_embedding = dim_embedding
        self.num_blocks = num_blocks
        self.p_dropout = p_dropout

        self.normalizer_x = preprocessing.StandardScaler()

    def _set_eps_model(self):
        self.eps_model = AutoEncoder(
            self.num_noise_steps,
            self.dim_input,
            self.dim_embedding,
            self.num_blocks,
            self.p_dropout,
        ).to(self.device)
        self.optimiser = torch.optim.Adam(self.eps_model.parameters(), lr=self.lr)

        # p1 = int(0.75 * self.epochs)
        # p2 = int(0.9 * self.epochs)
        # self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     self.optimiser, milestones=[p1, p2], gamma=0.1
        # )

    def fit(
        self,
        x: pd.DataFrame,
        epochs: int = 10,
        batch_size: int = 100,
        print_valid: bool = False,
        x_valid: pd.DataFrame = None,
        x_valid_mask: pd.DataFrame = None,
        metrics_valid: Dict[str, Callable] = {"mae": metrics.mean_absolute_error},
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.columns = x.columns.tolist()
        self.metrics_valid = metrics_valid
        self.print_valid = print_valid
        self.time_durations: List = []

        self._set_eps_model()

        self.summary: Dict[str, List] = {
            "epoch_loss": [],
            "num_params": [get_num_params(self.eps_model)],
        }

        self.normalizer_x.fit(x.values)

        x_processed, x_mask = self._process_data(x)

        if x_valid is not None:
            if x_valid_mask is None:
                x_valid_mask = missing_patterns.UniformHoleGenerator(
                    n_splits=1, ratio_masked=self.ratio_masked
                ).generate_mask(x_valid)
            x_processed_valid, x_mask_valid = self._process_data(x_valid, x_valid_mask)

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
            self.eps_model.train()
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

            # self.lr_scheduler.step()
            time_duration = time.time() - time_start
            self.summary["epoch_loss"].append(loss.item())
            if x_valid is not None:
                self.eps_model.eval()
                dict_loss = self._eval(x_processed_valid, x_mask_valid, x_valid, x_valid_mask)
                for name_loss, value_loss in dict_loss.items():
                    if name_loss not in self.summary:
                        self.summary[name_loss] = [value_loss]
                    else:
                        self.summary[name_loss].append(value_loss)

            self._print_valid(epoch, time_duration)

    def _print_valid(self, epoch: int, time_duration: float):
        self.time_durations.append(time_duration)
        print_step = 1 if int(self.epochs / 10) == 0 else int(self.epochs / 10)
        if epoch == 0:
            print(f'Num params: {self.summary["num_params"][0]}')
        if self.print_valid and epoch % print_step == 0:
            string_valid = f"Epoch {epoch}: "
            for s in self.summary:
                if s not in ["num_params"]:
                    string_valid += f" {s}={round(self.summary[s][epoch], 5)}"
            string_valid += f" | in {round(time_duration, 3)} secs"
            remaining_duration = np.mean(self.time_durations) * (self.epochs - epoch)
            string_valid += f" remaining {timedelta(seconds=remaining_duration)}"
            print(string_valid)

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

        scores = {}
        for name, metric in self.metrics_valid.items():
            scores[name] = metric(x_df, x_normalized, ~x_mask_obs_df).mean()
        return scores

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


class TabDDPMTS(TabDDPM):
    def __init__(
        self,
        dim_input: int,
        num_noise_steps: int = 50,
        size_window: int = 10,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        lr: float = 0.001,
        ratio_masked: float = 0.1,
        dim_embedding: int = 128,
        dim_feedforward: int = 64,
        num_blocks: int = 1,
        nheads_feature: int = 5,
        nheads_time: int = 8,
        num_layers_transformer: int = 1,
        p_dropout: float = 0.0,
    ):
        super().__init__(
            dim_input,
            num_noise_steps,
            beta_start,
            beta_end,
            lr,
            ratio_masked,
            dim_embedding,
            num_blocks,
            p_dropout,
        )

        self.size_window = size_window
        self.dim_feedforward = dim_feedforward
        self.nheads_feature = nheads_feature
        self.nheads_time = nheads_time
        self.num_layers_transformer = num_layers_transformer

    def _set_eps_model(self):
        self.eps_model = AutoEncoderTS(
            self.num_noise_steps,
            self.dim_input,
            self.size_window,
            self.dim_embedding,
            self.dim_feedforward,
            self.num_blocks,
            self.nheads_feature,
            self.nheads_time,
            self.num_layers_transformer,
            self.p_dropout,
        ).to(self.device)
        self.optimiser = torch.optim.Adam(self.eps_model.parameters(), lr=self.lr)

        # p1 = int(0.75 * self.epochs)
        # p2 = int(0.9 * self.epochs)
        # self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     self.optimiser, milestones=[p1, p2], gamma=0.1
        # )

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

                    sqrt_alpha_t = self.sqrt_alpha[t].view(-1, 1, 1)
                    beta_t = self.beta[t].view(-1, 1, 1)
                    sqrt_one_minus_alpha_hat_t = self.sqrt_one_minus_alpha_hat[t].view(-1, 1, 1)
                    epsilon_t = self.std_beta[t].view(-1, 1, 1)

                    random_noise = torch.randn_like(noise) if i > 1 else torch.zeros_like(noise)

                    noise = (
                        (1 / sqrt_alpha_t)
                        * (
                            noise
                            - ((beta_t / sqrt_one_minus_alpha_hat_t) * self.eps_model(noise, t))
                        )
                    ) + (epsilon_t * random_noise)
                    noise = mask_x_batch * x_batch + (1.0 - mask_x_batch) * noise

                for id_window, x_window in enumerate(noise.detach().cpu().numpy()):
                    if id_batch == 0 and id_window == 0:
                        outputs += list(x_window)
                    else:
                        outputs += [x_window[-1, :]]

        return np.array(outputs)

    def _process_data(self, x: pd.DataFrame, mask: pd.DataFrame = None):
        x_windows = list(x.rolling(window=self.size_window))[self.size_window - 1 :]

        x_windows_processed = []
        x_windows_mask_processed = []
        for x_w in x_windows:
            x_windows_mask_processed.append(~x_w.isna().to_numpy())
            x_w_fillna = x_w.fillna(x_w.mean())
            x_w_fillna = x_w_fillna.fillna(x.mean())
            x_w_norm = self.normalizer_x.transform(x_w_fillna.values)
            x_windows_processed.append(x_w_norm)

        if mask is not None:
            x_masks = list(mask.rolling(window=self.size_window))[self.size_window - 1 :]
            x_windows_mask_processed = []
            for x_m in x_masks:
                x_windows_mask_processed.append(x_m.to_numpy())

        x_windows_processed = np.array(x_windows_processed)
        x_windows_mask_processed = np.array(x_windows_mask_processed)

        return x_windows_processed, x_windows_mask_processed
