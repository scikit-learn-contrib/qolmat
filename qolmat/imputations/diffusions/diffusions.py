from typing import Dict, List, Callable, Tuple
import math
import numpy as np
import pandas as pd
import time
from datetime import timedelta
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing

from qolmat.imputations.diffusions.base import DDPM, AutoEncoder, AutoEncoderTS
from qolmat.imputations.diffusions.utils import get_num_params
from qolmat.benchmark import missing_patterns, metrics


class TabDDPM(DDPM):
    """Diffusion model for tabular data based on the works of
    Ho et al., 2020 (https://arxiv.org/abs/2006.11239),
    Tashiro et al., 2021 (https://arxiv.org/abs/2107.03502).
    This implementation follows the implementations found in
    https://github.com/quickgrid/pytorch-diffusion/tree/main,
    https://github.com/ermongroup/CSDI/tree/main
    """

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
        num_sampling: int = 1,
        is_clip: bool = True,
    ):
        """Diffusion model for tabular data based on the works of
        Ho et al., 2020 (https://arxiv.org/abs/2006.11239),
        Tashiro et al., 2021 (https://arxiv.org/abs/2107.03502).
        This implementation follows the implementations found in
        https://github.com/quickgrid/pytorch-diffusion/tree/main,
        https://github.com/ermongroup/CSDI/tree/main

        Parameters
        ----------
        dim_input : int
            Input dimension
        num_noise_steps : int, optional
            Number of noise steps, by default 50
        beta_start : float, optional
            Range of beta (noise scale value), by default 1e-4
        beta_end : float, optional
            Range of beta (noise scale value), by default 0.02
        lr : float, optional
            Learning rate, by default 0.001
        ratio_masked : float, optional
            Ratio of artificial nan for training and validation, by default 0.1
        dim_embedding : int, optional
            Embedding dimension, by default 128
        num_blocks : int, optional
            Number of residual block in epsilon model, by default 1
        p_dropout : float, optional
            Dropout probability, by default 0.0
        num_sampling : int, optional
            Number of samples generated for each cell, by default 1
        """

        super().__init__(num_noise_steps, beta_start, beta_end)

        self.lr = lr
        self.ratio_masked = ratio_masked
        self.num_noise_steps = num_noise_steps
        self.dim_input = dim_input
        self.dim_embedding = dim_embedding
        self.num_blocks = num_blocks
        self.p_dropout = p_dropout
        self.num_sampling = num_sampling
        self.is_clip = is_clip

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
        metrics_valid: Tuple[Tuple[str, Callable], ...] = (
            ("mae", metrics.mean_absolute_error),
            ("wasser", metrics.dist_wasserstein),
        ),
        round: int = 10,
        cols_imputed: Tuple[str, ...] = (),
    ):

        """Fit data

        Parameters
        ----------
        x : pd.DataFrame
            Input dataframe
        epochs : int, optional
            Number of epochs, by default 10
        batch_size : int, optional
            Batch size, by default 100
        print_valid : bool, optional
            Print model performance for after several epochs, by default False
        x_valid : pd.DataFrame, optional
            Dataframe for validation, by default None
        x_valid_mask : pd.DataFrame, optional
            Artificial nan for validation dataframe, by default None
        metrics_valid : Tuple[Tuple[str, Callable], ...], optional
            Set of validation metrics, by default ( ("mae", metrics.mean_absolute_error),
            ("wasser", metrics.dist_wasserstein), )
        round : int, optional
            Number of decimal places to round to, by default 10
        cols_imputed : Tuple[str, ...], optional
            Name of columns that need to be imputed, by default ()

        Raises
        ------
        ValueError
            Batch size is larger than data size
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.batch_size_predict = batch_size
        self.columns = x.columns.tolist()
        self.metrics_valid = dict((x, y) for x, y in metrics_valid)
        self.print_valid = print_valid
        self.cols_imputed = cols_imputed
        self.round = round

        self.time_durations: List = []
        self.cols_idx_not_imputed: List = []

        if len(self.cols_imputed) != 0:
            self.cols_idx_not_imputed = [
                idx for idx, col in enumerate(self.columns) if col not in self.cols_imputed
            ]

        self.interval_x = {col: [x[col].min(), x[col].max()] for col in self.columns}
        self.normalizer_x.fit(x.values)

        x_processed, x_mask = self._process_data(x)

        if self.batch_size >= x_processed.shape[0]:
            raise ValueError(f"Batch size {self.batch_size} larger than x size {x.shape[0]}")

        if x_valid is not None:
            if x_valid_mask is None:
                x_valid_mask = missing_patterns.UniformHoleGenerator(
                    n_splits=1,
                    ratio_masked=self.ratio_masked,
                ).split(x_valid)[0]
            x_processed_valid, x_mask_valid = self._process_data(x_valid, x_valid_mask)

        x_tensor = torch.from_numpy(x_processed).float().to(self.device)
        x_mask_tensor = torch.from_numpy(x_mask).float().to(self.device)
        dataloader = DataLoader(
            TensorDataset(x_tensor, x_mask_tensor),
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
        )

        self._set_eps_model()
        self.summary: Dict[str, List] = {
            "epoch_loss": [],
            "num_params": [get_num_params(self.eps_model)],
        }

        for epoch in range(epochs):
            loss_epoch = 0.0
            time_start = time.time()
            self.eps_model.train()
            for id_batch, (x_batch, mask_x_batch) in enumerate(dataloader):
                mask_rand = torch.FloatTensor(mask_x_batch.size()).uniform_() > self.ratio_masked
                for col in self.cols_idx_not_imputed:
                    mask_rand[:, col] = 0.0
                mask_x_batch = mask_x_batch * mask_rand.to(self.device)

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
            self.summary["epoch_loss"].append(np.mean(loss_epoch))
            if x_valid is not None:
                self.eps_model.eval()
                dict_loss = self._eval(x_processed_valid, x_mask_valid, x_valid, x_valid_mask)
                for name_loss, value_loss in dict_loss.items():
                    if name_loss not in self.summary:
                        self.summary[name_loss] = [value_loss]
                    else:
                        self.summary[name_loss].append(value_loss)
            time_duration = time.time() - time_start
            self._print_valid(epoch, time_duration)

    def _print_valid(self, epoch: int, time_duration: float) -> None:
        """Print model performance on validation data

        Parameters
        ----------
        epoch : int
            Epoch of the printed performance
        time_duration : float
            Duration for training step
        """
        self.time_durations.append(time_duration)
        print_step = 1 if int(self.epochs / 10) == 0 else int(self.epochs / 10)
        if self.print_valid and epoch == 0:
            print(f'Num params: {self.summary["num_params"][0]}')
        if self.print_valid and epoch % print_step == 0:
            string_valid = f"Epoch {epoch}: "
            for s in self.summary:
                if s not in ["num_params"]:
                    string_valid += f" {s}={round(self.summary[s][epoch], self.round)}"
            # string_valid += f" | in {round(time_duration, 3)} secs"
            remaining_duration = np.mean(self.time_durations) * (self.epochs - epoch)
            string_valid += f"| remaining {timedelta(seconds=remaining_duration)}"
            print(string_valid)

    def _impute(self, x: np.ndarray, x_mask_obs: np.ndarray) -> np.ndarray:
        """Impute data array

        Parameters
        ----------
        x : np.ndarray
            Input data
        x_mask_obs : np.ndarray
            Observed value mask

        Returns
        -------
        np.ndarray
            Imputed data
        """
        x_tensor = torch.from_numpy(x).float().to(self.device)
        x_mask_tensor = torch.from_numpy(x_mask_obs).float().to(self.device)
        dataloader = DataLoader(
            TensorDataset(x_tensor, x_mask_tensor),
            batch_size=self.batch_size_predict,
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

                # Generate data output, this activation function depends on normalizer_x
                x_out = noise
                outputs.append(x_out.detach().cpu().numpy())

        outputs = np.concatenate(outputs)
        return np.array(outputs)

    def _eval(
        self,
        x: np.ndarray,
        x_mask_obs: np.ndarray,
        x_df: pd.DataFrame,
        x_mask_obs_df: pd.DataFrame,
    ) -> Dict:
        """Evaluate the model

        Parameters
        ----------
        x : np.ndarray
            Input data - Array (after pre-processing)
        x_mask_obs : np.ndarray
            Observed value mask (after pre-processing)
        x_df : pd.DataFrame
            Reference dataframe before pre-processing
        x_mask_obs_df : pd.DataFrame
            Observed value mask before pre-processing

        Returns
        -------
        Dict
            Scores
        """

        list_x_imputed = []
        for i in tqdm(range(self.num_sampling), disable=True):
            x_imputed = self._impute(x, x_mask_obs)
            list_x_imputed.append(x_imputed)
        x_imputed = np.mean(np.array(list_x_imputed), axis=0)
        x_normalized = self.normalizer_x.inverse_transform(x_imputed)
        x_normalized = x_normalized[: x_df.shape[0]]
        x_out = pd.DataFrame(x_normalized, columns=self.columns, index=x_df.index)
        if self.is_clip:
            for col, interval in self.interval_x.items():
                x_out[col] = np.clip(x_out[col], interval[0], interval[1])

        columns_with_True = x_mask_obs_df.columns[(x_mask_obs_df == True).any()]
        scores = {}
        for name, metric in self.metrics_valid.items():
            scores[name] = metric(
                x_df[columns_with_True],
                x_out[columns_with_True],
                x_mask_obs_df[columns_with_True],
            ).mean()
        return scores

    def _set_hyperparams_predict(self, **kwargs) -> None:
        """Reset hyperparams for predition step"""
        if "num_sampling" in kwargs:
            self.num_sampling = kwargs["num_sampling"]
        if "is_clip" in kwargs:
            self.is_clip = kwargs["is_clip"]
        if "batch_size_predict" in kwargs:
            self.batch_size_predict = kwargs["batch_size_predict"]

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        """Predict/impute data

        Parameters
        ----------
        x : pd.DataFrame
            Data needs to be imputed

        Returns
        -------
        pd.DataFrame
            Imputed data
        """
        self.eps_model.eval()

        x_processed, x_mask = self._process_data(x)

        list_x_imputed = []
        for i in tqdm(range(self.num_sampling)):
            x_imputed = self._impute(x_processed, x_mask)
            list_x_imputed.append(x_imputed)
        x_imputed = np.mean(np.array(list_x_imputed), axis=0)

        x_normalized = self.normalizer_x.inverse_transform(x_imputed)
        x_normalized = x_normalized[: x.shape[0]]
        x_out = pd.DataFrame(x_normalized, columns=x.columns, index=x.index)
        if self.is_clip:
            for col, interval in self.interval_x.items():
                x_out[col] = np.clip(x_out[col], interval[0], interval[1])
        x_out = x.fillna(x_out)
        return x_out

    def _process_data(
        self, x: pd.DataFrame, mask: pd.DataFrame = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pre-process data

        Parameters
        ----------
        x : pd.DataFrame
            Input data
        mask : pd.DataFrame, optional
            Observed value mask, by default None

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Data and mask pre-processed
        """
        x_windows_processed = self.normalizer_x.transform(x.fillna(x.mean()).values)
        x_windows_mask_processed = ~x.isna().to_numpy()
        if mask is not None:
            x_windows_mask_processed = mask.to_numpy()
        return x_windows_processed, x_windows_mask_processed


class TabDDPMTS(TabDDPM):
    """Diffusion model for time-series data based on the works of
    Ho et al., 2020 (https://arxiv.org/abs/2006.11239),
    Tashiro et al., 2021 (https://arxiv.org/abs/2107.03502).
    This implementation follows the implementations found in
    https://github.com/quickgrid/pytorch-diffusion/tree/main,
    https://github.com/ermongroup/CSDI/tree/main
    """

    def __init__(
        self,
        dim_input: int,
        num_noise_steps: int = 50,
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
        num_sampling: int = 1,
    ):
        """Diffusion model for time-series data based on the works of
        Ho et al., 2020 (https://arxiv.org/abs/2006.11239),
        Tashiro et al., 2021 (https://arxiv.org/abs/2107.03502).
        This implementation follows the implementations found in
        https://github.com/quickgrid/pytorch-diffusion/tree/main,
        https://github.com/ermongroup/CSDI/tree/main

        Parameters
        ----------
        dim_input : int
            Input dimension
        num_noise_steps : int, optional
            Number of noise steps, by default 50
        beta_start : float, optional
            Range of beta (noise scale value), by default 1e-4
        beta_end : float, optional
            Range of beta (noise scale value), by default 0.02
        lr : float, optional
            Learning rate, by default 0.001
        ratio_masked : float, optional
            Ratio of artificial nan for training and validation, by default 0.1
        dim_embedding : int, optional
            Embedding dimension, by default 128
        dim_feedforward : int, optional
            Feedforward layer dimension in Transformers, by default 64
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
        num_sampling : int, optional
            Number of samples generated for each cell, by default 1
        """
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
            num_sampling,
        )

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
        """Impute data array

        Parameters
        ----------
        x : np.ndarray
            Input data
        x_mask_obs : np.ndarray
            Observed value mask

        Returns
        -------
        np.ndarray
            Imputed data
        """

        x_tensor = torch.from_numpy(x).float().to(self.device)
        x_mask_tensor = torch.from_numpy(x_mask_obs).float().to(self.device)
        dataloader = DataLoader(
            TensorDataset(x_tensor, x_mask_tensor),
            batch_size=self.batch_size_predict,
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

                # Generate data output, this activation function depends on normalizer_x
                x_out = noise.detach().cpu().numpy()
                outputs.append(x_out)

        outputs = np.concatenate(outputs)
        outputs_shape = np.shape(outputs)
        outputs_reshaped = np.reshape(
            outputs, (outputs_shape[0] * outputs_shape[1], outputs_shape[2])
        )
        return np.array(outputs_reshaped)

    def fit(
        self,
        x: pd.DataFrame,
        epochs: int = 10,
        batch_size: int = 100,
        print_valid: bool = False,
        x_valid: pd.DataFrame = None,
        x_valid_mask: pd.DataFrame = None,
        metrics_valid: Tuple[Tuple[str, Callable], ...] = (
            ("mae", metrics.mean_absolute_error),
            ("wasser", metrics.dist_wasserstein),
        ),
        round: int = 10,
        cols_imputed: Tuple[str, ...] = (),
        index_datetime: str = "",
        freq_str: str = "1D",
    ):
        """Fit data

        Parameters
        ----------
        x : pd.DataFrame
            Input dataframe
        epochs : int, optional
            Number of epochs, by default 10
        batch_size : int, optional
            Batch size, by default 100
        print_valid : bool, optional
            Print model performance for after several epochs, by default False
        x_valid : pd.DataFrame, optional
            Dataframe for validation, by default None
        x_valid_mask : pd.DataFrame, optional
            Artificial nan for validation dataframe, by default None
        metrics_valid : Tuple[Tuple[str, Callable], ...], optional
            Set of validation metrics, by default ( ("mae", metrics.mean_absolute_error),
            ("wasser", metrics.dist_wasserstein), )
        round : int, optional
            Number of decimal places to round to, by default 10
        cols_imputed : Tuple[str, ...], optional
            Name of columns that need to be imputed, by default ()
        index_datetime : str
            Name of datetime-like index
        freq_str : str
            Frequency string of DateOffset
        Raises
        ------
        ValueError
            Batch size is larger than data size
        """
        if index_datetime == "":
            raise ValueError(
                "Please set the params index_datetime (the name of datatime-like index column)."
                + f" Suggestions: {x.index.names}"
            )
        self.index_datetime = index_datetime
        self.freq_str = freq_str
        super().fit(
            x,
            epochs,
            batch_size,
            print_valid,
            x_valid,
            x_valid_mask,
            metrics_valid,
            round,
            cols_imputed,
        )

    def _process_data(
        self, x: pd.DataFrame, mask: pd.DataFrame = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pre-process data

        Parameters
        ----------
        x : pd.DataFrame
            Input data
        mask : pd.DataFrame, optional
            Observed value mask, by default None

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Data and mask pre-processed
        """
        x_windows = list(x.resample(rule=self.freq_str, level=self.index_datetime))
        x_windows_processed = []
        x_windows_mask_processed = []
        self.size_window = x_windows[0][1].shape[0]
        for x_w in x_windows:
            x_w = x_w[1]
            x_w_fillna = x_w.fillna(x_w.mean())
            x_w_fillna = x_w_fillna.fillna(x.mean())
            x_w_norm = self.normalizer_x.transform(x_w_fillna.values)
            x_w_mask = ~x_w.isna().to_numpy()

            x_w_shape = x_w.shape
            if x_w_shape[0] < self.size_window:
                npad = [(0, self.size_window - x_w_shape[0]), (0, 0)]
                x_w_norm = np.pad(x_w_norm, pad_width=npad, mode="mean")
                x_w_mask = np.pad(x_w_mask, pad_width=npad, mode="constant", constant_values=1)

            x_windows_processed.append(x_w_norm)
            x_windows_mask_processed.append(x_w_mask)

        if mask is not None:
            x_masks = list(mask.resample(rule=self.freq_str, level=self.index_datetime))
            x_windows_mask_processed = []
            for x_m in x_masks:
                x_m = x_m[1]
                x_m_mask = x_m.to_numpy()

                x_m_shape = x_m.shape
                if x_m_shape[0] < self.size_window:
                    x_m_mask = np.pad(x_m_mask, pad_width=npad, mode="constant", constant_values=1)
                x_windows_mask_processed.append(x_m_mask)

        return np.array(x_windows_processed), np.array(x_windows_mask_processed)
