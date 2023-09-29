from typing import Dict, List, Callable, Tuple
from typing_extensions import Self
import math
import numpy as np
import pandas as pd
import time
from datetime import timedelta
from tqdm import tqdm
import gc

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing

from qolmat.imputations.diffusions.base import AutoEncoder, ResidualBlock, ResidualBlockTS
from qolmat.imputations.diffusions.utils import get_num_params
from qolmat.benchmark import missing_patterns, metrics


class TabDDPM:
    """Diffusion model for tabular data based on
    Denoising Diffusion Probabilistic Models (DDPM) of
    Ho et al., 2020 (https://arxiv.org/abs/2006.11239),
    Tashiro et al., 2021 (https://arxiv.org/abs/2107.03502).
    This implementation follows the implementations found in
    https://github.com/quickgrid/pytorch-diffusion/tree/main,
    https://github.com/ermongroup/CSDI/tree/main
    """

    def __init__(
        self,
        num_noise_steps: int = 50,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        lr: float = 0.001,
        ratio_nan: float = 0.1,
        dim_embedding: int = 128,
        num_blocks: int = 1,
        p_dropout: float = 0.0,
        num_sampling: int = 1,
        is_clip: bool = True,
    ):
        """Diffusion model for tabular data based on
        Denoising Diffusion Probabilistic Models (DDPM) of
        Ho et al., 2020 (https://arxiv.org/abs/2006.11239),
        Tashiro et al., 2021 (https://arxiv.org/abs/2107.03502).
        This implementation follows the implementations found in
        https://github.com/quickgrid/pytorch-diffusion/tree/main,
        https://github.com/ermongroup/CSDI/tree/main

        Parameters
        ----------
        num_noise_steps : int, optional
            Number of noise steps, by default 50
        beta_start : float, optional
            Range of beta (noise scale value), by default 1e-4
        beta_end : float, optional
            Range of beta (noise scale value), by default 0.02
        lr : float, optional
            Learning rate, by default 0.001
        ratio_nan : float, optional
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
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Hyper-parameters for DDPM
        # Section 2, equation 1, num_noise_steps is T.
        self.num_noise_steps = num_noise_steps

        # Section 2, equation 4 and near explation for alpha, alpha hat, beta.
        self.beta_start = beta_start
        self.beta_end = beta_end
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

        # Hyper-parameters for bulding and training the model
        self.loss_func = torch.nn.MSELoss(reduction="none")

        self.lr = lr
        self.ratio_nan = ratio_nan
        self.num_noise_steps = num_noise_steps
        self.dim_embedding = dim_embedding
        self.num_blocks = num_blocks
        self.p_dropout = p_dropout
        self.num_sampling = num_sampling
        self.is_clip = is_clip

        self.normalizer_x = preprocessing.StandardScaler()

    def _q_sample(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Section 3.2, algorithm 1 formula implementation. Forward process, defined by `q`.
        Found in section 2. `q` gradually adds gaussian noise according to variance schedule. Also,
        can be seen on figure 2.
        Ho et al., 2020 (https://arxiv.org/abs/2006.11239)

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

        epsilon = torch.randn_like(x, device=self.device)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def _set_eps_model(self) -> None:
        self._eps_model = AutoEncoder(
            num_noise_steps=self.num_noise_steps,
            dim_input=self.dim_input,
            residual_block=ResidualBlock(self.dim_embedding, self.dim_embedding, self.p_dropout),
            dim_embedding=self.dim_embedding,
            num_blocks=self.num_blocks,
            p_dropout=self.p_dropout,
        ).to(self.device)

        self.optimiser = torch.optim.Adam(self._eps_model.parameters(), lr=self.lr)

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
            print(f"Num params of {self.__class__.__name__}: {self.num_params}")
        if self.print_valid and epoch % print_step == 0:
            string_valid = f"Epoch {epoch}: "
            for s in self.summary:
                string_valid += f" {s}={round(self.summary[s][epoch], self.round)}"
            # string_valid += f" | in {round(time_duration, 3)} secs"
            remaining_duration = np.mean(self.time_durations) * (self.epochs - epoch)
            string_valid += f" | remaining {timedelta(seconds=remaining_duration)}"
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
                    if len(x_batch.size()) == 3:
                        # Data are splited into chunks (i.e., Time-series data), a window of rows
                        # is processed.
                        sqrt_alpha_t = self.sqrt_alpha[t].view(-1, 1, 1)
                        beta_t = self.beta[t].view(-1, 1, 1)
                        sqrt_one_minus_alpha_hat_t = self.sqrt_one_minus_alpha_hat[t].view(
                            -1, 1, 1
                        )
                        epsilon_t = self.std_beta[t].view(-1, 1, 1)
                    else:
                        # Each row of data is separately processed.
                        sqrt_alpha_t = self.sqrt_alpha[t].view(-1, 1)
                        beta_t = self.beta[t].view(-1, 1)
                        sqrt_one_minus_alpha_hat_t = self.sqrt_one_minus_alpha_hat[t].view(-1, 1)
                        epsilon_t = self.std_beta[t].view(-1, 1)

                    random_noise = torch.randn_like(noise) if i > 1 else torch.zeros_like(noise)

                    noise = (
                        (1 / sqrt_alpha_t)
                        * (
                            noise
                            - ((beta_t / sqrt_one_minus_alpha_hat_t) * self._eps_model(noise, t))
                        )
                    ) + (epsilon_t * random_noise)
                    noise = mask_x_batch * x_batch + (1.0 - mask_x_batch) * noise

                # Generate data output, this activation function depends on normalizer_x
                x_out = noise.detach().cpu().numpy()
                outputs.append(x_out)

        outputs = np.concatenate(outputs)
        return np.array(outputs)

    def _eval(
        self,
        x: np.ndarray,
        x_mask_obs: np.ndarray,
        x_df: pd.DataFrame,
        x_mask_obs_df: pd.DataFrame,
        x_indices: List,
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
        x_indices : List
            List of row indices for batches

        Returns
        -------
        Dict
            Scores
        """

        list_x_imputed = []
        for i in tqdm(range(self.num_sampling), disable=True, leave=False):
            x_imputed = self._impute(x, x_mask_obs)
            list_x_imputed.append(x_imputed)
        x_imputed = np.mean(np.array(list_x_imputed), axis=0)

        x_out = self._process_reversely_data(x_imputed, x_df, x_indices)

        if self.is_clip:
            for col, interval in self.interval_x.items():
                x_out[col] = np.clip(x_out[col], interval[0], interval[1])

        x_final = x_df.copy()
        x_final.loc[x_out.index] = x_out.loc[x_out.index]

        x_mask_imputed_df = ~x_mask_obs_df
        columns_with_True = x_mask_imputed_df.columns[(x_mask_imputed_df == True).any()]
        scores = {}
        for metric in self.metrics_valid:
            scores[metric.__name__] = metric(
                x_df[columns_with_True],
                x_final[columns_with_True],
                x_mask_imputed_df[columns_with_True],
            ).mean()
        return scores

    def _process_data(
        self, x: pd.DataFrame, mask: pd.DataFrame = None, is_training: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, List]:
        """Pre-process data

        Parameters
        ----------
        x : pd.DataFrame
            Input data
        mask : pd.DataFrame, optional
            Observed value mask, by default None
        is_training : bool
            Processing data for training step

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Data and mask pre-processed
        """
        if is_training:
            self.normalizer_x.fit(x.values)
        x_windows_processed = self.normalizer_x.transform(x.fillna(x.mean()).values)
        x_windows_mask_processed = ~x.isna().to_numpy()
        if mask is not None:
            x_windows_mask_processed = mask.to_numpy()

        return x_windows_processed, x_windows_mask_processed, list(x.index)

    def _process_reversely_data(
        self, x_imputed: np.ndarray, x_input: pd.DataFrame, x_indices: List
    ):
        x_normalized = self.normalizer_x.inverse_transform(x_imputed)
        x_normalized = x_normalized[: x_input.shape[0]]
        x_out = pd.DataFrame(x_normalized, columns=self.columns, index=x_input.index)

        x_final = x_input.copy()
        x_final.loc[x_out.index] = x_out.loc[x_out.index]

        return x_final

    def fit(
        self,
        x: pd.DataFrame,
        epochs: int = 10,
        batch_size: int = 100,
        print_valid: bool = False,
        x_valid: pd.DataFrame = None,
        metrics_valid: Tuple[Callable, ...] = (
            metrics.mean_absolute_error,
            metrics.dist_wasserstein,
        ),
        round: int = 10,
        cols_imputed: Tuple[str, ...] = (),
    ) -> Self:

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
        metrics_valid : Tuple[Callable, ...], optional
            Set of validation metrics, by default ( metrics.mean_absolute_error,
            metrics.dist_wasserstein )
        round : int, optional
            Number of decimal places to round to, for better displaying model
            performance, by default 10
        cols_imputed : Tuple[str, ...], optional
            Name of columns that need to be imputed, by default ()

        Raises
        ------
        ValueError
            Batch size is larger than data size
        Returns
        -------
        Self
            Return Self
        """
        self.dim_input = len(x.columns)
        self.epochs = epochs
        self.batch_size = batch_size
        self.columns = x.columns.tolist()
        self.metrics_valid = metrics_valid
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

        # x_mask: 1 for observed values, 0 for nan
        x_processed, x_mask, _ = self._process_data(x, is_training=True)

        if self.batch_size > x_processed.shape[0]:
            raise ValueError(
                f"Batch size {self.batch_size} larger than size of pre-processed x"
                + f" size={x_processed.shape[0]}. Please reduce batch_size."
                + " In the case of TabDDPMTS, you can also reduce freq_str."
            )

        if x_valid is not None:
            # We reuse the UniformHoleGenerator to generate artificial holes (with one mask)
            # in validation dataset
            x_valid_mask = missing_patterns.UniformHoleGenerator(
                n_splits=1, ratio_masked=self.ratio_nan
            ).split(x_valid)[0]
            # x_valid_obs_mask is the mask for observed values
            x_valid_obs_mask = ~x_valid_mask
            (
                x_processed_valid,
                x_processed_valid_obs_mask,
                x_processed_valid_indices,
            ) = self._process_data(x_valid, x_valid_obs_mask, is_training=False)

        x_tensor = torch.from_numpy(x_processed).float().to(self.device)
        x_mask_tensor = torch.from_numpy(x_mask).float().to(self.device)
        dataloader = DataLoader(
            TensorDataset(x_tensor, x_mask_tensor),
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
        )

        self._set_eps_model()
        self.num_params: int = get_num_params(self._eps_model)
        self.summary: Dict[str, List] = {
            "epoch_loss": [],
        }

        for epoch in range(epochs):
            loss_epoch = 0.0
            time_start = time.time()
            self._eps_model.train()
            for id_batch, (x_batch, mask_x_batch) in enumerate(dataloader):
                mask_obs_rand = torch.FloatTensor(mask_x_batch.size()).uniform_() > self.ratio_nan
                for col in self.cols_idx_not_imputed:
                    mask_obs_rand[:, col] = 0.0
                mask_x_batch = mask_x_batch * mask_obs_rand.to(self.device)

                self.optimiser.zero_grad()
                t = torch.randint(
                    low=1,
                    high=self.num_noise_steps,
                    size=(x_batch.size(dim=0), 1),
                    device=self.device,
                )
                x_batch_t, noise = self._q_sample(x=x_batch, t=t)
                predicted_noise = self._eps_model(x=x_batch_t, t=t)
                loss = (self.loss_func(predicted_noise, noise) * mask_x_batch).mean()
                loss.backward()
                self.optimiser.step()
                loss_epoch += loss.item()

            self.summary["epoch_loss"].append(np.mean(loss_epoch))
            if x_valid is not None:
                self._eps_model.eval()
                dict_loss = self._eval(
                    x_processed_valid,
                    x_processed_valid_obs_mask,
                    x_valid,
                    x_valid_obs_mask,
                    x_processed_valid_indices,
                )
                for name_loss, value_loss in dict_loss.items():
                    if name_loss not in self.summary:
                        self.summary[name_loss] = [value_loss]
                    else:
                        self.summary[name_loss].append(value_loss)
            time_duration = time.time() - time_start
            self._print_valid(epoch, time_duration)

        return self

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
        self._eps_model.eval()

        x_processed, x_mask, x_indices = self._process_data(x, is_training=False)

        list_x_imputed = []
        for i in tqdm(range(self.num_sampling), leave=False):
            x_imputed = self._impute(x_processed, x_mask)
            list_x_imputed.append(x_imputed)
        x_imputed = np.mean(np.array(list_x_imputed), axis=0)

        x_out = self._process_reversely_data(x_imputed, x, x_indices)

        if self.is_clip:
            for col, interval in self.interval_x.items():
                x_out[col] = np.clip(x_out[col], interval[0], interval[1])
        x_out = x.fillna(x_out)
        return x_out


class TsDDPM(TabDDPM):
    """Diffusion model for time-series data based on
    Denoising Diffusion Probabilistic Models (DDPMs) of
    Ho et al., 2020 (https://arxiv.org/abs/2006.11239),
    Tashiro et al., 2021 (https://arxiv.org/abs/2107.03502).
    This implementation follows the implementations found in
    https://github.com/quickgrid/pytorch-diffusion/tree/main,
    https://github.com/ermongroup/CSDI/tree/main
    """

    def __init__(
        self,
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
        is_rolling: bool = False,
    ):
        """Diffusion model for time-series data based on the works of
        Ho et al., 2020 (https://arxiv.org/abs/2006.11239),
        Tashiro et al., 2021 (https://arxiv.org/abs/2107.03502).
        This implementation follows the implementations found in
        https://github.com/quickgrid/pytorch-diffusion/tree/main,
        https://github.com/ermongroup/CSDI/tree/main

        Parameters
        ----------
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
        is_rolling : bool, optional
            Use pandas.DataFrame.rolling for preprocessing data, by default False
        """
        super().__init__(
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
        self.is_rolling = is_rolling

    def _q_sample(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

        sqrt_alpha_hat = self.sqrt_alpha_hat[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_hat[t].view(-1, 1, 1)

        epsilon = torch.randn_like(x, device=self.device)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def _set_eps_model(self):
        self._eps_model = AutoEncoder(
            num_noise_steps=self.num_noise_steps,
            dim_input=self.dim_input,
            residual_block=ResidualBlockTS(
                self.dim_embedding,
                self.size_window,
                self.dim_embedding,
                self.dim_feedforward,
                self.nheads_feature,
                self.nheads_time,
                self.num_layers_transformer,
            ),
            dim_embedding=self.dim_embedding,
            num_blocks=self.num_blocks,
            p_dropout=self.p_dropout,
        ).to(self.device)

        self.optimiser = torch.optim.Adam(self._eps_model.parameters(), lr=self.lr)

    def _process_data(
        self, x: pd.DataFrame, mask: pd.DataFrame = None, is_training: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, List]:
        """Pre-process data

        Parameters
        ----------
        x : pd.DataFrame
            Input data
        mask : pd.DataFrame, optional
            Observed value mask, by default None
        is_training : bool
            Processing data for training step

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Data and mask pre-processed
        """
        if is_training:
            self.normalizer_x.fit(x.values)

        x_windows: List = []
        x_windows_indices: List = []
        columns_index = [col for col in x.index.names if col != self.index_datetime]
        if is_training:
            if self.is_rolling:
                if self.print_valid:
                    print(
                        "Preprocessing data with sliding window (pandas.DataFrame.rolling)"
                        + " can require more times than usual. Please be patient!"
                    )
                if len(columns_index) == 0:
                    x_windows = x.rolling(window=self.freq_str)
                else:
                    columns_index_ = columns_index[0] if len(columns_index) == 1 else columns_index
                    for x_group in tqdm(x.groupby(by=columns_index_), disable=True, leave=False):
                        x_windows += list(
                            x_group[1].droplevel(columns_index).rolling(window=self.freq_str)
                        )
            else:
                for x_w in x.resample(rule=self.freq_str, level=self.index_datetime):
                    x_windows.append(x_w[1])
        else:
            if self.is_rolling:
                if len(columns_index) == 0:
                    indices_nan = x.loc[x.isna().any(axis=1), :].index
                    x_group_rolling = x.rolling(window=self.freq_str)
                    for x_rolling in x_group_rolling:
                        if x_rolling.index[-1] in indices_nan:
                            x_windows.append(x_rolling)
                            x_windows_indices.append(x_rolling.index)
                else:
                    columns_index_ = columns_index[0] if len(columns_index) == 1 else columns_index
                    for x_group in tqdm(x.groupby(by=columns_index_), disable=True, leave=False):
                        x_group_index = [x_group[0]] if len(columns_index) == 1 else x_group[0]
                        x_group_value = x_group[1].droplevel(columns_index)
                        indices_nan = x_group_value.loc[x_group_value.isna().any(axis=1), :].index
                        x_group_rolling = x_group_value.rolling(window=self.freq_str)
                        for x_rolling in x_group_rolling:
                            if x_rolling.index[-1] in indices_nan:
                                x_windows.append(x_rolling)
                                x_rolling_ = x_rolling.copy()
                                for idx, col in enumerate(columns_index):
                                    x_rolling_[col] = x_group_index[idx]
                                x_rolling_ = x_rolling_.set_index(columns_index, append=True)
                                x_rolling_ = x_rolling_.reorder_levels(x.index.names)
                                x_windows_indices.append(x_rolling_.index)
            else:
                for x_w in x.resample(rule=self.freq_str, level=self.index_datetime):
                    x_windows.append(x_w[1])
                    x_windows_indices.append(x_w[1].index)

        x_windows_processed = []
        x_windows_mask_processed = []
        self.size_window = np.max([w.shape[0] for w in x_windows])
        for x_w in x_windows:
            x_w_fillna = x_w.fillna(method="bfill")
            x_w_fillna = x_w_fillna.fillna(x.mean())
            x_w_norm = self.normalizer_x.transform(x_w_fillna.values)
            x_w_mask = ~x_w.isna().to_numpy()

            x_w_shape = x_w.shape
            if x_w_shape[0] < self.size_window:
                npad = [(0, self.size_window - x_w_shape[0]), (0, 0)]
                x_w_norm = np.pad(x_w_norm, pad_width=npad, mode="wrap")
                x_w_mask = np.pad(x_w_mask, pad_width=npad, mode="constant", constant_values=1)

            x_windows_processed.append(x_w_norm)
            x_windows_mask_processed.append(x_w_mask)

        if mask is not None:
            x_windows_mask_processed = []
            for x_window_indices in x_windows_indices:
                x_m = mask.loc[x_window_indices]
                x_m_mask = x_m.to_numpy()

                x_m_shape = x_m.shape
                if x_m_shape[0] < self.size_window:
                    npad = [(0, self.size_window - x_m_shape[0]), (0, 0)]
                    x_m_mask = np.pad(x_m_mask, pad_width=npad, mode="constant", constant_values=1)
                x_windows_mask_processed.append(x_m_mask)

        return np.array(x_windows_processed), np.array(x_windows_mask_processed), x_windows_indices

    def _process_reversely_data(
        self, x_imputed: np.ndarray, x_input: pd.DataFrame, x_indices: List
    ):
        x_imputed_nan_only = []
        x_indices_nan_only = []
        for x_imputed_batch, x_indices_batch in zip(x_imputed, x_indices):
            imputed_index = x_indices_batch.shape[0] - 1
            x_imputed_nan_only.append(x_imputed_batch[imputed_index])
            x_indices_nan_only.append(x_indices_batch[imputed_index])

        if len(np.shape(x_indices_nan_only)) == 1:
            x_out_index = pd.Index(x_indices_nan_only, name=x_input.index.names[0])
        else:
            x_out_index = pd.MultiIndex.from_tuples(x_indices_nan_only, names=x_input.index.names)
        x_normalized = self.normalizer_x.inverse_transform(x_imputed_nan_only)
        x_out = pd.DataFrame(
            x_normalized,
            columns=self.columns,
            index=x_out_index,
        )

        x_final = x_input.copy()
        x_final.loc[x_out.index] = x_out.loc[x_out.index]

        return x_final

    def fit(
        self,
        x: pd.DataFrame,
        epochs: int = 10,
        batch_size: int = 100,
        print_valid: bool = False,
        x_valid: pd.DataFrame = None,
        metrics_valid: Tuple[Callable, ...] = (
            metrics.mean_absolute_error,
            metrics.dist_wasserstein,
        ),
        round: int = 10,
        cols_imputed: Tuple[str, ...] = (),
        index_datetime: str = "",
        freq_str: str = "1D",
    ) -> Self:
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
        metrics_valid : Tuple[Callable, ...], optional
            Set of validation metrics, by default ( metrics.mean_absolute_error,
            metrics.dist_wasserstein )
        round : int, optional
            Number of decimal places to round to, by default 10
        cols_imputed : Tuple[str, ...], optional
            Name of columns that need to be imputed, by default ()
        index_datetime : str
            Name of datetime-like index
        freq_str : str
            Frequency string of DateOffset of Pandas
        Raises
        ------
        ValueError
            Batch size is larger than data size
        Returns
        -------
        Self
            Return Self
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
            metrics_valid,
            round,
            cols_imputed,
        )
        return self
