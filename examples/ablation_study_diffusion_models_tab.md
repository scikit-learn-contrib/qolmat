---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: env_qolmat_dev
    language: python
    name: python3
---

# Ablation study for diffusion models

```python
cd ../
```

```python
%reload_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np

from qolmat.imputations import imputers
from qolmat.benchmark import comparator, missing_patterns
import qolmat.benchmark.metrics as mtr
from qolmat.utils import data, plot

from matplotlib import pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import inspect
import pickle

from sklearn.linear_model import LinearRegression

```

```python
dict_metrics = {
    "mae": mtr.mean_absolute_error,
    "wasser": mtr.wasserstein_distance,
}

def plot_errors(df_original, dfs_imputed, dfs_mask, dict_metrics, **kwargs):
    dict_errors_df = {}
    for ind, (name, df_imputed) in enumerate(list(dfs_imputed.items())):
        dict_errors_mtr = {}
        for name_metric in dict_metrics:
            metric_args = list(inspect.signature(dict_metrics[name_metric]).parameters)
            args_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in metric_args}
            if "wasser" in name_metric:
                dict_errors_mtr[name_metric] = dict_metrics[name_metric](df_original, df_imputed, dfs_mask)
            else:
                dict_errors_mtr[name_metric] = dict_metrics[name_metric](df_original, df_imputed, dfs_mask, **args_dict)
        dict_errors_df[name] = pd.concat(dict_errors_mtr.values(), keys=dict_errors_mtr.keys())

    return pd.DataFrame(dict_errors_df)

def plot_summaries(summaries, display='epoch_loss', xaxis_title='epoch', height=500):
    fig = go.Figure()

    for ind, (name, values) in enumerate(list(summaries.items())):
        values_selected = values[display]
        fig.add_trace(go.Scatter(x=list(range(len(values_selected))), y=values_selected, mode='lines', name=name))

    fig.update_layout(xaxis_title=xaxis_title,
                      yaxis_title=display, height=height)

    fig.update_yaxes(type="log")

    return fig

def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    return params
```

## **I. Load data**

```python
from sklearn.datasets import fetch_california_housing, fetch_covtype

df_data_raw = fetch_covtype(as_frame=True).data
display(df_data_raw.describe())
```

```python
df_data_raw_sample = df_data_raw.iloc[:, :10].sample(20000)
df_data_raw_sample = df_data_raw_sample.sample(frac = 1)
df_data_raw_sample = df_data_raw_sample.reset_index(drop=True)
cols_to_impute = df_data_raw_sample.columns.to_list()
ratio_masked = 0.5

df_mask = missing_patterns.UniformHoleGenerator(n_splits=1, subset=cols_to_impute, ratio_masked=ratio_masked).generate_mask(df_data_raw_sample)

df_data = df_data_raw_sample[df_mask]

display(df_data.describe())

print("Number of nans:", df_data.isna().sum().sum())
print("Number of rows with nans:", df_data.isna().sum(axis=1).size)
print("Number of rows without nans:", df_data.notna().sum(axis=1).size)
print("Number of nans in each column:")
display(df_data.isna().sum())

```

## Baseline imputers

```python
imputer_mean = imputers.ImputerMean()
imputer_median = imputers.ImputerMedian()
imputer_mode = imputers.ImputerMode()
imputer_locf = imputers.ImputerLOCF()
imputer_nocb = imputers.ImputerNOCB()
imputer_interpol = imputers.ImputerInterpolation(method="linear")
imputer_spline = imputers.ImputerInterpolation(method="spline", order=2)
imputer_shuffle = imputers.ImputerShuffle()
imputer_residuals = imputers.ImputerResiduals(period=7, model_tsa="additive", extrapolate_trend="freq", method_interpolation="linear")

imputer_rpca = imputers.ImputerRPCA(columnwise=True, period=365, max_iter=200, tau=2, lam=.3)
imputer_rpca_opti = imputers.ImputerRPCA(columnwise=True, period=365, max_iter=100)

imputer_ou = imputers.ImputerEM(model="multinormal", method="sample", max_iter_em=34, n_iter_ou=15, dt=1e-3)
imputer_tsou = imputers.ImputerEM(model="VAR1", method="sample", max_iter_em=34, n_iter_ou=15, dt=1e-3)
imputer_tsmle = imputers.ImputerEM(model="VAR1", method="mle", max_iter_em=34, n_iter_ou=15, dt=1e-3)

imputer_knn = imputers.ImputerKNN(k=10)
imputer_mice = imputers.ImputerMICE(estimator=LinearRegression(), sample_posterior=False, max_iter=100, missing_values=np.nan)
imputer_regressor = imputers.ImputerRegressor(estimator=LinearRegression())

dict_imputers_baseline = {
    "mean": imputer_mean,
    "median": imputer_median,
    "mode": imputer_mode,
    # "interpolation": imputer_interpol,
    # "spline": imputer_spline,
    "shuffle": imputer_shuffle,
    # "residuals": imputer_residuals,
    "OU": imputer_ou,
    "TSOU": imputer_tsou,
    "TSMLE": imputer_tsmle,
    # "RPCA": imputer_rpca,
    # "RPCA_opti": imputer_rpca_opti,
    # "locf": imputer_locf,
    # "nocb": imputer_nocb,
    "knn": imputer_knn,
    "ols": imputer_regressor,
    "mice_ols": imputer_mice,
}

n_imputers = len(dict_imputers_baseline)
```

## Baseline NN


### ImputerRegressor: feed-forward NN regressor

```python
from qolmat.imputations import imputers_pytorch
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing

class feedforward_regressor:
    def __init__(self, input_size):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.estimator = torch.nn.Sequential(
            torch.nn.Linear(input_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        ).to(self.device)
        self.loss_func = torch.nn.MSELoss()
        self.optimiser = torch.optim.Adam(self.estimator.parameters(), lr = 0.0001)

    def fit(self, x, y, epochs=10, batch_size=100):
        x = x.fillna(x.mean())
        self.normalizer_x = preprocessing.StandardScaler()
        x_normalized = self.normalizer_x.fit_transform(x.values)

        self.normalizer_y = preprocessing.StandardScaler()
        y_normalized = self.normalizer_y.fit_transform(np.expand_dims(y.values, axis=1))

        self.estimator.train()
        x_tensor = torch.from_numpy(x_normalized).float()
        y_tensor = torch.from_numpy(y_normalized).float()
        dataloader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size, shuffle=True)
        dataset_size = len(dataloader.dataset)
        for epoch in range(epochs):
            loss_epoch = 0.
            for id_batch, (x_batch, y_batch) in enumerate(dataloader):
                self.optimiser.zero_grad()
                outputs = self.estimator.forward(x_batch.to(self.device))
                loss = self.loss_func(outputs, y_batch.to(self.device))
                loss.backward()
                self.optimiser.step()
                loss_epoch += loss.item()
            # if epoch%20==0:
            #     print(f"Epoch {epoch}, Loss = {loss_epoch/dataset_size}")

    def predict(self, x):
        self.estimator.eval()
        x = x.fillna(x.mean())
        x_normalized = self.normalizer_x.transform(x.values)
        inputs = torch.from_numpy(x_normalized).float().to(self.device)
        outputs_normalized = self.estimator.forward(inputs).detach().cpu().numpy()
        outputs = self.normalizer_y.inverse_transform(outputs_normalized).squeeze()
        return outputs
```

### Auto-encoder

```python
class AutoEncoderSimple(torch.nn.Module):
    def __init__(self, input_size):
        super(AutoEncoderSimple, self).__init__()
        self.layer_x_1 = torch.nn.Linear(input_size, 256)
        self.layer_x_2 = torch.nn.Linear(256, 256)

        self.layer_out_1 = torch.nn.Linear(256, 256)
        self.layer_out_2 = torch.nn.Linear(256, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_1 = torch.nn.functional.relu(self.layer_x_1(x))
        x_2 = torch.nn.functional.relu(self.layer_x_2(x_1))

        out_1 = torch.nn.functional.relu(self.layer_out_1(x_2))
        out_2 = self.layer_out_2(out_1)
        return out_2

class AutoEncoderImputer:
    def __init__(self, input_size):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.estimator = AutoEncoderSimple(input_size).to(self.device)
        self.loss_func = torch.nn.MSELoss()
        self.optimiser = torch.optim.Adam(self.estimator.parameters(), lr = 0.0001)

    def fit(self, x, epochs=10, batch_size=100):
        x = x.fillna(x.mean())
        self.normalizer_x = preprocessing.StandardScaler()
        x_normalized = self.normalizer_x.fit_transform(x.values)

        self.estimator.train()
        x_tensor = torch.from_numpy(x_normalized).float()
        dataloader = DataLoader(TensorDataset(x_tensor), batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            loss_epoch = 0.
            for id_batch, x_batch in enumerate(dataloader):
                x_batch = x_batch[0].to(self.device)
                self.optimiser.zero_grad()
                outputs = self.estimator.forward(x_batch)
                loss = self.loss_func(outputs, x_batch)
                loss.backward()
                self.optimiser.step()
                loss_epoch += loss.item()
            # if epoch%20==0:
            #     print(f"Epoch {epoch}, Loss = {loss_epoch/dataset_size}")

    def predict(self, x):
        self.estimator.eval()
        x_input = x.fillna(x.mean())
        x_normalized = self.normalizer_x.transform(x_input.values)

        inputs = torch.from_numpy(x_normalized).float().to(self.device)
        outputs_normalized = self.estimator.forward(inputs).detach().cpu().numpy()

        outputs_real = self.normalizer_x.inverse_transform(outputs_normalized).squeeze()
        outputs = pd.DataFrame(outputs_real, columns=x.columns, index=x.index)
        
        x = x.fillna(outputs)
        return outputs
```

### Simple VAE

```python
# from functools import partial

# import torch
# import torch.nn as nn
# import torch.distributions as dists
# from torch.nn.functional import softplus
# from torch.distributions import constraints
# from torch.distributions.utils import logits_to_probs

# import pytorch_lightning as pl


# def init_weights(m, gain=1.):
#     if type(m) == nn.Linear:
#         nn.init.xavier_uniform_(m.weight, gain=gain)
#         m.bias.data.fill_(0.01)


# class Encoder(nn.Module):
#     def __init__(self, input_size, latent_size, hidden_size, dropout):
#         super().__init__()

#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Dropout(p=dropout), nn.BatchNorm1d(input_size),
#             nn.Linear(input_size, hidden_size), nn.Tanh(),
#             nn.Linear(hidden_size, hidden_size), nn.Tanh(),
#             nn.Linear(hidden_size, hidden_size), nn.Tanh(),
#         )

#         self.z_loc = nn.Linear(hidden_size, latent_size)
#         self.z_log_scale = nn.Linear(hidden_size, latent_size)

#         self.encoder.apply(partial(init_weights, gain=nn.init.calculate_gain('tanh')))
#         self.z_loc.apply(init_weights)
#         self.z_log_scale.apply(init_weights)

#     def q_z(self, loc, logscale):
#         scale = softplus(logscale)
#         return dists.Normal(loc, scale)

#     def forward(self, x):
#         h = self.encoder(x)

#         loc = self.z_loc(h)  # constraints.real
#         log_scale = self.z_log_scale(h)  # constraints.real

#         return loc, log_scale


# class VAE(torch.nn.Module):
#     def __init__(self, input_size, latent_size, hidden_size, dropout):
#         super().__init__()
#         self.samples = 1

#         # Prior
#         self.prior_z_loc = torch.zeros(latent_size)
#         self.prior_z_scale = torch.ones(latent_size)

#         # Encoder
#         self.encoder = Encoder(input_size, latent_size, hidden_size, dropout)

#         # Decoder
#         self.decoder = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size), nn.Tanh(),
#             nn.Linear(hidden_size, hidden_size), nn.Tanh(),
#             nn.Linear(hidden_size, input_size), nn.Tanh(),
#         )

#     def get_params_from_data(self, x: pd.DataFrame):
#         self.x_loc = x.mean().to_numpy()
#         self.x_std = x.std().to_numpy()

#     def _step(self, batch):
#         x, mask, _ = batch

#         z_params = self.encoder(x if mask is None else x * mask.float())
#         z = self.encoder.q_z(*z_params).rsample([self.samples])

#         y = self.decoder(z)

#         log_px_z = dists.Normal(self.x_loc, self.x_std).to

#         log_pz = dists.Normal(self.prior_z_loc, self.prior_z_scale).log_prob(z).sum(dim=-1)  # samples x batch_size
#         log_qz_x = self.encoder.q_z(*z_params).log_prob(z).sum(dim=-1)  # samples x batch_size
#         kl_z = log_qz_x - log_pz

#         elbo = sum(log_px_z) - kl_z
#         loss = -elbo.squeeze(dim=0).sum(dim=0)
#         assert loss.size() == torch.Size([])

#         logs = dict()
#         logs['loss'] = loss / x.size(0)

#         return loss, logs

#     def training_step(self, batch, batch_idx):
#         loss, logs = self._step(batch, batch_idx)
#         self.log_dict({f'training/{k}': v for k, v in logs.items()})
#         return loss

#     def validation_step(self, batch, batch_idx):
#         loss, logs = self._step(batch, batch_idx)
#         self.log_dict({f'validation/{k}': v for k, v in logs.items()})
#         return loss

#     def forward(self, batch, mode=True):
#         x, mask, _ = batch

#         z_params = self.encoder(x if mask is None else x * mask.float())
#         if mode:
#             z = z_params[0]  # Mode of a Normal distribution
#         else:
#             z = self.encoder.q_z(*z_params).sample()

#         y = self.decoder(z)

#         return y

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam([
#             {'params': self.parameters(), 'lr': self.hparams.learning_rate},
#         ])

#         return optimizer
```

## Diffusion models


### ImputerGenerativeModel: Denoising Diffusion Probabilistic Models - [DDPM](https://arxiv.org/abs/2006.11239)

- Forward: $x_0 \rightarrow x_1 \rightarrow \dots \rightarrow x_{T-1} \rightarrow x_T$
    - $q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)I)$
    - $x_t = \bar{\alpha}_t \times x_0 + \sqrt{1-\bar{\alpha}_t} \times \epsilon$ where
        - $\epsilon \sim \mathcal{N}(0,I)$
        - $\bar{\alpha}_t = \sum^t_{t=0} \alpha_t$
        - $\alpha$: noise scheduler

- Reserve: $x_T \rightarrow x_{t-1} \rightarrow \dots \rightarrow x_1 \rightarrow x_0$
    - $p_\theta (x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta (x_t, t), \Sigma_\theta (x_t, t))$
    - $x_{t-1} = \frac{1}{\sqrt{\alpha_t}} (x_t - \frac{1 - \alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)) + \sigma_t z$ where
        - $\epsilon$: our model to predict noise at t
        - $z \sim \mathcal{N}(0,I)$

- Objective function:
    - $E_{t \sim \mathcal{U} [[1,T]], x_0 \sim q(x_0), \epsilon \sim \mathcal{N}(0,I)} [|| \epsilon - \epsilon_\theta(x_t, t)||^2]$

```python
from qolmat.imputations import imputers_pytorch
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing
from typing import Tuple

```

#### Simple TabDDPM

- Training:
    - Fill real nan with mean
    - Time step t as a float
- Inference:
    - $\epsilon \rightarrow \hat{x}_0$
    - Fill nan with $\hat{x}_0$

```python
class AutoEncoder(torch.nn.Module):
    def __init__(self, input_size):
        super(AutoEncoder, self).__init__()
        self.layer_x_1 = torch.nn.Linear(input_size, 256)
        self.layer_x_2 = torch.nn.Linear(256, 256)

        self.layer_t_1 = torch.nn.Linear(1, 256)
        self.layer_t_2 = torch.nn.Linear(256, 256)

        self.layer_out_1 = torch.nn.Linear(256, 256)
        self.layer_out_2 = torch.nn.Linear(256, input_size)

    def forward(self, x: torch.Tensor, t: torch.LongTensor) -> torch.Tensor:
        x_1 = torch.nn.functional.relu(self.layer_x_1(x))
        x_2 = torch.nn.functional.relu(self.layer_x_2(x_1))

        t_1 = torch.nn.functional.relu(self.layer_t_1(t.float()))
        t_2 = torch.nn.functional.relu(self.layer_t_2(t_1))

        cat_x_t = x_2 + t_2

        out_1 = torch.nn.functional.relu(self.layer_out_1(cat_x_t))
        out_2 = self.layer_out_2(out_1)
        return out_2

class TabDDPM:
    def __init__(self, input_size, noise_steps, beta_start: float = 1e-4, beta_end: float = 0.02, lr: float = 0.0001):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.eps_model = AutoEncoder(input_size).to(self.device)
        self.loss_func = torch.nn.MSELoss(reduction='none')
        self.optimiser = torch.optim.Adam(self.eps_model.parameters(), lr = lr)

        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Section 2, equation 4 and near explation for alpha, alpha hat, beta.
        self.beta = torch.linspace(start=self.beta_start, end=self.beta_end, steps=self.noise_steps, device=self.device) # Linear noise schedule
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        # Section 3.2, algorithm 1 formula implementation. Generate values early reuse later.
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat)

        # Section 3.2, equation 2 precalculation values.
        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.std_beta = torch.sqrt(self.beta)

        #self.normalizer_x = preprocessing.MinMaxScaler()
        self.normalizer_x = preprocessing.StandardScaler()

        self.summary = {
            'epoch_loss': [],
            'eval_mae': []
        }

    def q_sample(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Section 3.2, algorithm 1 formula implementation. Forward process, defined by `q`.
        Found in section 2. `q` gradually adds gaussian noise according to variance schedule. Also,
        can be seen on figure 2.
        """
        sqrt_alpha_hat = self.sqrt_alpha_hat[t].view(-1, 1)
        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_hat[t].view(-1, 1)
        epsilon = torch.randn_like(x, device=self.device)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def fit(self, x, epochs=10, batch_size=100, x_valid=None):
        x = x.fillna(x.mean())
        x_normalized = self.normalizer_x.fit_transform(x.values)

        self.eps_model.train()
        x_tensor = torch.from_numpy(x_normalized).float()
        dataloader = DataLoader(TensorDataset(x_tensor), batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            loss_epoch = 0.
            for id_batch, x_batch in enumerate(dataloader):
                x_batch = x_batch[0].to(self.device)
                self.optimiser.zero_grad()
                t = torch.randint(low=1, high=self.noise_steps, size=(x_batch.size(dim=0), 1), device=self.device)
                x_batch_t, noise = self.q_sample(x=x_batch, t=t)
                predicted_noise = self.eps_model(x=x_batch_t, t=t)
                loss = self.loss_func(predicted_noise, noise).mean()
                loss.backward()
                self.optimiser.step()
                loss_epoch += loss.item()
            
            self.summary['epoch_loss'].append(loss.item())
            if x_valid is not None:
                self.summary['eval_mae'].append(self.eval(x_valid))

    def eval(self, x):
        mask_x = ~x.isna().to_numpy()

        x_normalized = self.normalizer_x.transform(x.values)
        x_tensor = torch.from_numpy(x_normalized).float().to(self.device)
        mask_x_tensor = torch.from_numpy(mask_x).float().to(self.device)

        with torch.no_grad():
            noise = torch.randn((x_tensor.size(dim=0), x_tensor.size(dim=1)), device=self.device)

            for i in reversed(range(1, self.noise_steps)):
                t = torch.ones((x_tensor.size(dim=0), 1), dtype=torch.long, device=self.device) * i

                sqrt_alpha_t = self.sqrt_alpha[t].view(-1, 1)
                beta_t = self.beta[t].view(-1, 1)
                sqrt_one_minus_alpha_hat_t = self.sqrt_one_minus_alpha_hat[t].view(-1, 1)
                epsilon_t = self.std_beta[t].view(-1, 1)

                random_noise = torch.randn_like(noise) if i > 1 else torch.zeros_like(noise)

                noise = ((1 / sqrt_alpha_t) * (noise - ((beta_t / sqrt_one_minus_alpha_hat_t) * self.eps_model(noise, t)))) + (epsilon_t * random_noise)

        return (torch.nn.L1Loss(reduction='none')(x_tensor, noise) * mask_x_tensor).nanmean().item()

    def predict(self, x):
        self.eps_model.eval()
        n_samples = len(x)
        n_features = x.columns.size

        with torch.no_grad():
            noise = torch.randn((n_samples, n_features), device=self.device)

            for i in reversed(range(1, self.noise_steps)):
                t = torch.ones((n_samples, 1), dtype=torch.long, device=self.device) * i

                sqrt_alpha_t = self.sqrt_alpha[t].view(-1, 1)
                beta_t = self.beta[t].view(-1, 1)
                sqrt_one_minus_alpha_hat_t = self.sqrt_one_minus_alpha_hat[t].view(-1, 1)
                epsilon_t = self.std_beta[t].view(-1, 1)

                random_noise = torch.randn_like(noise) if i > 1 else torch.zeros_like(noise)

                noise = ((1 / sqrt_alpha_t) * (noise - ((beta_t / sqrt_one_minus_alpha_hat_t) * self.eps_model(noise, t)))) + (epsilon_t * random_noise)

        outputs_normalized = noise.detach().cpu().numpy()
        outputs_real = self.normalizer_x.inverse_transform(outputs_normalized)
        outputs = pd.DataFrame(outputs_real, columns=x.columns, index=x.index)
        x = x.fillna(outputs)
        return x
```

#### TabDDPM with mask

- Training:
    - Fill real nan with mean
    - Time step t as a float
    - Add: Compute only loss values from observed data
- Inference:
    - Add: $\epsilon \rightarrow \hat{x}_t \rightarrow \hat{x}_0$ where
        - $\hat{x}_t = mask * x_0 + (1 - mask) * \hat{x}_t$
        - $mask$: 1 = observed values
    - Fill nan with $\hat{x}_0$

```python
class TabDDPM_Mask(TabDDPM):
    def __init__(self, input_size, noise_steps, beta_start: float = 1e-4, beta_end: float = 0.02, lr: float = 0.0001):
        super(TabDDPM_Mask, self).__init__(input_size, noise_steps, beta_start, beta_end, lr)

        self.eps_model = AutoEncoder(input_size).to(self.device)
        self.optimiser = torch.optim.Adam(self.eps_model.parameters(), lr = lr)
        
    def fit(self, x, epochs=10, batch_size=100, x_valid=None):
        mask_x = ~x.isna().to_numpy()
        x = x.fillna(x.mean())
        
        x_normalized = self.normalizer_x.fit_transform(x.values)

        self.eps_model.train()
        x_tensor = torch.from_numpy(x_normalized).float()
        mask_x_tensor = torch.from_numpy(mask_x)
        dataloader = DataLoader(TensorDataset(x_tensor, mask_x_tensor), batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            loss_epoch = 0.
            for id_batch, (x_batch, mask_x_batch) in enumerate(dataloader):
                x_batch = x_batch.to(self.device)
                mask_rand = torch.cuda.FloatTensor(mask_x_batch.size()).uniform_() > 0.2
                mask_x_batch = mask_x_batch.to(self.device).bool().float() * mask_rand

                self.optimiser.zero_grad()
                t = torch.randint(low=1, high=self.noise_steps, size=(x_batch.size(dim=0), 1), device=self.device)
                x_batch_t, noise = self.q_sample(x=x_batch, t=t)
                predicted_noise = self.eps_model(x=x_batch_t, t=t)
                loss = (self.loss_func(predicted_noise, noise) * mask_x_batch).mean()
                loss.backward()
                self.optimiser.step()
                loss_epoch += loss.item()

            self.summary['epoch_loss'].append(loss.item())
            if x_valid is not None:
                self.summary['eval_mae'].append(self.eval(x_valid))

    def eval(self, x):
        mask_x = ~x.isna().to_numpy()

        x_normalized = self.normalizer_x.transform(x.values)
        x_tensor = torch.from_numpy(x_normalized).float().to(self.device)
        mask_x_tensor = torch.from_numpy(mask_x).float().to(self.device)

        with torch.no_grad():
            noise = torch.randn((x_tensor.size(dim=0), x_tensor.size(dim=1)), device=self.device)

            for i in reversed(range(1, self.noise_steps)):
                t = torch.ones((x_tensor.size(dim=0), 1), dtype=torch.long, device=self.device) * i

                sqrt_alpha_t = self.sqrt_alpha[t].view(-1, 1)
                beta_t = self.beta[t].view(-1, 1)
                sqrt_one_minus_alpha_hat_t = self.sqrt_one_minus_alpha_hat[t].view(-1, 1)
                epsilon_t = self.std_beta[t].view(-1, 1)

                random_noise = torch.randn_like(noise) if i > 1 else torch.zeros_like(noise)
                noise = mask_x_tensor * x_tensor + (1.0 - mask_x_tensor) * noise
                noise = ((1 / sqrt_alpha_t) * (noise - ((beta_t / sqrt_one_minus_alpha_hat_t) * self.eps_model(noise, t)))) + (epsilon_t * random_noise)

        return (torch.nn.L1Loss(reduction='none')(x_tensor, noise) * mask_x_tensor).nanmean().item()
    
    def predict(self, x):
        self.eps_model.eval()
        n_samples = len(x)
        n_features = x.columns.size
        mask_x = ~x.isna().to_numpy()
        x_normalized = self.normalizer_x.transform(x.fillna(x.mean()).values)

        with torch.no_grad():
            noise = torch.randn((n_samples, n_features), device=self.device)
            mask_x_tensor = torch.from_numpy(mask_x).to(self.device).bool().float()
            x_tensor = torch.from_numpy(x_normalized).to(self.device).float()

            for i in reversed(range(1, self.noise_steps)):
                t = torch.ones((n_samples, 1), dtype=torch.long, device=self.device) * i

                sqrt_alpha_t = self.sqrt_alpha[t].view(-1, 1)
                beta_t = self.beta[t].view(-1, 1)
                sqrt_one_minus_alpha_hat_t = self.sqrt_one_minus_alpha_hat[t].view(-1, 1)
                epsilon_t = self.std_beta[t].view(-1, 1)

                random_noise = torch.randn_like(noise) if i > 1 else torch.zeros_like(noise)
                noise = mask_x_tensor * x_tensor + (1.0 - mask_x_tensor) * noise
                noise = ((1 / sqrt_alpha_t) * (noise - ((beta_t / sqrt_one_minus_alpha_hat_t) * self.eps_model(noise, t)))) + (epsilon_t * random_noise)

        outputs_normalized = noise.detach().cpu().numpy()
        outputs_real = self.normalizer_x.inverse_transform(outputs_normalized)
        outputs = pd.DataFrame(outputs_real, columns=x.columns, index=x.index)
        x = x.fillna(outputs)
        return x
```

#### TabDDPM with mask, more complex autoencoder

- Training:
    - Fill real nan with mean
    - Time step t as a float
    - Compute only loss values from observed data
    - Add: more complex autoencoder based on [Gorishniy et al., 2021](https://arxiv.org/abs/2106.11959) ([code](https://github.com/Yura52/rtdl))
    - Add: improve embedding of noise steps
- Inference:
    - $\epsilon \rightarrow \hat{x}_t \rightarrow \hat{x}_0$ where
        - $\hat{x}_t = mask * x_0 + (1 - mask) * \hat{x}_t$
        - $mask$: 1 = observed values
    - Fill nan with $\hat{x}_0$

```python
import math

class ResidualBlock(torch.nn.Module):
    def __init__(self, input_size, embedding_size):
        super(ResidualBlock, self).__init__()

        self.linear_in = torch.nn.Linear(input_size, embedding_size)
        self.linear_out = torch.nn.Linear(embedding_size, input_size)

    def forward(self, x, t):
        x_t = x + t
        x_t = self.linear_in(x_t)
        x_t = torch.nn.functional.relu(x_t)
        x_t = self.linear_out(x_t)
        return x + x_t, x_t

class AutoEncoder_ResNet(torch.nn.Module):
    def __init__(self, input_size, noise_steps, num_blocks=2):
        super(AutoEncoder_ResNet, self).__init__()

        embedding_size = 256
        self.layer_x = torch.nn.Linear(input_size, embedding_size)

        self.register_buffer(
            "pos_encoding",
            self._build_embedding(noise_steps, embedding_size / 2),
            persistent=False,
        )
        self.layer_t_1 = torch.nn.Linear(embedding_size, embedding_size)
        self.layer_t_2 = torch.nn.Linear(embedding_size, embedding_size)

        self.layer_out_1 = torch.nn.Linear(embedding_size, embedding_size)
        self.layer_out_2 = torch.nn.Linear(embedding_size, input_size)

        self.residual_layers = torch.nn.ModuleList([
            ResidualBlock(input_size=embedding_size, embedding_size=embedding_size) for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor, t: torch.LongTensor) -> torch.Tensor:
        # Time step embedding
        t_emb = self.pos_encoding[t].squeeze()
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
        out = self.layer_out_1(out)
        out = torch.nn.functional.relu(out)
        out = self.layer_out_2(out)

        return out
    
    def _build_embedding(self, noise_steps, dim=64):
        steps = torch.arange(noise_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table

class TabDDPM_Mask_ResNet(TabDDPM_Mask):
    def __init__(self, input_size, noise_steps, beta_start: float = 1e-4, beta_end: float = 0.02, lr: float = 0.0001, num_blocks: int = 2):
        super(TabDDPM_Mask_ResNet, self).__init__(input_size, noise_steps, beta_start, beta_end, lr)

        self.eps_model = AutoEncoder_ResNet(input_size, noise_steps=noise_steps, num_blocks=num_blocks).to(self.device)
        self.optimiser = torch.optim.Adam(self.eps_model.parameters(), lr = lr)
```

#### TabDDPM with mask, more complex, conditional autoencoder

- Training:
    - Fill real nan with mean
    - Time step t as a float
    - Compute only loss values from observed data
    - More complex autoencoder based on ResNet [Gorishniy et al., 2021](https://arxiv.org/abs/2106.11959) ([code](https://github.com/Yura52/rtdl))
    - Improve embedding of noise steps
    - Condition on $x_0$
- Inference:
    - $\epsilon \rightarrow \hat{x}_t \rightarrow \hat{x}_0$ where
        - $\hat{x}_t = mask * x_0 + (1 - mask) * \hat{x}_t$
        - $mask$: 1 = observed values
    - Fill nan with $\hat{x}_0$

```python
import math

class ResidualBlock_Cond(torch.nn.Module):
    def __init__(self, input_size, embedding_size):
        super(ResidualBlock_Cond, self).__init__()

        self.linear_in = torch.nn.Linear(input_size, embedding_size)
        self.linear_out = torch.nn.Linear(embedding_size, input_size)

    def forward(self, x, t, cond):
        x_t = x + t + cond
        x_t = self.linear_in(x_t)
        x_t = torch.nn.functional.relu(x_t)
        x_t = torch.nn.Dropout(0.1)(x_t)
        x_t = self.linear_out(x_t)
        x_t = torch.nn.Dropout(0.1)(x_t)
        return x + x_t, x_t

class AutoEncoder_ResNet_Cond(torch.nn.Module):
    def __init__(self, input_size, noise_steps, num_blocks=2):
        super(AutoEncoder_ResNet_Cond, self).__init__()

        embedding_size = 256
        self.layer_x = torch.nn.Linear(input_size, embedding_size)

        self.layer_cond = torch.nn.Linear(input_size, embedding_size)

        self.register_buffer(
            "pos_encoding",
            self._build_embedding(noise_steps, embedding_size / 2),
            persistent=False,
        )
        self.layer_t_1 = torch.nn.Linear(embedding_size, embedding_size)
        self.layer_t_2 = torch.nn.Linear(embedding_size, embedding_size)

        self.layer_out_1 = torch.nn.Linear(embedding_size, embedding_size)
        self.layer_out_2 = torch.nn.Linear(embedding_size, input_size)

        self.residual_layers = torch.nn.ModuleList([
            ResidualBlock_Cond(input_size=embedding_size, embedding_size=embedding_size) for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor, t: torch.LongTensor, cond: torch.Tensor) -> torch.Tensor:
        # Time step embedding
        t_emb = self.pos_encoding[t].squeeze()
        t_emb = self.layer_t_1(t_emb)
        t_emb = torch.nn.functional.silu(t_emb)
        t_emb = self.layer_t_2(t_emb)
        t_emb = torch.nn.functional.silu(t_emb)

        x_emb = torch.nn.functional.relu(self.layer_x(x))

        cond_emb = torch.nn.functional.relu(self.layer_cond(cond))

        skip = []
        for layer in self.residual_layers:
            x_emb, skip_connection = layer(x_emb, t_emb, cond_emb)
            skip.append(skip_connection)

        out = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        out = self.layer_out_1(out)
        out = torch.nn.functional.relu(out)
        out = self.layer_out_2(out)

        return out
    
    def _build_embedding(self, noise_steps, dim=64):
        steps = torch.arange(noise_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table

class TabDDPM_Mask_ResNet_Cond(TabDDPM_Mask):
    def __init__(self, input_size, noise_steps, beta_start: float = 1e-4, beta_end: float = 0.02, lr: float = 0.0001, num_blocks: int = 2):
        super(TabDDPM_Mask_ResNet_Cond, self).__init__(input_size, noise_steps, beta_start, beta_end, lr)

        self.eps_model = AutoEncoder_ResNet_Cond(input_size, noise_steps=noise_steps, num_blocks=num_blocks).to(self.device)
        self.optimiser = torch.optim.Adam(self.eps_model.parameters(), lr = lr)

    def fit(self, x, epochs=10, batch_size=100, x_valid=None):
        mask_x = ~x.isna().to_numpy()
        x = x.fillna(x.mean())
        
        x_normalized = self.normalizer_x.fit_transform(x.values)

        self.eps_model.train()
        x_tensor = torch.from_numpy(x_normalized).float()
        mask_x_tensor = torch.from_numpy(mask_x)
        dataloader = DataLoader(TensorDataset(x_tensor, mask_x_tensor), batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            loss_epoch = 0.
            for id_batch, (x_batch, mask_x_batch) in enumerate(dataloader):
                x_batch = x_batch.to(self.device)
                mask_rand = torch.cuda.FloatTensor(mask_x_batch.size()).uniform_() > 0.2
                mask_x_batch = mask_x_batch.to(self.device).bool().float() * mask_rand

                self.optimiser.zero_grad()
                t = torch.randint(low=1, high=self.noise_steps, size=(x_batch.size(dim=0), 1), device=self.device)
                x_batch_t, noise = self.q_sample(x=x_batch, t=t)
                predicted_noise = self.eps_model(x=x_batch_t, t=t, cond=x_batch)
                loss = (self.loss_func(predicted_noise, noise) * mask_x_batch).mean()
                loss.backward()
                self.optimiser.step()
                loss_epoch += loss.item()

            self.summary['epoch_loss'].append(loss.item())
            if x_valid is not None:
                self.summary['eval_mae'].append(self.eval(x_valid))

    def eval(self, x):
        mask_x = ~x.isna().to_numpy()

        x_normalized = self.normalizer_x.transform(x.values)
        x_tensor = torch.from_numpy(x_normalized).float().to(self.device)
        mask_x_tensor = torch.from_numpy(mask_x).float().to(self.device)

        with torch.no_grad():
            noise = torch.randn((x_tensor.size(dim=0), x_tensor.size(dim=1)), device=self.device)

            for i in reversed(range(1, self.noise_steps)):
                t = torch.ones((x_tensor.size(dim=0), 1), dtype=torch.long, device=self.device) * i

                sqrt_alpha_t = self.sqrt_alpha[t].view(-1, 1)
                beta_t = self.beta[t].view(-1, 1)
                sqrt_one_minus_alpha_hat_t = self.sqrt_one_minus_alpha_hat[t].view(-1, 1)
                epsilon_t = self.std_beta[t].view(-1, 1)

                random_noise = torch.randn_like(noise) if i > 1 else torch.zeros_like(noise)
                noise = mask_x_tensor * x_tensor + (1.0 - mask_x_tensor) * noise
                noise = ((1 / sqrt_alpha_t) * (noise - ((beta_t / sqrt_one_minus_alpha_hat_t) * self.eps_model(noise, t, x_tensor)))) + (epsilon_t * random_noise)

        return (torch.nn.L1Loss(reduction='none')(x_tensor, noise) * mask_x_tensor).nanmean().item()
    
    def predict(self, x):
        self.eps_model.eval()
        n_samples = len(x)
        n_features = x.columns.size
        mask_x = ~x.isna().to_numpy()
        x_normalized = self.normalizer_x.transform(x.fillna(x.mean()).values)

        with torch.no_grad():
            noise = torch.randn((n_samples, n_features), device=self.device)
            mask_x_tensor = torch.from_numpy(mask_x).to(self.device).bool().float()
            x_tensor = torch.from_numpy(x_normalized).to(self.device).float()

            for i in reversed(range(1, self.noise_steps)):
                t = torch.ones((n_samples, 1), dtype=torch.long, device=self.device) * i

                sqrt_alpha_t = self.sqrt_alpha[t].view(-1, 1)
                beta_t = self.beta[t].view(-1, 1)
                sqrt_one_minus_alpha_hat_t = self.sqrt_one_minus_alpha_hat[t].view(-1, 1)
                epsilon_t = self.std_beta[t].view(-1, 1)

                random_noise = torch.randn_like(noise) if i > 1 else torch.zeros_like(noise)
                noise = mask_x_tensor * x_tensor + (1.0 - mask_x_tensor) * noise
                noise = ((1 / sqrt_alpha_t) * (noise - ((beta_t / sqrt_one_minus_alpha_hat_t) * self.eps_model(noise, t, x_tensor)))) + (epsilon_t * random_noise)

        outputs_normalized = noise.detach().cpu().numpy()
        outputs_real = self.normalizer_x.inverse_transform(outputs_normalized)
        outputs = pd.DataFrame(outputs_real, columns=x.columns, index=x.index)
        x = x.fillna(outputs)
        return x
```

# Evaluation


## Noise steps

```python
# %%time

# summaries_noise_steps = {}

# tabddpm = TabDDPM(input_size=8, noise_steps=50)
# tabddpm.fit(df_data, batch_size=200, epochs=100, eval_size=1000)
# summaries_noise_steps["TabDDPM_ns50"] = tabddpm.summary

# tabddpm = TabDDPM(input_size=8, noise_steps=100)
# tabddpm.fit(df_data, batch_size=200, epochs=100, eval_size=1000)
# summaries_noise_steps["TabDDPM_ns100"] = tabddpm.summary

# tabddpm = TabDDPM(input_size=8, noise_steps=200)
# tabddpm.fit(df_data, batch_size=200, epochs=100, eval_size=1000)
# summaries_noise_steps["TabDDPM_ns200"] = tabddpm.summary

# tabddpm = TabDDPM(input_size=8, noise_steps=500)
# tabddpm.fit(df_data, batch_size=200, epochs=100, eval_size=1000)
# summaries_noise_steps["TabDDPM_ns500"] = tabddpm.summary

# tabddpm = TabDDPM(input_size=8, noise_steps=1000)
# tabddpm.fit(df_data, batch_size=200, epochs=100, eval_size=1000)
# summaries_noise_steps["TabDDPM_ns1000"] = tabddpm.summary
```

```python
# fig = plot_summaries(summaries_noise_steps, display='epoch_loss', height=300)
# with open("examples/figures/fig_noise_steps_epoch_loss.json", 'w') as outfile: outfile.write(fig.to_json())
# fig.show()
# fig = plot_summaries(summaries_noise_steps, display='eval_mae', height=300)
# with open("examples/figures/fig_noise_steps_eval_mae.json", 'w') as outfile: outfile.write(fig.to_json())
# fig.show()

# with open("examples/figures/fig_noise_steps_epoch_loss.json", 'r') as f: fig = pio.from_json(f.read())
# fig.show()

# with open("examples/figures/fig_noise_steps_eval_mae.json", 'r') as f: fig = pio.from_json(f.read())
# fig.show()


```

## Nan ratio


- Decrease ratio of nan in dataset: 0.1 -> 0.9

```python
dfs_imputed_ratio = {}
for ratio_masked in np.arange(0.1, 1.0, 0.1):
    print(ratio_masked)
    df_mask_ratio = missing_patterns.UniformHoleGenerator(n_splits=1, subset=cols_to_impute, ratio_masked=ratio_masked).generate_mask(df_data_raw_sample)
    df_data_ratio = df_data_raw_sample[df_mask_ratio]
    for name, imp in dict_imputers_baseline.items():
        dfs_imputed_ratio[(name, ratio_masked)] = imp.fit_transform(df_data_ratio)

    TabDDPM_simple = imputers_pytorch.ImputerGenerativeModelPytorch(model=TabDDPM(input_size=10, noise_steps=100), batch_size=500, epochs=100)
    dfs_imputed_ratio[("TabDDPM", ratio_masked)] = TabDDPM_simple.fit_transform(df_data_ratio)

    TabDDPM_mask = imputers_pytorch.ImputerGenerativeModelPytorch(model=TabDDPM_Mask(input_size=10, noise_steps=100), batch_size=500, epochs=100)
    dfs_imputed_ratio[("TabDDPM_mask", ratio_masked)] = TabDDPM_mask.fit_transform(df_data_ratio)

    TabDDPM_mask_resnet = imputers_pytorch.ImputerGenerativeModelPytorch(model=TabDDPM_Mask_ResNet(input_size=10, noise_steps=100, num_blocks=2), batch_size=500, epochs=100)
    dfs_imputed_ratio[("TabDDPM_mask_resnet", ratio_masked)] = TabDDPM_mask_resnet.fit_transform(df_data_ratio)

    TabDDPM_mask_resnet_cond = imputers_pytorch.ImputerGenerativeModelPytorch(model=TabDDPM_Mask_ResNet_Cond(input_size=10, noise_steps=100, num_blocks=2), batch_size=500, epochs=100)
    dfs_imputed_ratio[("TabDDPM_mask_resnet_cond", ratio_masked)] = TabDDPM_mask_resnet_cond.fit_transform(df_data_ratio)
```

```python
# with open('examples/figures/nan_ratio.pkl', 'wb') as handle:
#     pickle.dump(dfs_imputed_ratio, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('examples/figures/nan_ratio.pkl', 'rb') as handle:
    dfs_imputed_ratio = pickle.load(handle)
```

```python
dict_metrics = {
    "mae": mtr.mean_absolute_error,
    "wasser": mtr.wasserstein_distance,
    "corr": mtr.mean_difference_correlation_matrix_numerical_features,
    "KL": mtr.kl_divergence,
}

df_error_ratio = plot_errors(df_data_raw_sample, dfs_imputed_ratio, df_mask.replace(False, True), dict_metrics, use_p_value=False, method="random_forest").sort_index()
df_error_ratio = df_error_ratio.swaplevel(axis=1)
```

- For each feature, each metric and each model, compute number of other models beaten by this model.
- The table is grouped by metrics, mean values are displayed.

```python
df_error_ratio_vote = df_error_ratio.copy()
for col in df_error_ratio.columns:
    _col = list(col)
    _col[1] = f'{_col[1]}_vote'
    df_error_ratio_vote[tuple(_col)] = df_error_ratio.apply(lambda x: np.sum([1 if x[col] < v else 0 for v in x]), axis=1)

df_error_ratio_vote = df_error_ratio_vote.iloc[:, df_error_ratio_vote.columns.get_level_values(1).str.contains('vote')]
for ratio_masked in df_error_ratio_vote.columns.get_level_values(0).unique().to_list():
    display(df_error_ratio_vote.iloc[:, df_error_ratio_vote.columns.get_level_values(0)==ratio_masked].groupby(level=0).mean()\
    .style.apply(lambda x: ["background: green" if v == x.max() else "" for v in x], axis = 1))
```

```python
df_error_ratio_vote_plot = df_error_ratio_vote.groupby(level=0).mean()

# for m in df_error_ratio_vote_plot.index.values:
for m in ['mae', 'KL', 'corr']:
    fig = go.Figure()
    for model in df_error_ratio_vote_plot.columns.get_level_values(1).unique().to_list():
        x_plot = df_error_ratio_vote_plot.loc[m, pd.IndexSlice[:, model]].index.get_level_values(0).values
        y_plot = df_error_ratio_vote_plot.loc[m, pd.IndexSlice[:, model]].values
        fig.add_trace(go.Scatter(x=x_plot, y=y_plot, mode='markers+lines', name = model.replace('_vote','')))
    fig.update_layout(title=m, xaxis_title="Obs ratio", yaxis_title='Number of wins', legend_title="Models")
    fig.show()
```

## Architectures


### Train

```python
models = {}
df_data_valid = df_data.sample(1000)
```

```python
%%time
models["TabDDPM"] = TabDDPM(input_size=10, noise_steps=100)
models["TabDDPM"].fit(df_data, batch_size=500, epochs=50, x_valid=df_data_valid)
```

```python
%%time
models["TabDDPM_Mask"] = TabDDPM_Mask(input_size=10, noise_steps=100)
models["TabDDPM_Mask"].fit(df_data, batch_size=500, epochs=50, x_valid=df_data_valid)
```

```python
%%time
models["TabDDPM_Mask_ResNet"] = TabDDPM_Mask_ResNet(input_size=10, noise_steps=100, num_blocks=2)
models["TabDDPM_Mask_ResNet"].fit(df_data, batch_size=500, epochs=50, x_valid=df_data_valid)
```

```python
%%time
models["TabDDPM_Mask_ResNet_Cond"] = TabDDPM_Mask_ResNet_Cond(input_size=10, noise_steps=100, num_blocks=2)
models["TabDDPM_Mask_ResNet_Cond"].fit(df_data, batch_size=500, epochs=50, x_valid=df_data_valid)
```

```python
print('Number of trained parameters:')
for name_model, model in models.items():
    print(f"{name_model}: {get_num_params(model.eps_model)}")

summaries = {
    "TabDDPM": models["TabDDPM"].summary,
    "TabDDPM_Mask": models["TabDDPM_Mask"].summary,
    "TabDDPM_Mask_ResNet": models["TabDDPM_Mask_ResNet"].summary,
    "TabDDPM_Mask_ResNet_Cond": models["TabDDPM_Mask_ResNet_Cond"].summary
}

plot_summaries(summaries, display='epoch_loss', height=300).show()
plot_summaries(summaries, display='eval_mae', height=300).show()
```

### Test

```python
%%time

dfs_imputed = {name: imp.fit_transform(df_data) for name, imp in dict_imputers_baseline.items()}
```

```python
%%time

dict_imputers = {}

# dict_imputers["regressor_col"] = imputers.ImputerRegressor(estimator=LinearRegression(), handler_nan = "column")
# dfs_imputed["regressor_col"] = dict_imputers["regressor_col"].fit_transform(df_data)

# dict_imputers["MLP_col"] = imputers_pytorch.ImputerRegressorPytorch(estimator=feedforward_regressor(input_size=), handler_nan = "column", batch_size=500, epochs=100)
# dfs_imputed["MLP_col"] = dict_imputers["MLP_col"].fit_transform(df_data)

# dict_imputers["MLP_fit"] = imputers_pytorch.ImputerRegressorPytorch(estimator=feedforward_regressor(input_size=9), handler_nan = "fit", batch_size=500, epochs=100)
# dfs_imputed["MLP_fit"] = dict_imputers["MLP_fit"].fit_transform(df_data)
```

```python
%%time

dict_imputers["AutoEncoderImputer"] = imputers_pytorch.ImputerGenerativeModelPytorch(model=AutoEncoderImputer(input_size=10), batch_size=500, epochs=100)
dfs_imputed["AutoEncoderImputer"] = dict_imputers["AutoEncoderImputer"].fit_transform(df_data)
```

```python
%%time

dict_imputers["TabDDPM"] = imputers_pytorch.ImputerGenerativeModelPytorch(model=TabDDPM(input_size=10, noise_steps=100), batch_size=500, epochs=100)
dfs_imputed["TabDDPM"] = dict_imputers["TabDDPM"].fit_transform(df_data)
```

```python
%%time

dict_imputers["TabDDPM_mask"] = imputers_pytorch.ImputerGenerativeModelPytorch(model=TabDDPM_Mask(input_size=10, noise_steps=100), batch_size=500, epochs=100)
dfs_imputed["TabDDPM_mask"] = dict_imputers["TabDDPM_mask"].fit_transform(df_data)
```

```python
%%time

dict_imputers["TabDDPM_mask_resnet"] = imputers_pytorch.ImputerGenerativeModelPytorch(model=TabDDPM_Mask_ResNet(input_size=10, noise_steps=200, num_blocks=2), batch_size=500, epochs=100)
dfs_imputed["TabDDPM_mask_resnet"] = dict_imputers["TabDDPM_mask_resnet"].fit_transform(df_data)
```

```python
%%time

dict_imputers["TabDDPM_mask_resnet_cond"] = imputers_pytorch.ImputerGenerativeModelPytorch(model=TabDDPM_Mask_ResNet_Cond(input_size=10, noise_steps=200, num_blocks=2), batch_size=500, epochs=100)
dfs_imputed["TabDDPM_mask_resnet_cond"] = dict_imputers["TabDDPM_mask_resnet_cond"].fit_transform(df_data)
```

```python
dict_metrics = {
    "mae": mtr.mean_absolute_error,
    "KL": mtr.kl_divergence,
    "wasser": mtr.wasserstein_distance,
    "corr": mtr.mean_difference_correlation_matrix_numerical_features,
}

df_error = plot_errors(df_data_raw_sample, dfs_imputed, df_mask.replace(False, True), dict_metrics, use_p_value=False, method="gaussian").sort_index()
```

- Diffusion models only

```python
df_error[["TabDDPM", "TabDDPM_mask", "TabDDPM_mask_resnet", "TabDDPM_mask_resnet_cond"]].style.apply(lambda x: ["background: green" if v == x.min() else "" for v in x], axis = 1)
```

- Other models
- Distribution metrics (KL, Wasserten distance) and Accuracy metrics (MAE)

```python
cols_min_value = df_error.idxmin(axis=1).unique().tolist() + ["AutoEncoderImputer", "TabDDPM", "TabDDPM_mask", "TabDDPM_mask_resnet", "TabDDPM_mask_resnet_cond"]

# df_error.style\
# .apply(lambda x: ["background: green" if v == x.min() else "" for v in x], axis = 1)\
# .hide([col for col in df_error.columns.to_list() if col not in cols_min_value or col in ['shuffle']], axis=1)\

# Remove shuffle
display(df_error.loc[ ['KL', 'wasser'], [col for col in df_error.columns.to_list() if col in cols_min_value and col not in ['shuffle']]]\
.style.apply(lambda x: ["background: green" if v == x.min() else "" for v in x], axis = 1))


display(df_error.loc[ ['mae'], [col for col in df_error.columns.to_list() if col in cols_min_value]]\
.style.apply(lambda x: ["background: green" if v == x.min() else "" for v in x], axis = 1))
```

- For each feature, each metric and each model, compute number of other models beaten by this model.
- The first table is grouped by metrics, the mean values are displayed.
- The second table shows the mean values for all features and metrics.

```python
df_error_vote = df_error.copy()
for col in df_error.columns.to_list():
    df_error_vote[f'{col}_vote'] = df_error.apply(lambda x: np.sum([1 if x[col] < v else 0 for v in x]), axis=1)

display(df_error_vote.loc[:, df_error_vote.columns.str.contains('vote')].groupby(level=0).mean()\
.style.apply(lambda x: ["background: green" if v == x.max() else "" for v in x], axis = 1))

display(df_error_vote.loc[:, df_error_vote.columns.str.contains('vote')].mean())
```

- Plotting them

```python
col1 = cols_to_impute[0]
col2 = cols_to_impute[1]
fig = go.Figure()

index_nan = df_data.isna().any(axis=1).index
# fig.add_trace(go.Scatter(x=df_data_raw.iloc[index_nan][col1], y=df_data_raw.iloc[index_nan][col2], mode='markers', name='all original', marker=dict(color='black')))
fig.add_trace(go.Scatter(x=df_data_raw_sample.iloc[index_nan][col1], y=df_data_raw_sample.iloc[index_nan][col2], mode='markers', name='original', marker=dict(color='black')))

for ind, (name, data) in enumerate(list(dfs_imputed.items())):
    values_imp_col1 = data[col1].copy()
    values_imp_col1[df_data[col1].notna()] = np.nan

    values_imp_col2 = data[col2].copy()
    values_imp_col2[df_data[col2].notna()] = np.nan

    fig.add_trace(go.Scatter(x=values_imp_col1, y=values_imp_col2, mode='markers', name=name))

fig.update_layout(xaxis_title=col1,
                    yaxis_title=col2, height=500)
fig.show()
```

### CV

```python
%%time

search_params = {
    "RPCA_opti": {
        "tau": {"min": .5, "max": 5, "type":"Real"},
        "lam": {"min": .1, "max": 1, "type":"Real"},
    }
}

generator_holes = missing_patterns.EmpiricalHoleGenerator(n_splits=2, subset = cols_to_impute, ratio_masked=ratio_masked)

comparison = comparator.Comparator(
    dict_imputers,
    df_data.columns,
    generator_holes = generator_holes,
    n_calls_opt=10,
    search_params=search_params,
)
results = comparison.compare(df_data)
results
```

```python
fig = plt.figure(figsize=(24, 4))
plot.multibar(results.loc["mae"], decimals=4)
plt.ylabel("mae")
plt.show()
```
