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
%reload_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np

from qolmat.imputations import imputers
from qolmat.benchmark import comparator, missing_patterns
import qolmat.benchmark.metrics as mtr
from qolmat.utils import data, plot

from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression


import plotly.graph_objects as go
```

```python
dict_metrics = {
    "mae": mtr.mean_absolute_error,
    "wasser": mtr.wasser_distance,
    # "KL": mtr.kl_divergence,
}

def plot_errors(df_original, dfs_imputed, dict_metrics):
    dict_errors_df = {}
    for ind, (name, df_imputed) in enumerate(list(dfs_imputed.items())):
        dict_errors_mtr = {}
        for name_metric in dict_metrics:
            dict_errors_mtr[name_metric] = dict_metrics[name_metric](df_original, df_imputed)

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
```

## **I. Load data**

```python
from datasets import load_dataset
dataset = load_dataset("inria-soda/tabular-benchmark", data_files="reg_num/wine_quality.csv")
```

```python
df_data_raw = dataset["train"].to_pandas().sample(frac = 1)
cols_to_impute = df_data_raw.columns.to_list()
ratio_masked = 0.5

df_mask = missing_patterns.UniformHoleGenerator(n_splits=2, subset=cols_to_impute, ratio_masked=ratio_masked).generate_mask(df_data_raw)

df_data = df_data_raw[df_mask]
```

```python
df_data.isna().sum()
```

## Baseline imputers

```python
imputer_mean = imputers.ImputerMean()
imputer_median = imputers.ImputerMedian()
imputer_mode = imputers.ImputerMode()
# imputer_locf = imputers.ImputerLOCF()
# imputer_nocb = imputers.ImputerNOCB()
# imputer_interpol = imputers.ImputerInterpolation(method="linear")
# imputer_spline = imputers.ImputerInterpolation(method="spline", order=2)
imputer_shuffle = imputers.ImputerShuffle()
# imputer_residuals = imputers.ImputerResiduals(period=7, model_tsa="additive", extrapolate_trend="freq", method_interpolation="linear")

# imputer_rpca = imputers.ImputerRPCA(columnwise=True, period=365, max_iter=200, tau=2, lam=.3)
# imputer_rpca_opti = imputers.ImputerRPCA(columnwise=True, period=365, max_iter=100)

imputer_ou = imputers.ImputerEM(method="multinormal", strategy="ou", max_iter_em=34, n_iter_ou=15, dt=1e-3)
imputer_tsou = imputers.ImputerEM(method="VAR1", strategy="ou", max_iter_em=34, n_iter_ou=15, dt=1e-3)
imputer_tsmle = imputers.ImputerEM(method="VAR1", strategy="mle", max_iter_em=34, n_iter_ou=15, dt=1e-3)

imputer_knn = imputers.ImputerKNN(k=10)
imputer_mice = imputers.ImputerMICE(estimator=LinearRegression(), sample_posterior=False, max_iter=100, missing_values=np.nan)
imputer_regressor = imputers.ImputerRegressor(estimator=LinearRegression())

dict_imputers = {
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
    "mice": imputer_mice,
    # "regressor": imputer_regressor,
}

n_imputers = len(dict_imputers)
```

## Diffusion models


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
        self.normalizer_x = preprocessing.MinMaxScaler()
        x_normalized = self.normalizer_x.fit_transform(x.values)

        self.normalizer_y = preprocessing.MinMaxScaler()
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

        self.layer_out_1 = torch.nn.Linear(512, 512)
        self.layer_out_2 = torch.nn.Linear(512, input_size)

    def forward(self, x: torch.Tensor, t: torch.LongTensor) -> torch.Tensor:
        x_1 = torch.nn.functional.relu(self.layer_x_1(x))
        x_2 = torch.nn.functional.relu(self.layer_x_2(x_1))

        t_1 = torch.nn.functional.relu(self.layer_t_1(t.float()))
        t_2 = torch.nn.functional.relu(self.layer_t_2(t_1))

        cat_x_t = torch.cat([x_2, t_2], dim=1)

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

    def fit(self, x, epochs=10, batch_size=100, eval_size=100):
        x_valid = x.sample(eval_size)
        x = x.fillna(x.mean())

        self.normalizer_x = preprocessing.MinMaxScaler()
        x_normalized = self.normalizer_x.fit_transform(x.values)

        self.eps_model.train()
        x_tensor = torch.from_numpy(x_normalized).float()
        dataloader = DataLoader(TensorDataset(x_tensor), batch_size=batch_size, shuffle=True)
        size_dataset = len(dataloader.dataset)
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

            self.summary['eval_mae'].append(self.eval(x_valid))
            self.summary['epoch_loss'].append(loss.item())

    def eval(self, x):
        mask_x = ~x.isna().to_numpy()

        x_normalized = self.normalizer_x.fit_transform(x.values)
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

        self.loss_func = torch.nn.MSELoss(reduction="none")

    def fit(self, x, epochs=10, batch_size=100, eval_size=100):
        x_valid = x.sample(eval_size)

        mask_x = ~x.isna().to_numpy()
        x = x.fillna(x.mean())

        self.normalizer_x = preprocessing.MinMaxScaler()
        x_normalized = self.normalizer_x.fit_transform(x.values)

        self.eps_model.train()
        x_tensor = torch.from_numpy(x_normalized).float()
        mask_x_tensor = torch.from_numpy(mask_x)
        dataloader = DataLoader(TensorDataset(x_tensor, mask_x_tensor), batch_size=batch_size, shuffle=True)
        size_dataset = len(dataloader.dataset)
        for epoch in range(epochs):
            loss_epoch = 0.
            for id_batch, (x_batch, mask_x_batch) in enumerate(dataloader):
                x_batch = x_batch.to(self.device)
                mask_x_batch = mask_x_batch.to(self.device).bool().float()

                self.optimiser.zero_grad()
                t = torch.randint(low=1, high=self.noise_steps, size=(x_batch.size(dim=0), 1), device=self.device)
                x_batch_t, noise = self.q_sample(x=x_batch, t=t)
                predicted_noise = self.eps_model(x=x_batch_t, t=t)
                loss = (self.loss_func(predicted_noise, noise) * mask_x_batch).mean()
                loss.backward()
                self.optimiser.step()
                loss_epoch += loss.item()

            self.summary['epoch_loss'].append(loss.item()/size_dataset)
            self.summary['eval_mae'].append(self.eval(x_valid))

    def predict(self, x):
        self.eps_model.eval()
        n_samples = len(x)
        n_features = x.columns.size
        mask_x = ~x.isna().to_numpy()
        x_normalized = self.normalizer_x.transform(x.values)

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
- Inference:
    - $\epsilon \rightarrow \hat{x}_t \rightarrow \hat{x}_0$ where
        - $\hat{x}_t = mask * x_0 + (1 - mask) * \hat{x}_t$
        - $mask$: 1 = observed values
    - Fill nan with $\hat{x}_0$

```python
import math

class ResidualBlock(torch.nn.Module):
    def __init__(self, input_size_x, input_size_t, embedding_size):
        super(ResidualBlock, self).__init__()

        self.batch_norm = torch.nn.BatchNorm1d(input_size_x * 2)
        self.linear_in = torch.nn.Linear(input_size_x * 2, embedding_size)
        self.linear_out = torch.nn.Linear(embedding_size, input_size_x)

        self.linear_t = torch.nn.Linear(input_size_t, input_size_x)

    def forward(self, input):
        x, t = input[0], input[1]
        t_emb = self.linear_t(t)
        #x_t = x + t_emb
        x_t = torch.cat([x, t_emb], dim=1)
        x_batch_norm = self.batch_norm(x_t)
        x_linear_in = self.linear_in(x_batch_norm)
        x_relu = torch.nn.functional.relu(x_linear_in)
        x_linear_out = self.linear_out(x_relu)
        x_out = x + x_linear_out
        return {0: x_out, 1: t}

class AutoEncoder_ResNet(torch.nn.Module):
    def __init__(self, input_size_x, num_blocks=1):
        super(AutoEncoder_ResNet, self).__init__()

        self.seqs = torch.nn.Sequential(*[
            ResidualBlock(input_size_x=256, input_size_t=256, embedding_size=256) for i in range(num_blocks)
        ])

        self.layer_in = torch.nn.Linear(input_size_x, 256)
        self.layer_out = torch.nn.Linear(256, input_size_x)

        self.layer_t = torch.nn.Linear(1, 256)

    def forward(self, x: torch.Tensor, t: torch.LongTensor) -> torch.Tensor:
        x_emb = self.layer_in(x)
        t_emb = self.layer_t(t.float())
        out = self.seqs({0: x_emb, 1: t_emb})
        x_out = self.layer_out(out[0])
        return x_out

class TabDDPM_Mask_ResNet(TabDDPM_Mask):
    def __init__(self, input_size, noise_steps, beta_start: float = 1e-4, beta_end: float = 0.02, lr: float = 0.0001):
        super(TabDDPM_Mask_ResNet, self).__init__(input_size, noise_steps, beta_start, beta_end, lr)

        self.eps_model = AutoEncoder_ResNet(input_size).to(self.device)
```

#### TabDDPM with mask, time step emb

- Training:
    - Fill real nan with mean
    - Time step t as a float -> Better embedding of time step
    - With mask: Compute only loss values from observed data
- Inference:
    - With mask: $\epsilon \rightarrow \hat{x}_t \rightarrow \hat{x}_0$ where
        - $\hat{x}_t = mask * x_0 + (1 - mask) * \hat{x}_t$
        - $mask$: 1 = observed values
    - Fill nan with $\hat{x}_0$

```python
import math

class AutoEncoderPosition(torch.nn.Module):
    def __init__(self, input_size, num_steps):
        super(AutoEncoderPosition, self).__init__()

        self.register_buffer(
            "pos_encoding",
            self._build_embedding(num_steps, 256 / 2),
            persistent=False,
        )

        self.layer_x_1 = torch.nn.Linear(input_size, 256)
        self.layer_x_2 = torch.nn.Linear(256, 256)

        self.layer_t_1 = torch.nn.Linear(256, 256)
        self.layer_t_2 = torch.nn.Linear(256, 256)

        self.layer_out_1 = torch.nn.Linear(512, 512)
        self.layer_out_2 = torch.nn.Linear(512, input_size)

    def forward(self, x: torch.Tensor, t: torch.LongTensor) -> torch.Tensor:
        x_1 = torch.nn.functional.relu(self.layer_x_1(x))
        x_2 = torch.nn.functional.relu(self.layer_x_2(x_1))

        t_emb = self.pos_encoding[t].squeeze(1)

        t_1 = torch.nn.functional.relu(self.layer_t_1(t_emb))
        t_2 = torch.nn.functional.relu(self.layer_t_2(t_1))

        cat_x_t = torch.cat([x_2, t_2], dim=1)

        out_1 = torch.nn.functional.relu(self.layer_out_1(cat_x_t))
        out_2 = self.layer_out_2(out_1)
        return out_2

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class TabDDPM_Mask_Position(TabDDPM_Mask):
    def __init__(self, input_size, noise_steps, beta_start: float = 1e-4, beta_end: float = 0.02, lr: float = 0.0001):
        super(TabDDPM_Mask_Position, self).__init__(input_size, noise_steps, beta_start, beta_end, lr)

        self.eps_model = AutoEncoderPosition(input_size, noise_steps).to(self.device)
```

## Evaluation

```python
%%time

summaries = {}
```

```python
tabddpm = TabDDPM(input_size=12, noise_steps=100)
tabddpm.fit(df_data, batch_size=100, epochs=100)
summaries["TabDDPM"] = tabddpm.summary
```

```python
tabddpm_mask = TabDDPM_Mask(input_size=12, noise_steps=100)
tabddpm_mask.fit(df_data, batch_size=100, epochs=100)
summaries["TabDDPM_Mask"] = tabddpm_mask.summary
```

```python
tabddpm_mask_resnet = TabDDPM_Mask_ResNet(input_size=12, noise_steps=100)
tabddpm_mask_resnet.fit(df_data, batch_size=100, epochs=100)
summaries["TabDDPM_Mask_ResNet"] = tabddpm_mask_resnet.summary
```

```python
plot_summaries(summaries, display='epoch_loss', height=300).show()
plot_summaries(summaries, display='eval_mae', height=300).show()
```

```python
TabDDPM(input_size=12, noise_steps=100).eps_model
```

```python
TabDDPM_Mask_ResNet(input_size=12, noise_steps=100).eps_model
```

```python
%%time

dfs_imputed = {name: imp.fit_transform(df_data) for name, imp in dict_imputers.items()}
```

```python
%%time

dict_imputers["regressor_col"] = imputers.ImputerRegressor(estimator=LinearRegression(), handler_nan = "column")
dfs_imputed["regressor_col"] = dict_imputers["regressor_col"].fit_transform(df_data)

# dict_imputers["MLP_col"] = imputers_pytorch.ImputerRegressorPytorch(estimator=feedforward_regressor(input_size=11), handler_nan = "column", batch_size=100, epochs=100)
# dfs_imputed["MLP_col"] = dict_imputers["MLP_col"].fit_transform(df_data)

dict_imputers["MLP_fit"] = imputers_pytorch.ImputerRegressorPytorch(estimator=feedforward_regressor(input_size=11), handler_nan = "fit", batch_size=200, epochs=100)
dfs_imputed["MLP_fit"] = dict_imputers["MLP_fit"].fit_transform(df_data)
```

```python
%%time

dict_imputers["TabDDPM"] = imputers_pytorch.ImputerGenerativeModelPytorch(model=TabDDPM(input_size=12, noise_steps=100), batch_size=100, epochs=100)
dfs_imputed["TabDDPM"] = dict_imputers["TabDDPM"].fit_transform(df_data)
```

```python
%%time

dict_imputers["TabDDPM_mask"] = imputers_pytorch.ImputerGenerativeModelPytorch(model=TabDDPM_Mask(input_size=12, noise_steps=100, lr=0.001), batch_size=100, epochs=100)
dfs_imputed["TabDDPM_mask"] = dict_imputers["TabDDPM_mask"].fit_transform(df_data)
```

```python
%%time

dict_imputers["TabDDPM_mask_pos"] = imputers_pytorch.ImputerGenerativeModelPytorch(model=TabDDPM_Mask_Position(input_size=12, noise_steps=100), batch_size=100, epochs=100)
dfs_imputed["TabDDPM_mask_pos"] = dict_imputers["TabDDPM_mask_pos"].fit_transform(df_data)
```

```python
df_error = plot_errors(df_data_raw, dfs_imputed, dict_metrics).sort_index()
```

```python
df_error[["TabDDPM", "TabDDPM_mask"]].style.apply(lambda x: ["background: green" if v == x.min() else "" for v in x], axis = 1)
```

```python
#df_error = plot_errors(df_data_raw, dfs_imputed, dict_metrics)

cols_min_value = df_error.idxmin(axis=1).unique().tolist() + ["TabDDPM", "TabDDPM_mask"]
print("Method:", df_error.columns)
df_error.style.apply(lambda x: ["background: green" if v == x.min() else "" for v in x], axis = 1)\
.hide([col for col in df_error if col not in cols_min_value], axis=1)\
# .apply(lambda x: ["color: yellow" if v > x['TabDDPM'] else "" for v in x], axis = 1)\
# .apply(lambda x: ["color: cyan" if v > x['TabDDPM_mask'] else "" for v in x], axis = 1)
```

```python
col1 = cols_to_impute[0]
col2 = cols_to_impute[1]
fig = go.Figure()

fig.add_trace(go.Scatter(x=df_data[col1], y=df_data[col2], mode='markers', name='original', marker=dict(color='black')))

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

```python
%%time

search_params = {
    "RPCA_opti": {
        "tau": {"min": .5, "max": 5, "type":"Real"},
        "lam": {"min": .1, "max": 1, "type":"Real"},
    }
}

ratio_masked = 0.1

generator_holes = missing_patterns.EmpiricalHoleGenerator(n_splits=2, subset = cols_to_impute, ratio_masked=ratio_masked)

comparison = comparator.Comparator(
    dict_imputers,
    df_data.columns, #Number of columns
    generator_holes = generator_holes,
    n_calls_opt=10,
    search_params=search_params,
)

metrics = ['mae', 'wasser', 'KL']
results = comparison.compare(df_data, True, metrics)

results
```

```python
fig = plt.figure(figsize=(24, 4))
plot.multibar(results.loc["mae"], decimals=4)
plt.ylabel("mae")
plt.show()
```

```python

```
