---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
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

# import warnings
# warnings.filterwarnings('error')

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

def plot_errors(df_original, dfs_imputed, dfs_mask, dict_metrics, cols_to_impute, **kwargs):
    dict_errors_df = {}
    for ind, (name, df_imputed) in enumerate(list(dfs_imputed.items())):
        dict_errors_mtr = {}
        for name_metric in dict_metrics:
            metric_args = list(inspect.signature(dict_metrics[name_metric]).parameters)
            args_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in metric_args}
            if "wasser" in name_metric:
                dict_errors_mtr[name_metric] = dict_metrics[name_metric](df_original[cols_to_impute], df_imputed[cols_to_impute], dfs_mask[cols_to_impute])
            else:
                dict_errors_mtr[name_metric] = dict_metrics[name_metric](df_original[cols_to_impute], df_imputed[cols_to_impute], dfs_mask[cols_to_impute], **args_dict)
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
df_data_raw = data.get_data_corrupted("Beijing", ratio_masked=0., mean_size=120)
df_data = data.get_data_corrupted("Beijing", ratio_masked=.2, mean_size=120)

# cols_to_impute = ["TEMP", "PRES", "DEWP", "NO2", "CO", "O3", "WSPM"]
# cols_to_impute = df_data.columns[df_data.isna().any()]
cols_to_impute = ["TEMP", "PRES"]

n_stations = len(df_data.groupby("station").size())
n_cols = len(cols_to_impute)

df_mask = df_data.isna()
df_mask[df_data_raw.isna()] = False

```

```python
df_data.describe()
```

## Baseline imputers

```python
imputer_mean = imputers.ImputerMean(groups=["station"])
imputer_median = imputers.ImputerMedian(groups=["station"])
imputer_mode = imputers.ImputerMode(groups=["station"])
imputer_locf = imputers.ImputerLOCF(groups=["station"])
imputer_nocb = imputers.ImputerNOCB(groups=["station"])
imputer_interpol = imputers.ImputerInterpolation(groups=["station"], method="linear")
imputer_spline = imputers.ImputerInterpolation(groups=["station"], method="spline", order=2)
imputer_shuffle = imputers.ImputerShuffle(groups=["station"])
imputer_residuals = imputers.ImputerResiduals(groups=["station"], period=7, model_tsa="additive", extrapolate_trend="freq", method_interpolation="linear")

imputer_rpca = imputers.ImputerRPCA(groups=["station"], columnwise=True, period=365, max_iter=200, tau=2, lam=.3)
imputer_rpca_opti = imputers.ImputerRPCA(groups=["station"], columnwise=True, period=365, max_iter=100)

imputer_ou = imputers.ImputerEM(groups=["station"], model="multinormal", method="sample", max_iter_em=34, n_iter_ou=15, dt=1e-3)
imputer_tsou = imputers.ImputerEM(groups=["station"], model="VAR1", method="sample", max_iter_em=34, n_iter_ou=15, dt=1e-3)
imputer_tsmle = imputers.ImputerEM(groups=["station"], model="VAR1", method="mle", max_iter_em=34, n_iter_ou=15, dt=1e-3)

imputer_knn = imputers.ImputerKNN(groups=["station"], k=10)
imputer_mice = imputers.ImputerMICE(groups=["station"], estimator=LinearRegression(), sample_posterior=False, max_iter=100, missing_values=np.nan)
imputer_regressor = imputers.ImputerRegressor(groups=["station"], estimator=LinearRegression())

dict_imputers_baseline = {
    "mean": imputer_mean,
    # "median": imputer_median,
    # "mode": imputer_mode,
    # "interpolation": imputer_interpol,
    # "spline": imputer_spline,
    # "shuffle": imputer_shuffle,
    # "residuals": imputer_residuals,
    # "OU": imputer_ou,
    # "TSOU": imputer_tsou,
    # "TSMLE": imputer_tsmle,
    # "RPCA": imputer_rpca,
    # "RPCA_opti": imputer_rpca_opti,
    # "locf": imputer_locf,
    # "nocb": imputer_nocb,
    # "knn": imputer_knn,
    "mice": imputer_mice,
    "regressor": imputer_regressor,
}

n_imputers = len(dict_imputers_baseline)
```

## Diffusion models

```python
from qolmat.imputations import imputers_pytorch
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing
from typing import Tuple

```

```python
class TabDDPM:
    def __init__(self, noise_steps, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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

        self.normalizer_x = preprocessing.StandardScaler()
        self.loss_func = torch.nn.MSELoss(reduction='none')

        self.summary = {
            'epoch_loss': [],
        }

    def q_sample(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Section 3.2, algorithm 1 formula implementation. Forward process, defined by `q`.
        Found in section 2. `q` gradually adds gaussian noise according to variance schedule. Also,
        can be seen on figure 2.
        """
        sqrt_alpha_hat = self.sqrt_alpha_hat[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_hat[t].view(-1, 1, 1)
        epsilon = torch.randn_like(x, device=self.device)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon
```

```python
import math

class ResidualBlock_Cond_Trans(torch.nn.Module):
    def __init__(self, input_size, embedding_size, nheads_feature=5, nheads_time=8, num_layers_transformer=1, p_dropout=0.2, dim_feedforward=64, window_size=10):
        super(ResidualBlock_Cond_Trans, self).__init__()

        encoder_layer_feature = torch.nn.TransformerEncoderLayer(
        d_model=window_size, nhead=nheads_feature, dim_feedforward=dim_feedforward, activation="gelu")
        self.feature_layer = torch.nn.TransformerEncoder(encoder_layer_feature, num_layers=num_layers_transformer)

        encoder_layer_time = torch.nn.TransformerEncoderLayer(
        d_model=embedding_size, nhead=nheads_time, dim_feedforward=dim_feedforward, activation="gelu")
        self.time_layer = torch.nn.TransformerEncoder(encoder_layer_time, num_layers=num_layers_transformer)

        self.linear_out = torch.nn.Linear(embedding_size, input_size)

    def forward(self, x, t):
        batch_size, window_size, emb_size = x.shape

        x_emb = x.permute(0, 2, 1)
        x_emb = self.feature_layer(x_emb)
        x_emb = x_emb.permute(0, 2, 1)

        x_emb = self.time_layer(x_emb)
        t_emb = t.repeat(1, window_size).reshape(batch_size, window_size, emb_size)

        x_t = x_emb + t_emb
        x_t = self.linear_out(x_t)

        return x + x_t, x_t

class AutoEncoder_ResNet_Cond_Trans(torch.nn.Module):
    def __init__(self, input_size, noise_steps, num_blocks=2, p_dropout=0.2, embedding_size=256, window_size=10):
        super(AutoEncoder_ResNet_Cond_Trans, self).__init__()
        self.embedding_size = embedding_size
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
        self.dropout_out = torch.nn.Dropout(p_dropout)

        self.residual_layers = torch.nn.ModuleList([
            ResidualBlock_Cond_Trans(input_size=embedding_size, embedding_size=embedding_size, p_dropout=p_dropout, window_size=window_size) for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor, t: torch.LongTensor, cond) -> torch.Tensor:
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
        out = torch.nn.functional.relu(self.layer_out_1(out))
        out = self.dropout_out(out)
        out = self.layer_out_2(out)
        return out

    def _build_embedding(self, noise_steps, dim=64):
        steps = torch.arange(noise_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table

class TabDDPM_Mask_ResNet_Cond_Trans(TabDDPM):
    def __init__(self, input_size, noise_steps, beta_start: float = 1e-4, beta_end: float = 0.02, lr: float = 0.0001, num_blocks: int = 2, p_dropout=0.2, window_size=10):
        super(TabDDPM_Mask_ResNet_Cond_Trans, self).__init__(noise_steps, beta_start, beta_end)

        self.eps_model = AutoEncoder_ResNet_Cond_Trans(input_size, noise_steps=noise_steps, num_blocks=num_blocks, p_dropout=p_dropout, window_size=window_size).to(self.device)
        self.optimiser = torch.optim.Adam(self.eps_model.parameters(), lr = lr)
        self.normalizer_x = preprocessing.StandardScaler()
        self.window_size = window_size

    def fit(self, x: pd.DataFrame, epochs=10, batch_size=100, x_valid=None, x_valid_mask=None):
        self.batch_size = batch_size
        self.columns = x.columns.tolist()
        self.normalizer_x.fit(x.values)

        x_processed, x_mask = self.process_data(x, window_size=self.window_size)

        if x_valid is not None:
            if len(x_valid) < self.window_size:
                raise ValueError(f"Size of validation dataframe must be larger than window_size")
            x_processed_valid, x_mask_valid = self.process_data(x_valid, x_valid_mask, window_size=self.window_size)

        self.eps_model.train()
        x_tensor = torch.from_numpy(x_processed).float().to(self.device)
        x_mask_tensor = torch.from_numpy(x_mask).float().to(self.device)
        dataloader = DataLoader(TensorDataset(x_tensor, x_mask_tensor), batch_size=batch_size, drop_last=True, shuffle=True)
        for epoch in range(epochs):
            loss_epoch = 0.
            for id_batch, (x_batch, mask_x_batch) in enumerate(dataloader):
                mask_rand = torch.cuda.FloatTensor(mask_x_batch.size()).uniform_() > 0.2
                mask_x_batch = mask_x_batch * mask_rand

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
                dict_loss = self.eval(x_processed_valid, x_mask_valid, x_valid, x_valid_mask)
                for name_loss, value_loss in dict_loss.items():
                    if name_loss not in self.summary:
                        self.summary[name_loss] = [value_loss]
                    else:
                        self.summary[name_loss].append(value_loss)

            if epoch % 20 == 0:
                print(f"Epoch {epoch}: epoch_loss={self.summary['epoch_loss'][epoch]}")

    def impute(self, x: np.ndarray, x_mask_obs: np.ndarray):
        x_tensor = torch.from_numpy(x).float().to(self.device)
        x_mask_tensor = torch.from_numpy(x_mask_obs).float().to(self.device)
        dataloader = DataLoader(TensorDataset(x_tensor, x_mask_tensor), batch_size=self.batch_size, drop_last=False, shuffle=False)
        with torch.no_grad():
            outputs = []
            for id_batch, (x_batch, mask_x_batch) in enumerate(dataloader):
                noise = torch.randn(x_batch.size(), device=self.device)

                for i in reversed(range(1, self.noise_steps)):
                    t = torch.ones((x_batch.size(dim=0), 1), dtype=torch.long, device=self.device) * i

                    sqrt_alpha_t = self.sqrt_alpha[t].view(-1, 1, 1)
                    beta_t = self.beta[t].view(-1, 1, 1)
                    sqrt_one_minus_alpha_hat_t = self.sqrt_one_minus_alpha_hat[t].view(-1, 1, 1)
                    epsilon_t = self.std_beta[t].view(-1, 1, 1)

                    random_noise = torch.randn_like(noise) if i > 1 else torch.zeros_like(noise)

                    noise = ((1 / sqrt_alpha_t) * (noise - ((beta_t / sqrt_one_minus_alpha_hat_t) * self.eps_model(noise, t, x_batch)))) + (epsilon_t * random_noise)
                    noise = mask_x_batch * x_batch + (1.0 - mask_x_batch) * noise

                for id_window, x_window in enumerate(noise.detach().cpu().numpy()):
                    if id_batch == 0 and id_window == 0:
                        outputs += list(x_window)
                    else:
                        outputs += [x_window[-1, :]]

        return np.array(outputs)

    def eval(self, x: np.ndarray, x_mask_obs: np.ndarray, x_df: pd.DataFrame, x_mask_obs_df: pd.DataFrame):

        x_imputed = self.impute(x, x_mask_obs)
        x_normalized = pd.DataFrame(self.normalizer_x.inverse_transform(x_imputed), columns=self.columns, index=x_df.index)

        return {'MAE': mtr.mean_absolute_error(x_df, x_normalized, ~x_mask_obs_df).mean(),
                'KL': mtr.kl_divergence(x_df, x_normalized, ~x_mask_obs_df, method='gaussian').mean()
                }

    def predict(self, x):
        self.eps_model.eval()

        x_processed, x_mask = self.process_data(x, window_size=self.window_size)
        x_imputed = self.impute(x_processed, x_mask)

        x_out = self.normalizer_x.inverse_transform(x_imputed)
        x_out = pd.DataFrame(x_out, columns=x.columns, index=x.index)
        x_out = x.fillna(x_out)
        return x_out

    def process_data(self, x: pd.DataFrame, mask: pd.DataFrame = None, window_size=10):
        x_windows = list(x.rolling(window=window_size))[window_size-1:]

        x_windows_processed = []
        x_windows_mask_processed = []
        for x_w in x_windows:
            x_windows_mask_processed.append(~x_w.isna().to_numpy())
            x_w_fillna = x_w.fillna(x_w.mean())
            x_w_fillna = x_w_fillna.fillna(x.mean())
            x_w_norm = self.normalizer_x.transform(x_w_fillna.values)
            x_windows_processed.append(x_w_norm)

        if mask is not None:
            x_masks = list(mask.rolling(window=window_size))[window_size-1:]
            x_windows_mask_processed = []
            for x_m in x_masks:
                x_windows_mask_processed.append(x_m.to_numpy())

        x_windows_processed = np.array(x_windows_processed)
        x_windows_mask_processed = np.array(x_windows_mask_processed)

        return x_windows_processed, x_windows_mask_processed
```

## Evaluation


### Train

```python
models = {}
df_valid = df_data_raw.dropna().loc['Aotizhongxin'].sample(100)
df_valid_mask = df_mask.loc['Aotizhongxin'].loc[df_valid.index]

```

```python
%%time
models["TabDDPM_Mask_ResNet_Cond_Trans"] = TabDDPM_Mask_ResNet_Cond_Trans(input_size=11, noise_steps=100, num_blocks=1, window_size=60)
models["TabDDPM_Mask_ResNet_Cond_Trans"].fit(df_data.loc['Aotizhongxin'], batch_size=100, epochs=50, x_valid=df_valid, x_valid_mask=df_valid_mask)
```

```python
print('Number of trained parameters:')
for name_model, model in models.items():
    print(f"{name_model}: {get_num_params(model.eps_model)}")

summaries = {
    "TabDDPM_Mask_ResNet_Cond_Trans": models["TabDDPM_Mask_ResNet_Cond_Trans"].summary
}

plot_summaries(summaries, display='epoch_loss', height=300).show()
plot_summaries(summaries, display='MAE', height=300).show()
plot_summaries(summaries, display='KL', height=300).show()
```

### Test

```python
# df_data_dt = data.add_datetime_features(df_data)
# df_data = data.add_station_features(df_data)

df_data_st = df_data.loc[['Aotizhongxin']]
df_data_raw_st = df_data_raw.loc[['Aotizhongxin']]
df_mask_st = df_mask.loc[['Aotizhongxin']]
```

```python
df_data_st = df_data.loc[['Aotizhongxin']]
df_data_raw_st = df_data_raw.loc[['Aotizhongxin']]
df_mask_st = df_mask.loc[['Aotizhongxin']]
```

```python
dfs_imputed_baseline = {name: imp.fit_transform(df_data_st) for name, imp in dict_imputers_baseline.items()}
```

```python
%%time
torch.cuda.empty_cache()
dict_imputers = {}
dfs_imputed = {}

dict_imputers["TabDDPM_Mask_ResNet_Cond_Trans"] = imputers_pytorch.ImputerGenerativeModelPytorch(groups=['station'], model=TabDDPM_Mask_ResNet_Cond_Trans(input_size=11, noise_steps=100, num_blocks=1, p_dropout=0.0, window_size=180, lr=0.001), batch_size=300, epochs=100)
dfs_imputed["TabDDPM_Mask_ResNet_Cond_Trans"] = dict_imputers["TabDDPM_Mask_ResNet_Cond_Trans"].fit_transform(df_data_st)
```

```python
dict_metrics = {
    "mae": mtr.mean_absolute_error,
    "KL": mtr.kl_divergence,
    "wasser": mtr.wasserstein_distance,
    # "corr": mtr.mean_difference_correlation_matrix_numerical_features,
}

df_error = plot_errors(df_data_raw_st, {**dfs_imputed_baseline, **dfs_imputed}, df_mask_st, dict_metrics, cols_to_impute, use_p_value=False, method="gaussian").sort_index()
```

```python
cols_min_value = df_error.idxmin(axis=1).unique().tolist() + list(dict_imputers.keys())

print("Metrics: ", df_error.columns)
# df_error.style\
# .apply(lambda x: ["background: green" if v == x.min() else "" for v in x], axis = 1)\
# .hide([col for col in df_error.columns.to_list() if col not in cols_min_value or col in ['shuffle']], axis=1)\

# Remove shuffle
display(df_error.loc[ ['KL', 'wasser'], [col for col in df_error.columns.to_list() if col in cols_min_value and col not in ['shuffle']]]\
.style.apply(lambda x: ["background: green" if v == x.min() else "" for v in x], axis = 1))

display(df_error.loc[ ['mae'], [col for col in df_error.columns.to_list() if col in cols_min_value]]\
.style.apply(lambda x: ["background: green" if v == x.min() else "" for v in x], axis = 1))
```

```python
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import matplotlib.ticker as plticker

tab10 = plt.get_cmap("tab10")
plt.rcParams.update({'font.size': 18})

df_plot = df_data
station = df_plot.index.get_level_values("station")[0]
print(station)
df_station = df_plot.loc[station]
dfs_imputed_station = {name: df_plot.loc[station] for name, df_plot in {**dfs_imputed_baseline, **dfs_imputed}.items()}

for col in cols_to_impute:
    fig, ax = plt.subplots(figsize=(10, 3))
    values_orig = df_station[col]

    plt.plot(values_orig, ".", color='black', label="original")

    for ind, (name, model) in enumerate(list(dfs_imputed_station.items())):
        values_imp = dfs_imputed_station[name][col].copy()
        values_imp[values_orig.notna()] = np.nan
        plt.plot(values_imp, ".", color=tab10(ind), label=name, alpha=1)
    plt.ylabel(col, fontsize=16)
    plt.legend(loc=[1, 0], fontsize=18)
    loc = plticker.MultipleLocator(base=2*365)
    ax.xaxis.set_major_locator(loc)
    ax.tick_params(axis='both', which='major', labelsize=17)
    plt.show()
```

```python

```

```python

```
