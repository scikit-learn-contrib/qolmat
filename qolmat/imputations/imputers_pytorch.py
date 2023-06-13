from typing import Dict, List, Optional

from sklearn.base import BaseEstimator

from qolmat.imputations.imputers import ImputerRegressor, ImputerGenerativeModel
from qolmat.utils.exceptions import PytorchNotInstalled

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing
from typing import Tuple
import pandas as pd

# try:
#     from tensorflow.keras.callbacks import EarlyStopping
# except ModuleNotFoundError:
#     raise PytorchNotInstalled


class ImputerRegressorPytorch(ImputerRegressor):
    def __init__(
        self,
        groups: List[str] = [],
        estimator: Optional[BaseEstimator] = None,
        handler_nan: str = "column",
        epochs: int = 100,
        batch_size: int = 100,
        **hyperparams,
    ):
        super().__init__(
            groups=groups, estimator=estimator, handler_nan=handler_nan, **hyperparams
        )
        self.epochs = epochs
        self.batch_size = batch_size

    def get_params_fit(self) -> Dict:
        return {"epochs": self.epochs, "batch_size": self.batch_size}


class ImputerGenerativeModelPytorch(ImputerGenerativeModel):
    def __init__(
        self,
        groups: List[str] = [],
        model: Optional[BaseEstimator] = None,
        epochs: int = 100,
        batch_size: int = 100,
        **hyperparams,
    ):
        super().__init__(groups=groups, model=model, **hyperparams)
        self.epochs = epochs
        self.batch_size = batch_size

    def get_params_fit(self) -> Dict:
        return {"epochs": self.epochs, "batch_size": self.batch_size}


import math 

class ResidualBlockTS(torch.nn.Module):
    def __init__(self, input_size, embedding_size, nheads=8, num_layers_transformer=1, p_dropout=0.2, dim_feedforward=64, batch_size=100):
        super(ResidualBlockTS, self).__init__()

        self.linear_in = torch.nn.Linear(input_size, embedding_size)
        self.linear_out = torch.nn.Linear(embedding_size, input_size)
        self.dropout = torch.nn.Dropout(p_dropout)

        encoder_layer_feature = torch.nn.TransformerEncoderLayer(
        d_model=input_size, nhead=nheads, dim_feedforward=dim_feedforward, activation="gelu")
        
        self.feature_layer = torch.nn.TransformerEncoder(encoder_layer_feature, num_layers=num_layers_transformer)

        encoder_layer_time = torch.nn.TransformerEncoderLayer(
        d_model=batch_size, nhead=nheads, dim_feedforward=dim_feedforward, activation="gelu")
        self.time_layer = torch.nn.TransformerEncoder(encoder_layer_time, num_layers=num_layers_transformer)

    def forward(self, x, t, cond):
        cond_feature = self.feature_layer(cond)
        
        cond_time = self.time_layer(cond.permute(1, 0)).permute(1, 0)

        x_t = x + t + cond_feature + cond_time
        x_t = self.linear_in(x_t)
        x_t = torch.nn.functional.relu(x_t)
        x_t = self.dropout(x_t)
        x_t = self.linear_out(x_t)
        return x + x_t, x_t

class AutoEncoderTS(torch.nn.Module):
    def __init__(self, input_size, noise_steps, embedding_size=256, num_blocks=2, p_dropout=0.2, batch_size=100):
        super(AutoEncoderTS, self).__init__()


        self.layer_x = torch.nn.Linear(input_size, embedding_size)

        self.layer_cond = torch.nn.Linear(input_size, embedding_size)

        self.register_buffer(
            "pos_encoding",
            self._build_embedding(noise_steps, embedding_size / 2),
            persistent=False,
        )
        self.layer_t_1 = torch.nn.Linear(embedding_size, embedding_size)
        self.layer_t_2 = torch.nn.Linear(embedding_size, embedding_size)
        self.dropout_t = torch.nn.Dropout(p_dropout)
        
        self.layer_out_1 = torch.nn.Linear(embedding_size, embedding_size)
        self.layer_out_2 = torch.nn.Linear(embedding_size, input_size)
        self.dropout_out = torch.nn.Dropout(p_dropout)

        self.dropout_cond = torch.nn.Dropout(p_dropout)

        self.residual_layers = torch.nn.ModuleList([
            ResidualBlockTS(input_size=embedding_size, embedding_size=embedding_size, p_dropout=p_dropout, batch_size=batch_size) for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor, t: torch.LongTensor, cond: torch.Tensor) -> torch.Tensor:
        # Time step embedding
        t_emb = self.pos_encoding[t].squeeze()
        t_emb = self.layer_t_1(t_emb)
        t_emb = torch.nn.functional.silu(t_emb)
        t_emb = self.layer_t_2(t_emb)
        t_emb = torch.nn.functional.silu(t_emb)
        t_emb = self.dropout_t(t_emb)

        x_emb = torch.nn.functional.relu(self.layer_x(x))

        cond_emb = torch.nn.functional.relu(self.layer_cond(cond))
        cond_emb = self.dropout_cond(cond_emb)

        skip = []
        for layer in self.residual_layers:
            x_emb, skip_connection = layer(x_emb, t_emb, cond_emb)
            skip.append(skip_connection)

        out = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        out = self.layer_out_1(out)
        out = torch.nn.functional.relu(out)
        out = self.dropout_t(out)
        out = self.layer_out_2(out)

        return out
    
    def _build_embedding(self, noise_steps, dim=64):
        steps = torch.arange(noise_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table

class TabDDPM:
    def __init__(self, input_size, noise_steps, beta_start: float = 1e-4, beta_end: float = 0.02, lr: float = 0.0001):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.loss_func = torch.nn.MSELoss(reduction='none')

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
            'eval_mae': [],
            'eval_kl': []
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
    
class TabDDPMTS(TabDDPM):
    def __init__(self, input_size, noise_steps, beta_start: float = 1e-4, beta_end: float = 0.02, lr: float = 0.0001, num_blocks: int = 2, p_dropout=0.2, batch_size=100):
        super(TabDDPMTS, self).__init__(input_size, noise_steps, beta_start, beta_end, lr)

        self.eps_model = AutoEncoderTS(input_size, noise_steps=noise_steps, num_blocks=num_blocks, p_dropout=p_dropout, batch_size=batch_size).to(self.device)
        self.optimiser = torch.optim.Adam(self.eps_model.parameters(), lr = lr)

    def fit(self, x: pd.DataFrame, epochs=10, batch_size=100, x_valid=None, x_valid_mask=None):
        self.batch_size = batch_size
        mask_x = ~x.isna().to_numpy()
        x = x.fillna(x.mean())
        
        x_normalized = self.normalizer_x.fit_transform(x.values)

        self.eps_model.train()
        x_tensor = torch.from_numpy(x_normalized).float()
        mask_x_tensor = torch.from_numpy(mask_x)
        dataloader = DataLoader(TensorDataset(x_tensor, mask_x_tensor), batch_size=batch_size, drop_last=True, shuffle=False)
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
                valid_loss = self.eval(x_valid, x_valid_mask)
                self.summary['eval_mae'].append(valid_loss['L1Loss'])
                self.summary['eval_kl'].append(valid_loss['KLDivLoss'])

    def eval(self, x, mask_x_obs):
        mask_x = mask_x_obs.to_numpy()

        x_normalized = self.normalizer_x.transform(x.fillna(x.mean()).values)
        x_tensor = torch.from_numpy(x_normalized).float().to(self.device)
        mask_x_tensor = torch.from_numpy(mask_x).float().to(self.device)
        dataloader = DataLoader(TensorDataset(x_tensor, mask_x_tensor), batch_size=self.batch_size, drop_last=True, shuffle=False)
        with torch.no_grad():
            loss_eval = {'L1Loss': [],
                         'KLDivLoss': []}
            for id_batch, (x_batch, mask_x_batch) in enumerate(dataloader):
                noise = torch.randn((x_batch.size(dim=0), x_batch.size(dim=1)), device=self.device)

                for i in reversed(range(1, self.noise_steps)):
                    t = torch.ones((x_batch.size(dim=0), 1), dtype=torch.long, device=self.device) * i

                    sqrt_alpha_t = self.sqrt_alpha[t].view(-1, 1)
                    beta_t = self.beta[t].view(-1, 1)
                    sqrt_one_minus_alpha_hat_t = self.sqrt_one_minus_alpha_hat[t].view(-1, 1)
                    epsilon_t = self.std_beta[t].view(-1, 1)

                    random_noise = torch.randn_like(noise) if i > 1 else torch.zeros_like(noise)
                    
                    noise = ((1 / sqrt_alpha_t) * (noise - ((beta_t / sqrt_one_minus_alpha_hat_t) * self.eps_model(noise, t, x_batch)))) + (epsilon_t * random_noise)
                    noise = mask_x_batch * x_batch + (1.0 - mask_x_batch) * noise
                
                mask_x_batch_df = pd.DataFrame(mask_x_batch.detach().cpu().numpy(), columns=x.columns).astype('bool')
                loss_eval['L1Loss'].append((torch.nn.L1Loss(reduction='none')(x_batch, noise) * (1.0 - mask_x_batch)).nanmean().item())
                loss_eval['KLDivLoss'].append(mtr.kl_divergence(pd.DataFrame(x_normalized), pd.DataFrame(noise.detach().cpu().numpy()), ~mask_x_batch_df, method='gaussian').mean())
        
        return {'L1Loss': np.mean(loss_eval['L1Loss']),
                'KLDivLoss': np.mean(loss_eval['KLDivLoss'])}
    
    def predict(self, x):
        self.eps_model.eval()
        mask_x = ~x.isna()
        size_x = len(x)
        max_size = int(self.batch_size*(np.round(len(x)/self.batch_size)+1))
        
        x_pad = pd.concat([x, x.iloc[:(max_size-size_x), :]], axis=0)
        mask_x_pad = pd.concat([mask_x, mask_x.iloc[:(max_size-size_x), :]], axis=0).to_numpy()
        x_normalized = self.normalizer_x.transform(x_pad.fillna(x.mean()).values)
        
        x_tensor = torch.from_numpy(x_normalized).to(self.device).float()
        mask_x_tensor = torch.from_numpy(mask_x_pad).to(self.device).bool().float()
        dataloader = DataLoader(TensorDataset(x_tensor, mask_x_tensor), batch_size=self.batch_size, shuffle=False)

        outputs_normalized = []
        with torch.no_grad():
            for id_batch, (x_batch, mask_x_batch) in enumerate(dataloader):
                noise = torch.randn((x_batch.size(dim=0), x_batch.size(dim=1)), device=self.device)

                for i in reversed(range(1, self.noise_steps)):
                    t = torch.ones((x_batch.size(dim=0), 1), dtype=torch.long, device=self.device) * i

                    sqrt_alpha_t = self.sqrt_alpha[t].view(-1, 1)
                    beta_t = self.beta[t].view(-1, 1)
                    sqrt_one_minus_alpha_hat_t = self.sqrt_one_minus_alpha_hat[t].view(-1, 1)
                    epsilon_t = self.std_beta[t].view(-1, 1)

                    random_noise = torch.randn_like(noise) if i > 1 else torch.zeros_like(noise)
                    
                    noise = ((1 / sqrt_alpha_t) * (noise - ((beta_t / sqrt_one_minus_alpha_hat_t) * self.eps_model(noise, t, x_batch)))) + (epsilon_t * random_noise)
                    noise = mask_x_batch * x_batch + (1.0 - mask_x_batch) * noise

                outputs_normalized.append(noise.detach().cpu().numpy())
        
        outputs_normalized = np.concatenate(outputs_normalized)[:size_x]
        outputs_real = self.normalizer_x.inverse_transform(outputs_normalized)
        outputs = pd.DataFrame(outputs_real, columns=x.columns, index=x.index)
        x = x.fillna(outputs)
        return x