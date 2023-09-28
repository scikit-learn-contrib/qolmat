"""
The implementation of Transformer for the partially-observed time-series imputation task.

Refer to the paper "Du, W., Cote, D., & Liu, Y. (2023). SAITS: Self-Attention-based Imputation for
Time Series.
Expert systems with applications."

Notes
-----
Partial implementation uses code from https://github.com/WenjieDu/SAITS.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GPL-v3

from typing import Tuple, Union, Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from pypots.imputation.saits.data import DatasetForSAITS
from pypots.imputation.base import BaseNNImputer
from pypots.data.base import BaseDataset
from pypots.optim.adam import Adam
from pypots.optim.base import Optimizer
from pypots.utils.metrics import cal_mae


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention"""

    def __init__(self, temperature: float, attn_dropout: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 1, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """original Transformer multi-head attention"""

    def __init__(
        self,
        n_head: int,
        d_model: int,
        d_k: int,
        d_v: int,
        attn_dropout: float,
    ):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        self.attention = ScaledDotProductAttention(d_k**0.5, attn_dropout)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if attn_mask is not None:
            # this mask is imputation mask, which is not generated from each batch, so needs
            # broadcasting on batch dim
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(
                1
            )  # For batch and head axis broadcasting.

        v, attn_weights = self.attention(q, k, v, attn_mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        v = v.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        v = self.fc(v)
        return v, attn_weights


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_in: int, d_hid: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        return x


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_time: int,
        d_feature: int,
        d_model: int,
        d_inner: int,
        n_head: int,
        d_k: int,
        d_v: int,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        diagonal_attention_mask: bool = False,
    ):
        super().__init__()

        self.diagonal_attention_mask = diagonal_attention_mask
        self.d_time = d_time
        self.d_feature = d_feature

        self.layer_norm = nn.LayerNorm(d_model)
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, attn_dropout)
        self.dropout = nn.Dropout(dropout)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout)

    def forward(self, enc_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.diagonal_attention_mask:
            mask_time = torch.eye(self.d_time).to(enc_input.device)
        else:
            mask_time = None

        residual = enc_input
        # here we apply LN before attention cal, namely Pre-LN, refer paper
        # https://arxiv.org/abs/2002.04745
        enc_input = self.layer_norm(enc_input)
        enc_output, attn_weights = self.slf_attn(
            enc_input, enc_input, enc_input, attn_mask=mask_time
        )
        enc_output = self.dropout(enc_output)
        enc_output += residual

        enc_output = self.pos_ffn(enc_output)
        return enc_output, attn_weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid: int, n_position: int = 200):
        super().__init__()
        # Not a parameter
        self.register_buffer("pos_table", self._get_sinusoid_encoding_table(n_position, d_hid))

    @staticmethod
    def _get_sinusoid_encoding_table(n_position: int, d_hid: int) -> torch.Tensor:
        """Sinusoid position encoding table"""

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, : x.size(1)].clone().detach()


class _TransformerEncoder(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_time: int,
        d_feature: int,
        d_model: int,
        d_inner: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        dropout: float,
        attn_dropout: float,
        ORT_weight: float = 1,
        MIT_weight: float = 1,
    ):
        super().__init__()
        self.n_layers = n_layers
        actual_d_feature = d_feature * 2
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight

        self.layer_stack = nn.ModuleList(
            [
                EncoderLayer(
                    d_time,
                    actual_d_feature,
                    d_model,
                    d_inner,
                    n_heads,
                    d_k,
                    d_v,
                    dropout,
                    attn_dropout,
                    False,
                )
                for _ in range(n_layers)
            ]
        )

        self.embedding = nn.Linear(actual_d_feature, d_model)
        self.position_enc = PositionalEncoding(d_model, n_position=d_time)
        self.dropout = nn.Dropout(p=dropout)
        self.reduce_dim = nn.Linear(d_model, d_feature)

    def _process(self, inputs: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        X, masks = inputs["X"], inputs["missing_mask"]
        input_X = torch.cat([X, masks], dim=2)
        input_X = self.embedding(input_X)
        enc_output = self.dropout(self.position_enc(input_X))

        list_attn_weights = []
        for encoder_layer in self.layer_stack:
            enc_output, attn_weights = encoder_layer(enc_output)
            list_attn_weights.append(torch.mean(attn_weights, dim=1))

        attn_matrices = torch.sum(torch.stack(list_attn_weights), dim=0)
        learned_presentation = self.reduce_dim(enc_output)
        imputed_data = (
            masks * X + (1 - masks) * learned_presentation
        )  # replace non-missing part with original data
        return imputed_data, learned_presentation, attn_matrices

    def forward(self, inputs: dict, training: bool = True) -> dict:
        X, masks = inputs["X"], inputs["missing_mask"]
        imputed_data, learned_presentation, attn_matrices = self._process(inputs)

        if not training:
            # if not in training mode, return the classification result only
            return {"imputed_data": imputed_data, "attn_matrices": attn_matrices}

        ORT_loss = cal_mae(learned_presentation, X, masks)
        MIT_loss = cal_mae(learned_presentation, inputs["X_intact"], inputs["indicating_mask"])

        # `loss` is always the item for backward propagating to update the model
        loss = self.ORT_weight * ORT_loss + self.MIT_weight * MIT_loss

        results = {
            "imputed_data": imputed_data,
            "ORT_loss": ORT_loss,
            "MIT_loss": MIT_loss,
            "loss": loss,
            "attn_matrices": attn_matrices,
        }
        return results


class TransformerExplaination(BaseNNImputer):
    """The PyTorch implementation of the Transformer model.
    Transformer is originally proposed by Vaswani et al. in :cite:`vaswani2017Transformer`,
    and gets re-implemented as a time-series imputation model by Du et al. in :cite:`du2023SAITS`.
    Here you should refer to :cite:`du2023SAITS` for details about this
    Transformer imputation model.

    Parameters
    ----------
    n_steps :
        The number of time steps in the time-series data sample.

    n_features :
        The number of features in the time-series data sample.

    n_layers :
        The number of layers in the 1st and 2nd DMSA blocks in the SAITS model.

    d_model :
        The dimension of the model's backbone.
        It is the input dimension of the multi-head self-attention layers.

    d_inner :
        The dimension of the layer in the Feed-Forward Networks (FFN).

    n_heads :
        The number of heads in the multi-head self-attention mechanism.
        ``d_model`` must be divisible by ``n_heads``, and the result should be equal to ``d_k``.

    d_k :
        The dimension of the `keys` (K) and the `queries` (Q) in the DMSA mechanism.
        ``d_k`` should be the result of ``d_model`` divided by ``n_heads``. Although ``d_k`` can
        be directly calculated with given ``d_model`` and ``n_heads``, we want it be explicitly
        given together with ``d_v`` by users to ensure
        users be aware of them and to avoid any potential mistakes.

    d_v :
        The dimension of the `values` (V) in the DMSA mechanism.

    dropout :
        The dropout rate for all fully-connected layers in the model.

    attn_dropout :
        The dropout rate for DMSA.

    ORT_weight :
        The weight for the ORT loss.

    MIT_weight :
        The weight for the MIT loss.

    batch_size :
        The batch size for training and evaluating the model.

    epochs :
        The number of epochs for training the model.

    patience :
        The patience for the early-stopping mechanism. Given a positive integer, the training
        process will be stopped when the model does not perform better after that number of epochs.
        Leaving it default as None will disable the early-stopping.

    optimizer :
        The optimizer for model training.
        If not given, will use a default Adam optimizer.

    num_workers :
        The number of subprocesses to use for data loading.
        `0` means data loading will be in the main process, i.e. there won't be subprocesses.

    device :
        The device for the model to run on. It can be a string, a :class:`torch.device` object, or
        a list of them.
        If not given, will try to use CUDA devices first (will use the default CUDA device if
        there are multiple), then CPUs, considering CUDA and CPU are so far the main devices for
        people to train ML models.
        If given a list of devices, e.g. ['cuda:0', 'cuda:1'], or [torch.device('cuda:0'),
        torch.device('cuda:1')] , the model will be parallely trained on the multiple devices
        (so far only support parallel training on CUDA devices).
        Other devices like Google TPU and Apple Silicon accelerator MPS may be added in the future.

    saving_path :
        The path for automatically saving model checkpoints and tensorboard files (i.e. loss
        values recorded during training into a tensorboard file). Will not save if not given.

    model_saving_strategy :
        The strategy to save model checkpoints. It has to be one of [None, "best", "better"].
        No model will be saved when it is set as None.
        The "best" strategy will only automatically save the best model
        after the training finished.
        The "better" strategy will automatically save the model during training whenever
        the model performs better than in previous epochs.

    Attributes
    ----------
    model : :class:`torch.nn.Module`
        The underlying Transformer model.

    optimizer : :class:`pypots.optim.Optimizer`
        The optimizer for model training.

    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_layers: int,
        d_model: int,
        d_inner: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        dropout: float = 0,
        attn_dropout: float = 0,
        ORT_weight: int = 1,
        MIT_weight: int = 1,
        batch_size: int = 32,
        epochs: int = 100,
        patience: Optional[int] = None,
        optimizer: Optional[Optimizer] = Adam(),
        num_workers: int = 0,
        device: Optional[Union[str, torch.device, list]] = None,
        saving_path: Optional[str] = None,
        model_saving_strategy: Optional[str] = "best",
    ):
        super().__init__(
            batch_size,
            epochs,
            patience,
            num_workers,
            device,
            saving_path,
            model_saving_strategy,
        )

        self.n_steps = n_steps
        self.n_features = n_features
        # model hype-parameters
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight

        # set up the model
        self.model = _TransformerEncoder(
            self.n_layers,
            self.n_steps,
            self.n_features,
            self.d_model,
            self.d_inner,
            self.n_heads,
            self.d_k,
            self.d_v,
            self.dropout,
            self.ORT_weight,
            self.MIT_weight,
        )
        self._send_model_to_given_device()
        self._print_model_size()

        # set up the optimizer
        self.optimizer = optimizer
        self.optimizer.init_optimizer(self.model.parameters())

    def _assemble_input_for_training(self, data: dict) -> dict:
        (
            indices,
            X_intact,
            X,
            missing_mask,
            indicating_mask,
        ) = self._send_data_to_given_device(data)

        inputs = {
            "X": X,
            "X_intact": X_intact,
            "missing_mask": missing_mask,
            "indicating_mask": indicating_mask,
        }

        return inputs

    def _assemble_input_for_validating(self, data: list) -> dict:
        indices, X, missing_mask = self._send_data_to_given_device(data)

        inputs = {
            "X": X,
            "missing_mask": missing_mask,
        }

        return inputs

    def _assemble_input_for_testing(self, data: list) -> dict:
        return self._assemble_input_for_validating(data)

    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "h5py",
    ) -> None:
        # Step 1: wrap the input data with classes Dataset and DataLoader
        training_set = DatasetForSAITS(train_set, return_labels=False, file_type=file_type)
        training_loader = DataLoader(
            training_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = None
        if val_set is not None:
            if isinstance(val_set, str):
                with h5py.File(val_set, "r") as hf:
                    # Here we read the whole validation set from the file
                    # to mask a portion for validation.
                    # In PyPOTS, using a file usually because the data is too big.
                    # However, the validation set is generally shouldn't be too large.
                    # For example, we have 1 billion samples for model training.
                    # We won't take 20% of them as the validation set
                    # because we want as much as possible data for the
                    # training stage to enhance the model's generalization ability.
                    # Therefore, 100,000 representative samples will be enough
                    # to validate the model.
                    val_set = {
                        "X": hf["X"][:],
                        "X_intact": hf["X_intact"][:],
                        "indicating_mask": hf["indicating_mask"][:],
                    }

            val_set = BaseDataset(val_set, return_labels=False, file_type=file_type)
            val_loader = DataLoader(
                val_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        # Step 2: train the model and freeze it
        self._train_model(training_loader, val_loader)
        self.model.load_state_dict(self.best_model_dict)
        self.model.eval()  # set the model as eval status to freeze it.

        # Step 3: save the model if necessary
        self._auto_save_model_if_necessary(training_finished=True)

    def impute(self, X: Union[dict, str], file_type: str = "h5py") -> np.ndarray:
        self.model.eval()  # set the model as eval status to freeze it.
        test_set = BaseDataset(X, return_labels=False, file_type=file_type)
        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        imputation_collector = []
        attention_matrix_collector = []
        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                inputs = self._assemble_input_for_testing(data)
                results = self.model.forward(inputs, training=False)
                imputed_data = results["imputed_data"]
                attn_matrices = results["attn_matrices"]
                imputation_collector.append(imputed_data)
                attention_matrix_collector.append(attn_matrices)

        imputation_collector_ = torch.cat(imputation_collector)
        attention_matrix_collector_ = torch.cat(attention_matrix_collector)
        return (
            imputation_collector_.cpu().detach().numpy(),
            attention_matrix_collector_.cpu().detach().numpy(),
        )
