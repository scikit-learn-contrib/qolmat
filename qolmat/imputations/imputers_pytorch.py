import pandas as pd
import numpy as np

from typing import Callable, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator

from qolmat.imputations.imputers import ImputerRegressor
from qolmat.utils.exceptions import PyTorchExtraNotInstalled

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ModuleNotFoundError:
    raise PyTorchExtraNotInstalled


def build_mlp_example(
    input_dim: int,
    list_num_neurons: List[int],
    output_dim: int = 1,
    activation: Callable = nn.ReLU,
) -> nn.Sequential:
    """
    Constructs a multi-layer perceptron (MLP) with a custom architecture.

    Parameters
    ----------
    input_dim : int
        Dimension of the input layer.
    list_num_neurons : List[int]
        List specifying the number of neurons in each hidden layer.
    output_dim : int, optional
        Dimension of the output layer, defaults to 1.
    activation : nn.Module, optional
        Activation function to use between hidden layers, defaults to nn.ReLU().

    Returns
    -------
    nn.Sequential
        PyTorch model representing the MLP.

    Raises
    ------
    TypeError
        If `input_dim` is not an integer or `list_num_neurons` is not a list.

    Examples
    --------
    >>> model = build_mlp_example(input_dim=10, list_num_neurons=[32, 64, 128], output_dim=1)
    >>> print(model)
    Sequential(
      (0): Linear(in_features=10, out_features=32, bias=True)
      (1): ReLU()
      (2): Linear(in_features=32, out_features=64, bias=True)
      (3): ReLU()
      (4): Linear(in_features=64, out_features=128, bias=True)
      (5): ReLU()
      (6): Linear(in_features=128, out_features=1, bias=True)
    )
    """
    layers = []
    for num_neurons in list_num_neurons:
        layers.append(nn.Linear(input_dim, num_neurons))
        layers.append(activation())
        input_dim = num_neurons
    layers.append(nn.Linear(input_dim, output_dim))

    estimator = nn.Sequential(*layers)
    return estimator


def build_autoencoder_example(
    input_dim: int,
    latent_dim: int,
    list_num_neurons: List[int],
    output_dim: int = 1,
    activation: Callable = nn.ReLU,
) -> Tuple[nn.Sequential, nn.Sequential]:
    """
    Constructs an autoencoder with a custom architecture.

    Parameters
    ----------
    input_dim : int
        Dimension of the input layer.
    latent_dim : int
        Dimension of the latent space.
    list_num_neurons : List[int]
        List specifying the number of neurons in each hidden layer.
    output_dim : int, optional
        Dimension of the output layer, defaults to 1.
    activation : nn.Module, optional
        Activation function to use between hidden layers, defaults to nn.ReLU().

    Returns
    -------
    Tuple[nn.Sequential, nn.Sequential]
        Tuple containing the encoder and decoder models.

    Raises
    ------
    TypeError
        If `input_dim` is not an integer or `list_num_neurons` is not a list.

    Examples
    --------
    >>> encoder, decoder = build_autoencoder_example(
                                                        input_dim=10,
                                                        latent_dim=4,
                                                        list_num_neurons=[32, 64, 128],
                                                        output_dim=252
                                                    )
    >>> print(encoder)
    Sequential(
      (0): Linear(in_features=10, out_features=128, bias=True)
      (1): ReLU()
      (2): Linear(in_features=128, out_features=64, bias=True)
      (3): ReLU()
      (4): Linear(in_features=64, out_features=32, bias=True)
      (5): ReLU()
      (6): Linear(in_features=32, out_features=4, bias=True)
    )
    >>> print(decoder)
    Sequential(
      (0): Linear(in_features=4, out_features=32, bias=True)
      (1): ReLU()
      (2): Linear(in_features=32, out_features=64, bias=True)
      (3): ReLU()
      (4): Linear(in_features=64, out_features=128, bias=True)
      (5): ReLU()
      (6): Linear(in_features=128, out_features=252, bias=True)
    )
    """

    encoder = build_mlp_example(
        input_dim=input_dim,
        output_dim=latent_dim,
        list_num_neurons=np.sort(list_num_neurons)[::-1].tolist(),
        activation=activation,
    )
    decoder = build_mlp_example(
        input_dim=latent_dim,
        output_dim=output_dim,
        list_num_neurons=np.sort(list_num_neurons).tolist(),
        activation=activation,
    )
    return encoder, decoder


class ImputerRegressorPyTorch(ImputerRegressor):
    def __init__(
        self,
        groups: Tuple[str, ...] = (),
        estimator: Optional[BaseEstimator] = None,
        handler_nan: str = "column",
        epochs: int = 100,
        learning_rate: float = 0.001,
        loss_fn: Callable = nn.L1Loss(),
    ):
        super().__init__(
            imputer_params=("handler_nan", "epochs", "monitor", "patience"),
            groups=groups,
            handler_nan=handler_nan,
        )
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.estimator = estimator

    def _fit_estimator(self, X: pd.DataFrame, y: pd.DataFrame, **hp):
        """
        Fit the PyTorch estimator using the provided input and target data.

        Parameters
        ----------
        X : pd.DataFrame
            The input data for training.
        y : pd.DataFrame
            The target data for training.
        """
        optimizer = optim.Adam(self.estimator.parameters(), lr=self.learning_rate)
        loss_fn = self.loss_fn
        if self.estimator is None:
            assert AssertionError("Estimator is not provided.")
        else:
            for epoch in range(self.epochs):
                self.estimator.train()
                optimizer.zero_grad()

                input_data = torch.Tensor(X.values)
                target_data = torch.Tensor(y.values)
                target_data = target_data.unsqueeze(1)
                outputs = self.estimator(input_data)
                loss = loss_fn(outputs, target_data)

                loss.backward()
                optimizer.step()
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}")

    def _predict_estimator(self, X: pd.DataFrame) -> pd.Series:
        """
        Perform predictions using the trained PyTorch estimator.

        Parameters
        ----------
        X : pd.DataFrame
            The input data for prediction.

        Returns
        -------
        pd.Series
            The predicted values.
        """
        if self.estimator:
            input_data = torch.Tensor(X.values)
            output_data = self.estimator(input_data)
            y = pd.Series(output_data.detach().numpy().flatten())
        return y


class Autoencoder(nn.Module):
    def __init__(
        self,
        encoder: nn.Sequential,
        decoder: nn.Sequential,
        epochs: int = 100,
        learning_rate: float = 0.001,
        loss_fn: Callable = nn.L1Loss(),
    ):
        super(Autoencoder, self).__init__()

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.encoder = encoder
        self.decoder = decoder

        self.loss: List[List[float]] = []

    def forward(self, x: pd.DataFrame) -> pd.DataFrame:
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return decode

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_fn = self.loss_fn
        list_loss = []
        for epoch in range(self.epochs):
            self.train()
            optimizer.zero_grad()

            input_data = torch.Tensor(X.values)
            target_data = torch.Tensor(y.values)
            outputs = self(input_data)
            loss = loss_fn(outputs, target_data)

            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}")
            list_loss.append(loss.item())
        self.loss.extend([list_loss])

    def imputation_data(
        self, df: pd.DataFrame, mask: pd.DataFrame, lamb: float, iterations: int
    ) -> pd.DataFrame:
        df_train = df.copy()
        df_train = df_train.fillna(df_train.mean())
        scaler = StandardScaler()
        df_train_scaler = pd.DataFrame(
            scaler.fit_transform(df_train), index=df_train.index, columns=df_train.columns
        )
        df_itt = df_train_scaler

        for _ in range(iterations):
            self.fit(df_itt, df_itt)
            encode_tensor = self.encoder(torch.Tensor(df_itt.values))
            encode = encode_tensor.detach().numpy()

            scaler_encode = StandardScaler()
            encode_scaler = scaler_encode.fit_transform(encode)
            W = np.sqrt(lamb) * torch.randn(encode.shape).detach().numpy()
            Z_itt = (1 - lamb) * encode_scaler + W
            Z_itt = scaler_encode.inverse_transform(Z_itt)
            y_itt_tensor = self.decoder(torch.Tensor(Z_itt))
            y_itt = pd.DataFrame(
                y_itt_tensor.detach().numpy(), index=df_itt.index, columns=df_itt.columns
            )
            df_itt[mask] = y_itt[mask]
        df_imputed = pd.DataFrame(
            scaler.inverse_transform(df_itt), index=df_train.index, columns=df_train.columns
        )
        return df_imputed
