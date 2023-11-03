import pandas as pd
import numpy as np

from typing import Any, Callable, List, Optional, Tuple, Union, Dict
from typing_extensions import Self
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator

from qolmat.imputations.imputers import _Imputer, ImputerRegressor
from qolmat.utils.exceptions import EstimatorNotDefined, PyTorchExtraNotInstalled
from qolmat.benchmark import metrics

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ModuleNotFoundError:
    raise PyTorchExtraNotInstalled


class ImputerRegressorPyTorch(ImputerRegressor):
    """
    This class inherits from the class ImputerRegressor and allows for PyTorch regressors.

    Parameters
    ----------
    groups: Tuple[str, ...]
        List of column names to group by, by default []
    estimator : torch.nn.Sequential, optional
        PyTorch estimator for imputing a column based on the others
    handler_nan : str
        Can be `fit, `row` or `column`:
        - if `fit`, the estimator is assumed to be fitted on parcelar data,
        - if `row` all non complete rows will be removed from the train dataset, and will not be
        used for the inferance,
        - if `column`all non complete columns will be ignored.
    epochs: int
        Number of epochs when fitting the autoencoder, by default 100
    learning_rate: float
        Learning rate hen fitting the autoencoder, by default 0.001
    loss_fn: Callable
        Loss used when fitting the autoencoder, by default nn.L1Loss()
    """

    def __init__(
        self,
        groups: Tuple[str, ...] = (),
        estimator: Optional[nn.Sequential] = None,
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

    def _fit_estimator(self, estimator: nn.Sequential, X: pd.DataFrame, y: pd.DataFrame) -> Any:
        """
        Fit the PyTorch estimator using the provided input and target data.

        Parameters
        ----------
        X : pd.DataFrame
            The input data for training.
        y : pd.DataFrame
            The target data for training.

        Returns
        -------
        Any
            Return fitted PyTorch estimator.
        """
        if not estimator:
            raise EstimatorNotDefined()
        optimizer = optim.Adam(estimator.parameters(), lr=self.learning_rate)
        loss_fn = self.loss_fn
        if estimator is None:
            assert EstimatorNotDefined()
        else:
            for epoch in range(self.epochs):
                estimator.train()
                optimizer.zero_grad()

                input_data = torch.Tensor(X.values)
                target_data = torch.Tensor(y.values)
                target_data = target_data.unsqueeze(1)
                outputs = estimator(input_data)
                loss = loss_fn(outputs, target_data)

                loss.backward()
                optimizer.step()
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}")
        return estimator

    def _predict_estimator(self, estimator: nn.Sequential, X: pd.DataFrame) -> pd.Series:
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

        Raises
        ------
        EstimatorNotDefined
            Raises an error if the attribute estimator is not defined.
        """
        if not estimator:
            raise EstimatorNotDefined()
        input_data = torch.Tensor(X.values)
        output_data = estimator(input_data)
        y = pd.Series(output_data.detach().numpy().flatten())
        return y


class Autoencoder(nn.Module):
    """
    Wrapper of a PyTorch autoencoder allowing to encode

    Parameters
    ----------
    encoder : nn.Sequential
        The encoder module.
    decoder : nn.Sequential
        The decoder module.
    epochs : int, optional
        Number of epochs for training, by default 100.
    learning_rate : float, optional
        Learning rate for optimization, by default 0.001.
    loss_fn : Callable, optional
        Loss function for training, by default nn.L1Loss().
    """

    def __init__(
        self,
        encoder: nn.Sequential,
        decoder: nn.Sequential,
        epochs: int = 100,
        learning_rate: float = 0.001,
        loss_fn: Callable = nn.L1Loss(),
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss: List[List[float]] = []
        self.scaler = StandardScaler()

    def forward(self, x: NDArray) -> nn.Sequential:
        """
        Forward pass through the autoencoder.

        Parameters
        ----------
        x : pd.DataFrame
            Input data.

        Returns
        -------
        pd.DataFrame
            Decoded data.
        """
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return decode

    def fit(self, X: NDArray, y: NDArray) -> Self:
        """
        Fit the autoencoder to the data.

        Parameters
        ----------
        X : ndarray
            Input data for training.
        y : ndarray
            Target data for training.

        Returns
        -------
        Self
            Return Self
        """
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_fn = self.loss_fn
        list_loss = []
        for epoch in range(self.epochs):
            self.train()
            optimizer.zero_grad()

            input_data = torch.Tensor(X)
            target_data = torch.Tensor(y)
            outputs = self(input_data)
            loss = loss_fn(outputs, target_data)

            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}")
            list_loss.append(loss.item())
        self.loss.extend([list_loss])
        return self

    def decode(self, Z: NDArray) -> NDArray:
        """
        Decode encoded data.

        Parameters
        ----------
        Z : ndarray
            Encoded data.

        Returns
        -------
        ndarray
            Decoded data.
        """
        Z_decoded = self.scaler.inverse_transform(Z)
        Z_decoded = self.decoder(torch.Tensor(Z_decoded))
        Z_decoded = Z_decoded.detach().numpy()
        return Z_decoded

    def encode(self, X: NDArray) -> NDArray:
        """
        Encode input data.

        Parameters
        ----------
        X : ndarray
            Input data.

        Returns
        -------
        ndarray
            Encoded data.
        """
        X_encoded = self.encoder(torch.Tensor(X))
        X_encoded = X_encoded.detach().numpy()
        X_encoded = self.scaler.fit_transform(X_encoded)
        return X_encoded


class ImputerAutoencoder(_Imputer):
    """Impute by the mean of the column.

    Parameters
    ----------
    groups: Tuple[str, ...]
        List of column names to group by, by default []
    lamb: float
        Sampling step
    max_iterations: int
        Maximal number of iterations in the sampling process
    epochs: int
        Number of epochs when fitting the autoencoder, by default 100
    learning_rate: float
        Learning rate hen fitting the autoencoder, by default 0.001
    loss_fn: Callable
        Loss used when fitting the autoencoder, by default nn.L1Loss()
    """

    def __init__(
        self,
        encoder: nn.Sequential,
        decoder: nn.Sequential,
        groups: Tuple[str, ...] = (),
        random_state: Union[None, int, np.random.RandomState] = None,
        lamb: float = 1e-2,
        max_iterations: int = 100,
        epochs: int = 100,
        learning_rate: float = 0.001,
        loss_fn: Callable = nn.L1Loss(),
    ) -> None:
        super().__init__(groups=groups, columnwise=False, shrink=False, random_state=random_state)
        self.loss_fn = loss_fn
        self.lamb = lamb
        self.max_iterations = max_iterations
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.encoder = encoder
        self.decoder = decoder

    def _fit_element(self, df: pd.DataFrame, col: str = "__all__", ngroup: int = 0) -> Autoencoder:
        """
        Fits the imputer on `df`, at the group and/or column level depending onself.groups and
        self.columnwise.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe on which the imputer is fitted
        col : str, optional
            Column on which the imputer is fitted, by default "__all__"
        ngroup : int, optional
            Id of the group on which the method is applied

        Returns
        -------
        Any
            Return fitted encoder

        Raises
        ------
        NotDataFrame
            Input has to be a pandas.DataFrame.
        """
        self._check_dataframe(df)
        autoencoder = Autoencoder(
            self.encoder,
            self.decoder,
            self.epochs,
            self.learning_rate,
            self.loss_fn,
        )
        X = df.fillna(df.mean()).values
        return autoencoder.fit(X, X)

    def _transform_element(
        self, df: pd.DataFrame, col: str = "__all__", ngroup: int = 0
    ) -> pd.DataFrame:
        """
        Transforms the dataframe `df`, at the group and/or column level depending onself.groups and
        self.columnwise.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe or column to impute
        col : str, optional
            Column transformed by the imputer, by default "__all__"
        ngroup : int, optional
            Id of the group on which the method is applied

        Returns
        -------
        pd.DataFrame
            Imputed dataframe

        Raises
        ------
        NotDataFrame
            Input has to be a pandas.DataFrame.
        """
        autoencoder = self._dict_fitting[col][ngroup]
        df_train = df.copy()
        df_train = df_train.fillna(df_train.mean())
        scaler = StandardScaler()
        df_train_scaler = pd.DataFrame(
            scaler.fit_transform(df_train), index=df_train.index, columns=df_train.columns
        )
        X = df_train_scaler.values
        mask = df.isna().values
        for _ in range(self.max_iterations):
            self.fit(X, X)
            Z = autoencoder.encode(X)
            W = np.sqrt(self.lamb) * self._rng.normal(0, 1, size=Z.shape)
            Z_next = (1 - self.lamb) * Z + W
            X_next = autoencoder.decode(Z_next)
            X[mask] = X_next[mask]
        df_imputed = pd.DataFrame(
            scaler.inverse_transform(X), index=df_train.index, columns=df_train.columns
        )
        return df_imputed


def build_mlp(
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
    >>> model = build_mlp(input_dim=10, list_num_neurons=[32, 64, 128], output_dim=1)
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


def build_autoencoder(
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
    >>> encoder, decoder = build_autoencoder(input_dim=10,
    ...                                      latent_dim=4,
    ...                                      list_num_neurons=[32, 64, 128],
    ...                                      output_dim=252)
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

    encoder = build_mlp(
        input_dim=input_dim,
        output_dim=latent_dim,
        list_num_neurons=np.sort(list_num_neurons)[::-1].tolist(),
        activation=activation,
    )
    decoder = build_mlp(
        input_dim=latent_dim,
        output_dim=output_dim,
        list_num_neurons=np.sort(list_num_neurons).tolist(),
        activation=activation,
    )
    return encoder, decoder


class ImputerDiffusion(_Imputer):
    """This class inherits from the class _Imputer.
    It is a wrapper for imputers based on diffusion models.
    """

    def __init__(
        self,
        groups: Tuple[str, ...] = (),
        model: Optional[BaseEstimator] = None,
        epochs: int = 100,
        batch_size: int = 100,
        x_valid: pd.DataFrame = None,
        print_valid: bool = False,
        metrics_valid: Tuple[Callable, ...] = (
            metrics.mean_absolute_error,
            metrics.dist_wasserstein,
        ),
        round: int = 10,
        cols_imputed: Tuple[str, ...] = (),
        index_datetime: str = "",
        freq_str: str = "1D",
    ):
        """This class inherits from the class _Imputer.
        It is a wrapper for imputers based on diffusion models.

        Parameters
        ----------
        groups : Tuple[str, ...], optional
            List of column names to group by, by default ()
        model : Optional[BaseEstimator], optional
            Imputer based on diffusion models (e.g., TabDDPM, TsDDPM),
            by default None
        epochs : int, optional
            Number of epochs, by default 10
        batch_size : int, optional
            Batch size, by default 100
        x_valid : pd.DataFrame, optional
            Dataframe for validation, by default None
        print_valid : bool, optional
            Print model performance for after several epochs, by default False
        metrics_valid : Tuple[Callable, ...], optional
            Set of validation metrics, by default ( metrics.mean_absolute_error,
            metrics.dist_wasserstein )
        round : int, optional
            Number of decimal places to round to, for better displaying model
            performance, by default 10
        cols_imputed : Tuple[str, ...], optional
            Name of columns that need to be imputed, by default ()
        index_datetime : str
            Name of datetime-like index.
            It is for processing time-series data, used in diffusion models e.g., TsDDPM.
        freq_str : str
            Frequency string of DateOffset of Pandas.
            It is for processing time-series data, used in diffusion models e.g., TsDDPM.
        """
        super().__init__(groups=groups, columnwise=False)
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.x_valid = x_valid
        self.print_valid = print_valid
        self.metrics_valid = metrics_valid
        self.round = round
        self.cols_imputed = cols_imputed
        self.index_datetime = index_datetime
        self.freq_str = freq_str

    def _more_tags(self):
        return {
            "non_deterministic": True,
            "_xfail_checks": {
                "check_estimators_pickle": "Diffusion models can return\
                                  different outputs",
                "check_estimators_overwrite_params": "Diffusion models can\
                                    return different outputs",
            },
        }

    def _fit_element(self, df: pd.DataFrame, col: str = "__all__", ngroup: int = 0):
        """
        Fits the imputer on `df`, at the group and/or column level depending onself.groups and
        self.columnwise.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe on which the imputer is fitted
        col : str, optional
            Column on which the imputer is fitted, by default "__all__"
        ngroup : int, optional
            Id of the group on which the method is applied

        Returns
        -------
        Any
            Return fitted model

        Raises
        ------
        NotDataFrame
            Input has to be a pandas.DataFrame.
        """
        self._check_dataframe(df)
        hp = self._get_params_fit()
        return self.model.fit(df, **hp)

    def _transform_element(
        self, df: pd.DataFrame, col: str = "__all__", ngroup: int = 0
    ) -> pd.DataFrame:
        """
        Transforms the dataframe `df`, at the group and/or column level depending on self.groups
        and self.columnwise.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe or column to impute
        col : str, optional
            Column transformed by the imputer, by default "__all__"
        ngroup : int, optional
            Id of the group on which the method is applied

        Returns
        -------
        pd.DataFrame
            Imputed dataframe

        Raises
        ------
        NotDataFrame
            Input has to be a pandas.DataFrame.
        """
        df_imputed = self.model.predict(df)
        return df_imputed

    def _get_params_fit(self) -> Dict:
        hyperparams = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "x_valid": self.x_valid,
            "print_valid": self.print_valid,
            "metrics_valid": self.metrics_valid,
            "round": self.round,
            "cols_imputed": self.cols_imputed,
        }
        if self.index_datetime != "":
            hyperparams = {
                **hyperparams,
                **{
                    "index_datetime": self.index_datetime,
                    "freq_str": self.freq_str,
                },
            }

        return hyperparams

    def get_summary_training(self) -> Dict:
        return self.model.summary

    def get_summary_architecture(self) -> Dict:
        return {"number_parameters": self.model.num_params, "epsilon_model": self.model._eps_model}
