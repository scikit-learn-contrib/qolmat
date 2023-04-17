from tensorflow.keras.callbacks import EarlyStopping


class Hyperparam_dl:
    """
    This class implements a MLP imputer in the multivariate case.
    It imputes each Series with missing value within a DataFrame using the complete ones.

    Parameters
    ----------
    model :
        MLP model

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from qolmat.imputations.models import ImputeRegressor
    >>> from tensorflow.keras import ExtraTreesRegressor
    >>> estimator = tf.keras.Sequential([
    >>>                                tf.keras.layers.Dense(128, activation='relu'),
    >>>                                tf.keras.layers.Dense(64, activation='relu'),
    >>>                                tf.keras.layers.Dense(1)
    >>>                                ])
    >>> estimator.compile(optimizer='adam',
    >>>                 loss='mse',
    >>>                 metrics=['mae'])
    >>> df = pd.DataFrame(data=[[1, 1, 1, 1],
    >>>                       [np.nan, np.nan, 2, 3],
    >>>                       [1, 2, 2, 5], [2, 2, 2, 2]],
    >>>                       columns=["var1", "var2", "var3", "var4"])
    >>> estimator.fit_transform(df)
    """

    def __init__(
        self, epoch: int = 100, verbose: int = 0, monitor: str = "loss", patience: int = 5
    ):

        es = EarlyStopping(monitor=monitor, patience=patience, verbose=verbose, mode="min")
        self.epochs = epoch
        self.callbacks = [es]
        self.verbose = 0
