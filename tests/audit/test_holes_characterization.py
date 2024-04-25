import numpy as np
import pandas as pd
import pytest

from qolmat.audit.holes_characterization import MCARTest
from qolmat.imputations.imputers import ImputerEM


np.random.seed(11)
matrix = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=100)
matrix_1, matrix_2, matrix_3 = map(np.copy, [matrix] * 3)

# Case 1 : MCAR case detected by Little
matrix_1.ravel()[np.random.choice(matrix_1.size, size=20, replace=False)] = np.nan
df_1 = pd.DataFrame(matrix_1)

# Case 2 : MAR case detected by Little
matrix_2[np.argwhere(matrix_2[:, 0] > 1.96), 1] = np.nan
df_2 = pd.DataFrame(matrix_2)

# Case 3 : MAR case undetected by Little
matrix_3[np.argwhere(abs(matrix_3[:, 0]) >= 1.95), 1] = np.nan
df_3 = pd.DataFrame(matrix_3)


@pytest.mark.parametrize("df_input, expected", [(df_1, True), (df_2, False), (df_3, True)])
def test_little_mcar_test(df_input: pd.DataFrame, expected: bool):
    mcar_test_little = MCARTest(method="little", imputer=ImputerEM(random_state=42))
    result = mcar_test_little.test(df_input)
    assert expected == (result > 0.05)
