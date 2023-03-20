import numpy as np
import pandas as pd
from qolmat.imputations import imputers

df = pd.DataFrame({"col1": [0, 1, 2, 3, 4], "col2": [1, 2, np.nan, np.nan, np.nan]})

df_incomplete = pd.DataFrame(
    {"col1": [0, 1, np.nan, np.nan, 3, 4], "col2": [np.nan, np.nan, 0.5, np.nan, 1.5, 4]}
)

# print(df.mean())

imputer = imputers.ImputerNOCB()
print(imputer.fit_transform(df_incomplete))

# imputer = imputers.ImputerShuffle()
# print(imputer.fit_transform(df))
