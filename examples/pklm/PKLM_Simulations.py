#!/usr/bin/env python
# coding: utf-8

# ### L'objectif de ce notebook est de récréer les résultats du papiers.
#
# Il reste encore pas mal de travail.

# In[18]:


import numpy as np
import pandas as pd
from scipy.stats import multivariate_t

from qolmat.audit.pklm import PKLMtest


# In[4]:


# Set the parameters
n = 1000  # Number of samples
p = 5  # Number of dimensions

# Mean vector of zeros
mean = np.zeros(p)

# Identity covariance matrix
cov = np.eye(p)

# Generate the multivariate normal distribution
data = np.random.multivariate_normal(mean, cov, n)

# Create a DataFrame
df = pd.DataFrame(data, columns=[f"Feature_{i+1}" for i in range(p)])

# Display the first few rows of the DataFrame
df.head()


# In[10]:


def generate_df_case_1(n: int, p: int) -> pd.DataFrame:
    mean = np.zeros(p)
    cov = np.eye(p)
    data = np.random.multivariate_normal(mean, cov, n)
    return pd.DataFrame(data, columns=[f"Feature_{i+1}" for i in range(p)])


# In[16]:


def generate_df_case_2(n: int, p: int) -> pd.DataFrame:
    cov = np.full((p, p), 0.7)
    np.fill_diagonal(cov, 1)
    mean = np.zeros(p)
    data = np.random.multivariate_normal(mean, cov, n)
    return pd.DataFrame(data, columns=[f"Feature_{i+1}" for i in range(p)])


# In[27]:


def generate_df_case_3(n: int, p: int, df: int = 4) -> pd.DataFrame:
    mean = np.zeros(4)
    cov = np.eye(4)
    rv = multivariate_t(loc=mean, shape=cov, df=4)
    data = rv.rvs(size=n)
    return pd.DataFrame(data, columns=[f"Feature_{i+1}" for i in range(p)])


# In[30]:


def generate_df_case_4(n: int, p: int, df: int = 4) -> pd.DataFrame:
    cov = np.full((p, p), 0.7)
    np.fill_diagonal(cov, 1)
    mean = np.zeros(p)
    rv = multivariate_t(loc=mean, shape=cov, df=4)
    data = rv.rvs(size=n)
    return pd.DataFrame(data, columns=[f"Feature_{i+1}" for i in range(p)])


# In[32]:


def generate_df_case_5(n: int, p: int) -> pd.DataFrame:
    return pd.DataFrame(np.random.rand(n, p), columns=[f"Feature_{i+1}" for i in range(p)])


# ### Generate the MCAR holes

# In[7]:


def gen_mcar_holes(df: pd.DataFrame, r: float) -> pd.DataFrame:
    _, n_col = df.shape
    mask = np.random.rand(*df.shape) < (1 - r ** (1 / n_col))
    df[mask] = np.nan
    return df


# In[12]:


df_1 = generate_df_case_1(n=200, p=4)
df_1_mcar = gen_mcar_holes(df_1, 0.65)
PKLMtest(df_1_mcar, nb_projections=100, nb_permutations=30)


# In[15]:


PKLMtest(df_1_mcar, nb_projections=100, nb_permutations=30)


# Ça a vraiment pris beaucoup de temps. (environ 10 minutes).

# ### Generate the MAR holes

# In[38]:
n = 200
p = 4
M = pd.DataFrame(np.zeros((n, p)), columns=[f"Feature_{i+1}" for i in range(p)])


def generate_m(df: pd.DataFrame, r: float) -> pd.DataFrame:
    _, n_col = df.shape
    mask = np.random.rand(*df.shape) < (1 - r ** (1 / (n_col - 1)))
    df[mask] = 1
    df.iloc[:, 0] = 0
    return df


generate_m(M, 0.65)

# In[39]:


my_df = generate_df_case_1(200, 4)


# In[40]:


my_df.head(1)


# In[42]:


my_df["Feature_1"].apply(lambda x: x > my_df["Feature_1"].mean()).sort_values()
