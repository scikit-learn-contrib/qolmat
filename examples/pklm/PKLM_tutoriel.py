#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd
import numpy as np
from scipy.stats import norm

from qolmat.analysis.pklm import PKLMtest
from qolmat.benchmark.missing_patterns import UniformHoleGenerator

# ### DataFrame definition

# In[5]:


nb_rows = 40
nb_col = 5

# Création du DataFrame avec des données aléatoires
data = {f"Colonne_{i}": np.random.randint(0, 100, nb_rows).astype(float) for i in range(nb_col)}

df = pd.DataFrame(data)

# Introduction de valeurs manquantes
nb_valeurs_manquantes = int(0.1 * df.size)
indices_valeurs_manquantes = np.random.choice(df.size, nb_valeurs_manquantes, replace=False)
df.values.flat[indices_valeurs_manquantes] = np.nan


# __Remarque__ : On remarque que parfois le calcul du U_hat n'est pas possible pour une
# certaine projection.

# In[5]:


PKLMtest(df, 10, 10)


# ### Tests sur des matrices typiques (utilisées dans le test de Little)

# ### The MCAR real case

# In[2]:


np.random.seed(42)
matrix = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=200)
df = pd.DataFrame(data=matrix, columns=["Column_1", "Column_2"])

hole_gen = UniformHoleGenerator(n_splits=1, random_state=42, subset=["Column_2"], ratio_masked=0.2)
df_mask = hole_gen.generate_mask(df)
df_unmasked = ~df_mask
df_unmasked["Column_1"] = False

df_observed = df.mask(df_mask).dropna()
df_hidden = df.mask(df_unmasked).dropna(subset="Column_2")


# In[4]:


PKLMtest(df.mask(df_mask), 10, 10)


# __Résultat__ : Très cohérent les trous sont bien MCAR.

# ### The MAR case  : Heterogeneity in means

# In[11]:


np.random.seed(42)
quantile_95 = norm.ppf(0.975)

matrix = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=200)
df = pd.DataFrame(matrix, columns=["Column_1", "Column_2"])
df_nan = df.copy()
df_nan.loc[df_nan["Column_1"] > quantile_95, "Column_2"] = np.nan

df_mask = df_nan.isna()
df_unmasked = ~df_mask
df_unmasked["Column_1"] = False

df_observed = df.mask(df_mask).dropna()
df_hidden = df.mask(df_unmasked).dropna(subset="Column_2")


# In[12]:


PKLMtest(df.mask(df_mask), 20, 20)


# __Résultat__ : Cohérent les trous sont MAR.

# ### The MAR case  : Heterogeneity in covariance

# In[13]:


np.random.seed(42)
quantile_95 = norm.ppf(0.975)

matrix = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=200)
df = pd.DataFrame(matrix, columns=["Column_1", "Column_2"])
df_nan = df.copy()
df_nan.loc[abs(df_nan["Column_1"]) > quantile_95, "Column_2"] = np.nan

df_mask = df_nan.isna()
df_unmasked = ~df_mask
df_unmasked["Column_1"] = False

df_observed = df.mask(df_mask).dropna()
df_hidden = df.mask(df_unmasked).dropna(subset="Column_2")


# In[14]:


PKLMtest(df.mask(df_mask), 20, 20)


# __Résultat__ : Cohérent les trous sont MAR.
#
# Pourquoi la p-value du test est-elle éxacetement égale à la précédente
