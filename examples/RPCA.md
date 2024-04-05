---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: env_qolmat_dev
    language: python
    name: env_qolmat_dev
---

```python tags=[]
%reload_ext autoreload
%autoreload 2

import numpy as np
# import timesynth as ts # package for generating time series

import matplotlib.pyplot as plt

import sys

from math import pi

from qolmat.utils import utils, plot, data
from qolmat.imputations.rpca.rpca_pcp import RpcaPcp
from qolmat.imputations.rpca.rpca_noisy import RpcaNoisy
from qolmat.imputations.softimpute import SoftImpute
from qolmat.imputations.rpca import rpca_utils
from qolmat.utils.data import generate_artificial_ts
```

**Generate synthetic data**

```python tags=[]
n_samples = 10000
periods = [100, 20]
amp_anomalies = 0.5
ratio_anomalies = 0.05
amp_noise = 0.1

X_true, A_true, E_true = generate_artificial_ts(n_samples, periods, amp_anomalies, ratio_anomalies, amp_noise)

signal = X_true + A_true + E_true

# Adding missing data
signal[120:180] = np.nan
signal[:20] = np.nan
# signal[80:220] = np.nan
# mask = np.random.choice(len(signal), round(len(signal) / 20))
# signal[mask] = np.nan

```

```python tags=[]
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(4, 1, 1)
ax.title.set_text("Low-rank signal")
plt.plot(X_true)

ax = fig.add_subplot(4, 1, 2)
ax.title.set_text("Corruption signal")
plt.plot(A_true)

ax = fig.add_subplot(4, 1, 3)
ax.title.set_text("Noise")
plt.plot(E_true)

ax = fig.add_subplot(4, 1, 4)
ax.title.set_text("Corrupted signal")
plt.plot(signal)

plt.show()
```

<!-- #region tags=[] -->
# Fit RPCA Noisy
<!-- #endregion -->

```python tags=[]
rpca_noisy = RpcaNoisy(tau=1, lam=.4, rank=1, norm="L2")
```

```python tags=[]
period = 100
D = utils.prepare_data(signal, period)
Omega = ~np.isnan(D)
D = utils.linear_interpolation(D)
```

```python tags=[]
M, A, L, Q = rpca_noisy.decompose_with_basis(D, Omega)
M2, A2 = rpca_noisy.decompose_on_basis(D, Omega, Q)
```

```python tags=[]
M_final = utils.get_shape_original(M, signal.shape)
A_final = utils.get_shape_original(A, signal.shape)
D_final = utils.get_shape_original(D, signal.shape)
signal_imputed = M_final + A_final
```

```python tags=[]
fig = plt.figure(figsize=(12, 4))

plt.plot(signal_imputed, label="Imputed signal with anomalies")
plt.plot(M_final, label="Imputed signal without anomalies")
plt.plot(A_final, label="Anomalies")
# plt.plot(D_final, label="D")
plt.plot(signal, color="black", label="Original signal")
plt.xlim(0, 400)
plt.legend()
plt.show()
```

## PCP RPCA

```python tags=[]
rpca_pcp = RpcaPcp(max_iterations=1000, lam=.1)
```

```python tags=[]
period = 100
D = utils.prepare_data(signal, period)
Omega = ~np.isnan(D)
D = utils.linear_interpolation(D)
```

```python tags=[]
M, A = rpca_pcp.decompose(D, Omega)
```

```python tags=[]
M_final = utils.get_shape_original(M, signal.shape)
A_final = utils.get_shape_original(A, signal.shape)
D_final = utils.get_shape_original(D, signal.shape)
# Y_final = utils.get_shape_original(Y, signal.shape)
signal_imputed = M_final + A_final
```

```python tags=[]
fig = plt.figure(figsize=(12, 4))

plt.plot(signal_imputed, label="Imputed signal with anomalies")
plt.plot(M_final, label="Imputed signal without anomalies")
plt.plot(A_final, label="Anomalies")

plt.plot(signal, color="black", label="Original signal")
plt.xlim(0, 400)
# plt.gca().twinx()
# plt.plot(Y_final, label="Y")
plt.legend()
plt.show()
```

## Soft Impute

```python tags=[]
imputer = SoftImpute(max_iterations=1000, tau=.1)
```

```python tags=[]
period = 100
D = utils.prepare_data(signal, period)
Omega = ~np.isnan(D)
D = utils.linear_interpolation(D)
```

```python tags=[]
M, A = imputer.decompose(D, Omega)
```

```python tags=[]
M_final = utils.get_shape_original(M, signal.shape)
A_final = utils.get_shape_original(A, signal.shape)
D_final = utils.get_shape_original(D, signal.shape)
# Y_final = utils.get_shape_original(Y, signal.shape)
signal_imputed = M_final + A_final
```

```python tags=[]
fig = plt.figure(figsize=(12, 4))

plt.plot(signal_imputed, label="Imputed signal with anomalies")
plt.plot(M_final, label="Imputed signal without anomalies")
plt.plot(A_final, label="Anomalies")

plt.plot(signal, color="black", label="Original signal")
plt.xlim(0, 400)
plt.legend()
plt.show()
```

## Temporal RPCA

```python
%%time
# rpca_noisy = RpcaNoisy(period=10, tau=1, lam=0.4, rank=2, list_periods=[10], list_etas=[0.01], norm="L2")
rpca_noisy = RpcaNoisy(tau=1, lam=0.4, rank=2, norm="L2")
M, A = rpca_noisy.decompose(D, Omega)
# imputed = X
```

```python tags=[]
fig = plt.figure(figsize=(12, 4))

plt.plot(signal_imputed, label="Imputed signal with anomalies")
plt.plot(M_final, label="Imputed signal without anomalies")
plt.plot(A_final, label="Anomalies")

plt.plot(signal, color="black", label="Original signal")
plt.xlim(0, 400)
# plt.gca().twinx()
# plt.plot(Y_final, label="Y")
plt.legend()
plt.show()
```

# EM VAR(p)

```python
from qolmat.imputations import em_sampler
```

```python
p = 1
model = em_sampler.VARpEM(method="mle", max_iter_em=10, n_iter_ou=512, dt=1e-1, p=p)
```

```python
D = signal.reshape(-1, 1)
M_final = model.fit_transform(D)
```

```python
fig = plt.figure(figsize=(12, 4))
plt.plot(signal_imputed, label="Imputed signal with anomalies")
plt.plot(M_final, label="Imputed signal without anomalies")
plt.xlim(0, 400)
plt.legend()
plt.show()
```

```python

```
