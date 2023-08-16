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
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
%reload_ext autoreload
%autoreload 2

import numpy as np
# import timesynth as ts # package for generating time series

import matplotlib.pyplot as plt

import sys

from math import pi

from qolmat.utils import plot, data
from qolmat.imputations.rpca.rpca_pcp import RPCAPCP
from qolmat.imputations.rpca.rpca_noisy import RPCANoisy
from qolmat.imputations.rpca import rpca_utils
from qolmat.utils.data import generate_artificial_ts
```

**Generate synthetic data**

```python
n_samples = 1000
periods = [100, 20]
amp_anomalies = 0.5
ratio_anomalies = 0.05
amp_noise = 0.1

X_true, A_true, E_true = generate_artificial_ts(n_samples, periods, amp_anomalies, ratio_anomalies, amp_noise)

signal = X_true + A_true + E_true

# Adding missing data
#signal[5:20] = np.nan
mask = np.random.choice(len(signal), round(len(signal) / 20))
signal[mask] = np.nan

```

```python
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

## PCP RPCA

```python
%%time
rpca_pcp = RPCAPCP(period=100, max_iterations=100, mu=.5, lam=0.1)
X, A = rpca_pcp.decompose_rpca_signal(signal)
imputed = signal - A
```

```python
fig = plt.figure(figsize=(12, 4))
plt.plot(X, color="black")
plt.plot(imputed)
```

## Temporal RPCA

```python
signal.shape
```

```python
%%time
# rpca_noisy = RPCANoisy(period=10, tau=1, lam=0.4, rank=2, list_periods=[10], list_etas=[0.01], norm="L2")
rpca_noisy = RPCANoisy(period=10, tau=1, lam=0.4, rank=2, norm="L2")
X, A = rpca_noisy.decompose_rpca_signal(signal)
imputed =
```

```python
fig = plt.figure(figsize=(12, 4))
plt.plot(signal, color="black")
plt.plot(X_true)
plt.plot(X)
```

```python

```
