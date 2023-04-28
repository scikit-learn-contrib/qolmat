---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: env_qolmat_dev
    language: python
    name: python3
---

# Computation time of Energy distance between Scipy, Dcor and the implementation of Qolmat

```python
from qolmat.benchmark import metrics
import dcor
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy
import pandas as pd
```

```python
a = pd.DataFrame(np.random.rand(10000, 10))
print('Float:', metrics._sum_manhattan_distances(a) * 2, np.sum(scipy.spatial.distance.cdist(a, a, metric="cityblock")))

a = pd.DataFrame(np.rint(a))
print('Integer:', metrics._sum_manhattan_distances(a) * 2, np.sum(scipy.spatial.distance.cdist(a, a, metric="cityblock")))

errors = []
for i in range(5):
    a = pd.DataFrame(np.random.rand(10000, 10))
    errors.append( (metrics._sum_manhattan_distances(a) * 2) - np.sum(scipy.spatial.distance.cdist(a, a, metric="cityblock")))

print('Sum of errors:', np.sum(errors))
```

## Sum of Manhattan distances between all pairs of points: computation time between Scipy and the implementation of Qolmat

```python
ncol = 10
nrows = np.logspace(1, 4, num=20, dtype=int)

manhattan_scipy = []
runtime_scipy = []

manhattan_qolmat = []
runtime_qolmat = []
for nrow in nrows:
    b = pd.DataFrame(np.random.randn(nrow, ncol)).abs()
    start = time.time()
    dist = np.sum(scipy.spatial.distance.cdist(b, b, metric="cityblock"))
    end = time.time()
    manhattan_scipy.append(dist)
    runtime_scipy.append(end - start)

    start = time.time()
    dist = metrics._sum_manhattan_distances(b)
    end = time.time()
    manhattan_qolmat.append(dist)
    runtime_qolmat.append(end - start)

fig, ax = plt.subplots(figsize=(5,2))
ax.plot(nrows, runtime_scipy, 'k--', label='Scipy')
ax.plot(nrows, runtime_qolmat, 'k:', label='Qolmat')
ax.legend()
plt.xlabel('number of lines')
plt.ylabel('runtime')
plt.title('Manhattan dist: scipy vs qolmat')
plt.grid()

#assert energy_dist_scipy == energy_dist_qolmat
#[[i,j*2] for i,j in zip(manhattan_scipy, manhattan_qolmat)]
```

## Energy distance between all pairs of points:

```python
ncol = 10
nrows = np.logspace(1, 4, num=20, dtype=int)

energy_dist_scipy = []
runtime_scipy = []

energy_dist_qolmat = []
runtime_qolmat = []
a = pd.DataFrame(np.random.randn(10000, ncol))
for nrow in nrows:
    b = pd.DataFrame(np.random.randn(nrow, ncol)).abs()
    start = time.time()
    dist_a = np.sum(scipy.spatial.distance.cdist(a, a, metric="cityblock"))
    dist_b = np.sum(scipy.spatial.distance.cdist(b, b, metric="cityblock"))
    dist_ab = np.sum(scipy.spatial.distance.cdist(a, b, metric="cityblock"))
    e_dist = 2 * dist_ab - dist_a - dist_b
    end = time.time()
    energy_dist_scipy.append(e_dist)
    runtime_scipy.append(end - start)

    start = time.time()
    e_dist = sum_energy_distances(a, b)
    end = time.time()
    energy_dist_qolmat.append(e_dist)
    runtime_qolmat.append(end - start)

fig, ax = plt.subplots(figsize=(5,2))
ax.plot(nrows, runtime_scipy, 'k--', label='Scipy')
ax.plot(nrows, runtime_qolmat, 'k:', label='Qolmat')
ax.legend()
plt.xlabel('number of lines')
plt.ylabel('runtime')
plt.title('Energy dist with Manhattan: scipy vs qolmat')
plt.grid()

#assert energy_dist_scipy == energy_dist_qolmat
#[[i,j] for i,j in zip(energy_dist_scipy, energy_dist_qolmat)]
```

```python
ncol = 10
nrows = np.logspace(1, 4, num=20, dtype=int)

energy_dist_scipy = []
runtime_scipy = []

energy_dist_dcor = []
runtime_dcor = []
a = pd.DataFrame(np.random.randn(10000, ncol))
for nrow in nrows:
    b = pd.DataFrame(np.random.randn(nrow, ncol)).abs()
    start = time.time()
    dist_a = np.sum(scipy.spatial.distance.cdist(a, a, metric="euclidean"))
    dist_b = np.sum(scipy.spatial.distance.cdist(b, b, metric="euclidean"))
    dist_ab = np.sum(scipy.spatial.distance.cdist(a, b, metric="euclidean"))
    e_dist = 2 * dist_ab - dist_a - dist_b
    end = time.time()
    energy_dist_scipy.append(e_dist)
    runtime_scipy.append(end - start)

    start = time.time()
    e_dist = dcor.energy_distance(a, b, average=np.sum)
    end = time.time()
    energy_dist_dcor.append(e_dist)
    runtime_dcor.append(end - start)

fig, ax = plt.subplots(figsize=(5,2))
ax.plot(nrows, runtime_scipy, 'k--', label='Scipy')
ax.plot(nrows, runtime_dcor, 'k:', label='Dcor')
ax.legend()
plt.xlabel('number of lines')
plt.ylabel('runtime')
plt.title('Energy dist with Euclidean: scipy vs qolmat')
plt.grid()

#assert energy_dist_scipy == energy_dist_dcor
#[[i,j] for i,j in zip(energy_dist_scipy, energy_dist_dcor)]
```

```python

```
