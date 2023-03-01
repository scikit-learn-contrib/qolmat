
import numpy as np
import timesynth as ts # package for generating time series

import matplotlib.pyplot as plt

from qolmat.utils import plot
from qolmat.imputations.rpca.pcp_rpca import PcpRPCA
from qolmat.imputations.rpca.temporal_rpca import TemporalRPCA, OnlineTemporalRPCA
np.random.seed(402)

################################################################################

time_sampler = ts.TimeSampler(stop_time=20)
irregular_time_samples = time_sampler.sample_irregular_time(num_points=5_000, keep_percentage=100)
sinusoid = ts.signals.Sinusoidal(frequency=2)
white_noise = ts.noise.GaussianNoise(std=0.1)
timeseries = ts.TimeSeries(sinusoid, noise_generator=white_noise)
samples, signals, errors = timeseries.sample(irregular_time_samples)

n = len(samples)
pc = 0.02
indices_ano1 = np.random.choice(n, int(n*pc))
samples[indices_ano1] = [np.random.uniform(low=2*np.min(samples), high=2*np.max(samples)) for i in range(int(n*pc))]
indices = np.random.choice(n, int(n*pc))
samples[indices] = np.nan


################################################################################

time_sampler = ts.TimeSampler(stop_time=20)
irregular_time_samples = time_sampler.sample_irregular_time(num_points=5_000, keep_percentage=100)
sinusoid = ts.signals.Sinusoidal(frequency=3)
white_noise = ts.noise.GaussianNoise(std=0)
timeseries = ts.TimeSeries(sinusoid, noise_generator=white_noise)
samples2, signals2, errors2 = timeseries.sample(irregular_time_samples)

n2 = len(samples2)
indices_ano2 = np.random.choice(n2, int(n*pc))
samples2[indices_ano2] = [np.random.uniform(low=2*np.min(samples2), high=2*np.max(samples2)) for i in range(int(n2*pc))]
indices = np.random.choice(n2, int(n*pc))
samples2[indices] = np.nan

samples += samples2
signals += signals2
errors += errors2



online_temp_rpca = OnlineTemporalRPCA(n_rows=25, tau=1, lam=0.3, list_periods=[20], list_etas=[0.01],
                       burnin=0.2, online_list_etas=[0.3], nwin=20)
X, A = online_temp_rpca.fit_transform(X=samples)
plot.plot_sig
nal([samples, X, A], style="matplotlib")
len(samples)
