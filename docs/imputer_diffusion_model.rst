Imputers based on Diffusion models
##################################

Qolmat proposes two imputers based on Denoising Diffusion Probabilistic Models (DDPMs) of `Ho et al. (2020) <https://arxiv.org/abs/2006.11239>`_: :class:`qolmat.diffusions.TabDDPM` for tabular data and :class:`qolmat.diffusions.TabDDPMTs` for time-series data. Our implementations mainly follow the work of `Tashiro et al. (2021) <https://arxiv.org/abs/2107.03502>`_.

1. Denoising Diffusion Probabilistic Models (DDPMs)
***************************************************
Diffusion models are a class of probabilistic generative models used to describe the dynamic evolution of data over time or space. Inspired by physical processes like diffusion, these models capture how information, attributes, or features spread gradually through a sequence of steps. Instead of explicitly modeling the data's generation process, diffusion models focus on modeling the process of data transitions from noisy or incomplete observations to the underlying true data. They find applications in diverse fields such as image synthesis, anomaly detection, time series prediction and particularly data imputation.

Introduced by Jonathan Ho et al. (2020), Denoising Diffusion Probabilistic Models (DDPMs) tackle data synthesis by leveraging the concept of diffusion, wherein a noisy data is iteratively transformed to remove noise, revealing the underlying true data.

1.1 TabDDPM architecture
========================

1.2 TabDDPMTS architecture
==========================

2. User guide
*************

2.1 Architecture configuration
==============================

2.2 Imputation by samling
=========================

2.3 Focus on time-series data
=============================

3. References
*************

[1] Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." Advances in neural information processing systems 33 (2020): 6840-6851.
[2] Tashiro, Yusuke, et al. "Csdi: Conditional score-based diffusion models for probabilistic time series imputation." Advances in Neural Information Processing Systems 34 (2021): 24804-24816.