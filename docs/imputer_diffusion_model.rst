Imputers based on Diffusion models
##################################

Qolmat proposes two imputers based on Denoising Diffusion Probabilistic Models (DDPMs) of `Ho et al. (2020) <https://arxiv.org/abs/2006.11239>`_: :class:`qolmat.diffusions.TabDDPM` for tabular data and :class:`qolmat.diffusions.TabDDPMTs` for time-series data. Our implementations mainly follow the works of `Tashiro et al. (2021) <https://arxiv.org/abs/2107.03502>`_ and `Kotelnikov et al., (2022) <https://arxiv.org/abs/2209.15421>`_.

1. Denoising Diffusion Probabilistic Models (DDPMs)
***************************************************
Diffusion models are a class of generative models used to describe the dynamic evolution of data. Inspired by physical processes like diffusion, these models capture how information (e.g., features in tabular data) spread gradually through a sequence of steps. Instead of explicitly modeling the data's generation process, diffusion models focus on modeling the process of data transitions from noisy or incomplete observations to the underlying true data. They find applications in diverse fields such as image synthesis, anomaly detection, time series prediction and particularly data imputation.

Introduced by Jonathan Ho et al. (2020), Denoising Diffusion Probabilistic Models (DDPMs) tackle data synthesis by leveraging the concept of diffusion, wherein a noisy data is iteratively transformed to remove noise, revealing the underlying true data.

- Forward: :math:`x_0 \rightarrow x_1 \rightarrow \dots \rightarrow x_{T-1} \rightarrow x_T`
    - :math:`q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)I)`
    - :math:`x_t = \bar{\alpha}_t \times x_0 + \sqrt{1-\bar{\alpha}_t} \times \epsilon` where
        - :math:`\epsilon \sim \mathcal{N}(0,I)`
        - :math:`\bar{\alpha}_t = \sum^t_{t=0} \alpha_t`
        - :math:`\alpha`: noise scheduler

- Reserve: :math:`x_T \rightarrow x_{t-1} \rightarrow \dots \rightarrow x_1 \rightarrow x_0`
    - :math:`p_\theta (x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta (x_t, t), \Sigma_\theta (x_t, t))`
    - :math:`x_{t-1} = \frac{1}{\sqrt{\alpha_t}} (x_t - \frac{1 - \alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)) + \sigma_t z` where
        - :math:`\epsilon`: our model to predict noise at t
        - :math:`z \sim \mathcal{N}(0,I)`

- Objective function:
    - :math:`E_{t \sim \mathcal{U} [[1,T]], x_0 \sim q(x_0), \epsilon \sim \mathcal{N}(0,I)} [|| \epsilon - \epsilon_\theta(x_t, t)||^2]`
    - 
1.1. TabDDPM architecture
=========================

In training, several key preprocessing and model architecture choices are employed to ensure the robustness and effectiveness of the neural network. First and foremost, the presence of missing values, represented as "nan," is addressed by filling them with the mean value of the observed data. This imputation technique helps prevent the network from being adversely affected by missing information and ensures that the training process can proceed smoothly.

One of the innovative techniques applied during self-training, referred to as Compute-only loss values from observed data, plays a pivotal role in the training pipeline. CSDI focuses on computing loss values exclusively from the observed data, effectively allowing the model to concentrate on the available information while disregarding the missing data points.

Furthermore, the training phase incorporates a more complex autoencoder architecture based on ResNet, as introduced by Gorishniy et al. (2021). This state-of-the-art autoencoder architecture is designed to capture intricate patterns and dependencies within the data, which can be particularly valuable when dealing with complex and high-dimensional data sources.

Moving on to the inference phase, the process for imputing missing values is outlined. Here, the neural network leverages the information it has learned during training to make predictions in the absence of observed data. The inference pipeline involves a series of steps, starting with the estimation of $\hat{x}_t$ based on the available information. This estimated value $\hat{x}_t$ is obtained using a masking mechanism that combines the initial data point $x_0$ and the predicted value $\hat{x}_t$ based on the available information. The mask variable, which equals 1 for observed values, controls the blending of the two sources of information.

- :math:`\epsilon \rightarrow \hat{x}_t \rightarrow \hat{x}_0` where
        - :math:`\hat{x}_t = mask * x_0 + (1 - mask) * \hat{x}_t`
        - :math:`mask`: 1 = observed values
    - Fill nan with :math:`\hat{x}_0`

1.2. TabDDPMTS architecture
===========================

- Sliding window method: obtain a list of data chunks
- Apply Transformer Encoder to encode the relationship between times in a chunk

1. References
*************

[1] Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." Advances in neural information processing systems 33 (2020): 6840-6851.
[2] Tashiro, Yusuke, et al. "Csdi: Conditional score-based diffusion models for probabilistic time series imputation." Advances in Neural Information Processing Systems 34 (2021): 24804-24816.