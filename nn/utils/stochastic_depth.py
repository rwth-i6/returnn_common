"""
Stochastic depth
"""

from __future__ import annotations
from typing import Callable
from ... import nn


def stochastic_depth(func: Callable[[], nn.Tensor], p: float, mode: str = "batch") -> nn.Tensor:
    """
    Implements Stochastic Depth (sometimes also called "layer drop")
    for randomly dropping residual branches of residual architectures.

    Code adopted from here: https://github.com/pytorch/vision/blob/main/torchvision/ops/stochastic_depth.py

    Only applied when in training.

    For some further discussion, also see: https://github.com/rwth-i6/returnn_common/issues/99
    Relevant papers:
    - `"Deep Networks with Stochastic Depth" <https://arxiv.org/abs/1603.09382>`__
    - `"Very Deep Self-Attention Networks for End-to-End Speech Recognition" <https://arxiv.org/abs/1904.13377>`__
    - `"Reducing Transformer Depth on Demand with Structured Dropout" <https://arxiv.org/abs/1909.11556>`__
    - `"Intermediate Loss Regularization for CTC-based Speech Recognition" <https://arxiv.org/abs/1904.09751>`__
    - `"Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" <https://arxiv.org/abs/2103.14030>`__

    Args:
        func (() -> Tensor[...]): Module or function for input tensor or arbitrary dimensions
        p (float): probability of the input to be zeroed.
        mode (str): ``"batch"`` or ``"row"``.
                    ``"batch"`` randomly zeroes the entire input and performs the computation only when necessary,
                    ``"row"`` zeroes randomly selected rows (batch indices) from the batch.
    Returns:
        Tensor[...]: The randomly zeroed tensor.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
    if mode not in ["batch", "row"]:
        raise ValueError(f"mode has to be either 'batch' or 'row', but got {mode}")
    if p == 0.0:
        return func()

    training = nn.train_flag()

    survival_rate = 1.0 - p
    if mode == "row":
        true_value = func()
        with nn.Cond(training) as cond:
            # Not efficient.
            noise = nn.random_bernoulli([true_value.batch_dim], p=survival_rate)
            if survival_rate > 0.0:
                noise /= survival_rate
            cond.true = true_value * noise
            cond.false = true_value
        return cond.result
    elif mode == "batch":
        with nn.Cond(training) as cond_train:
            noise = nn.random_bernoulli((), p=survival_rate)
            with nn.Cond(noise) as cond_noise:
                true_value = func()
                if survival_rate > 0.0:
                    true_value /= survival_rate
                cond_noise.true = true_value
                cond_noise.false = nn.zeros_like(true_value)
            cond_train.true = cond_noise.result
            cond_train.false = func()
        return cond_train.result
    else:
        raise ValueError(f"mode {mode!r} invalid")
