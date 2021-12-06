"""
Attention layers used in CNNs
"""
import math
from typing import Optional

import tensorflow as tf

from .factory import act_layer_factory, norm_layer_factory


def make_divisible(
    v: int, divisor: int, min_value: Optional[int] = None, round_limit: float = 0.9
) -> int:
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


class SEModule(tf.keras.layers.Layer):
    """
    SE Module as defined in original SE-Nets with a few additions

    Paper: Squeeze-and-Excitation Networks
    Link: https://arxiv.org/abs/1709.01507

    Additions include:
        * divisor can be specified to keep channels % div == 0 (default: 8)
        * reduction channels can be specified directly by arg (if rd_channels is set)
        * reduction channels can be specified by float rd_ratio (default: 1/16)
        * global max pooling can be added to the squeeze aggregation
        * customizable activation, normalization, and gate layer
    """

    def __init__(
        self,
        rd_ratio=1.0 / 16,
        rd_channels=None,
        rd_divisor=8,
        act_layer="relu",
        norm_layer="",
        gate_layer="sigmoid",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rd_ratio = rd_ratio
        self.rd_channels = rd_channels
        self.rd_divisor = rd_divisor
        act_layer = act_layer_factory(act_layer)
        norm_layer = norm_layer_factory(norm_layer)
        gate_layer = act_layer_factory(gate_layer)

        self.bn = norm_layer(name="bn")
        self.act = act_layer()
        self.gate = gate_layer()

    def build(self, input_shape):
        channels = input_shape[-1]
        rd_channels = self.rd_channels or make_divisible(
            channels * self.rd_ratio, self.rd_divisor, round_limit=0.0
        )
        self.fc1 = tf.keras.layers.Conv2D(
            filters=rd_channels,
            kernel_size=1,
            use_bias=True,
            name="fc1",
        )
        self.fc2 = tf.keras.layers.Conv2D(
            filters=channels,
            kernel_size=1,
            use_bias=True,
            name="fc2",
        )

    def call(self, x, training=False):
        x_se = tf.reduce_mean(x, axis=(1, 2), keepdims=True)
        x_se = self.fc1(x_se)
        x_se = self.bn(x_se, training=training)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        x_se = self.gate(x_se)
        x = x * x_se
        return x


class EcaModule(tf.keras.layers.Layer):
    """Constructs an ECA module.

    Paper: ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
    Link: https://arxiv.org/pdf/1910.03151.pdf

    Args:
        gamma: Used in kernel_size calc, see original paper for details
            refer to original paper https://arxiv.org/pdf/1910.03151.pdf
        beta: Used in kernel_size calc, see above
        gate_layer: Gating non-linearity to use
    """

    def __init__(
        self,
        gamma: int = 2,
        beta: int = 1,
        gate_layer: str = "sigmoid",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.beta = beta

        gate_layer = act_layer_factory(gate_layer)
        self.gate = gate_layer()

    def build(self, input_shape):
        # Kernel size is determined dynamically from number of input channels
        channels = input_shape[-1]
        t = int(abs(math.log(channels, 2) + self.beta) / self.gamma)
        kernel_size = max(t if t % 2 else t + 1, 3)
        padding = (kernel_size - 1) // 2

        self.pad = tf.keras.layers.ZeroPadding1D(padding=padding)
        self.conv = tf.keras.layers.Conv1D(
            filters=1,
            kernel_size=kernel_size,
            use_bias=False,
            name="conv",
        )

    def call(self, x, training=False):
        # x is (N, H, W, C)
        y = tf.reduce_mean(x, axis=(1, 2))  # (N, C)
        y = tf.expand_dims(y, axis=-1)  # (N, C, 1)
        y = self.pad(y)
        y = self.conv(y)  # (N, C, 1)
        y = self.gate(y)
        y = tf.transpose(y, perm=(0, 2, 1))  # (N, 1, C)
        y = tf.expand_dims(y, axis=1)  # (N, 1, 1, C)
        x = y * x
        return x


def attn_layer_factory(attn_layer: str):
    """Returns a function that creates the required attention layer."""
    if attn_layer == "":
        return lambda **kwargs: tf.keras.layers.Activation("linear")
    # Lightweight attention modules (channel and/or coarse spatial).
    # Typically added to existing network architecture blocks in addition to existing
    # convolutions.
    elif attn_layer == "eca":
        return lambda **kwargs: EcaModule(**kwargs)
    elif attn_layer == "se":
        return lambda **kwargs: SEModule(**kwargs)
    else:
        raise ValueError(f"Unknown attention layer: {attn_layer}.")
