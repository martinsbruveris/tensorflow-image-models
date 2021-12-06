"""
Attention layers used in CNNs
"""
import math

import tensorflow as tf

from .factory import act_layer_factory


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
    else:
        raise ValueError(f"Unknown attention layer: {attn_layer}.")
