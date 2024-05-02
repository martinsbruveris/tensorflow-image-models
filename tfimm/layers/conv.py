"""
Convolution with Weight Standardization

Paper: Micro-Batch Training with Batch-Channel Normalization and Weight Standardization
Link: https://arxiv.org/abs/1903.10520v2
Code: https://github.com/joe-siyuan-qiao/WeightStandardization
"""
from typing import Tuple, Union

import tensorflow as tf
import tf_keras

from tfimm.utils.etc import to_2tuple


def get_padding(
    kernel_size: Union[int, tuple],
    strides: Union[int, tuple] = 1,
    dilation_rate: Union[int, tuple] = 1,
) -> Tuple[int, int]:
    """Calculate pytorch-style symmetric padding for a convolution"""
    kernel_size = to_2tuple(kernel_size)
    strides = to_2tuple(strides)
    dilation_rate = to_2tuple(dilation_rate)
    padding = (
        ((strides[0] - 1) + dilation_rate[0] * (kernel_size[0] - 1)) // 2,
        ((strides[1] - 1) + dilation_rate[1] * (kernel_size[1] - 1)) // 2,
    )
    return padding


class PadConv2D(tf_keras.layers.Conv2D):
    """
    Conv2D with support for PyTorch-style symmetric padding. Used to convert PyTorch
    models.
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        data_format=None,
        dilation_rate=(1, 1),
        groups=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding if padding != "symmetric" else "valid",
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )
        self.pad = (
            tf_keras.layers.ZeroPadding2D(
                padding=get_padding(kernel_size, strides, dilation_rate)
            )
            if padding == "symmetric"
            else None
        )

    def call(self, x):
        if self.pad is not None:
            x = self.pad(x)
        x = super().call(x)
        return x


class PadDepthwiseConv2D(tf_keras.layers.DepthwiseConv2D):
    """
    DepthwiseConv2D with support for PyTorch-style symmetric padding. Used to convert
    PyTorch models.
    """

    def __init__(
        self,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        depth_multiplier=1,
        data_format=None,
        dilation_rate=(1, 1),
        groups=1,
        activation=None,
        use_bias=True,
        depthwise_initializer="glorot_uniform",
        bias_initializer="zeros",
        depthwise_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        depthwise_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(
            kernel_size=kernel_size,
            strides=strides,
            padding=padding if padding != "symmetric" else "valid",
            depth_multiplier=depth_multiplier,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_bias=use_bias,
            depthwise_initializer=depthwise_initializer,
            bias_initializer=bias_initializer,
            depthwise_regularizer=depthwise_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            depthwise_constraint=depthwise_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )
        self.pad = (
            tf_keras.layers.ZeroPadding2D(
                padding=get_padding(kernel_size, strides, dilation_rate)
            )
            if padding == "symmetric"
            else None
        )

    def call(self, x):
        if self.pad is not None:
            x = self.pad(x)
        x = super().call(x)
        return x


class StdConv2D(tf_keras.layers.Conv2D):
    """
    Conv2D with Weight Standardization. Used for BiT ResNet-V2 models.

    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight
        Standardization`
    Link: https://arxiv.org/abs/1903.10520v2
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        data_format=None,
        dilation_rate=(1, 1),
        groups=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        eps=1e-8,
        **kwargs,
    ):
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding if padding != "symmetric" else "valid",
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )
        self.eps = eps
        self.pad = (
            tf_keras.layers.ZeroPadding2D(
                padding=get_padding(kernel_size, strides, dilation_rate)
            )
            if padding == "symmetric"
            else None
        )

    def call(self, x):
        if self.pad is not None:
            x = self.pad(x)

        orig_kernel = self.kernel
        kernel_mean = tf.math.reduce_mean(self.kernel, axis=[0, 1, 2], keepdims=True)
        kernel = self.kernel - kernel_mean
        kernel_var = tf.math.reduce_variance(self.kernel, axis=[0, 1, 2], keepdims=True)
        kernel = kernel / tf.math.sqrt(kernel_var + self.eps)
        # We change the kernel temporarily to make use of `call(x)` of the super-class
        self.kernel = kernel
        x = super().call(x)
        self.kernel = orig_kernel
        return x
