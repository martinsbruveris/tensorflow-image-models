import numpy as np
import tensorflow as tf


class BlurPool2D(tf.keras.layers.Layer):
    """
    BlurPool layer as described in the paper "Making Convolutional Networks
    Shift-Invariant Again" by Zhang et al.

    Link: https://arxiv.org/pdf/1904.11486.pdf

    This implementation is adapted from:
    https://github.com/csvance/blur-pool-keras/blob/master/blurpool.py
    """

    def __init__(self, kernel_size: int = 3, stride: int = 2, **kwargs):
        super().__init__(**kwargs)

        self.kernel_size = kernel_size
        self.stride = stride
        p = (kernel_size + stride) // 2 - 1
        self.paddings = [[0, 0], [p, p], [p, p], [0, 0]]
        self.blur_kernel = None

    def build(self, input_shape):
        if self.kernel_size == 3:
            bk = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
            bk = bk / np.sum(bk)
        elif self.kernel_size == 5:
            bk = np.array(
                [
                    [1, 4, 6, 4, 1],
                    [4, 16, 24, 16, 4],
                    [6, 24, 36, 24, 6],
                    [4, 16, 24, 16, 4],
                    [1, 4, 6, 4, 1],
                ]
            )
            bk = bk / np.sum(bk)
        else:
            raise ValueError(f"Kernel size {self.kernel_size} not implemented.")

        channels = input_shape[-1]
        bk = np.repeat(bk, channels)
        bk = np.reshape(bk, (self.kernel_size, self.kernel_size, channels, 1))

        self.blur_kernel = self.add_weight(
            name="blur_kernel",
            shape=(self.kernel_size, self.kernel_size, channels, 1),
            initializer=tf.keras.initializers.constant(bk),
            trainable=False,
        )

    def call(self, x):
        x = tf.pad(x, paddings=self.paddings, mode="REFLECT")
        x = tf.nn.depthwise_conv2d(
            input=x,
            filter=self.blur_kernel,
            strides=[1, self.stride, self.stride, 1],
            padding="VALID",
        )
        return x

    def cast_inputs(self, inputs):
        # Casts to float16, the policy's lowest-precision dtype
        return self._mixed_precision_policy.cast_to_lowest(inputs)
