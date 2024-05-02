import tensorflow as tf
import tf_keras

LORA_WEIGHT_NAMES = ["kernel_lora_a", "kernel_lora_b"]
"""
List of patterns to match LoRA weights, so they can be excluded from weight transfers
between a model and its LoRA version.
"""


class LoRADense(tf_keras.layers.Dense):
    """LoRA version of the ``Dense`` layer."""

    is_lora_layer: bool = True

    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        lora_rank: int = 4,
        lora_alpha: float = 1,
        lora_initializer="glorot_uniform",
        lora_regularizer=None,
        **kwargs,
    ):
        super().__init__(
            units=units,
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
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_initializer = lora_initializer
        self.lora_regularizer = lora_regularizer
        self.scaling = lora_alpha / lora_rank

        self.kernel_lora_a = None
        self.kernel_lora_b = None

        self.merged = False
        self.full_rank_kernel = None

    def build(self, input_shape):
        super().build(input_shape)
        last_dim = self.kernel.shape[0]
        self.kernel_lora_a = self.add_weight(
            "kernel_lora_a",
            shape=[last_dim, self.lora_rank],
            initializer=self.lora_initializer,
            regularizer=self.lora_regularizer,
            # We don't support constraints on the low-rank updates at the moment.
            constraint=None,
            dtype=self.dtype,
            trainable=True,
        )
        self.kernel_lora_b = self.add_weight(
            "kernel_lora_b",
            shape=[self.lora_rank, self.units],
            initializer=tf_keras.initializers.Zeros(),
            regularizer=self.lora_regularizer,
            constraint=None,
            dtype=self.dtype,
            trainable=True,
        )

    def call(self, x):
        # If LoRA weights are merged, inference is the same as for the original layer.
        if self.merged:
            return super().call(x)

        if x.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            x = tf.cast(x, dtype=self._compute_dtype_object)

        rank = x.shape.rank
        shape = x.shape.as_list()
        if rank == 2 or rank is None:
            x1 = tf.matmul(a=x, b=self.kernel)
            x2 = tf.matmul(a=x, b=self.kernel_lora_a)
            x2 = tf.matmul(a=x2, b=self.kernel_lora_b)
        else:
            x1 = tf.tensordot(x, self.kernel, [[rank - 1], [0]])
            x2 = tf.tensordot(x, self.kernel_lora_a, [[rank - 1], [0]])
            x2 = tf.tensordot(x2, self.kernel_lora_b, [[rank - 1], [0]])

        x = x1 + self.scaling * x2

        # Reshape the output back to the original ndim of the input.
        if rank is not None and rank != 2 and not tf.executing_eagerly():
            output_shape = shape[:-1] + [self.kernel.shape[-1]]
            x.set_shape(output_shape)

        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias)

        if self.activation is not None:
            x = self.activation(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({"lora_rank": self.lora_rank, "lora_alpha": self.lora_alpha})
        return config

    def merge_weights(self):
        if self.merged:
            raise ValueError("LoRA updates have already been merged.")
        self.full_rank_kernel = self.kernel.value()
        self.kernel.assign_add(
            self.scaling * tf.matmul(self.kernel_lora_a, self.kernel_lora_b)
        )
        self.merged = True

    def unmerge_weights(self):
        if not self.merged:
            raise ValueError("LoRA updates have not been merged yet.")
        self.kernel.assign(self.full_rank_kernel)
        self.merged = False

    def lora_trainable_weights(self, train_bias: bool):
        trainable_variables = [self.kernel_lora_a, self.kernel_lora_b]
        if train_bias and self.use_bias:
            trainable_variables += [self.bias]
        return trainable_variables


class LoRAConv2D(tf_keras.layers.Conv2D):
    """LoRA version of the ``Conv2D`` layer."""

    is_lora_layer: bool = True

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
        lora_rank: int = 4,
        lora_alpha: float = 1,
        lora_initializer="glorot_uniform",
        lora_regularizer=None,
        **kwargs,
    ):
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
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
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_initializer = tf_keras.initializers.get(lora_initializer)
        self.lora_regularizer = tf_keras.regularizers.get(lora_regularizer)
        self.scaling = lora_alpha / lora_rank

        self.kernel_lora_a = None
        self.kernel_lora_b = None

        self.merged = False
        self.full_rank_kernel = None

    def build(self, input_shape):
        super().build(input_shape)
        kernel_height, kernel_width, in_channels, out_channels = self.kernel.shape
        self.kernel_lora_a = self.add_weight(
            "kernel_lora_a",
            shape=(kernel_height, kernel_width, in_channels, self.lora_rank),
            initializer=self.lora_initializer,
            regularizer=self.lora_regularizer,
            # We don't support constraints on the low-rank updates at the moment.
            constraint=None,
            dtype=self.dtype,
            trainable=True,
        )
        self.kernel_lora_b = self.add_weight(
            "kernel_lora_b",
            shape=(kernel_height, kernel_width, self.lora_rank, out_channels),
            initializer=tf_keras.initializers.Zeros(),
            regularizer=self.lora_regularizer,
            constraint=None,
            dtype=self.dtype,
            trainable=True,
        )

    def call(self, x):
        if self.merged:
            return super().call(x)

        full_kernel = self.kernel.value()  # Save the existing kernel

        # Apply low-rank update to kernel
        kernel_update = tf.matmul(self.kernel_lora_a, self.kernel_lora_b)
        self.kernel.assign_add(self.scaling * kernel_update)
        x = super().call(x)  # Re-use the call function of the Keras Cond2D layer

        self.kernel.assign(full_kernel)  # Restore kernel again
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"lora_rank": self.lora_rank, "lora_alpha": self.lora_alpha})
        return config

    def merge_weights(self):
        if self.merged:
            raise ValueError("LoRA updates have already been merged.")
        self.full_rank_kernel = self.kernel.value()
        self.kernel.assign_add(
            self.scaling * tf.matmul(self.kernel_lora_a, self.kernel_lora_b)
        )
        self.merged = True

    def unmerge_weights(self):
        if not self.merged:
            raise ValueError("LoRA updates have not been merged yet.")
        self.kernel.assign(self.full_rank_kernel)
        self.merged = False

    def lora_trainable_weights(self, train_bias: bool):
        trainable_variables = [self.kernel_lora_a, self.kernel_lora_b]
        if train_bias and self.use_bias:
            trainable_variables += [self.bias]
        return trainable_variables


def convert_to_lora_layer(
    layer: tf_keras.layers.Layer, **kwargs
) -> tf_keras.layers.Layer:
    """
    Convenience function to convert supported layer types to their LoRA counterparts.

    Args:
        layer: Layer to be converted.
        **kwargs: LoRA specific parameters such as ``lora_rank`` have to be passed as
            kwargs.

    Returns:
        LoRA layer instance.
    """
    layer_lookup = {tf_keras.layers.Dense: LoRADense}
    if type(layer) in layer_lookup:
        lora_layer = layer_lookup[type(layer)](**layer.get_config(), **kwargs)
    else:
        raise ValueError(
            f"Unsupported layer type for conversion to LoRA: {type(layer)}."
        )
    return lora_layer
