import tensorflow as tf


class LoRADense(tf.keras.layers.Dense):
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
        self.scaling = lora_alpha / lora_rank
        self.merged = False
        self.kernel_lora_a = None
        self.kernel_lora_b = None

    def build(self, input_shape):
        super().build(input_shape)
        last_dim = self.kernel.shape[0]
        self.kernel_lora_a = self.add_weight(
            "kernel_lora_a",
            shape=[last_dim, self.lora_rank],
            # For now, we are reusing the class default parameter. We could make this
            # customisable. Note that we cannot simply use self.kernel_initializer here,
            # because initializers should only be used once.
            initializer="glorot_uniform",
            regularizer=self.kernel_regularizer,
            # We don't support constraints on the low-rank updates at the moment.
            constraint=None,
            dtype=self.dtype,
            trainable=True,
        )
        self.kernel_lora_b = self.add_weight(
            "kernel_lora_b",
            shape=[self.lora_rank, self.units],
            initializer=tf.keras.initializers.Zeros(),
            regularizer=self.kernel_regularizer,
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
            raise ValueError("LoRA updates have already been merged")
        self.kernel.assign_add(self.kernel_lora_a @ self.kernel_lora_b * self.scaling)
        self.merged = True

    def unmerge_weights(self):
        if not self.merged:
            raise ValueError("LoRA updates have not been merged yet")
        self.kernel.assign_add(-self.kernel_lora_a @ self.kernel_lora_b * self.scaling)
        self.merged = False

    def lora_trainable_weights(self, train_bias: bool):
        trainable_variables = [self.kernel_lora_a, self.kernel_lora_b]
        if train_bias and self.use_bias:
            trainable_variables += [self.bias]
        return trainable_variables


def convert_to_lora_layer(
    layer: tf.keras.layers.Layer, **kwargs
) -> tf.keras.layers.Layer:
    """
    Convenience function to convert supported layer types to their LoRA counterparts.

    Args:
        layer: Layer to be converted.
        **kwargs: LoRA specific parameters such as ``lora_rank`` have to be passed as
            kwargs.

    Returns:
        LoRA layer instance.
    """
    if type(layer) is tf.keras.layers.Dense:
        lora_layer = LoRADense(**layer.get_config(), **kwargs)
    else:
        raise ValueError(
            f"Unsupported layer type for conversion to LoRA: {type(layer)}."
        )
    return lora_layer
