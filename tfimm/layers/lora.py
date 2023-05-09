import tensorflow as tf


class LoraDense(tf.keras.layers.Dense):
    """
    https://github.com/keras-team/keras/blob/v2.12.0/keras/layers/core/dense.py#L33-L301
    """

    lora_rank = 10
    lora_alpha = 1
    scaling = lora_alpha / lora_rank

    def build(self, input_shape):
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())  # tf.keras
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError(
                "A Dense layer can only be built with a floating-point "
                f"dtype. Received: dtype={dtype}"
            )

        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError(
                "The last dimension of the inputs to a Dense layer "
                "should be defined. Found None. "
                f"Full input shape received: {input_shape}"
            )
        self.input_spec = tf.keras.layers.InputSpec(
            min_ndim=2, axes={-1: last_dim}
        )  # tf.keras
        self.kernel_0 = self.add_weight(
            "kernel",
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=False,  # LoRA change
        )
        self.kernel_lora_a = self.add_weight(
            "kernel_lora_a",
            shape=[last_dim, self.lora_rank],
            initializer=self.kernel_initializer,  # random initialisation
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )
        self.kernel_lora_b = self.add_weight(
            "kernel_lora_b",
            shape=[self.lora_rank, self.units],
            initializer=tf.keras.initializers.Zeros(),  # initialise to 0
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )
        # for higher dim weights use, tf.tensordot()
        self.kernel = (
            self.kernel_0 + self.kernel_lora_a @ self.kernel_lora_b * self.scaling
        )

        if self.use_bias:
            self.bias = self.add_weight(
                "bias",
                shape=[
                    self.units,
                ],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )
        else:
            self.bias = None
        self.built = True

    def call(self, x):
        print("BBB")
        return super().call(x)
