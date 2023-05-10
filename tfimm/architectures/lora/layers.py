import tensorflow as tf


class LoRADense(tf.keras.layers.Dense):
    def __init__(self, *args, lora_rank: int = 4, lora_alpha: float = 1, **kwargs):
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / lora_rank
        self.merged = False
        self.kernel = None
        self.bias = None
        self.kernel_lora_a = None
        self.kernel_lora_b = None
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        last_dim = self.kernel.shape[0]
        self.kernel_lora_a = self.add_weight(
            "kernel_lora_a",
            shape=[last_dim, self.lora_rank],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )
        self.kernel_lora_b = self.add_weight(
            "kernel_lora_b",
            shape=[self.lora_rank, self.units],
            initializer=tf.keras.initializers.Zeros(),
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )

    def call(self, x):
        if x.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            x = tf.cast(x, dtype=self._compute_dtype_object)

        if self.merged:
            x = super().call(x)
        else:
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

    def merge_lora_weights(self):
        if self.merged:
            raise ValueError("LoRA updates have already been merged")
        self.kernel.assign_add(self.kernel_lora_a @ self.kernel_lora_b * self.scaling)
        self.merged = True

    def unmerge_lora_weights(self):
        if not self.merged:
            raise ValueError("LoRA updates have not been merged yet")
        self.kernel.assign_add(-self.kernel_lora_a @ self.kernel_lora_b * self.scaling)
        self.merged = False

    def set_only_lora_weights_trainable(self, train_bias: bool):
        self.kernel = tf.Variable(self.kernel, trainable=False, name=self.kernel.name)
        if not train_bias:
            self.bias = tf.Variable(self.bias, trainable=False, name=self.bias.name)
