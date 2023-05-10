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
        if self.merged:
            x = super().call(x)
        else:
            x1 = tf.linalg.matvec(self.kernel, x)
            x2 = tf.linalg.matvec(self.kernel_lora_b, x)
            x2 = tf.linalg.matvec(self.kernel_lora_a, x2)
            x = x1 + self.scaling * x2 + self.bias
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"lora_rank": self.lora_rank, "lora_alpha": self.lora_alpha})
        return config

    def merge_lora_weights(self):
        self.kernel += self.kernel_lora_a @ self.kernel_lora_b * self.scaling
        self.merged = True

    def unmerge_lora_weights(self):
        self.kernel -= self.kernel_lora_a @ self.kernel_lora_b * self.scaling
        self.merged = False

    def mark_only_lora_as_trainable(self, train_bias: bool):
        self.kernel = tf.Variable(self.kernel, trainable=False)
        if not train_bias:
            self.bias = tf.Variable(self.bias, trainable=False)
