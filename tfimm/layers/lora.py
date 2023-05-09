import tensorflow as tf


class LoraDense(tf.keras.layers.Dense):
    """
    https://github.com/keras-team/keras/blob/v2.12.0/keras/layers/core/dense.py#L33-L301
    """
    # TODO: move to config
    lora_rank = 4
    lora_alpha = 1
    scaling = lora_alpha / lora_rank

    def build(self, input_shape):
        super().build(input_shape)
        last_dim = self.kernel.shape[0]
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
            initializer=tf.keras.initializers.RandomNormal(stddev=1/self.lora_rank),  # random initialisation
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

    def call(self, x):
        print("BBB")
        res = super().call(x)
        print(len(self.variables),len(self.trainable_variables))
        return res
