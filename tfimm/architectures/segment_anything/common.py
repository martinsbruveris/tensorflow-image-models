import tf_keras
from tfimm.layers import act_layer_factory


class MLPBlock(tf_keras.layers.Layer):
    """MLP block. Only difference with tfimm version is naming of layers."""

    def __init__(
        self,
        hidden_dim: int,
        embed_dim: int,
        drop_rate: float,
        act_layer: str,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        **kwargs,
    ):
        super().__init__(**kwargs)
        act_layer = act_layer_factory(act_layer)

        self.fc1 = tf_keras.layers.Dense(
            units=hidden_dim,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name="lin1",
        )
        self.act = act_layer()
        self.drop1 = tf_keras.layers.Dropout(rate=drop_rate)
        self.fc2 = tf_keras.layers.Dense(
            units=embed_dim,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name="lin2",
        )
        self.drop2 = tf_keras.layers.Dropout(rate=drop_rate)

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x, training=training)
        x = self.fc2(x)
        x = self.drop2(x, training=training)
        return x
