import tensorflow as tf


class Affine(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.alpha = self.add_weight(
            shape=(input_shape[-1],),
            initializer="ones",
            trainable=True,
            name="alpha",
        )
        self.beta = self.add_weight(
            shape=(input_shape[-1],),
            initializer="zeros",
            trainable=True,
            name="beta",
        )

    def call(self, x):
        x = self.alpha * x + self.beta
        return x
