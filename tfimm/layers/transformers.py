"""
Common layers shared between transformer architectures.

Copyright 2021 Martins Bruveris
"""
import tensorflow as tf

from tfimm.layers.factory import act_layer_factory, norm_layer_factory


class PatchEmbeddings(tf.keras.layers.Layer):
    """
    Image to Patch Embedding.
    """

    def __init__(self, patch_size: int, embed_dim: int, norm_layer: str = "", **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.norm_layer = norm_layer_factory(norm_layer)

        self.projection = tf.keras.layers.Conv2D(
            filters=self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            use_bias=True,
            name="proj",
        )
        self.norm = self.norm_layer(name="norm")

    def call(self, x, training=False):
        emb = self.projection(x)

        # Change the 2D spatial dimensions to a single temporal dimension.
        # shape = (batch_size, num_patches, out_channels=embed_dim)
        batch_size, height, width = tf.unstack(tf.shape(x)[:3])
        num_patches = (width // self.patch_size) * (height // self.patch_size)
        emb = tf.reshape(tensor=emb, shape=(batch_size, num_patches, -1))

        emb = self.norm(emb, training=training)
        return emb


class MLP(tf.keras.layers.Layer):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        hidden_dim: int,
        embed_dim: int,
        drop_rate: float,
        act_layer: str,
        **kwargs
    ):
        super().__init__(**kwargs)
        act_layer = act_layer_factory(act_layer)

        self.fc1 = tf.keras.layers.Dense(units=hidden_dim, name="fc1")
        self.act = act_layer()
        self.drop1 = tf.keras.layers.Dropout(rate=drop_rate)
        self.fc2 = tf.keras.layers.Dense(units=embed_dim, name="fc2")
        self.drop2 = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x, training=training)
        x = self.fc2(x)
        x = self.drop2(x, training=training)
        return x
