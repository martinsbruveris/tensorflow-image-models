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
        **kwargs,
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


class GatedBiasInitializer(tf.keras.initializers.Initializer):
    """
    Splits tensor in half along last axis (channels). Initializes second half with
    ones.

    Used for bias term in Gated Linear Units.
    """

    def __init__(self, initializer="zeros"):
        self.initializer = tf.keras.initializers.get(initializer)

    def __call__(self, shape, dtype=None, **kwargs):
        if dtype is None:
            dtype = tf.keras.backend.floatx()

        assert shape[-1] % 2 == 0
        split_shape = shape[:-1] + [
            shape[-1] // 2,
        ]
        x1 = self.initializer(split_shape, dtype=dtype)
        x2 = tf.ones(split_shape, dtype=dtype)
        x = tf.concat([x1, x2], axis=-1)
        return x


class GatedKernelInitializer(tf.keras.initializers.Initializer):
    """
    Splits tensor in half along last axis (channels). Initializes second half with
    normal distribution with stddev=1e-6.

    Used for kernel term in Gated Linear Units.
    """

    def __init__(self, initializer="glorot_uniform"):
        self.normal = tf.keras.initializers.RandomNormal(stddev=1e-6)
        self.initializer = tf.keras.initializers.get(initializer)

    def __call__(self, shape, dtype=None, **kwargs):
        if dtype is None:
            dtype = tf.keras.backend.floatx()

        assert shape[-1] % 2 == 0
        split_shape = shape[:-1] + [
            shape[-1] // 2,
        ]
        x1 = self.initializer(split_shape, dtype=dtype)
        x2 = self.normal(split_shape, dtype=dtype)
        x = tf.concat([x1, x2], axis=-1)
        return x


class GluMLP(tf.keras.layers.Layer):
    """
    MLP w/ GLU style gating
    See: https://arxiv.org/abs/1612.08083, https://arxiv.org/abs/2002.05202
    """

    def __init__(
        self,
        hidden_dim: int,
        embed_dim: int,
        drop_rate: float,
        act_layer: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        act_layer = act_layer_factory(act_layer)
        assert hidden_dim % 2 == 0

        self.fc1 = tf.keras.layers.Dense(
            units=hidden_dim,
            bias_initializer=GatedBiasInitializer(),
            kernel_initializer=GatedKernelInitializer(),
            name="fc1",
        )
        self.act = act_layer()
        self.drop1 = tf.keras.layers.Dropout(rate=drop_rate)
        self.fc2 = tf.keras.layers.Dense(units=embed_dim, name="fc2")
        self.drop2 = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, training=False):
        x = self.fc1(x)
        x, gates = tf.split(x, num_or_size_splits=2, axis=-1)
        x = x * self.act(gates)
        x = self.drop1(x, training=training)
        x = self.fc2(x)
        x = self.drop2(x, training=training)
        return x


class SpatialGatingUnit(tf.keras.layers.Layer):
    """
    Spatial Gating Unit

    Based on: Pay Attention to MLPs - https://arxiv.org/abs/2105.08050
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        norm_layer = norm_layer_factory("layer_norm")
        self.norm = norm_layer(name="norm")

    def build(self, input_shape):
        seq_len = input_shape[-2]
        self.proj = tf.keras.layers.Dense(
            units=seq_len,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-6),
            bias_initializer=tf.keras.initializers.Ones(),
            name="proj",
        )

    def call(self, x, training=False):
        u, v = tf.split(x, num_or_size_splits=2, axis=-1)
        v = self.norm(v, training=training)
        v = tf.transpose(v, perm=(0, 2, 1))
        v = self.proj(v)
        v = tf.transpose(v, perm=(0, 2, 1))
        x = u * v
        return x


class GatedMLP(tf.keras.layers.Layer):
    """MLP as used in gMLP"""

    def __init__(
        self,
        hidden_dim: int,
        embed_dim: int,
        drop_rate: float,
        act_layer: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        act_layer = act_layer_factory(act_layer)

        self.fc1 = tf.keras.layers.Dense(units=hidden_dim, name="fc1")
        self.act = act_layer()
        self.drop1 = tf.keras.layers.Dropout(rate=drop_rate)
        self.gate = SpatialGatingUnit(name="gate")
        self.fc2 = tf.keras.layers.Dense(units=embed_dim, name="fc2")
        self.drop2 = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x, training=training)
        x = self.gate(x, training=training)
        x = self.fc2(x)
        x = self.drop2(x, training=training)
        return x
