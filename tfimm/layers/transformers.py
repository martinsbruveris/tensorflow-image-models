"""
Common layers shared between transformer architectures.

Copyright 2021 Martins Bruveris
"""
from typing import Optional, Tuple

import tensorflow as tf

from tfimm.layers.factory import act_layer_factory, norm_layer_factory


def interpolate_pos_embeddings(
    pos_embed: tf.Tensor,
    src_grid_size: Tuple[int, int],
    tgt_grid_size: Tuple[int, int],
    nb_tokens: int = 0,
) -> tf.Tensor:
    """
    This method allows to interpolate the pre-trained position encodings, to be
    able to use the model on higher resolution images.

    Args:
        pos_embed: Positional embeddings to interpolate, shape (1, N, D)
        src_grid_size: Grid size of given embeddings.
        tgt_grid_size: Input size to which position embeddings should be adapted
        nb_tokens: How many token should be ignored for interpolation (e.g., class or
            distillation tokens)

    Returns:
        Position embeddings (including class tokens) appropriate to input_size
    """
    if src_grid_size == tgt_grid_size:
        return pos_embed  # No interpolation needed

    src_pos_embed = pos_embed[:, nb_tokens:]
    src_pos_embed = tf.reshape(src_pos_embed, shape=(1, *src_grid_size, -1))
    tgt_pos_embed = tf.image.resize(
        images=src_pos_embed,
        size=tgt_grid_size,
        method="bicubic",
    )
    tgt_pos_embed = tf.cast(tgt_pos_embed, dtype=src_pos_embed.dtype)
    nb_pos_tokens = tgt_grid_size[0] * tgt_grid_size[1]
    tgt_pos_embed = tf.reshape(tgt_pos_embed, shape=(1, nb_pos_tokens, -1))
    tgt_pos_embed = tf.concat((pos_embed[:, :nb_tokens], tgt_pos_embed), axis=1)
    return tgt_pos_embed


def interpolate_pos_embeddings_grid(
    pos_embed: tf.Tensor,
    tgt_grid_size: Tuple[int, int],
) -> tf.Tensor:
    """
    This method allows to interpolate the pre-trained position encodings, to be
    able to use the model on higher resolution images. We use this function if position
    embeddings are already given as a grid.

    Args:
        pos_embed: Positional embeddings to interpolate, shape (B, H, W, D)
        tgt_grid_size: Input size to which position embeddings should be adapted

    Returns:
        Position embeddings (including class tokens) appropriate to input_size
    """
    height, width = tf.unstack(tf.shape(pos_embed)[1:3])
    if (height, width) == tgt_grid_size:
        return pos_embed  # No interpolation needed

    tgt_pos_embed = tf.image.resize(
        images=pos_embed,
        size=tgt_grid_size,
        method="bicubic",
    )
    tgt_pos_embed = tf.cast(tgt_pos_embed, dtype=pos_embed.dtype)
    return tgt_pos_embed


class PatchEmbeddings(tf.keras.layers.Layer):
    """
    Image to Patch Embedding.

    Supports overlapping patches when stride is specified. Used, e.g., in Pyramid
    Vision Transformer V2 or PoolFormer.

    Parameters:
        patch_size: Patchifying the image is implemented via a convolutional layer with
            kernel size equal to ``patch_size``.
        embed_dim: Number of embedding dimensions.
        stride: If ``None``, we use non-overlapping patches, i.e.,
            ``stride=patch_size``. Other values are used e.g., in PVT v2 or PoolFormer.
        padding: Padding is only applied when overlapping patches are used. In that
            case we default to ``padding = patch_size // 2``, but other values can be
            specified via this parameter. For non-overlapping patches, padding is
            always 0.
        flatten: If ``True``, we reshape the patchified image from ``(H', W', D)`` into
            ``(H'*W', D)``.
        norm_layer: Normalization layer to be applied after the patch projection.
        kernel_initializer: Initializer for kernel weights
        bias_initializer: Initializer for bias weights
        **kwargs: Other arguments are passed to the parent class.
    """

    def __init__(
        self,
        patch_size: int,
        embed_dim: int,
        stride: Optional[int] = None,
        padding: Optional[int] = None,
        flatten: bool = True,
        norm_layer: str = "",
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # If stride=None we default to non-overlapping patches
        self.stride = stride or patch_size
        # Default padding for PVT transformer
        self.padding = padding or patch_size // 2
        self.flatten = flatten
        self.norm_layer = norm_layer_factory(norm_layer)

        # We only apply padding, if we use overlapping patches. For non-overlapping
        # patches we assume image size is divisible by patch size.
        self.pad = tf.keras.layers.ZeroPadding2D(
            padding=self.padding if self.stride != self.patch_size else 0
        )
        self.projection = tf.keras.layers.Conv2D(
            filters=self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.stride,
            use_bias=True,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name="proj",
        )
        self.norm = self.norm_layer(name="norm")

    def call(self, x, training=False, return_shape=False):
        """
        Forward pass through the layer.

        An image of shape ``(H, W, C)`` will be mapped to patches of shape
        ``(H', W', D)``, where ``D`` is ``embed_dim``. The output is then optionally
        flattened to ``(H'*W', D)``.

        Arguments:
             x: Input to layer
             training: Training or inference phase?
             return_shape: If ``True``, we additionally return the spatial shape of
                the image as well as the tokens.

        Returns:
            We return the flattened tokens and if ``return_shape=True``, additionally
            the shape ``(H', W')``.

            This information is used by models that use convolutional layers in addition
            to attention layers and convolutional layers need to know the original shape
            of the token list.
        """
        x = self.pad(x)
        x = self.projection(x)

        batch_size, height, width = tf.unstack(tf.shape(x)[:3])
        if self.flatten:
            # Change the 2D spatial dimensions to a single temporal dimension.
            x = tf.reshape(tensor=x, shape=(batch_size, height * width, -1))

        x = self.norm(x, training=training)
        return (x, (height, width)) if return_shape else x


class MLP(tf.keras.layers.Layer):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

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

        self.fc1 = tf.keras.layers.Dense(
            units=hidden_dim,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name="fc1",
        )
        self.act = act_layer()
        self.drop1 = tf.keras.layers.Dropout(rate=drop_rate)
        self.fc2 = tf.keras.layers.Dense(
            units=embed_dim,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name="fc2",
        )
        self.drop2 = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x, training=training)
        x = self.fc2(x)
        x = self.drop2(x, training=training)
        return x


class ConvMLP(tf.keras.layers.Layer):
    """
    ConvMLP is the same block as MLP, but it uses 1x1 convolutions instead of fully
    connected layers.

    Used in ``ConvNeXt`` models.
    """

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

        self.fc1 = tf.keras.layers.Conv2D(
            filters=hidden_dim,
            kernel_size=1,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name="fc1",
        )
        self.act = act_layer()
        self.drop1 = tf.keras.layers.Dropout(rate=drop_rate)
        self.fc2 = tf.keras.layers.Conv2D(
            filters=embed_dim,
            kernel_size=1,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name="fc2",
        )
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
