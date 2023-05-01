from collections import OrderedDict
from typing import Tuple

import tensorflow as tf

from tfimm.layers import DropPath, PatchEmbeddings, norm_layer_factory

from .common import MLPBlock


def window_partition(
    x: tf.Tensor, window_size: int
) -> Tuple[tf.Tensor, Tuple[int, int]]:
    """
    Partition 4D tensor into non-overlapping windows with padding if needed.

    Args:
        x: Input tensor of shape (N, H, W, C).
        window_size: Window size.

    Returns:
        windows: Tensor after partition of shape
            (N * nb_windows, window_size, window_size, C).
        (Hp, Hw): Padded height and width to make spatial dimensions of x
            be multiples of window_size.
    """
    n, h, w, c = tf.unstack(tf.shape(x))
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size

    x = tf.cond(
        (pad_h > 0) | (pad_w > 0),
        lambda: tf.pad(x, [(0, 0), (0, pad_h), (0, pad_w), (0, 0)]),
        lambda: x,
    )
    _, hp, wp, _ = tf.unstack(tf.shape(x))

    x = tf.reshape(
        x, (n, hp // window_size, window_size, wp // window_size, window_size, c)
    )
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, (-1, window_size, window_size, c))
    return x, (hp, wp)


def window_unpartition(
    windows: tf.Tensor, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> tf.Tensor:
    """
    Unpartitions windows into original sequences and removes padding.

    Args:
        windows: Tensor of shape (N * nb_windows, window_size, window_size, C).
        pad_hw: Padded height and width (Hp, Wp) of x.
        hw: Original height and width (H, W) of x before padding.

    Returns:
        x: Unpartitioned tensor of shape (B, H, W, C).
    """
    hp, wp = pad_hw
    h, w = hw[0], hw[1]
    window_size = tf.shape(windows)[1]
    nb_windows = (hp // window_size) * (wp // window_size)
    n = tf.shape(windows)[0] // nb_windows

    x = tf.reshape(
        windows, (n, hp // window_size, wp // window_size, window_size, window_size, -1)
    )
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, (n, hp, wp, -1))

    x = tf.cond((hp > h) | (wp > w), lambda: x[:, :h, :w, :], lambda: x)
    return x


def get_rel_pos(
    q_size: int,
    k_size: int,
    rel_pos: tf.Tensor,
    interpolate_pos: bool,
) -> tf.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.

    Args:
        q_size: Size of query q.
        k_size: Size of key k.
        rel_pos: Relative position embeddings (M, C).
        interpolate_pos: If True, we interpolate positional embeddings to fit size.

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    m = tf.shape(rel_pos)[0]
    max_rel_dist = tf.cast(2 * tf.math.maximum(q_size, k_size) - 1, tf.int32)

    if interpolate_pos:
        # Interpolate positional embeddings if needed.
        rel_pos = tf.reshape(rel_pos, (1, m, -1))  # (1, M, C)
        rel_pos = tf.image.resize(
            rel_pos,
            size=(1, max_rel_dist),
            method="bilinear",
        )
        rel_pos = tf.reshape(rel_pos, (max_rel_dist, -1))  # (M', C)

    q_coords = tf.expand_dims(tf.range(q_size, dtype=tf.float32), axis=-1)
    k_coords = tf.expand_dims(tf.range(k_size, dtype=tf.float32), axis=0)
    # Scale the coords with short length if shapes for q and k are different.
    q_coords = q_coords * tf.cast(tf.math.maximum(k_size / q_size, 1.0), tf.float32)
    k_coords = k_coords * tf.cast(tf.math.maximum(q_size / k_size, 1.0), tf.float32)

    lambda_ = tf.cast(tf.math.maximum(q_size / k_size, 1.0), tf.float32)
    offset = tf.cast(k_size - 1, tf.float32) * lambda_
    relative_coords = (q_coords - k_coords) + offset
    relative_coords = tf.cast(relative_coords, tf.int32)
    return tf.gather(rel_pos, relative_coords)


def add_decomposed_rel_pos(
    attn: tf.Tensor,
    q: tf.Tensor,
    rel_pos_h: tf.Tensor,
    rel_pos_w: tf.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
    interpolate_pos: bool,
) -> tf.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from MVITV2 paper.

    Paper: MViTv2: Improved Multiscale Vision Transformers for Classification and
        Detection. https://arxiv.org/abs/2112.01526

    Code: https://github.com/facebookresearch/mvit

    Args:
        attn: Attention map, shape (N, H*W, H*W)
        q: Query q in the attention layer with shape (N, H*W, C).
        rel_pos_h: Relative position embeddings (Mh, C) for height axis.
        rel_pos_w: Relative position embeddings (Mw, C) for width axis.
        q_size: Spatial sequence size of query q with (q_h, q_w), usually (H, W).
        k_size: Spatial sequence size of key k with (k_h, k_w), usually (H, W).
        interpolate_pos: If True, we interpolate positional embeddings to fit size.

    Returns:
        attn: Attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size

    n, _, c = tf.unstack(tf.shape(q))  # (N, H*W, C)
    q = tf.reshape(q, (n, q_h, q_w, c))  # (N, H, W, C)

    r_h = get_rel_pos(q_h, k_h, rel_pos_h, interpolate_pos)  # (H, H, C)
    r_h = tf.einsum("nhwc,hkc->nhwk", q, r_h)  # (N, H, W, H)
    r_h = tf.expand_dims(r_h, axis=-1)  # (N, H, W, H, 1)

    r_w = get_rel_pos(q_w, k_w, rel_pos_w, interpolate_pos)  # (W, W, C)
    r_w = tf.einsum("nhwc,wkc->nhwk", q, r_w)  # (N, H, W, W)
    r_w = tf.expand_dims(r_w, axis=-2)  # (N, H, W, 1, W)

    attn = tf.reshape(attn, (n, q_h, q_w, k_h, k_w))  # (N, H, W, H, W)
    attn = attn + r_h + r_w
    attn = tf.reshape(attn, (n, q_h * q_w, k_h * k_w))  # (N, H*W, H*W)

    return attn


class RelPosAttention(tf.keras.layers.Layer):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        fixed_input_size: bool,
        embed_dim: int,
        nb_heads: int,
        qkv_bias: bool,
        use_rel_pos: bool,
        drop_rate: float,
        attn_drop_rate: float,
        **kwargs,
    ):
        """
        Args:
            fixed_input_size: If False, we interpolate positional embeddings to fit
                input size.
            embed_dim: Number of input channels.
            nb_heads: Number of attention heads.
            qkv_bias:  If True, add a learnable bias to query, key, value.
            use_rel_pos: If True, add relative positional embeddings to attention.
        """
        super().__init__(**kwargs)
        self.fixed_input_size = fixed_input_size
        self.embed_dim = embed_dim
        self.nb_heads = nb_heads
        self.qkv_bias = qkv_bias
        self.use_rel_pos = use_rel_pos
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate

        self.head_dim = self.embed_dim // self.nb_heads
        self.scale = self.head_dim**-0.5

        self.qkv = tf.keras.layers.Dense(embed_dim * 3, use_bias=qkv_bias, name="qkv")
        self.attn_drop = tf.keras.layers.Dropout(rate=attn_drop_rate)
        self.proj = tf.keras.layers.Dense(embed_dim, use_bias=True, name="proj")
        self.proj_drop = tf.keras.layers.Dropout(rate=drop_rate)

        self.rel_pos_h = None
        self.rel_pos_w = None

    def build(self, input_shape):
        # input_shape is (N, H, W, C)
        if self.use_rel_pos:
            # Initialize relative positional embeddings
            self.rel_pos_h = self.add_weight(
                shape=(2 * input_shape[1] - 1, self.head_dim),
                initializer="zeros",
                trainable=True,
                name="rel_pos_h",
            )
            self.rel_pos_w = self.add_weight(
                shape=(2 * input_shape[2] - 1, self.head_dim),
                initializer="zeros",
                trainable=True,
                name="rel_pos_w",
            )

    def call(self, x, training=False):
        n, h, w, _ = tf.unstack(tf.shape(x))  # (N, H, W, C)

        qkv = self.qkv(x)  # (N, H, W, 3*C)
        qkv = tf.reshape(qkv, (n, h * w, 3, self.nb_heads, -1))  # (N, H*W, 3, Hd, C/Hd)
        qkv = tf.transpose(qkv, (2, 0, 3, 1, 4))  # (3, N, Hd, H*W, C/Hd)
        qkv = tf.reshape(qkv, (3, n * self.nb_heads, h * w, -1))  # (3, N*Hd, H*W, C/Hd)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = tf.linalg.matmul(q, k, transpose_b=True)  # (N*Hd, H*W, H*W)
        attn *= self.scale

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(
                attn=attn,
                q=q,
                rel_pos_h=self.rel_pos_h,
                rel_pos_w=self.rel_pos_w,
                q_size=(h, w),
                k_size=(h, w),
                interpolate_pos=not self.fixed_input_size,
            )

        attn = tf.nn.softmax(attn, axis=-1)  # (N*Hd, H*W, H*W)
        attn = self.attn_drop(attn, training=training)
        x = tf.linalg.matmul(attn, v)  # (N*Hd, H*W, C/Hd)
        x = tf.reshape(x, (n, self.nb_heads, h, w, -1))  # (N, Hd, H, W, C/Hd)
        x = tf.transpose(x, (0, 2, 3, 1, 4))  # (N, H, W, Hd, C/Hd)
        x = tf.reshape(x, (n, h, w, -1))  # (N, H, W, C)

        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        return x


class ImageEncoderBlock(tf.keras.layers.Layer):
    """
    Transformer blocks with support for window attention and residual propagation.
    """

    def __init__(
        self,
        fixed_input_size: bool,
        embed_dim: int,
        nb_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        norm_layer: str,
        act_layer: str,
        use_rel_pos: bool,
        window_size: int,
        drop_rate: float,
        attn_drop_rate: float,
        drop_path_rate: float,
        **kwargs,
    ) -> None:
        """
        Args:
            fixed_input_size: If False, we interpolate positional embeddings to fit
                input size.
            embed_dim: Number of embedding dimensions.
            nb_heads: Number of attention heads in each ViT block.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            norm_layer: Normalization layer.
            act_layer: Activation layer.
            use_rel_pos: If True, add relative positional embeddings to the attention
                map.
            window_size: Window size for window attention blocks. If it equals 0, then
                use global attention.
        """
        super().__init__(**kwargs)
        self.fixed_input_size = fixed_input_size
        self.embed_dim = embed_dim
        self.nb_heads = nb_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        self.use_rel_pos = use_rel_pos
        self.window_size = window_size
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate

        norm_layer = norm_layer_factory(norm_layer)

        self.norm1 = norm_layer(name="norm1")
        self.attn = RelPosAttention(
            fixed_input_size=self.fixed_input_size,
            embed_dim=self.embed_dim,
            nb_heads=self.nb_heads,
            qkv_bias=self.qkv_bias,
            use_rel_pos=self.use_rel_pos,
            drop_rate=self.drop_rate,
            attn_drop_rate=self.attn_drop_rate,
            name="attn",
        )
        self.drop_path1 = DropPath(drop_prob=self.drop_path_rate)
        self.norm2 = norm_layer(name="norm2")
        self.mlp = MLPBlock(
            hidden_dim=int(self.embed_dim * self.mlp_ratio),
            embed_dim=self.embed_dim,
            act_layer=self.act_layer,
            drop_rate=self.drop_rate,
            name="mlp",
        )
        self.drop_path2 = DropPath(drop_prob=self.drop_path_rate)

    def call(self, x, training=False):
        shortcut = x
        x = self.norm1(x, training=training)
        if self.window_size > 0:
            hw = tf.shape(x)[1:3]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x, training=training)

        if self.window_size > 0:
            x = window_unpartition(x, pad_hw, hw)
        x = self.drop_path1(x, training=training)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x, training=training)
        x = self.mlp(x, training=training)
        x = self.drop_path2(x, training=training)
        x = shortcut + x

        return x


class ImageEncoder(tf.keras.Model):
    def __init__(
        self,
        input_size: Tuple[int, int] = (1024, 1024),
        fixed_input_size: bool = True,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        nb_blocks: int = 12,
        nb_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_channels: int = 256,
        qkv_bias: bool = True,
        norm_layer: str = "layer_norm",
        act_layer: str = "gelu",
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        global_attn_indices: Tuple[int, ...] = (),
        window_size: int = 0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        **kwargs,
    ):
        """
        Args:
            input_size: Input image size.
            fixed_input_size: If False, we allow arbitrary input sizes and interpolate
                the positional embeddings accordingly. If True, only one input size
                is supported.
            patch_size: Patch size.
            in_channels: Number of input image channels.
            embed_dim: Patch embedding dimension.
            nb_blocks: Depth of ViT.
            nb_heads: Number of attention heads in each ViT block.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            out_channels: Number of channels in image embedding.
            qkv_bias: If True, add a learnable bias to query, key, value.
            norm_layer: Normalization layer.
            act_layer: Activation layer.
            use_abs_pos: If True, use absolute positional embeddings.
            use_rel_pos: If True, add relative positional embeddings to attention.
            global_attn_indices: Indexes for blocks using global attention.
            window_size: Window size for window attention blocks.
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            drop_path_rate: Dropout rate for stochastic depth
        """
        super().__init__(**kwargs)
        self.input_size = input_size
        self.fixed_input_size = fixed_input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.nb_blocks = nb_blocks
        self.nb_heads = nb_heads
        self.mlp_ratio = mlp_ratio
        self.out_channels = out_channels
        self.qkv_bias = qkv_bias
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        self.use_abs_pos = use_abs_pos
        self.use_rel_pos = use_rel_pos
        self.global_attn_indices = global_attn_indices
        self.window_size = window_size
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate

        self.patch_embed = PatchEmbeddings(
            patch_size=patch_size,
            embed_dim=embed_dim,
            flatten=False,
            name="patch_embed",
        )
        self.pos_embed = None

        self.blocks = [
            ImageEncoderBlock(
                fixed_input_size=self.fixed_input_size,
                embed_dim=self.embed_dim,
                nb_heads=self.nb_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                drop_path_rate=self.drop_path_rate,
                norm_layer=self.norm_layer,
                act_layer=self.act_layer,
                use_rel_pos=self.use_rel_pos,
                window_size=(
                    self.window_size if j not in self.global_attn_indices else 0
                ),
                name=f"blocks/{j}",
            )
            for j in range(self.nb_blocks)
        ]

        # Note that this norm layer is not affected by the norm_layer parameter.
        neck_norm_layer = norm_layer_factory("layer_norm_eps_1e-6")
        self.neck = [
            tf.keras.layers.Conv2D(
                filters=self.out_channels, kernel_size=1, use_bias=False, name="neck/0"
            ),
            neck_norm_layer(name="neck/1"),
            tf.keras.layers.Conv2D(
                filters=self.out_channels,
                kernel_size=3,
                padding="same",
                use_bias=False,
                name="neck/2",
            ),
            neck_norm_layer(name="neck/3"),
        ]

    @property
    def grid_size(self):
        return (
            self.input_size[0] // self.patch_size,
            self.input_size[1] // self.patch_size,
        )

    def build(self, input_shape):
        if self.use_abs_pos:
            self.pos_embed = self.add_weight(
                shape=(1, *self.grid_size, self.embed_dim),
                initializer="zeros",
                trainable=True,
                name="pos_embed",
            )

    def call(self, x, training=False, return_features=False):
        features = OrderedDict()
        x = self.patch_embed(x, training=training)
        if self.use_abs_pos:
            if not self.fixed_input_size:
                pos_embed = tf.image.resize(
                    self.pos_embed, size=tf.shape(x)[1:3], method="bilinear"
                )
            else:
                pos_embed = self.pos_embed
            x = x + pos_embed
        features["patch_embedding"] = x

        for j, block in enumerate(self.blocks):
            x = block(x, training=training)
            features[f"block_{j}"] = x

        for j, layer in enumerate(self.neck):
            x = layer(x, training=training)
        features["neck"] = x

        return (x, features) if return_features else x
