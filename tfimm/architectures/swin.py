"""
TensorFlow implementation of the Swin Transformer

Based on the implementation by Rishigami
Source: https://github.com/rishigami/Swin-Transformer-TF

Paper: Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
Link: https://arxiv.org/abs/2103.14030
Official implementation: https://github.com/microsoft/Swin-Transformer

Copyright: 2021 Martins Bruveris
Copyright: 2021 Rishigami
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import tensorflow as tf

from tfimm.layers import DropPath, MLP, PatchEmbeddings, norm_layer_factory
from tfimm.models import ModelConfig, keras_serializable, register_model
from tfimm.utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


# TODO: old imports
from tensorflow.keras.layers import Dense, Conv2D, LayerNormalization

# model_registry will add each entrypoint fn to this
__all__ = ["SwinTransformer", "SwinTransformerConfig"]


@dataclass
class SwinTransformerConfig(ModelConfig):
    nb_classes: int = 1000
    in_chans: int = 3
    input_size: Tuple[int, int] = (224, 224)
    patch_size: int = 4
    embed_dim: int = 96
    nb_blocks: Tuple = (2, 2, 6, 2)
    nb_heads: Tuple = (3, 6, 12, 24)
    window_size: int = 7
    mlp_ratio: float = 4.
    qkv_bias: bool = True
    # Regularization
    drop_rate: float = 0.
    attn_drop_rate: float = 0.
    drop_path_rate: float = 0.1
    # Other parameters
    norm_layer: str = "layer_norm"
    act_layer: str = "gelu"
    ape: bool = False  # Absolute position embedding
    patch_norm: bool = True
    # Parameters for inference
    interpolate_input: bool = False
    crop_pct: float = 0.9
    interpolation: str = "bicubic"
    # Preprocessing
    mean: Tuple[float, float, float] = IMAGENET_DEFAULT_MEAN
    std: Tuple[float, float, float] = IMAGENET_DEFAULT_STD
    # Weight transfer
    first_conv: str = "patch_embed/proj"
    classifier: str = "head"

    @property
    def patch_resolution(self):
        """Resolution of grid of patches."""
        return (
            self.input_size[0] // self.patch_size,
            self.input_size[1] // self.patch_size
        )

    @property
    def nb_patches(self):
        return self.patch_resolution[0] * self.patch_resolution[1]


def window_partition(x: tf.Tensor, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): Window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    b, h, w, c = tf.unstack(tf.shape(x))
    x = tf.reshape(
        x, shape=(-1, h // window_size, window_size, w // window_size, window_size, c)
    )
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
    windows = tf.reshape(x, shape=(-1, window_size, window_size, c))
    return windows


def window_reverse(windows, window_size, h, w, c):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image
        c (int): Number of channels in image

    Returns:
        x: (B, H, W, C)
    """
    x = tf.reshape(
        windows,
        shape=(-1, h // window_size, w // window_size, window_size, window_size, c)
    )
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, shape=(-1, h, w, c))
    return x


class WindowAttention(tf.keras.layers.Layer):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0., **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = Dense(dim * 3, use_bias=qkv_bias,
                         name=f'qkv')
        self.attn_drop = tf.keras.layers.Dropout(attn_drop)
        self.proj = Dense(dim, name=f'proj')
        self.proj_drop = tf.keras.layers.Dropout(proj_drop)

    def build(self, input_shape):
        self.relative_position_bias_table = self.add_weight(f'relative_position_bias_table',
                                                            shape=(
                                                                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads),
                                                            initializer=tf.initializers.Zeros(), trainable=True)

        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :,
                                         None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).astype(np.int64)
        self.relative_position_index = tf.Variable(initial_value=tf.convert_to_tensor(
            relative_position_index), trainable=False, name=f'relative_position_index')
        self.built = True

    def call(self, x, mask=None):
        B_, N, C = x.get_shape().as_list()
        qkv = tf.transpose(tf.reshape(self.qkv(
            x), shape=[-1, N, 3, self.num_heads, C // self.num_heads]), perm=[2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ tf.transpose(k, perm=[0, 1, 3, 2]))
        relative_position_bias = tf.gather(self.relative_position_bias_table, tf.reshape(
            self.relative_position_index, shape=[-1]))
        relative_position_bias = tf.reshape(relative_position_bias, shape=[
                                            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1])
        relative_position_bias = tf.transpose(
            relative_position_bias, perm=[2, 0, 1])
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.get_shape()[0]  # tf.shape(mask)[0]
            attn = tf.reshape(attn, shape=[-1, nW, self.num_heads, N, N]) + tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32)
            attn = tf.reshape(attn, shape=[-1, self.num_heads, N, N])
            attn = tf.nn.softmax(attn, axis=-1)
        else:
            attn = tf.nn.softmax(attn, axis=-1)

        attn = self.attn_drop(attn)

        x = tf.transpose((attn @ v), perm=[0, 2, 1, 3])
        x = tf.reshape(x, shape=[-1, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.,
                 qkv_bias=True, drop=0., attn_drop=0., drop_path_prob=0., norm_layer=LayerNormalization, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(epsilon=1e-5, name=f'norm1')
        self.attn = WindowAttention(dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                                    name="attn")
        self.drop_path = DropPath(drop_prob=drop_path_prob)
        self.norm2 = norm_layer(epsilon=1e-5, name=f'norm2')
        self.mlp = MLP(
            hidden_dim=int(dim * mlp_ratio),
            embed_dim=dim,
            drop_rate=drop,
            act_layer="gelu",  # TODO: Replace by config
            name="mlp"
        )

    def build(self, input_shape):
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = np.zeros([1, H, W, 1])
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            img_mask = tf.convert_to_tensor(img_mask)
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = tf.reshape(
                mask_windows, shape=[-1, self.window_size * self.window_size])
            attn_mask = tf.expand_dims(
                mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(
                initial_value=attn_mask, trainable=False, name=f'attn_mask')
        else:
            self.attn_mask = None

        self.built = True

    def call(self, x):
        H, W = self.input_resolution
        B, L, C = x.get_shape().as_list()
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = tf.reshape(x, shape=[-1, H, W, C])

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(
            x_windows, shape=[-1, self.window_size * self.window_size, C])

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = tf.reshape(
            attn_windows, shape=[-1, self.window_size, self.window_size, C])
        shifted_x = window_reverse(attn_windows, self.window_size, H, W, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = tf.roll(shifted_x, shift=[
                        self.shift_size, self.shift_size], axis=[1, 2])
        else:
            x = shifted_x
        x = tf.reshape(x, shape=[-1, H * W, C])

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(tf.keras.layers.Layer):
    def __init__(self, input_resolution, dim, norm_layer=LayerNormalization, **kwargs):
        super().__init__(**kwargs)
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = Dense(2 * dim, use_bias=False,
                               name=f'reduction')
        self.norm = norm_layer(epsilon=1e-5, name=f'norm')

    def call(self, x):
        H, W = self.input_resolution
        B, L, C = x.get_shape().as_list()
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = tf.reshape(x, shape=[-1, H, W, C])

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = tf.concat([x0, x1, x2, x3], axis=-1)
        x = tf.reshape(x, shape=[-1, (H // 2) * (W // 2), 4 * C])

        x = self.norm(x)
        x = self.reduction(x)

        return x


class SwinTransformerStage(tf.keras.layers.Layer):
    def __init__(
        self,
        cfg: SwinTransformerConfig,
        input_size: Tuple[int, int],
        embed_dim: int,
        nb_blocks: int,
        nb_heads: int,
        drop_path_prob: np.ndarray,
        downsample: bool,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.cfg = cfg

        input_resolution = input_size
        dim = embed_dim
        depth = nb_blocks
        num_heads = nb_heads
        window_size = cfg.window_size
        mlp_ratio = cfg.mlp_ratio
        qkv_bias = cfg.qkv_bias
        drop = cfg.drop_rate
        attn_drop = cfg.attn_drop_rate
        norm_layer = LayerNormalization  # TODO: Replace with factory
        # downsample = PatchMerging if downsample else None

        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = [SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                           num_heads=num_heads, window_size=window_size,
                                           shift_size=0 if (
                                               i % 2 == 0) else window_size // 2,
                                           mlp_ratio=mlp_ratio,
                                           qkv_bias=qkv_bias,
                                           drop=drop, attn_drop=attn_drop,
                                           drop_path_prob=drop_path_prob[i],
                                           norm_layer=norm_layer,
                                            name=f"blocks/{i}") for i in range(depth)]
        if downsample:
            self.downsample = PatchMerging(
                input_resolution, dim=dim, norm_layer=norm_layer, name="downsample")
        else:
            self.downsample = tf.keras.layers.Activation("linear")

    def call(self, x, training=False):
        for block in self.blocks:
            x = block(x, training=training)
        x = self.downsample(x, training=training)
        return x


@keras_serializable
class SwinTransformer(tf.keras.Model):
    cfg_class = SwinTransformerConfig

    def __init__(self, cfg: SwinTransformerConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.norm_layer = norm_layer_factory(cfg.norm_layer)

        # Split image into non-overlapping patches
        self.patch_embed = PatchEmbeddings(
            patch_size=cfg.patch_size,
            embed_dim=cfg.embed_dim,
            norm_layer=cfg.norm_layer,
            name="patch_embed",
        )
        if cfg.ape:  # Absolute position embedding
            self.absolute_pos_embed = self.add_weight(
                name="absolute_pos_embed",
                shape=(1, cfg.nb_patches, cfg.embed_dim),
                initializer=tf.initializers.Zeros()
            )
        self.drop = tf.keras.layers.Dropout(cfg.drop_rate)

        # Stochastic depth
        dpr = np.linspace(0., cfg.drop_path_rate, sum(cfg.nb_blocks))

        # Build stages
        self.stages = []
        nb_stages = len(cfg.nb_blocks)
        block_idx_to = 0
        for idx in range(nb_stages):
            block_idx_from = block_idx_to
            block_idx_to = block_idx_to + cfg.nb_blocks[idx]

            self.stages.append(
                SwinTransformerStage(
                    cfg=cfg,
                    input_size=(
                        cfg.patch_resolution[0] // (2 ** idx),
                        cfg.patch_resolution[1] // (2 ** idx)
                    ),
                    embed_dim=int(cfg.embed_dim * 2 ** idx),
                    nb_blocks=cfg.nb_blocks[idx],
                    nb_heads=cfg.nb_heads[idx],
                    drop_path_prob=dpr[block_idx_from:block_idx_to],
                    downsample=idx < nb_stages - 1,
                    name=f"layers/{idx}",
                    ))
        self.norm = self.norm_layer(name="norm")
        self.pool = tf.keras.layers.GlobalAveragePooling1D()
        self.head = (
            tf.keras.layers.Dense(units=cfg.nb_classes, name="head")
            if cfg.nb_classes > 0
            else tf.keras.layers.Activation("linear")  # Identity layer
        )

    @property
    def dummy_inputs(self) -> tf.Tensor:
        return tf.zeros((1, *self.cfg.input_size, self.cfg.in_chans))

    def forward_features(self, x, training=False):
        x = self.patch_embed(x, training=training)
        if self.cfg.ape:
            x = x + self.absolute_pos_embed
        x = self.drop(x, training=training)

        for stage in self.stages:
            x = stage(x, training=training)
        x = self.norm(x, training=training)
        x = self.pool(x)
        return x

    def call(self, x, training=False):
        x = self.forward_features(x, training=training)
        x = self.head(x)
        return x


@register_model
def swin_tiny_patch4_window7_224():
    """ Swin-T @ 224x224, trained ImageNet-1k
    """
    cfg = SwinTransformerConfig(
        name="swin_tiny_patch4_window7_224",
        url="",
        patch_size=4,
        embed_dim=96,
        nb_blocks=(2, 2, 6, 2),
        nb_heads=(3, 6, 12, 24),
        window_size=7,
    )
    return SwinTransformer, cfg


@register_model
def swin_small_patch4_window7_224():
    """ Swin-S @ 224x224, trained ImageNet-1k
    """
    cfg = SwinTransformerConfig(
        name="swin_small_patch4_window7_224",
        url="",
        patch_size=4,
        embed_dim=96,
        nb_blocks=(2, 2, 18, 2),
        nb_heads=(3, 6, 12, 24),
        window_size=7,
    )
    return SwinTransformer, cfg


@register_model
def swin_base_patch4_window12_384():
    """ Swin-B @ 384x384, pretrained ImageNet-22k, fine tune 1k
    """
    cfg = SwinTransformerConfig(
        name="swin_base_patch4_window12_384",
        url="",
        input_size=(384, 384),
        patch_size=4,
        embed_dim=128,
        nb_blocks=(2, 2, 18, 2),
        nb_heads=(4, 8, 16, 32),
        window_size=12,
        crop_pct=1.0
    )
    return SwinTransformer, cfg


@register_model
def swin_base_patch4_window7_224():
    """ Swin-B @ 224x224, pretrained ImageNet-22k, fine tune 1k
    """
    cfg = SwinTransformerConfig(
        name="swin_base_patch4_window7_224",
        url="",
        patch_size=4,
        embed_dim=128,
        nb_blocks=(2, 2, 18, 2),
        nb_heads=(4, 8, 16, 32),
        window_size=7,
    )
    return SwinTransformer, cfg


@register_model
def swin_base_patch4_window12_384_in22k():
    """ Swin-B @ 384x384, trained ImageNet-22k
    """
    cfg = SwinTransformerConfig(
        name="swin_base_patch4_window12_384_in22k",
        url="",
        nb_classes=21841,
        input_size=(384, 384),
        patch_size=4,
        embed_dim=128,
        nb_blocks=(2, 2, 18, 2),
        nb_heads=(4, 8, 16, 32),
        window_size=12,
        crop_pct=1.0,
    )
    return SwinTransformer, cfg


@register_model
def swin_base_patch4_window7_224_in22k():
    """ Swin-B @ 224x224, trained ImageNet-22k
    """
    cfg = SwinTransformerConfig(
        name="swin_base_patch4_window7_224_in22k",
        url="",
        nb_classes=21841,
        patch_size=4,
        embed_dim=128,
        nb_blocks=(2, 2, 18, 2),
        nb_heads=(4, 8, 16, 32),
        window_size=7,
    )
    return SwinTransformer, cfg


@register_model
def swin_large_patch4_window12_384():
    """ Swin-L @ 384x384, pretrained ImageNet-22k, fine tune 1k
    """
    cfg = SwinTransformerConfig(
        name="swin_large_patch4_window12_384",
        url="",
        input_size=(384, 384),
        patch_size=4,
        embed_dim=192,
        nb_blocks=(2, 2, 18, 2),
        nb_heads=(6, 12, 24, 48),
        window_size=12,
        crop_pct=1.0
    )
    return SwinTransformer, cfg


@register_model
def swin_large_patch4_window7_224():
    """ Swin-L @ 224x224, pretrained ImageNet-22k, fine tune 1k
    """
    cfg = SwinTransformerConfig(
        name="swin_large_patch4_window7_224",
        url="",
        patch_size=4,
        embed_dim=192,
        nb_blocks=(2, 2, 18, 2),
        nb_heads=(6, 12, 24, 48),
        window_size=7,
    )
    return SwinTransformer, cfg


@register_model
def swin_large_patch4_window12_384_in22k():
    """ Swin-L @ 384x384, trained ImageNet-22k
    """
    cfg = SwinTransformerConfig(
        name="swin_large_patch4_window12_384_in22k",
        url="",
        nb_classes=21841,
        input_size=(384, 384),
        patch_size=4,
        embed_dim=192,
        nb_blocks=(2, 2, 18, 2),
        nb_heads=(6, 12, 24, 48),
        window_size=12,
        crop_pct=1.0,
    )
    return SwinTransformer, cfg


@register_model
def swin_large_patch4_window7_224_in22k():
    """ Swin-L @ 224x224, trained ImageNet-22k
    """
    cfg = SwinTransformerConfig(
        name="swin_large_patch4_window7_224_in22k",
        url="",
        nb_classes=21841,
        patch_size=4,
        embed_dim=192,
        nb_blocks=(2, 2, 18, 2),
        nb_heads=(6, 12, 24, 48),
        window_size=7,
    )
    return SwinTransformer, cfg
