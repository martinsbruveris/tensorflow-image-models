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
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from tfimm.layers import MLP, DropPath, PatchEmbeddings, norm_layer_factory
from tfimm.models import ModelConfig, keras_serializable, register_model
from tfimm.utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# model_registry will add each entrypoint fn to this
__all__ = ["SwinTransformer", "SwinTransformerConfig"]


@dataclass
class SwinTransformerConfig(ModelConfig):
    nb_classes: int = 1000
    in_channels: int = 3
    input_size: Tuple[int, int] = (224, 224)
    patch_size: int = 4
    embed_dim: int = 96
    nb_blocks: Tuple = (2, 2, 6, 2)
    nb_heads: Tuple = (3, 6, 12, 24)
    window_size: int = 7
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    # Regularization
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    # Other parameters
    norm_layer: str = "layer_norm"
    act_layer: str = "gelu"
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
            self.input_size[1] // self.patch_size,
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
        shape=(-1, h // window_size, w // window_size, window_size, window_size, c),
    )
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, shape=(-1, h, w, c))
    return x


class WindowAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        cfg: SwinTransformerConfig,
        embed_dim: int,
        nb_heads: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.embed_dim = embed_dim
        self.nb_heads = nb_heads

        self.qkv = tf.keras.layers.Dense(
            embed_dim * 3, use_bias=cfg.qkv_bias, name="qkv"
        )
        self.attn_drop = tf.keras.layers.Dropout(cfg.attn_drop_rate)
        self.proj = tf.keras.layers.Dense(embed_dim, name="proj")
        self.proj_drop = tf.keras.layers.Dropout(cfg.drop_rate)

    def build(self, input_shape):
        window_size = self.cfg.window_size

        # The weights have to be created inside the build() function for the right
        # name scope to be set.
        self.relative_position_bias_table = self.add_weight(
            name="relative_position_bias_table",
            shape=((2 * window_size - 1) * (2 * window_size - 1), self.nb_heads),
            initializer=tf.initializers.Zeros(),
            trainable=True,
        )

        coords_h = np.arange(window_size)
        coords_w = np.arange(window_size)
        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose((1, 2, 0))
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1).astype(np.int64)
        self.relative_position_index = tf.Variable(
            name="relative_position_index",
            initial_value=tf.convert_to_tensor(relative_position_index),
            trainable=False,
        )

    def call(self, inputs, training=False):
        nb_heads = self.nb_heads
        window_size = self.cfg.window_size

        # Inputs are the batch and the attention mask
        x, mask = inputs[0], inputs[1]
        _, n, c = tf.unstack(tf.shape(x))

        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, shape=(-1, n, 3, nb_heads, c // nb_heads))
        qkv = tf.transpose(qkv, perm=(2, 0, 3, 1, 4))
        q, k, v = tf.unstack(qkv)

        scale = (self.embed_dim // nb_heads) ** -0.5
        q = q * scale
        attn = q @ tf.transpose(k, perm=(0, 1, 3, 2))
        relative_position_bias = tf.gather(
            self.relative_position_bias_table,
            tf.reshape(self.relative_position_index, shape=(-1,)),
        )
        relative_position_bias = tf.reshape(
            relative_position_bias,
            shape=(window_size ** 2, window_size ** 2, -1),
        )
        relative_position_bias = tf.transpose(relative_position_bias, perm=(2, 0, 1))
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        nw = mask.get_shape()[0]  # tf.shape(mask)[0]
        mask = tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0)
        mask = tf.cast(mask, attn.dtype)
        attn = tf.reshape(attn, shape=(-1, nw, nb_heads, n, n)) + mask
        attn = tf.reshape(attn, shape=(-1, nb_heads, n, n))
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)

        x = tf.transpose((attn @ v), perm=(0, 2, 1, 3))
        x = tf.reshape(x, shape=(-1, n, c))
        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        return x


class SwinTransformerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        cfg: SwinTransformerConfig,
        input_size: Tuple[int, int],
        embed_dim: int,
        nb_heads: int,
        drop_path_rate: float,
        shift_size: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.input_size = input_size
        self.shift_size = shift_size
        self.norm_layer = norm_layer_factory(cfg.norm_layer)
        self.window_size = cfg.window_size

        # If the image resolution is smaller than the window size, there is no point
        # shifting windows, since we already capture the global context in that case.
        if min(self.input_size) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_size)

        self.norm1 = self.norm_layer(name="norm1")
        self.attn = WindowAttention(
            cfg=cfg,
            embed_dim=embed_dim,
            nb_heads=nb_heads,
            name="attn",
        )
        self.attn_mask = None
        self.drop_path = DropPath(drop_prob=drop_path_rate)
        self.norm2 = self.norm_layer(name="norm2")
        self.mlp = MLP(
            hidden_dim=int(embed_dim * cfg.mlp_ratio),
            embed_dim=embed_dim,
            drop_rate=cfg.drop_rate,
            act_layer=cfg.act_layer,
            name="mlp",
        )

    def build(self, input_shape):
        h, w = self.input_size
        window_size = self.window_size
        shift_size = self.shift_size

        if shift_size > 0:
            img_mask = np.zeros([1, h, w, 1])
            h_slices = (
                slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None),
            )
            w_slices = (
                slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None),
            )
            cnt = 0
            for h_slice in h_slices:
                for w_slice in w_slices:
                    img_mask[:, h_slice, w_slice, :] = cnt
                    cnt += 1

            img_mask = tf.convert_to_tensor(img_mask)
            mask_windows = window_partition(img_mask, window_size)
            mask_windows = tf.reshape(mask_windows, shape=(-1, window_size ** 2))
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(
                mask_windows, axis=2
            )
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            # Only the non-trivial attention mask is a Variable, because in the PyTorch
            # model, only the non-trivial attention mask exists. The trivial maks
            # is replaced by None and if-statements in the call function.
            self.attn_mask = tf.Variable(
                initial_value=attn_mask, trainable=False, name="attn_mask"
            )
        else:
            # Attention mask is applied additively, so zero-mask has no effect
            # Broadcasting will take care of mapping it to the correct dimensions.
            self.attn_mask = tf.Variable(
                initial_value=tf.zeros((1,)), trainable=False, name="attn_mask"
            )

    def call(self, x, training=False):
        window_size = self.window_size
        shift_size = self.shift_size

        h, w = self.input_size
        b, l, c = tf.unstack(tf.shape(x))

        shortcut = x
        x = self.norm1(x, training=training)
        x = tf.reshape(x, shape=(-1, h, w, c))

        # Cyclic shift (Identify, if shift_size == 0)
        shifted_x = tf.roll(x, shift=(-shift_size, -shift_size), axis=[1, 2])

        # Partition windows
        x_windows = window_partition(shifted_x, window_size)
        x_windows = tf.reshape(x_windows, shape=(-1, window_size ** 2, c))

        # W-MSA/SW-MSA
        attn_windows = self.attn([x_windows, self.attn_mask])

        # Merge windows
        attn_windows = tf.reshape(attn_windows, shape=(-1, window_size, window_size, c))
        shifted_x = window_reverse(attn_windows, window_size, h, w, c)

        # Reverse cyclic shift
        x = tf.roll(shifted_x, shift=(shift_size, shift_size), axis=(1, 2))
        x = tf.reshape(x, shape=[-1, h * w, c])

        # Residual connection
        x = self.drop_path(x, training=training)
        x = x + shortcut

        # MLP
        shortcut = x
        x = self.norm2(x, training=training)
        x = self.mlp(x, training=training)
        x = self.drop_path(x, training=training)
        x = x + shortcut

        return x


class PatchMerging(tf.keras.layers.Layer):
    def __init__(
        self,
        cfg: SwinTransformerConfig,
        input_size: Tuple[int, int],
        embed_dim: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.input_size = input_size
        self.norm_layer = norm_layer_factory(cfg.norm_layer)

        self.reduction = tf.keras.layers.Dense(
            units=2 * embed_dim, use_bias=False, name="reduction"
        )
        self.norm = self.norm_layer(name="norm")

    def call(self, x, training=False):
        h, w = self.input_size
        b, l, c = tf.unstack(tf.shape(x))

        x = tf.reshape(x, shape=(-1, h, w, c))
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = tf.concat((x0, x1, x2, x3), axis=-1)
        x = tf.reshape(x, shape=(-1, (h // 2) * (w // 2), 4 * c))

        x = self.norm(x, training=training)
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
        drop_path_rate: np.ndarray,
        downsample: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cfg = cfg

        self.blocks = [
            SwinTransformerBlock(
                cfg=cfg,
                input_size=input_size,
                embed_dim=embed_dim,
                nb_heads=nb_heads,
                drop_path_rate=drop_path_rate[idx],
                shift_size=0 if idx % 2 == 0 else cfg.window_size // 2,
                name=f"blocks/{idx}",
            )
            for idx in range(nb_blocks)
        ]
        if downsample:
            self.downsample = PatchMerging(
                cfg=cfg, input_size=input_size, embed_dim=embed_dim, name="downsample"
            )
        else:
            self.downsample = tf.keras.layers.Activation("linear")

    def call(self, x, training=False, return_features=False):
        features = {}
        for j, block in enumerate(self.blocks):
            x = block(x, training=training)
            features[f"block_{j}"] = x

        x = self.downsample(x, training=training)
        features["features"] = x
        return (x, features) if return_features else x


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
        self.drop = tf.keras.layers.Dropout(cfg.drop_rate)

        # Stochastic depth
        dpr = np.linspace(0.0, cfg.drop_path_rate, sum(cfg.nb_blocks))

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
                        cfg.patch_resolution[1] // (2 ** idx),
                    ),
                    embed_dim=int(cfg.embed_dim * 2 ** idx),
                    nb_blocks=cfg.nb_blocks[idx],
                    nb_heads=cfg.nb_heads[idx],
                    drop_path_rate=dpr[block_idx_from:block_idx_to],
                    downsample=idx < nb_stages - 1,  # Don't downsample the last stage
                    name=f"layers/{idx}",
                )
            )
        self.norm = self.norm_layer(name="norm")
        self.pool = tf.keras.layers.GlobalAveragePooling1D()
        self.head = (
            tf.keras.layers.Dense(units=cfg.nb_classes, name="head")
            if cfg.nb_classes > 0
            else tf.keras.layers.Activation("linear")  # Identity layer
        )

    @property
    def dummy_inputs(self) -> tf.Tensor:
        return tf.zeros((1, *self.cfg.input_size, self.cfg.in_channels))

    @property
    def feature_names(self) -> List[str]:
        names = ["patch_embedding"]
        k = 0
        for j in range(len(self.cfg.nb_blocks)):
            for _ in range(self.cfg.nb_blocks[j]):
                names.append(f"block_{k}")
                k += 1
            names.append(f"stage_{j}")
        names += ["features_all", "features", "logits"]
        return names

    @property
    def keys_to_ignore_on_load_missing(self) -> List[str]:
        names = []
        for j in range(len(self.cfg.nb_blocks)):
            for k in range(self.cfg.nb_blocks[j]):
                names.append(f"layers/{j}/blocks/{k}/attn_mask")
                names.append(f"layers/{j}/blocks/{k}/attn/relative_position_index")
        return names

    def forward_features(self, x, training=False, return_features=False):
        features = {}
        x = self.patch_embed(x, training=training)
        x = self.drop(x, training=training)
        features["patch_embedding"] = x

        block_idx = 0
        for stage_idx, stage in enumerate(self.stages):
            x = stage(x, training=training, return_features=return_features)
            if return_features:
                x, stage_features = x
                for k in range(self.cfg.nb_blocks[stage_idx]):
                    features[f"block_{block_idx}"] = stage_features[f"block_{k}"]
                    block_idx += 1
                features[f"stage_{stage_idx}"] = stage_features["features"]

        x = self.norm(x, training=training)
        features["features_all"] = x
        x = self.pool(x)
        features["features"] = x
        return (x, features) if return_features else x

    def call(self, x, training=False, return_features=False):
        features = {}
        x = self.forward_features(x, training, return_features)
        if return_features:
            x, features = x
        x = self.head(x)
        features["logits"] = x
        return (x, features) if return_features else x


@register_model
def swin_tiny_patch4_window7_224():
    """Swin-T @ 224x224, trained ImageNet-1k"""
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
    """Swin-S @ 224x224, trained ImageNet-1k"""
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
    """Swin-B @ 384x384, pretrained ImageNet-22k, fine tune 1k"""
    cfg = SwinTransformerConfig(
        name="swin_base_patch4_window12_384",
        url="",
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
def swin_base_patch4_window7_224():
    """Swin-B @ 224x224, pretrained ImageNet-22k, fine tune 1k"""
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
    """Swin-B @ 384x384, trained ImageNet-22k"""
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
    """Swin-B @ 224x224, trained ImageNet-22k"""
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
    """Swin-L @ 384x384, pretrained ImageNet-22k, fine tune 1k"""
    cfg = SwinTransformerConfig(
        name="swin_large_patch4_window12_384",
        url="",
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
def swin_large_patch4_window7_224():
    """Swin-L @ 224x224, pretrained ImageNet-22k, fine tune 1k"""
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
    """Swin-L @ 384x384, trained ImageNet-22k"""
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
    """Swin-L @ 224x224, trained ImageNet-22k"""
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
