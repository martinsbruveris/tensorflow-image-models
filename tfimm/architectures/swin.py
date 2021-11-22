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

from tfimm.models import ModelConfig, keras_serializable, register_model
from tfimm.utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


# TODO: old imports
from tensorflow.keras.layers import Dense, Dropout, Conv2D, LayerNormalization, GlobalAveragePooling1D


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


class Mlp(tf.keras.layers.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., prefix='', **kwargs):
        super().__init__(**kwargs)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Dense(hidden_features, name=f'{prefix}/mlp/fc1')
        self.fc2 = Dense(out_features, name=f'{prefix}/mlp/fc2')
        self.drop = Dropout(drop)

    def call(self, x):
        x = self.fc1(x)
        x = tf.keras.activations.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.get_shape().as_list()
    x = tf.reshape(x, shape=[-1, H // window_size,
                   window_size, W // window_size, window_size, C])
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    windows = tf.reshape(x, shape=[-1, window_size, window_size, C])
    return windows


def window_reverse(windows, window_size, H, W, C):
    x = tf.reshape(windows, shape=[-1, H // window_size,
                   W // window_size, window_size, window_size, C])
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, shape=[-1, H, W, C])
    return x


class WindowAttention(tf.keras.layers.Layer):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., prefix='', **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.prefix = prefix

        self.qkv = Dense(dim * 3, use_bias=qkv_bias,
                         name=f'{self.prefix}/attn/qkv')
        self.attn_drop = Dropout(attn_drop)
        self.proj = Dense(dim, name=f'{self.prefix}/attn/proj')
        self.proj_drop = Dropout(proj_drop)

    def build(self, input_shape):
        self.relative_position_bias_table = self.add_weight(f'{self.prefix}/attn/relative_position_bias_table',
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
            relative_position_index), trainable=False, name=f'{self.prefix}/attn/relative_position_index')
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


def drop_path(inputs, drop_prob, is_training):
    if (not is_training) or (drop_prob == 0.):
        return inputs

    # Compute keep_prob
    keep_prob = 1.0 - drop_prob

    # Compute drop_connect tensor
    random_tensor = keep_prob
    shape = (tf.shape(inputs)[0],) + (1,) * \
        (len(tf.shape(inputs)) - 1)
    random_tensor += tf.random.uniform(shape, dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = tf.math.divide(inputs, keep_prob) * binary_tensor
    return output


class DropPath(tf.keras.layers.Layer):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        return drop_path(x, self.drop_prob, training)


class SwinTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path_prob=0., norm_layer=LayerNormalization, prefix=''):
        super().__init__()
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
        self.prefix = prefix

        self.norm1 = norm_layer(epsilon=1e-5, name=f'{self.prefix}/norm1')
        self.attn = WindowAttention(dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, prefix=self.prefix,
                                    name="remove")
        self.drop_path = DropPath(
            drop_path_prob if drop_path_prob > 0. else 0.)
        self.norm2 = norm_layer(epsilon=1e-5, name=f'{self.prefix}/norm2')
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       drop=drop, prefix=self.prefix, name="remove")

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
                initial_value=attn_mask, trainable=False, name=f'{self.prefix}/attn_mask')
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
    def __init__(self, input_resolution, dim, norm_layer=LayerNormalization, prefix='', **kwargs):
        super().__init__(**kwargs)
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = Dense(2 * dim, use_bias=False,
                               name=f'{prefix}/downsample/reduction')
        self.norm = norm_layer(epsilon=1e-5, name=f'{prefix}/downsample/norm')

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


class BasicLayer(tf.keras.layers.Layer):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path_prob=0., norm_layer=LayerNormalization, downsample=None, use_checkpoint=False, prefix=''):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = tf.keras.Sequential([SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                           num_heads=num_heads, window_size=window_size,
                                           shift_size=0 if (
                                               i % 2 == 0) else window_size // 2,
                                           mlp_ratio=mlp_ratio,
                                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           drop=drop, attn_drop=attn_drop,
                                           drop_path_prob=drop_path_prob[i] if isinstance(
                                               drop_path_prob, list) else drop_path_prob,
                                           norm_layer=norm_layer,
                                           prefix=f'{prefix}.blocks.{i}') for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer, prefix=prefix, name="remove")
        else:
            self.downsample = None

    def call(self, x):
        x = self.blocks(x)

        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(tf.keras.layers.Layer):
    def __init__(self, img_size=(224, 224), patch_size=(4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__(name='patch_embed')
        patches_resolution = [img_size[0] //
                              patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = Conv2D(embed_dim, kernel_size=patch_size,
                           strides=patch_size, name='proj')
        if norm_layer is not None:
            self.norm = norm_layer(epsilon=1e-5, name='norm')
        else:
            self.norm = None

    def call(self, x):
        B, H, W, C = x.get_shape().as_list()
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = tf.reshape(
            x, shape=[-1, (H // self.patch_size[0]) * (W // self.patch_size[0]), self.embed_dim])
        if self.norm is not None:
            x = self.norm(x)
        return x


@keras_serializable
class SwinTransformer(tf.keras.Model):
    cfg_class = SwinTransformerConfig

    def __init__(self, cfg: SwinTransformerConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg

        include_top = cfg.nb_classes > 0
        img_size = cfg.input_size
        patch_size = (cfg.patch_size, cfg.patch_size)
        in_chans = cfg.in_chans
        num_classes = cfg.nb_classes
        embed_dim = cfg.embed_dim
        depths = cfg.nb_blocks
        num_heads = cfg.nb_heads
        window_size = cfg.window_size
        mlp_ratio = cfg.mlp_ratio
        qkv_bias = cfg.qkv_bias
        qk_scale = None
        drop_rate = cfg.drop_rate
        attn_drop_rate = cfg.attn_drop_rate
        drop_path_rate = cfg.drop_path_rate
        norm_layer = LayerNormalization  # TODO: Replace with norm_layer_factory
        ape = cfg.ape
        patch_norm = cfg.patch_norm
        use_checkpoint = False

        self.include_top = include_top

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute postion embedding
        if self.ape:
            self.absolute_pos_embed = self.add_weight('absolute_pos_embed',
                                                      shape=(
                                                          1, num_patches, embed_dim),
                                                      initializer=tf.initializers.Zeros())

        self.pos_drop = Dropout(drop_rate)

        # stochastic depth
        dpr = [x for x in np.linspace(0., drop_path_rate, sum(depths))]

        # build layers
        self.basic_layers = tf.keras.Sequential([BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                                  patches_resolution[1] // (2 ** i_layer)),
                                                depth=depths[i_layer],
                                                num_heads=num_heads[i_layer],
                                                window_size=window_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path_prob=dpr[sum(depths[:i_layer]):sum(
                                                    depths[:i_layer + 1])],
                                                norm_layer=norm_layer,
                                                downsample=PatchMerging if (
                                                    i_layer < self.num_layers - 1) else None,
                                                use_checkpoint=use_checkpoint,
                                                prefix=f'layers.{i_layer}') for i_layer in range(self.num_layers)])
        self.norm = norm_layer(epsilon=1e-5, name='norm')
        self.avgpool = GlobalAveragePooling1D()
        if self.include_top:
            self.head = Dense(num_classes, name='head')
        else:
            self.head = None

    @property
    def dummy_inputs(self) -> tf.Tensor:
        return tf.zeros((1, *self.cfg.input_size, self.cfg.in_chans))

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        x = self.basic_layers(x)
        x = self.norm(x)
        x = self.avgpool(x)
        return x

    def call(self, x):
        x = self.forward_features(x)
        if self.include_top:
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
