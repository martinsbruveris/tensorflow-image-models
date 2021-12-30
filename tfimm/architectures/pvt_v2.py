"""
TensorFlow implementation of the Pyramid Vision Transformer

Paper: PVTv2: Improved Baselines with Pyramid Vision Transformer
Link: https://arxiv.org/pdf/2106.13797.pdf
Official implementation (pytorch): https://github.com/whai362/PVT

Copyright: 2021 Martins Bruveris
"""
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from tfimm.layers import (
    DropPath,
    PatchEmbeddings,
    act_layer_factory,
    norm_layer_factory,
)
from tfimm.models import ModelConfig, keras_serializable, register_model
from tfimm.utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# model_registry will add each entrypoint fn to this
__all__ = ["PyramidVisionTransformerV2", "PyramidVisionTransformerV2Config"]


@dataclass
class PyramidVisionTransformerV2Config(ModelConfig):
    nb_classes: int = 1000
    in_channels: int = 3
    input_size: Tuple[int, int] = (224, 224)
    embed_dim: Tuple = (64, 128, 256, 512)
    nb_blocks: Tuple = (3, 4, 6, 3)
    nb_heads: Tuple = (1, 2, 5, 8)
    mlp_ratio: Tuple = (8.0, 8.0, 4.0, 4.0)
    sr_ratio: Tuple = (8, 4, 2, 1)
    linear_sr: bool = False
    qkv_bias: bool = True
    # Regularization
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    attn_drop_rate: float = 0.0
    # Other parameters
    norm_layer: str = "layer_norm_eps_1e-6"
    act_layer: str = "gelu"
    # Parameters for inference
    crop_pct: float = 0.9
    interpolation: str = "bicubic"
    mean: Tuple[float, float, float] = IMAGENET_DEFAULT_MEAN
    std: Tuple[float, float, float] = IMAGENET_DEFAULT_STD
    first_conv: str = "patch_embed1/proj"
    classifier: str = "head"

    """
    Args:
        nb_classes: Number of classes for classification head
        in_channels: Number of input channels
        input_size: Input image size
        patch_size: Patch size
        embed_dim: Embedding dimension per stage
        nb_blocks: Number of encoder blocks per stage
        nb_heads: Number of self-attention heads per stage
        mlp_ratio: Ratio of mlp hidden dim to embedding dim per stage
        sr_ratio: Spatial reduction ratio
        linear_sr: Apply linear spatial reduction layer
        qkv_bias: Enable bias for qkv if True
        drop_rate: Dropout rate
        drop_path_rate: Dropout rate for stochastic depth
        attn_drop_rate: Attention dropout rate
        norm_layer: Normalization layer
        act_layer: Activation function
    """


class DWConv(tf.keras.layers.Layer):
    """
    Depthwise convolution applied on patch representations.

    Reshape patches into image form, applies convolution and flattens into sequence.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dwconv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=3,
            padding="same",
            name="dwconv",
        )

    def call(self, x):
        x, grid_size = x  # We need to pass information about spatial shape of input
        # x.shape = (B, N, D)
        batch_size, _, embed_dim = tf.unstack(tf.shape(x))
        x = tf.reshape(x, (batch_size, *grid_size, embed_dim))  # (B, h, w, D)
        x = self.dwconv(x)
        x = tf.reshape(x, (batch_size, -1, embed_dim))
        return x


class MLP(tf.keras.layers.Layer):
    """
    MLP as used in Pyramid Vision Transformer.

    Differs from usual MLP layer by the addition of a depthwise convolution.
    """

    def __init__(
        self,
        hidden_dim: int,
        embed_dim: int,
        linear_sr: bool,
        drop_rate: float,
        act_layer: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        act_layer = act_layer_factory(act_layer)

        self.fc1 = tf.keras.layers.Dense(units=hidden_dim, name="fc1")
        self.relu = (
            act_layer_factory("relu")() if linear_sr else act_layer_factory("linear")()
        )
        self.dwconv = DWConv(name="dwconv")
        self.act = act_layer()
        self.fc2 = tf.keras.layers.Dense(units=embed_dim, name="fc2")
        self.drop = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, training=False):
        x, grid_size = x  # We need to pass information about spatial shape of input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dwconv((x, grid_size))
        x = self.act(x)
        x = self.drop(x, training=training)
        x = self.fc2(x)
        x = self.drop(x, training=training)
        return x


class SpatialReductionAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        embed_dim: int,
        nb_heads: int,
        sr_ratio: int,
        linear_sr: bool,
        qkv_bias: bool,
        attn_drop_rate: float,
        proj_drop_rate: float,
        norm_layer: str,
        act_layer: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if embed_dim % nb_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} should be divisible by nb_heads={nb_heads}."
            )
        self.nb_heads = nb_heads
        self.sr_ratio = sr_ratio
        self.linear_sr = linear_sr
        head_dim = embed_dim // nb_heads
        self.scale = head_dim ** -0.5
        self.norm_layer = norm_layer_factory(norm_layer)
        self.act_layer = act_layer_factory(act_layer)

        self.q = tf.keras.layers.Dense(units=embed_dim, use_bias=qkv_bias, name="q")
        self.kv = tf.keras.layers.Dense(
            units=2 * embed_dim, use_bias=qkv_bias, name="kv"
        )
        self.attn_drop = tf.keras.layers.Dropout(rate=attn_drop_rate)
        self.proj = tf.keras.layers.Dense(units=embed_dim, name="proj")
        self.proj_drop = tf.keras.layers.Dropout(rate=proj_drop_rate)

        if not linear_sr and sr_ratio > 1:
            self.sr = tf.keras.layers.Conv2D(
                filters=embed_dim,
                kernel_size=sr_ratio,
                strides=sr_ratio,
                name="sr",
            )
            self.norm = self.norm_layer(name="norm")
        elif linear_sr:
            # This should be an AdaptiveAveragePooling2D layer instead.
            self.pool = tf.keras.layers.AveragePooling2D(pool_size=7)
            self.sr = tf.keras.layers.Conv2D(
                filters=embed_dim,
                kernel_size=1,
                strides=1,
                name="sr",
            )
            self.norm = self.norm_layer(name="norm")
            self.act = self.act_layer()
        else:
            self.sr = tf.keras.layers.Activation("linear")
            self.norm = tf.keras.layers.Activation("linear")

    def call(self, x, training=False):
        x, grid_size = x
        batch_size, seq_length, embed_dim = tf.unstack(tf.shape(x))
        head_dim = embed_dim // self.nb_heads

        q = self.q(x)  # (B, N, D)
        q = tf.reshape(q, (batch_size, seq_length, self.nb_heads, -1))
        q = tf.transpose(q, (0, 2, 1, 3))  # (B, H, N, D/H)

        # Spatial reduction happens here
        x = tf.reshape(x, (batch_size, *grid_size, embed_dim))  # (B, h, w, D)
        if self.linear_sr:
            x = self.pool(x)
        x = self.sr(x)  # (B, h', w', D)
        x = tf.reshape(x, (batch_size, -1, embed_dim))  # (B, N', D)
        x = self.norm(x, training=training)
        if self.linear_sr:
            x = self.act(x)

        kv = self.kv(x)  # (B, N', 2 * D)
        kv = tf.reshape(kv, (batch_size, -1, 2, self.nb_heads, head_dim))
        kv = tf.transpose(kv, (2, 0, 3, 1, 4))  # (2, B, H, N', D/H)
        k, v = kv[0], kv[1]

        attn = self.scale * tf.linalg.matmul(q, k, transpose_b=True)  # (B, H, N, N')
        attn = tf.nn.softmax(attn, axis=-1)  # (B, H, N, N')
        attn = self.attn_drop(attn, training=training)

        x = tf.linalg.matmul(attn, v)  # (B, H, N, D/H)
        x = tf.transpose(x, (0, 2, 1, 3))  # (B, N, H, D/H)
        x = tf.reshape(x, (batch_size, seq_length, -1))  # (B, N, D)

        x = self.proj(x)
        x = self.proj_drop(x, training=training)

        return x


class Block(tf.keras.layers.Layer):
    def __init__(
        self,
        embed_dim: int,
        nb_heads: int,
        mlp_ratio: float,
        sr_ratio: int,
        linear_sr: bool,
        qkv_bias: bool,
        drop_rate: float,
        attn_drop_rate: float,
        drop_path_rate: float,
        norm_layer: str,
        act_layer: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.norm_layer = norm_layer_factory(norm_layer)

        self.norm1 = self.norm_layer(name="norm1")
        self.attn = SpatialReductionAttention(
            embed_dim=embed_dim,
            nb_heads=nb_heads,
            sr_ratio=sr_ratio,
            linear_sr=linear_sr,
            qkv_bias=qkv_bias,
            proj_drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            # We don't pass the norm_layer parameter, because that is set to
            # layer_norm_eps_1e-6 in this module
            norm_layer="layer_norm",
            act_layer=act_layer,
            name="attn",
        )
        self.drop_path = DropPath(drop_prob=drop_path_rate)
        self.norm2 = self.norm_layer(name="norm2")
        self.mlp = MLP(
            hidden_dim=int(embed_dim * mlp_ratio),
            embed_dim=embed_dim,
            linear_sr=linear_sr,
            drop_rate=drop_rate,
            act_layer=act_layer,
            name="mlp",
        )

    def call(self, x, training=False):
        x, grid_size = x

        shortcut = x
        x = self.norm1(x, training=training)
        x = self.attn((x, grid_size), training=training)
        x = self.drop_path(x, training=training)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x, training=training)
        x = self.mlp((x, grid_size), training=training)
        x = self.drop_path(x, training=training)
        x = x + shortcut
        return x


@keras_serializable
class PyramidVisionTransformerV2(tf.keras.Model):
    cfg_class = PyramidVisionTransformerV2Config

    def __init__(self, cfg: PyramidVisionTransformerV2Config, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        norm_layer = norm_layer_factory(cfg.norm_layer)

        # Stochastic depth
        dpr = np.linspace(0, cfg.drop_path_rate, sum(cfg.nb_blocks))
        block_idx = 0

        self.patch_embed = []
        self.blocks = []
        self.norms = []
        nb_stages = len(cfg.nb_blocks)
        for j in range(nb_stages):
            self.patch_embed.append(
                PatchEmbeddings(
                    patch_size=7 if j == 0 else 3,
                    embed_dim=cfg.embed_dim[j],
                    stride=4 if j == 0 else 2,
                    norm_layer="layer_norm",
                    name=f"patch_embed{j+1}",
                )
            )
            for k in range(cfg.nb_blocks[j]):
                self.blocks.append(
                    Block(
                        embed_dim=cfg.embed_dim[j],
                        nb_heads=cfg.nb_heads[j],
                        mlp_ratio=cfg.mlp_ratio[j],
                        sr_ratio=cfg.sr_ratio[j],
                        linear_sr=cfg.linear_sr,
                        qkv_bias=cfg.qkv_bias,
                        drop_rate=cfg.drop_rate,
                        attn_drop_rate=cfg.attn_drop_rate,
                        drop_path_rate=dpr[block_idx],
                        norm_layer=cfg.norm_layer,
                        act_layer=cfg.act_layer,
                        name=f"block{j+1}/{k}",
                    )
                )
                block_idx += 1
            self.norms.append(norm_layer(name=f"norm{j+1}"))

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
        names = []
        k = 0
        for j in range(len(self.cfg.nb_blocks)):
            names.append(f"patch_embedding_{j}")
            for _ in range(self.cfg.nb_blocks[j]):
                names.append(f"block_{k}")
                k += 1
            names.append(f"stage_{j}")
        names += ["features_all", "features", "logits"]
        return names

    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {
    #         "pos_embed1",
    #         "pos_embed2",
    #         "pos_embed3",
    #         "pos_embed4",
    #         "cls_token",
    #     }  # has pos_embed may be better

    def forward_features(self, x, training=False, return_features=False):
        features = {}
        batch_size = tf.shape(x)[0]

        nb_stages = len(self.cfg.nb_blocks)
        k = 0
        for j in range(nb_stages):
            x, grid_size = self.patch_embed[j](x, training=training, return_shape=True)
            features[f"patch_embedding_{j}"] = x

            for _ in range(self.cfg.nb_blocks[j]):
                x = self.blocks[k]((x, grid_size), training=training)
                features[f"block_{k}"] = x
                k += 1

            x = self.norms[j](x, training=training)
            # Reshape to image form, ready for patch embedding
            x = tf.reshape(x, (batch_size, *grid_size, -1))
            features[f"stage_{j}"] = x

        x = tf.reshape(x, (batch_size, -1, self.cfg.embed_dim[-1]))
        features["features_all"] = x
        x = tf.reduce_mean(x, axis=1)
        features["features"] = x
        return (x, features) if return_features else x

    def call(self, x, training=False, return_features=False):
        features = {}
        x = self.forward_features(x, training, return_features)
        if return_features:
            x, features = x
        x = self.head(x, training=training)
        features["logits"] = x
        return (x, features) if return_features else x


@register_model
def pvt_v2_b0():
    cfg = PyramidVisionTransformerV2Config(
        name="pvt_v2_b0",
        url="https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b0.pth",
        embed_dim=(32, 64, 160, 256),
        nb_blocks=(2, 2, 2, 2),
    )
    return PyramidVisionTransformerV2, cfg


@register_model
def pvt_v2_b1():
    cfg = PyramidVisionTransformerV2Config(
        name="pvt_v2_b1",
        url="https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b1.pth",
        embed_dim=(64, 128, 320, 512),
        nb_blocks=(2, 2, 2, 2),
    )
    return PyramidVisionTransformerV2, cfg


@register_model
def pvt_v2_b2():
    cfg = PyramidVisionTransformerV2Config(
        name="pvt_v2_b2",
        url="https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth",
        embed_dim=(64, 128, 320, 512),
        nb_blocks=(3, 4, 6, 3),
    )
    return PyramidVisionTransformerV2, cfg


@register_model
def pvt_v2_b3():
    cfg = PyramidVisionTransformerV2Config(
        name="pvt_v2_b3",
        url="https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b3.pth",
        embed_dim=(64, 128, 320, 512),
        nb_blocks=(3, 4, 18, 3),
    )
    return PyramidVisionTransformerV2, cfg


@register_model
def pvt_v2_b4():
    cfg = PyramidVisionTransformerV2Config(
        name="pvt_v2_b4",
        url="https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b4.pth",
        embed_dim=(64, 128, 320, 512),
        nb_blocks=(3, 8, 27, 3),
    )
    return PyramidVisionTransformerV2, cfg


@register_model
def pvt_v2_b5():
    cfg = PyramidVisionTransformerV2Config(
        name="pvt_v2_b5",
        url="https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b5.pth",
        embed_dim=(64, 128, 320, 512),
        nb_blocks=(3, 6, 40, 3),
        mlp_ratio=(4.0, 4.0, 4.0, 4.0),
    )
    return PyramidVisionTransformerV2, cfg


# This model is broken at the moment, since I don't have an implementation of an
# AdaptiveAveragePooling2D layer.
# @register_model
# def pvt_v2_b2_li():
#     cfg = PyramidVisionTransformerV2Config(
#         name="pvt_v2_b2_li",
#         url="https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2_li.pth",
#         embed_dim=(64, 128, 320, 512),
#         nb_blocks=(3, 4, 6, 3),
#         linear_sr=True,
#     )
#     return PyramidVisionTransformerV2, cfg
