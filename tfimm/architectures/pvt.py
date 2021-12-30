"""
TensorFlow implementation of the Pyramid Vision Transformer

Paper: Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without
    Convolutions.
Link: https://arxiv.org/pdf/2102.12122.pdf
Official implementation (pytorch): https://github.com/whai362/PVT

Copyright: 2021 Martins Bruveris
"""
from dataclasses import dataclass
from functools import partial
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from tfimm.layers import (
    MLP,
    DropPath,
    PatchEmbeddings,
    act_layer_factory,
    interpolate_pos_embeddings,
    norm_layer_factory,
)
from tfimm.models import ModelConfig, keras_serializable, register_model
from tfimm.utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# model_registry will add each entrypoint fn to this
__all__ = ["PyramidVisionTransformer", "PyramidVisionTransformerConfig"]


@dataclass
class PyramidVisionTransformerConfig(ModelConfig):
    nb_classes: int = 1000
    in_channels: int = 3
    input_size: Tuple[int, int] = (224, 224)
    patch_size: Tuple = (4, 2, 2, 2)
    embed_dim: Tuple = (64, 128, 256, 512)
    nb_blocks: Tuple = (3, 4, 6, 3)
    nb_heads: Tuple = (1, 2, 5, 8)
    mlp_ratio: Tuple = (8.0, 8.0, 4.0, 4.0)
    sr_ratio: Tuple = (8, 4, 2, 1)
    qkv_bias: bool = True
    # Regularization
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    # Other parameters
    norm_layer: str = "layer_norm_eps_1e-6"
    act_layer: str = "gelu"
    # Parameters for inference
    interpolate_input: bool = False
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
        qkv_bias: Enable bias for qkv if True
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        drop_path_rate: Dropout rate for stochastic depth
        norm_layer: Normalization layer
        act_layer: Activation function
    """

    @property
    def nb_tokens(self) -> Tuple:
        """Number of special tokens. Class token is added during the last stage."""
        return 0, 0, 0, 1

    @property
    def grid_size(self) -> Tuple:
        grid_sizes = []
        input_size = self.input_size
        for patch_size in self.patch_size:
            grid_sizes.append(
                (input_size[0] // patch_size, input_size[1] // patch_size)
            )
            input_size = grid_sizes[-1]
        return tuple(grid_sizes)

    @property
    def nb_patches(self) -> Tuple:
        """Number of patches without class token."""
        return tuple(grid_size[0] * grid_size[1] for grid_size in self.grid_size)

    @property
    def transform_weights(self):
        return {
            f"pos_embed{j+1}": partial(
                PyramidVisionTransformer.transform_pos_embed, stage=j
            )
            for j in range(len(self.nb_blocks))
        }


class SpatialReductionAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        embed_dim: int,
        nb_heads: int,
        sr_ratio: int,
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

        if sr_ratio > 1:
            self.sr = tf.keras.layers.Conv2D(
                filters=embed_dim,
                kernel_size=sr_ratio,
                strides=sr_ratio,
                name="sr",
            )
            self.norm = self.norm_layer(name="norm")

    def call(self, x, training=False):
        x, grid_size = x  # We need to pass information about spatial shape of input
        batch_size, seq_length, embed_dim = tf.unstack(tf.shape(x))
        head_dim = embed_dim // self.nb_heads
        height, width = tf.unstack(grid_size)

        q = self.q(x)  # (B, N, D)
        q = tf.reshape(q, (batch_size, seq_length, self.nb_heads, -1))
        q = tf.transpose(q, (0, 2, 1, 3))  # (B, H, N, D/H)

        # Spatial reduction happens here. In the last stage, after adding the class
        # token, `sr_ratio=1`. Note that after adding the class token, the resize
        # operation will fail.
        if self.sr_ratio > 1:
            x = tf.reshape(x, (batch_size, height, width, embed_dim))  # (B, h, w, D)
            x = self.sr(x)  # (B, h', w', D)
            x = tf.reshape(x, (batch_size, -1, embed_dim))  # (B, N', D)
            x = self.norm(x, training=training)

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
        x = self.mlp(x, training=training)
        x = self.drop_path(x, training=training)
        x = x + shortcut
        return x


@keras_serializable
class PyramidVisionTransformer(tf.keras.Model):
    cfg_class = PyramidVisionTransformerConfig

    def __init__(self, cfg: PyramidVisionTransformerConfig, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg

        self.patch_embed = None
        self.pos_embed = None
        self.pos_drop = None
        self.blocks = None
        self.norm = None
        self.cls_token = None
        self.head = None

    def build(self, input_shape):
        norm_layer = norm_layer_factory(self.cfg.norm_layer)

        # Stochastic depth
        dpr = np.linspace(0, self.cfg.drop_path_rate, sum(self.cfg.nb_blocks))
        block_idx = 0

        self.patch_embed = []
        self.pos_embed = []
        self.pos_drop = []
        self.blocks = []
        nb_stages = len(self.cfg.nb_blocks)
        for j in range(nb_stages):
            self.patch_embed.append(
                PatchEmbeddings(
                    patch_size=self.cfg.patch_size[j],
                    embed_dim=self.cfg.embed_dim[j],
                    norm_layer="layer_norm",
                    name=f"patch_embed{j+1}",
                )
            )
            nb_tokens = self.cfg.nb_patches[j] + self.cfg.nb_tokens[j]
            self.pos_embed.append(
                self.add_weight(
                    shape=(1, nb_tokens, self.cfg.embed_dim[j]),
                    initializer="zeros",
                    trainable=True,
                    name=f"pos_embed{j+1}",
                )
            )
            self.pos_drop.append(tf.keras.layers.Dropout(rate=self.cfg.drop_rate))
            for k in range(self.cfg.nb_blocks[j]):
                self.blocks.append(
                    Block(
                        embed_dim=self.cfg.embed_dim[j],
                        nb_heads=self.cfg.nb_heads[j],
                        mlp_ratio=self.cfg.mlp_ratio[j],
                        sr_ratio=self.cfg.sr_ratio[j],
                        qkv_bias=self.cfg.qkv_bias,
                        drop_rate=self.cfg.drop_rate,
                        attn_drop_rate=self.cfg.attn_drop_rate,
                        drop_path_rate=dpr[block_idx],
                        norm_layer=self.cfg.norm_layer,
                        act_layer=self.cfg.act_layer,
                        name=f"block{j+1}/{k}",
                    )
                )
                block_idx += 1
        self.norm = norm_layer(name="norm")
        self.cls_token = self.add_weight(
            shape=(1, 1, self.cfg.embed_dim[-1]),
            initializer="zeros",
            trainable=True,
            name="cls_token",
        )
        self.head = (
            tf.keras.layers.Dense(units=self.cfg.nb_classes, name="head")
            if self.cfg.nb_classes > 0
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
            names += [f"patch_embedding_{j}", f"pos_embedding_{j}"]
            for _ in range(self.cfg.nb_blocks[j]):
                names.append(f"block_{k}")
                k += 1
            names.append(f"stage_{j}")
        names += ["features_all", "features", "logits"]
        return names

    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     # return {'pos_embed', 'cls_token'} # has pos_embed may be better
    #     return {'cls_token'}

    def transform_pos_embed(
        self, target_cfg: PyramidVisionTransformerConfig, stage: int
    ):
        return interpolate_pos_embeddings(
            pos_embed=self.pos_embed[stage],
            src_grid_size=self.cfg.grid_size[stage],
            tgt_grid_size=target_cfg.grid_size[stage],
            nb_tokens=self.cfg.nb_tokens[stage],
        )

    def forward_features(self, x, training=False, return_features=False):
        features = {}
        batch_size = tf.shape(x)[0]

        nb_stages = len(self.cfg.nb_blocks)
        k = 0
        for j in range(nb_stages):
            x, grid_size = self.patch_embed[j](x, training=training, return_shape=True)
            features[f"patch_embedding_{j}"] = x

            if j == nb_stages - 1:
                cls_token = tf.repeat(self.cls_token, repeats=batch_size, axis=0)
                x = tf.concat((cls_token, x), axis=1)
            if not self.cfg.interpolate_input:
                x = x + self.pos_embed[j]
            else:
                pos_embed = interpolate_pos_embeddings(
                    self.pos_embed[j],
                    src_grid_size=self.cfg.grid_size[j],
                    tgt_grid_size=grid_size,
                    nb_tokens=self.cfg.nb_tokens[j],
                )
                x = x + pos_embed
            x = self.pos_drop[j](x, training=training)
            features[f"pos_embedding_{j}"] = x

            for _ in range(self.cfg.nb_blocks[j]):
                x = self.blocks[k]((x, grid_size), training=training)
                features[f"block_{k}"] = x
                k += 1

            if j != nb_stages - 1:
                # Reshape to image form, ready for patch embedding
                x = tf.reshape(x, (batch_size, *grid_size, -1))
            features[f"stage_{j}"] = x

        x = self.norm(x, training=training)
        features["features_all"] = x
        x = x[:, 0]
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
def pvt_tiny():
    cfg = PyramidVisionTransformerConfig(
        name="pvt_tiny",
        url="https://github.com/whai362/PVT/releases/download/v2/pvt_tiny.pth",
        embed_dim=(64, 128, 320, 512),
        nb_blocks=(2, 2, 2, 2),
    )
    return PyramidVisionTransformer, cfg


@register_model
def pvt_small():
    cfg = PyramidVisionTransformerConfig(
        name="pvt_small",
        url="https://github.com/whai362/PVT/releases/download/v2/pvt_small.pth",
        embed_dim=(64, 128, 320, 512),
        nb_blocks=(3, 4, 6, 3),
    )
    return PyramidVisionTransformer, cfg


@register_model
def pvt_medium():
    cfg = PyramidVisionTransformerConfig(
        name="pvt_medium",
        url="https://github.com/whai362/PVT/releases/download/v2/pvt_medium.pth",
        embed_dim=(64, 128, 320, 512),
        nb_blocks=(3, 4, 18, 3),
    )
    return PyramidVisionTransformer, cfg


@register_model
def pvt_large():
    cfg = PyramidVisionTransformerConfig(
        name="pvt_large",
        url="https://github.com/whai362/PVT/releases/download/v2/pvt_large.pth",
        embed_dim=(64, 128, 320, 512),
        nb_blocks=(3, 8, 27, 3),
    )
    return PyramidVisionTransformer, cfg
