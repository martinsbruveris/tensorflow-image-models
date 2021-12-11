"""
Class-Attention in Image Transformers (CaiT)

Paper: Going deeper with Image Transformers
Link: https://arxiv.org/abs/2103.17239

Based on timm/models/cait.py by Ross Wightman.
Original code and weights from https://github.com/facebookresearch/deit

Copyright 2021 Martins Bruveris
Copyright 2021 Ross Wightman
Copyright 2015-present, Facebook, Inc.
"""
from dataclasses import dataclass
from typing import List, Tuple

import tensorflow as tf

from tfimm.layers import MLP, DropPath, PatchEmbeddings, norm_layer_factory
from tfimm.models import ModelConfig, keras_serializable, register_model
from tfimm.utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# Model registry will add each entrypoint function to this
__all__ = ["CaiT", "CaiTConfig"]


@dataclass
class CaiTConfig(ModelConfig):
    nb_classes: int = 1000
    in_chans: int = 3
    input_size: Tuple[int, int] = (224, 224)
    patch_size: int = 16
    embed_dim: int = 768
    nb_blocks: int = 12
    nb_heads: int = 12
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    # Regularization
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    attn_drop_rate: float = 0.0
    # Other parameters
    norm_layer: str = "layer_norm_eps_1e-6"
    act_layer: str = "gelu"
    init_scale: float = 1e-4  # TODO: Meaning??
    # Parameters for inference
    crop_pct: float = 1.0
    interpolation: str = "bicubic"
    mean: Tuple[float, float, float] = IMAGENET_DEFAULT_MEAN
    std: Tuple[float, float, float] = IMAGENET_DEFAULT_STD
    # Weight transfer
    first_conv: str = "patch_embed/proj"
    classifier: str = "head"

    @property
    def grid_size(self) -> Tuple[int, int]:
        return (
            self.input_size[0] // self.patch_size,
            self.input_size[1] // self.patch_size,
        )

    @property
    def nb_patches(self) -> int:
        return self.grid_size[0] * self.grid_size[1]


class ClassAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        embed_dim: int,
        nb_heads: int,
        qkv_bias: bool,
        attn_drop_rate: float,
        proj_drop_rate: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.nb_heads = nb_heads
        self.scale = (embed_dim // nb_heads) ** -0.5

        self.q = tf.keras.layers.Dense(units=embed_dim, use_bias=qkv_bias, name="q")
        self.k = tf.keras.layers.Dense(units=embed_dim, use_bias=qkv_bias, name="k")
        self.v = tf.keras.layers.Dense(units=embed_dim, use_bias=qkv_bias, name="v")
        self.attn_drop = tf.keras.layers.Dropout(rate=attn_drop_rate)
        self.proj = tf.keras.layers.Dense(units=embed_dim, name="proj")
        self.proj_drop = tf.keras.layers.Dropout(rate=proj_drop_rate)

    def call(self, x, training=False):
        # B (batch size), N (sequence length), D (embedding dimension),
        # H (number of heads)
        batch_size, seq_length = tf.unstack(tf.shape(x)[:2])
        q = self.q(x[:, 0])  # (B, D)
        q = tf.expand_dims(q, axis=1)  # (B, 1, D)
        q = tf.reshape(q, (batch_size, 1, self.nb_heads, -1))  # (B, 1, H, D/H)
        q = tf.transpose(q, (0, 2, 1, 3))  # (B, H, 1, B/H)
        q = q * self.scale  # (B, H, 1, B/H)

        k = self.k(x)  # (B, N, D)
        k = tf.reshape(k, (batch_size, seq_length, self.nb_heads, -1))  # (B, N, H, D/H)
        k = tf.transpose(k, (0, 2, 1, 3))  # (B, H, N, D/H)

        v = self.v(x)  # (B, N, D)
        v = tf.reshape(v, (batch_size, seq_length, self.nb_heads, -1))  # (B, N, H, D/H)
        v = tf.transpose(v, (0, 2, 1, 3))  # (B, H, N, D/H)

        attn = tf.linalg.matmul(q, k, transpose_b=True)  # (B, H, 1, N)
        attn = tf.nn.softmax(attn, axis=-1)  # (B, H, 1, N)
        attn = self.attn_drop(attn, training=training)  # (B, H, 1, N)

        x = tf.linalg.matmul(attn, v)  # (B, H, 1, D/H)
        x = tf.transpose(x, (0, 2, 1, 3))  # (B, 1, H, D/H)
        x = tf.reshape(x, (batch_size, 1, -1))  # (B, 1, D)

        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        return x


class LayerScaleBlockClassAttention(tf.keras.layers.Layer):
    def __init__(self, cfg: CaiTConfig, drop_path_rate: float, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg

        norm_layer = norm_layer_factory(cfg.norm_layer)
        self.norm1 = norm_layer(name="norm1")
        self.attn = ClassAttention(
            embed_dim=cfg.embed_dim,
            nb_heads=cfg.nb_heads,
            qkv_bias=cfg.qkv_bias,
            attn_drop_rate=0.0,
            proj_drop_rate=0.0,
            name="attn",
        )
        self.drop_path = DropPath(drop_prob=drop_path_rate)
        self.norm2 = norm_layer(name="norm2")
        self.mlp = MLP(
            hidden_dim=int(cfg.embed_dim * cfg.mlp_ratio),
            embed_dim=cfg.embed_dim,
            drop_rate=0.0,
            act_layer=cfg.act_layer,
            name="mlp",
        )

    def build(self, input_shape):
        self.gamma_1 = self.add_weight(
            shape=(self.cfg.embed_dim,),
            initializer=tf.keras.initializers.Constant(value=self.cfg.init_scale),
            trainable=True,
            name="gamma_1",
        )
        self.gamma_2 = self.add_weight(
            shape=(self.cfg.embed_dim,),
            initializer=tf.keras.initializers.Constant(value=self.cfg.init_scale),
            trainable=True,
            name="gamma_2",
        )

    def call(self, x, training=False):
        x_cls = tf.expand_dims(x[:, 0], axis=1)

        u = self.norm1(x, training=training)
        u = self.gamma_1 * self.attn(u)
        u = self.drop_path(u, training=training)
        x_cls = x_cls + u

        shortcut = x_cls
        x_cls = self.norm2(x_cls, training=training)
        x_cls = self.mlp(x_cls, training=training)
        x_cls = self.gamma_2 * x_cls
        x_cls = self.drop_path(x_cls, training=training)
        x_cls = x_cls + shortcut

        x = tf.concat((x_cls, x[:, 1:]), axis=1)
        return x


class TalkingHeadAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        embed_dim: int,
        nb_heads: int,
        qkv_bias: bool,
        attn_drop_rate: float,
        proj_drop_rate: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.nb_heads = nb_heads
        self.scale = (embed_dim // nb_heads) ** -0.5

        self.qkv = tf.keras.layers.Dense(
            units=3 * embed_dim,
            use_bias=qkv_bias,
            name="qkv",
        )
        self.attn_drop = tf.keras.layers.Dropout(rate=attn_drop_rate)
        self.proj = tf.keras.layers.Dense(units=embed_dim, name="proj")
        self.proj_l = tf.keras.layers.Dense(units=nb_heads, name="proj_l")
        self.proj_w = tf.keras.layers.Dense(units=nb_heads, name="proj_w")
        self.proj_drop = tf.keras.layers.Dropout(rate=proj_drop_rate)

    def call(self, x, training=False):
        # B (batch size), N (sequence length), D (embedding dimension),
        # H (number of heads)
        batch_size, seq_length = tf.unstack(tf.shape(x)[:2])
        qkv = self.qkv(x)  # (B, N, 3*D)
        qkv = tf.reshape(qkv, (batch_size, seq_length, 3, self.nb_heads, -1))
        qkv = tf.transpose(qkv, (2, 0, 3, 1, 4))  # (3, B, H, N, D/H)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.scale * q

        attn = tf.linalg.matmul(q, k, transpose_b=True)  # (B, H, N, N)
        attn = tf.transpose(attn, (0, 2, 3, 1))  # (B, N, N, H)
        attn = self.proj_l(attn)  # (B, N, N, H)
        attn = tf.transpose(attn, (0, 3, 1, 2))  # (B, H, N, N)
        attn = tf.nn.softmax(attn, axis=-1)  # (B, H, N, N)
        attn = tf.transpose(attn, (0, 2, 3, 1))  # (B, N, N, H)
        attn = self.proj_w(attn)  # (B, N, N, H)
        attn = tf.transpose(attn, (0, 3, 1, 2))  # (B, H, N, N)
        attn = self.attn_drop(attn, training=training)

        x = tf.linalg.matmul(attn, v)  # (B, H, N, D/H)
        x = tf.transpose(x, (0, 2, 1, 3))  # (B, N, H, D/H)
        x = tf.reshape(x, (batch_size, seq_length, -1))  # (B, N, D)

        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        return x


class LayerScaleBlock(tf.keras.layers.Layer):
    def __init__(self, cfg: CaiTConfig, drop_path_rate: float, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg

        norm_layer = norm_layer_factory(cfg.norm_layer)
        self.norm1 = norm_layer(name="norm1")
        self.attn = TalkingHeadAttention(
            embed_dim=cfg.embed_dim,
            nb_heads=cfg.nb_heads,
            qkv_bias=cfg.qkv_bias,
            attn_drop_rate=cfg.attn_drop_rate,
            proj_drop_rate=cfg.drop_rate,
            name="attn",
        )
        self.drop_path = DropPath(drop_prob=drop_path_rate)
        self.norm2 = norm_layer(name="norm2")
        self.mlp = MLP(
            hidden_dim=int(cfg.embed_dim * cfg.mlp_ratio),
            embed_dim=cfg.embed_dim,
            drop_rate=cfg.drop_rate,
            act_layer=cfg.act_layer,
            name="mlp",
        )

    def build(self, input_shape):
        self.gamma_1 = self.add_weight(
            shape=(self.cfg.embed_dim,),
            initializer=tf.keras.initializers.Constant(value=self.cfg.init_scale),
            trainable=True,
            name="gamma_1",
        )
        self.gamma_2 = self.add_weight(
            shape=(self.cfg.embed_dim,),
            initializer=tf.keras.initializers.Constant(value=self.cfg.init_scale),
            trainable=True,
            name="gamma_2",
        )

    def call(self, x, training=False):
        shortcut = x
        x = self.norm1(x, training=training)
        # noinspection PyCallingNonCallable
        x = self.attn(x, training=training)
        x = self.gamma_1 * x
        # noinspection PyCallingNonCallable
        x = self.drop_path(x, training=training)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x, training=training)
        # noinspection PyCallingNonCallable
        x = self.mlp(x, training=training)
        x = self.gamma_2 * x
        # noinspection PyCallingNonCallable
        x = self.drop_path(x, training=training)
        x = x + shortcut
        return x


@keras_serializable
class CaiT(tf.keras.Model):
    cfg_class = CaiTConfig

    def __init__(self, cfg: CaiTConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.nb_features = cfg.embed_dim

        norm_layer = norm_layer_factory(cfg.norm_layer)

        self.patch_embed = PatchEmbeddings(
            patch_size=cfg.patch_size,
            embed_dim=cfg.embed_dim,
            norm_layer="",
            name="patch_embed",
        )
        self.pos_drop = tf.keras.layers.Dropout(rate=cfg.drop_rate)

        dpr = [cfg.drop_path_rate for _ in range(cfg.nb_blocks)]
        self.blocks = [
            LayerScaleBlock(cfg=cfg, drop_path_rate=rate, name=f"blocks/{j}")
            for j, rate in zip(range(cfg.nb_blocks), dpr)
        ]
        self.block_token_only = [
            LayerScaleBlockClassAttention(
                cfg=cfg, drop_path_rate=0.0, name=f"blocks_token_only/{j}"
            )
            for j in range(2)
        ]
        self.norm = norm_layer(name="norm")
        self.head = (
            tf.keras.layers.Dense(units=cfg.nb_classes, name="head")
            if cfg.nb_classes > 0
            else tf.keras.layers.Activation("linear")  # Identity layer
        )

    @property
    def dummy_inputs(self) -> tf.Tensor:
        return tf.zeros((1, *self.cfg.input_size, self.cfg.in_chans))

    @property
    def feature_names(self) -> List[str]:
        return (
            ["patch_embedding"]
            + [f"block_{j}" for j in range(self.cfg.nb_blocks)]
            + ["features_cls_token"]
            + [f"block_cls_token_{j}" for j in range(2)]
            + ["features_all", "features", "logits"]
        )

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            shape=(1, 1, self.cfg.embed_dim),
            initializer="zeros",
            trainable=True,
            name="cls_token",
        )
        self.pos_embed = self.add_weight(
            shape=(1, self.cfg.nb_patches, self.cfg.embed_dim),
            initializer="zeros",
            trainable=True,
            name="pos_embed",
        )

    def forward_features(self, x, training=False, return_features=False):
        features = {}
        # noinspection PyCallingNonCallable
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x, training=training)
        features["patch_embedding"] = x

        for j, block in enumerate(self.blocks):
            # noinspection PyCallingNonCallable
            x = block(x, training=training)
            features[f"block_{j}"] = x

        batch_size = tf.shape(x)[0]
        cls_token = tf.repeat(self.cls_token, repeats=batch_size, axis=0)
        x = tf.concat((cls_token, x), axis=1)
        features["features_cls_token"] = x

        for j, block in enumerate(self.block_token_only):
            # noinspection PyCallingNonCallable
            x = block(x, training=training)
            features[f"block_cls_token_{j}"] = x

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
        x = self.head(x)
        features["logits"] = x
        return (x, features) if return_features else x


@register_model
def cait_xxs24_224():
    cfg = CaiTConfig(
        name="cait_xxs24_224",
        url="",
        input_size=(224, 224),
        patch_size=16,
        embed_dim=192,
        nb_blocks=24,
        nb_heads=4,
        init_scale=1e-5,
    )
    return CaiT, cfg


@register_model
def cait_xxs24_384():
    cfg = CaiTConfig(
        name="cait_xxs24_384",
        url="",
        input_size=(384, 384),
        patch_size=16,
        embed_dim=192,
        nb_blocks=24,
        nb_heads=4,
        init_scale=1e-5,
    )
    return CaiT, cfg


@register_model
def cait_xxs36_224():
    cfg = CaiTConfig(
        name="cait_xxs36_224",
        url="",
        input_size=(224, 224),
        patch_size=16,
        embed_dim=192,
        nb_blocks=36,
        nb_heads=4,
        init_scale=1e-5,
    )
    return CaiT, cfg


@register_model
def cait_xxs36_384():
    cfg = CaiTConfig(
        name="cait_xxs36_384",
        url="",
        input_size=(384, 384),
        patch_size=16,
        embed_dim=192,
        nb_blocks=36,
        nb_heads=4,
        init_scale=1e-5,
    )
    return CaiT, cfg


@register_model
def cait_xs24_384():
    cfg = CaiTConfig(
        name="cait_xs24_384",
        url="",
        input_size=(384, 384),
        patch_size=16,
        embed_dim=288,
        nb_blocks=24,
        nb_heads=6,
        init_scale=1e-5,
    )
    return CaiT, cfg


@register_model
def cait_s24_224():
    cfg = CaiTConfig(
        name="cait_s24_224",
        url="",
        input_size=(224, 224),
        patch_size=16,
        embed_dim=384,
        nb_blocks=24,
        nb_heads=8,
        init_scale=1e-5,
    )
    return CaiT, cfg


@register_model
def cait_s24_384():
    cfg = CaiTConfig(
        name="cait_s24_384",
        url="",
        input_size=(384, 384),
        patch_size=16,
        embed_dim=384,
        nb_blocks=24,
        nb_heads=8,
        init_scale=1e-5,
    )
    return CaiT, cfg


@register_model
def cait_s36_384():
    cfg = CaiTConfig(
        name="cait_s36_384",
        url="",
        input_size=(384, 384),
        patch_size=16,
        embed_dim=384,
        nb_blocks=36,
        nb_heads=8,
        init_scale=1e-6,
    )
    return CaiT, cfg


@register_model
def cait_m36_384():
    cfg = CaiTConfig(
        name="cait_m36_384",
        url="",
        input_size=(384, 384),
        patch_size=16,
        embed_dim=768,
        nb_blocks=36,
        nb_heads=16,
        init_scale=1e-6,
    )
    return CaiT, cfg


@register_model
def cait_m48_448():
    cfg = CaiTConfig(
        name="cait_m48_448",
        url="",
        input_size=(448, 448),
        patch_size=16,
        embed_dim=768,
        nb_blocks=48,
        nb_heads=16,
        init_scale=1e-6,
    )
    return CaiT, cfg
