"""
TensorFlow implementation of ConvMixer

Based on timm/models/convmixer.py by Ross Wightman.

Copyright 2021 Martins Bruveris
"""
from dataclasses import dataclass
from typing import List, Tuple

import tensorflow as tf

from tfimm.layers import act_layer_factory, norm_layer_factory
from tfimm.models import ModelConfig, keras_serializable, register_model
from tfimm.utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# model_registry will add each entrypoint fn to this
__all__ = ["ConvMixer", "ConvMixerConfig"]


@dataclass
class ConvMixerConfig(ModelConfig):
    nb_classes: int = 1000
    in_channels: int = 3
    input_size: Tuple[int, int] = (224, 224)
    patch_size: Tuple[int, int] = (7, 7)
    embed_dim: int = 768
    depth: int = 32
    kernel_size: int = 9
    norm_layer: str = "batch_norm"
    act_layer: str = "gelu"
    # Parameters for inference
    crop_pct: float = 0.96
    interpolation: str = "bicubic"
    mean: Tuple[float, float, float] = IMAGENET_DEFAULT_MEAN
    std: Tuple[float, float, float] = IMAGENET_DEFAULT_STD
    first_conv: str = "stem/0"
    classifier: str = "head"


class Block(tf.keras.layers.Layer):
    def __init__(self, cfg: ConvMixerConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        act_layer = act_layer_factory(cfg.act_layer)
        norm_layer = norm_layer_factory(cfg.norm_layer)

        self.conv1 = tf.keras.layers.DepthwiseConv2D(
            kernel_size=cfg.kernel_size,
            padding="same",
            name="0/fn/0",
        )
        self.act1 = act_layer()
        self.bn1 = norm_layer(name="0/fn/2")

        self.conv2 = tf.keras.layers.Conv2D(
            filters=cfg.embed_dim,
            kernel_size=1,
            name="1",
        )
        self.act2 = act_layer()
        self.bn2 = norm_layer(name="3")

    def call(self, x, training=False):
        x_residual = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.bn1(x, training=training)
        x = x + x_residual

        x = self.conv2(x)
        x = self.act2(x)
        x = self.bn2(x, training=training)
        return x


@keras_serializable
class ConvMixer(tf.keras.Model):
    cfg_class = ConvMixerConfig

    def __init__(self, cfg: ConvMixerConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nb_features = cfg.embed_dim  # For consistency with other models
        self.act_layer = act_layer_factory(cfg.act_layer)
        self.norm_layer = norm_layer_factory(cfg.norm_layer)
        self.cfg = cfg

        conv1 = tf.keras.layers.Conv2D(
            filters=cfg.embed_dim,
            kernel_size=cfg.patch_size,
            strides=cfg.patch_size,
            name=f"{self.name}/stem/0",
        )
        act1 = self.act_layer()
        bn1 = self.norm_layer(name=f"{self.name}/stem/2")
        self.stem = tf.keras.Sequential([conv1, act1, bn1])

        self.blocks = [Block(cfg, name=f"blocks/{j}") for j in range(cfg.depth)]
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()

        self.head = (
            tf.keras.layers.Dense(units=cfg.nb_classes, name="head")
            if cfg.nb_classes > 0
            else tf.keras.layers.Activation("linear")
        )

    @property
    def dummy_inputs(self) -> tf.Tensor:
        return tf.zeros((1, *self.cfg.input_size, self.cfg.in_channels))

    @property
    def feature_names(self) -> List[str]:
        return (
            ["stem"]
            + [f"block_{j}" for j in range(self.cfg.depth)]
            + ["features_all", "features", "logits"]
        )

    def forward_features(self, x, training=False, return_features=False):
        features = {}
        x = self.stem(x, training=training)
        features["stem"] = x
        for j, block in enumerate(self.blocks):
            # noinspection PyCallingNonCallable
            x = block(x, training=training)
            features[f"block_{j}"] = x
        features["features_all"] = x
        x = self.pool(x)
        x = self.flatten(x)
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
def convmixer_768_32():
    """
    ConvMixer-768/32.
    Source: https://github.com/tmp-iclr/convmixer
    Note: This network uses ReLU activation instead of GELU
    """
    cfg = ConvMixerConfig(
        name="convmixer_768_32",
        url="",
        patch_size=(7, 7),
        embed_dim=768,
        depth=32,
        kernel_size=7,
        act_layer="relu",
    )
    return ConvMixer, cfg


@register_model
def convmixer_1024_20_ks9_p14():
    """
    ConvMixer-1024/20.
    Source: https://github.com/tmp-iclr/convmixer
    """
    cfg = ConvMixerConfig(
        name="convmixer_1024_20_ks9_p14",
        url="",
        patch_size=(14, 14),
        embed_dim=1024,
        depth=20,
        kernel_size=9,
    )
    return ConvMixer, cfg


@register_model
def convmixer_1536_20():
    """
    ConvMixer-1536/20.
    Source: https://github.com/tmp-iclr/convmixer
    """
    cfg = ConvMixerConfig(
        name="convmixer_1536_20",
        url="",
        patch_size=(7, 7),
        embed_dim=1536,
        depth=20,
        kernel_size=9,
    )
    return ConvMixer, cfg
