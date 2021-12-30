"""
Pre-Activation ResNet v2 with GroupNorm and Weight Standardization.

A TensorFlow implementation of ResNetV2 adapted from the Google Big-Transfoer (BiT)
source code at https://github.com/google-research/big_transfer to match `tfimm`
interfaces. The BiT weights have been included here as pretrained models from the `timm`
versions.

Additionally, supports non pre-activation bottleneck for use as a backbone for
Vision Transfomers (ViT) and extra padding support to allow porting of official
Hybrid ResNet pretrained weights from
https://github.com/google-research/vision_transformer

Thanks to the Google team for the above two repositories and associated papers:
* Big Transfer (BiT): General Visual Representation Learning
  Link: https://arxiv.org/abs/1912.11370
* An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
  Link: https://arxiv.org/abs/2010.11929
* Knowledge distillation: A good teacher is patient and consistent
  Link: https://arxiv.org/abs/2106.05237

Copyright 2021 Martins Bruveris
Copyright 2020 Ross Wightman
Copyright 2020 Google LLC
"""
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from tfimm.layers import (
    ClassifierHead,
    DropPath,
    StdConv2D,
    act_layer_factory,
    norm_layer_factory,
)
from tfimm.models import ModelConfig, keras_serializable, register_model
from tfimm.utils import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

# model_registry will add each entrypoint fn to this
__all__ = ["ResNetV2", "ResNetV2Config", "ResNetV2Stem"]


@dataclass
class ResNetV2Config(ModelConfig):
    nb_classes: int = 1000
    in_channels: int = 3
    input_size: Tuple[int, int] = (224, 224)
    # Residual blocks
    nb_blocks: Tuple = (2, 2, 2, 2)
    nb_channels: Tuple = (256, 512, 1024, 2048)
    width_factor: int = 1
    preact: bool = True  # Preactivation structure for res blocks
    # Stem
    stem_width: int = 64
    stem_type: str = "fixed"
    # Pool
    global_pool: str = "avg"
    # Other params
    conv_padding: str = "symmetric"
    act_layer: str = "relu"
    norm_layer: str = "group_norm"
    # Regularization
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    # Parameters for inference
    pool_size: int = 7  # For test-time pooling (not implemented yet)
    crop_pct: float = 0.875
    interpolation: str = "bilinear"
    # Preprocessing
    mean: Tuple[float, float, float] = IMAGENET_INCEPTION_MEAN
    std: Tuple[float, float, float] = IMAGENET_INCEPTION_STD
    # Weight transfer
    first_conv: str = "stem/conv"
    classifier: str = "head/fc"


def make_divisible(v, divisor=8):
    min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class PreActBottleneck(tf.keras.layers.Layer):
    """
    Pre-activation (v2) bottleneck block.

    Follows the implementation of "Identity Mappings in Deep Residual Networks":
    https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua

    Except it puts the stride on 3x3 conv when available.
    """

    def __init__(
        self,
        nb_channels,
        strides: int,
        downsample: bool,
        conv_padding: str,
        act_layer: str,
        norm_layer: str,
        drop_path_rate: float,
        bottleneck_ratio=0.25,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.act_layer = act_layer_factory(act_layer)
        self.norm_layer = norm_layer_factory(norm_layer)
        mid_channels = make_divisible(nb_channels * bottleneck_ratio)

        if downsample:
            self.downsample = DownsampleConv(
                nb_channels=nb_channels,
                strides=strides,
                preact=True,
                conv_padding=conv_padding,
                norm_layer=norm_layer,
                name="downsample",
            )
        else:
            self.downsample = None

        self.norm1 = self.norm_layer(name="norm1")
        self.act1 = self.act_layer()
        self.conv1 = StdConv2D(
            filters=mid_channels,
            kernel_size=1,
            padding=conv_padding,
            use_bias=False,
            name="conv1",
        )
        self.norm2 = self.norm_layer(name="norm2")
        self.act2 = self.act_layer()
        self.conv2 = StdConv2D(
            filters=mid_channels,
            kernel_size=3,
            strides=strides,
            padding=conv_padding,
            use_bias=False,
            name="conv2",
        )
        self.norm3 = self.norm_layer(name="norm3")
        self.act3 = self.act_layer()
        self.conv3 = StdConv2D(
            filters=nb_channels,
            kernel_size=1,
            padding=conv_padding,
            use_bias=False,
            name="conv3",
        )
        self.drop_path = DropPath(drop_prob=drop_path_rate)

    def call(self, x, training=False):
        # Pre-activation
        y = self.norm1(x, training=training)
        y = self.act1(y)

        if self.downsample is not None:
            shortcut = self.downsample(y, training=training)
        else:
            shortcut = x

        # Residual branch
        x = self.conv1(y)
        x = self.norm2(x, training=training)
        x = self.act2(x)
        x = self.conv2(x)
        x = self.norm3(x, training=training)
        x = self.act3(x)
        x = self.conv3(x)
        x = self.drop_path(x, training=training)

        x = x + shortcut
        return x


class Bottleneck(tf.keras.layers.Layer):
    """
    Non Pre-activation bottleneck block, equiv to V1.5/V1b Bottleneck. Used for ViT.
    """

    def __init__(
        self,
        nb_channels,
        strides: int,
        downsample: bool,
        conv_padding: str,
        act_layer: str,
        norm_layer: str,
        drop_path_rate: float,
        bottleneck_ratio=0.25,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.act_layer = act_layer_factory(act_layer)
        self.norm_layer = norm_layer_factory(norm_layer)
        mid_channels = make_divisible(nb_channels * bottleneck_ratio)

        if downsample:
            self.downsample = DownsampleConv(
                nb_channels=nb_channels,
                strides=strides,
                preact=False,
                conv_padding=conv_padding,
                norm_layer=norm_layer,
                name="downsample",
            )
        else:
            self.downsample = None

        self.conv1 = StdConv2D(
            filters=mid_channels,
            kernel_size=1,
            padding=conv_padding,
            use_bias=False,
            name="conv1",
        )
        self.norm1 = self.norm_layer(name="norm1")
        self.act1 = self.act_layer()
        self.conv2 = StdConv2D(
            filters=mid_channels,
            kernel_size=3,
            strides=strides,
            padding=conv_padding,
            use_bias=False,
            name="conv2",
        )
        self.norm2 = self.norm_layer(name="norm2")
        self.act2 = self.act_layer()
        self.conv3 = StdConv2D(
            filters=nb_channels,
            kernel_size=1,
            padding=conv_padding,
            use_bias=False,
            name="conv3",
        )
        self.norm3 = self.norm_layer(name="norm3")
        self.drop_path = DropPath(drop_prob=drop_path_rate)
        self.act3 = self.act_layer()

    def call(self, x, training=False):
        # Shortcut branch
        if self.downsample is not None:
            shortcut = self.downsample(x, training=training)
        else:
            shortcut = x

        # Residual branch
        x = self.conv1(x)
        x = self.norm1(x, training=training)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x, training=training)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.norm3(x, training=training)
        x = self.drop_path(x, training=training)

        x = x + shortcut
        x = self.act3(x)
        return x


class DownsampleConv(tf.keras.layers.Layer):
    def __init__(
        self,
        nb_channels,
        strides,
        preact: bool,
        conv_padding: str,
        norm_layer: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.conv = StdConv2D(
            filters=nb_channels,
            kernel_size=1,
            strides=strides,
            padding=conv_padding,
            use_bias=False,
            name="conv",
        )
        if not preact:
            norm_layer = norm_layer_factory(norm_layer)
            self.norm = norm_layer(name="norm")
        else:
            self.norm = tf.keras.layers.Activation("linear")

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.norm(x, training=training)
        return x


class ResNetV2Stem(tf.keras.layers.Layer):
    def __init__(
        self,
        stem_type: str,
        stem_width: int,
        conv_padding: str,
        preact: bool,
        act_layer: str,
        norm_layer: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.preact = preact

        self.conv = StdConv2D(
            filters=stem_width,
            kernel_size=7,
            strides=2,
            padding=conv_padding,
            use_bias=False,
            name="conv",
        )

        act_layer = act_layer_factory(act_layer)
        norm_layer = norm_layer_factory(norm_layer)
        self.norm = norm_layer(name="norm") if not preact else None
        self.act = act_layer() if not preact else None

        if stem_type == "fixed":
            # 'fixed' SAME padding approximation that is used in BiT models
            self.pad = tf.keras.layers.ZeroPadding2D(padding=1)
            self.pool = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)
        elif stem_type == "same":
            self.pad = tf.keras.layers.Activation("linear")
            self.pool = tf.keras.layers.MaxPool2D(
                pool_size=3, strides=2, padding="same"
            )
        else:
            raise ValueError(f"Unknow value for stem_type: {stem_type}")

    def call(self, x, training=False):
        x = self.conv(x)
        if not self.preact:
            x = self.norm(x, training=training)
            x = self.act(x)
        x = self.pad(x)
        x = self.pool(x)
        return x


@keras_serializable
class ResNetV2(tf.keras.Model):
    """
    Implementation of Pre-activation (v2) ResNet models.
    """

    cfg_class = ResNetV2Config

    def __init__(self, cfg: ResNetV2Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg

        self.stem = ResNetV2Stem(
            stem_type=cfg.stem_type,
            stem_width=make_divisible(cfg.stem_width * cfg.width_factor),
            conv_padding=cfg.conv_padding,
            preact=cfg.preact,
            act_layer=cfg.act_layer,
            norm_layer=cfg.norm_layer,
            name="stem",
        )

        # Stochastic depth
        dpr = np.linspace(0.0, cfg.drop_path_rate, sum(cfg.nb_blocks))
        block_idx = 0

        self.blocks = []
        nb_stages = len(cfg.nb_blocks)
        block_layer = PreActBottleneck if cfg.preact else Bottleneck
        for j in range(nb_stages):
            nb_channels = make_divisible(cfg.nb_channels[j] * cfg.width_factor)
            for k in range(cfg.nb_blocks[j]):
                self.blocks.append(
                    block_layer(
                        nb_channels=nb_channels,
                        strides=2 if (j > 0) & (k == 0) else 1,
                        downsample=k == 0,
                        conv_padding=cfg.conv_padding,
                        act_layer=cfg.act_layer,
                        norm_layer=cfg.norm_layer,
                        drop_path_rate=dpr[block_idx],
                        name=f"stages/{j}/blocks/{k}",
                    )
                )
                block_idx += 1

        if cfg.preact:
            norm_layer = norm_layer_factory(cfg.norm_layer)
            act_layer = act_layer_factory(cfg.act_layer)
            self.norm = norm_layer(name="norm")
            self.act = act_layer()
        else:
            self.norm = None
            self.act = None

        self.head = ClassifierHead(
            nb_classes=cfg.nb_classes,
            pool_type=cfg.global_pool,
            drop_rate=cfg.drop_rate,
            use_conv=False,
            name="head",
        )

    @property
    def dummy_inputs(self) -> tf.Tensor:
        return tf.zeros((1, *self.cfg.input_size, self.cfg.in_channels))

    @property
    def feature_names(self) -> List[str]:
        return (
            ["stem"]
            + [f"block_{j}" for j in range(sum(self.cfg.nb_blocks))]
            + ["features", "logits"]
        )

    def forward_features(self, x, training=False, return_features=False):
        features = {}
        x = self.stem(x)
        features["stem"] = x

        for j, block in enumerate(self.blocks):
            x = block(x, training=training)
            features[f"block_{j}"] = x

        if self.cfg.preact:
            x = self.norm(x, training=training)
            x = self.act(x)
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
def resnetv2_50x1_bitm():
    cfg = ResNetV2Config(
        name="resnetv2_50x1_bitm",
        url="",
        input_size=(448, 448),
        nb_blocks=(3, 4, 6, 3),
        width_factor=1,
        pool_size=14,
        crop_pct=1.0,
    )
    return ResNetV2, cfg


@register_model
def resnetv2_50x3_bitm():
    cfg = ResNetV2Config(
        name="resnetv2_50x3_bitm",
        url="",
        input_size=(448, 448),
        nb_blocks=(3, 4, 6, 3),
        width_factor=3,
        pool_size=14,
        crop_pct=1.0,
    )
    return ResNetV2, cfg


@register_model
def resnetv2_101x1_bitm():
    cfg = ResNetV2Config(
        name="resnetv2_101x1_bitm",
        url="",
        input_size=(448, 448),
        nb_blocks=(3, 4, 23, 3),
        width_factor=1,
        pool_size=14,
        crop_pct=1.0,
    )
    return ResNetV2, cfg


@register_model
def resnetv2_101x3_bitm():
    cfg = ResNetV2Config(
        name="resnetv2_101x3_bitm",
        url="",
        input_size=(448, 448),
        nb_blocks=(3, 4, 23, 3),
        width_factor=3,
        pool_size=14,
        crop_pct=1.0,
    )
    return ResNetV2, cfg


@register_model
def resnetv2_152x2_bitm():
    cfg = ResNetV2Config(
        name="resnetv2_152x2_bitm",
        url="",
        input_size=(448, 448),
        nb_blocks=(3, 8, 36, 3),
        width_factor=2,
        pool_size=14,
        crop_pct=1.0,
    )
    return ResNetV2, cfg


@register_model
def resnetv2_152x4_bitm():
    cfg = ResNetV2Config(
        name="resnetv2_152x4_bitm",
        url="",
        input_size=(480, 480),
        nb_blocks=(3, 8, 36, 3),
        width_factor=4,
        pool_size=15,
        crop_pct=1.0,
    )
    return ResNetV2, cfg


@register_model
def resnetv2_50x1_bitm_in21k():
    cfg = ResNetV2Config(
        name="resnetv2_50x1_bitm_in21k",
        url="",
        nb_classes=21843,
        nb_blocks=(3, 4, 6, 3),
        width_factor=1,
    )
    return ResNetV2, cfg


@register_model
def resnetv2_50x3_bitm_in21k():
    cfg = ResNetV2Config(
        name="resnetv2_50x3_bitm_in21k",
        url="",
        nb_classes=21843,
        nb_blocks=(3, 4, 6, 3),
        width_factor=3,
    )
    return ResNetV2, cfg


@register_model
def resnetv2_101x1_bitm_in21k():
    cfg = ResNetV2Config(
        name="resnetv2_101x1_bitm_in21k",
        url="",
        nb_classes=21843,
        nb_blocks=(3, 4, 23, 3),
        width_factor=1,
    )
    return ResNetV2, cfg


@register_model
def resnetv2_101x3_bitm_in21k():
    cfg = ResNetV2Config(
        name="resnetv2_101x3_bitm_in21k",
        url="",
        nb_classes=21843,
        nb_blocks=(3, 4, 23, 3),
        width_factor=3,
    )
    return ResNetV2, cfg


@register_model
def resnetv2_152x2_bitm_in21k():
    cfg = ResNetV2Config(
        name="resnetv2_152x2_bitm_in21k",
        url="",
        nb_classes=21843,
        nb_blocks=(3, 8, 36, 3),
        width_factor=2,
    )
    return ResNetV2, cfg


@register_model
def resnetv2_152x4_bitm_in21k():
    cfg = ResNetV2Config(
        name="resnetv2_152x4_bitm_in21k",
        url="",
        nb_classes=21843,
        nb_blocks=(3, 8, 36, 3),
        width_factor=4,
    )
    return ResNetV2, cfg


@register_model
def resnetv2_50x1_bit_distilled():
    """
    ResNetV2-50x1-BiT Distilled
    Paper: Knowledge distillation: A good teacher is patient and consistent
    Link: https://arxiv.org/abs/2106.05237
    """
    cfg = ResNetV2Config(
        name="resnetv2_50x1_bit_distilled",
        url="",
        nb_blocks=(3, 4, 6, 3),
        width_factor=1,
        interpolation="bicubic",
    )
    return ResNetV2, cfg


@register_model
def resnetv2_152x2_bit_teacher():
    """
    ResNetV2-152x2-BiT Teacher
    Paper: Knowledge distillation: A good teacher is patient and consistent
    Link: https://arxiv.org/abs/2106.05237
    """
    cfg = ResNetV2Config(
        name="resnetv2_152x2_bit_teacher",
        url="",
        nb_blocks=(3, 8, 36, 3),
        width_factor=2,
        interpolation="bicubic",
    )
    return ResNetV2, cfg


@register_model
def resnetv2_152x2_bit_teacher_384():
    """
    ResNetV2-152x2-BiT Teacher @ 384x384
    Paper: Knowledge distillation: A good teacher is patient and consistent
    Link: https://arxiv.org/abs/2106.05237
    """
    cfg = ResNetV2Config(
        name="resnetv2_152x2_bit_teacher_384",
        url="",
        input_size=(384, 384),
        nb_blocks=(3, 8, 36, 3),
        width_factor=2,
        pool_size=12,
        crop_pct=1.0,
        interpolation="bicubic",
    )
    return ResNetV2, cfg
