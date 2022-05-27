"""
We provide an implementation and pretrained weights for the EfficientNet family of
models.

Paper: EfficientNet: Rethinking Model Scaling for CNNs.
`[arXiv:1905.11946] <https://arxiv.org/abs/1905.11946>`_.

This code and weights have been ported from the
`timm <https://github.com/rwightman/pytorch-image-models>`_ implementation. It does mean
that some model weights have undergone the journey from TF (original weights from the
Google Brain team) to PyTorch (timm library) back to TF (tfimm port of timm).

The following models are available.

* Original EfficientNet models.

  * ``efficientnet_{b0, b1, b2, b3, b4, b5, b6, b7}``. These models correspond to
    the models ``tf_efficientnet_{b0, ...}`` in timm.
"""
# Hacked together by / Copyright 2019, Ross Wightman
# Copyright 2022 Marting Bruveris
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Tuple

import tensorflow as tf

from tfimm.layers import norm_layer_factory, act_layer_factory
from tfimm.models import ModelConfig, keras_serializable, register_model
from tfimm.utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from tfimm.architectures.efficientnet_builder import (
    EfficientNetBuilder,
    round_channels,
    decode_architecture,
)
from .efficientnet_blocks import create_conv2d

# Model registry will add each entrypoint fn to this
__all__ = ["EfficientNetConfig", "EfficientNet"]


@dataclass
class EfficientNetConfig(ModelConfig):
    """
    Configuration class for EfficientNet models.

    Parameters:
        name: Name of the model.
        url: URL for pretrained weights.
        nb_classes: Number of classes for classification head.
        in_channels: Number of input image channels.
        input_size: Input image size (height, width)

        stem_size: Number of filters in first convolution.
        architecture: Tuple of tuple of strings defining the architecture of residual
            blocks. The outer tuple defines the stages while the inner tuple defines
            the blocks per stage.
        channel_multiplier: Multiplier for channel scaling. One of the three dimensions
            of EfficientNet scaling.
        depth_multiplier: Multiplier for depth scaling. One of the three dimensions of
            EfficientNet scaling.
        nb_features: Number of features before the classifier layer.

        drop_rate: Dropout rate.
        drop_path_rate: Dropout rate for stochastic depth.

        norm_layer: Normalization layer. See :func:`~norm_layer_factory` for possible
            values.
        act_layer: Activation function. See :func:`~act_layer_factory` for possible
            values.
        padding: Type of padding to use for convolutional layers. Can be one of
            "same", "valid" or "symmetric" (PyTorch-style symmetric padding).

        crop_pct: Crop percentage for ImageNet evaluation.
        interpolation: Interpolation method for ImageNet evaluation.
        mean: Defines preprocessing function. If ``x`` is an image with pixel values
            in (0, 1), the preprocessing function is ``(x - mean) / std``.
        std: Defines preprpocessing function.

        first_conv: Name of first convolutional layer. Used by
            :func:`~tfimm.create_model` to adapt the number in input channels when
            loading pretrained weights.
        classifier: Name of classifier layer. Used by :func:`~tfimm.create_model` to
            adapt the classifier when loading pretrained weights.
    """

    nb_classes: int = 1000
    in_channels: int = 3
    input_size: Tuple[int, int] = (224, 224)
    # Architecture
    stem_size: int = 32
    architecture: Tuple[Tuple[str, ...], ...] = ()
    channel_multiplier: float = 1.0
    depth_multiplier: float = 1.0
    nb_features: int = 1280
    # Regularization
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    # Other params
    norm_layer: str = "batch_norm"
    act_layer: str = "swish"
    padding: str = "symmetric"  # One of "symmetric", "same", "valid"
    # Parameters for inference
    crop_pct: float = 0.875
    interpolation: str = "bicubic"
    # Preprocessing
    mean: Tuple[float, float, float] = IMAGENET_DEFAULT_MEAN
    std: Tuple[float, float, float] = IMAGENET_DEFAULT_STD
    # Weight transfer
    first_conv: str = "conv_stem"
    classifier: str = "classifier"


# TODO: Add unit test for all model names
# TODO: Fix naming of all models
# TODO: Add unit tests for EfficientNet models
@keras_serializable
class EfficientNet(tf.keras.Model):
    """
    Generic EfficientNet implementation supporting depth and width scaling and flexible
    architecture definitions, including

    * EfficientNet B0-B8.

    Parameters:
        cfg: Configuration class for the model.
        **kwargs: Arguments are passed to ``tf.keras.Model``.
    """

    cfg_class = EfficientNetConfig

    def __init__(self, cfg: EfficientNetConfig, **kwargs):
        kwargs["name"] = kwargs.get("name", cfg.name)
        super().__init__(**kwargs)
        self.cfg = cfg
        self.norm_layer = norm_layer_factory(cfg.norm_layer)
        self.act_layer = act_layer_factory(cfg.act_layer)

        # Stem
        self.conv_stem = create_conv2d(
            filters=cfg.stem_size,
            kernel_size=3,
            strides=2,
            padding=cfg.padding,
            name="conv_stem",
        )
        self.bn1 = self.norm_layer(name="bn1")
        self.act1 = self.act_layer()

        # Middle stages (IR/ER/DS Blocks)
        builder = EfficientNetBuilder(
            output_stride=32,
            padding=cfg.padding,
            act_layer=cfg.act_layer,
            norm_layer=cfg.norm_layer,
            drop_path_rate=cfg.drop_path_rate,
        )
        architecture = decode_architecture(
            architecture=cfg.architecture,
            depth_multiplier=cfg.depth_multiplier,
            depth_truncation="ceil",
            experts_multiplier=1,
            fix_first_last=False,
            group_size=None,
        )
        self.blocks = builder(architecture)

        # Head
        self.conv_head = create_conv2d(
            filters=cfg.nb_features,
            kernel_size=1,
            padding=cfg.padding,
            name="conv_head",
        )
        self.bn2 = self.norm_layer(name="bn2")
        self.act2 = self.act_layer()

        # Pooling + Classifier
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.drop = tf.keras.layers.Dropout(rate=cfg.drop_rate)
        self.classifier = (
            tf.keras.layers.Dense(units=cfg.nb_classes, name="classifier")
            if cfg.nb_classes > 0
            else tf.keras.layers.Activation("linear")  # Identity layer
        )

    @property
    def dummy_inputs(self) -> tf.Tensor:
        """Returns a tensor of the correct shape for inference."""
        return tf.zeros((1, *self.cfg.input_size, self.cfg.in_channels))

    @property
    def feature_names(self) -> List[str]:
        """
        Names of features, returned when calling ``call`` with ``return_features=True``.
        """
        _, features = self(self.dummy_inputs, return_features=True)
        return list(features.keys())

    def forward_features(
        self, x, training: bool = False, return_features: bool = False
    ):
        """
        Forward pass through model, excluding the classifier layer. This function is
        useful if the model is used as input for downstream tasks such as object
        detection.

        Arguments:
             x: Input to model
             training: Training or inference phase?
             return_features: If ``True``, we return not only the model output, but a
                dictionary with intermediate features.

        Returns:
            If ``return_features=True``, we return a tuple ``(y, features)``, where
            ``y`` is the model output and ``features`` is a dictionary with
            intermediate features.

            If ``return_features=False``, we return only ``y``.
        """
        features = OrderedDict()
        x = self.conv_stem(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        features["stem"] = x

        for key, block in self.blocks.items():
            x = block(x, training=training)
            features[key] = x

        x = self.conv_head(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)
        features["conv_features"] = x

        return (x, features) if return_features else x

    def call(self, x, training: bool = False, return_features: bool = False):
        """
        Forward pass through the full model.

        Arguments:
             x: Input to model
             training: Training or inference phase?
             return_features: If ``True``, we return not only the model output, but a
                dictionary with intermediate features.

        Returns:
            If ``return_features=True``, we return a tuple ``(y, features)``, where
            ``y`` is the model output and ``features`` is a dictionary with
            intermediate features.

            If ``return_features=False``, we return only ``y``.
        """
        features = OrderedDict()
        x = self.forward_features(x, training, return_features)
        if return_features:
            x, features = x

        x = self.pool(x)
        x = self.flatten(x)
        features["features"] = x

        x = self.drop(x, training=training)
        x = self.classifier(x)
        features["logits"] = x
        return (x, features) if return_features else x


def _efficientnet_cfg(
    name: str,
    timm_name: str,
    input_size: Tuple[int, int],
    channel_multiplier: float,
    depth_multiplier: float,
    drop_rate: float,
    drop_path_rate: float,
    framework: str,
    crop_pct: float,
):
    """
    Creates the config for an EfficientNet model.

    Ref impl: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
    Paper: https://arxiv.org/abs/1905.11946

    EfficientNet params
    name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
    "efficientnet-b0": (1.0, 1.0, 224, 0.2),
    "efficientnet-b1": (1.0, 1.1, 240, 0.2),
    "efficientnet-b2": (1.1, 1.2, 260, 0.3),
    "efficientnet-b3": (1.2, 1.4, 300, 0.3),
    "efficientnet-b4": (1.4, 1.8, 380, 0.4),
    "efficientnet-b5": (1.6, 2.2, 456, 0.4),
    "efficientnet-b6": (1.8, 2.6, 528, 0.5),
    "efficientnet-b7": (2.0, 3.1, 600, 0.5),
    "efficientnet-b8": (2.2, 3.6, 672, 0.5),
    "efficientnet-l2": (4.3, 5.3, 800, 0.5),

    Args:
        name: Model name
        channel_multiplier: Multiplier to number of channels per layer
        depth_multiplier: Multiplier to number of repeats per stage
        framework: "tf" or "pytorch" for BN params and padding
    """
    assert framework in {"tf", "pytorch"}
    cfg = EfficientNetConfig(
        name=name,
        url="[timm]" + timm_name,
        input_size=input_size,
        stem_size=round_channels(32, multiplier=channel_multiplier),
        architecture=(
            ("ds_r1_k3_s1_e1_c16_se0.25",),
            ("ir_r2_k3_s2_e6_c24_se0.25",),
            ("ir_r2_k5_s2_e6_c40_se0.25",),
            ("ir_r3_k3_s2_e6_c80_se0.25",),
            ("ir_r3_k5_s1_e6_c112_se0.25",),
            ("ir_r4_k5_s2_e6_c192_se0.25",),
            ("ir_r1_k3_s1_e6_c320_se0.25",),
        ),
        channel_multiplier=channel_multiplier,
        depth_multiplier=depth_multiplier,
        nb_features=round_channels(1280, multiplier=channel_multiplier),
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        norm_layer="batch_norm_tf" if framework == "tf" else "batch_norm",
        act_layer="swish",
        padding="same" if framework == "tf" else "symmetric",
        crop_pct=crop_pct,
    )
    return cfg


@register_model
def efficientnet_b0():
    """EfficientNet-B0. Tensorflow compatible variant."""
    cfg = _efficientnet_cfg(
        name="efficientnet_b0",
        timm_name="tf_efficientnet_b0",
        input_size=(224, 224),
        channel_multiplier=1.0,
        depth_multiplier=1.0,
        drop_rate=0.2,
        drop_path_rate=0.2,
        framework="tf",
        crop_pct=0.875,
    )
    return EfficientNet, cfg


@register_model
def efficientnet_b1():
    """EfficientNet-B1. Tensorflow compatible variant."""
    cfg = _efficientnet_cfg(
        name="efficientnet_b1",
        timm_name="tf_efficientnet_b1",
        input_size=(240, 240),
        channel_multiplier=1.0,
        depth_multiplier=1.1,
        drop_rate=0.2,
        drop_path_rate=0.2,
        framework="tf",
        crop_pct=0.882,
    )
    return EfficientNet, cfg


@register_model
def efficientnet_b2():
    """EfficientNet-B2. Tensorflow compatible variant."""
    cfg = _efficientnet_cfg(
        name="efficientnet_b2",
        timm_name="tf_efficientnet_b2",
        input_size=(260, 260),
        channel_multiplier=1.1,
        depth_multiplier=1.2,
        drop_rate=0.3,
        drop_path_rate=0.3,
        framework="tf",
        crop_pct=0.890,
    )
    return EfficientNet, cfg


@register_model
def efficientnet_b3():
    """EfficientNet-B3. Tensorflow compatible variant."""
    cfg = _efficientnet_cfg(
        name="efficientnet_b3",
        timm_name="tf_efficientnet_b3",
        input_size=(300, 300),
        channel_multiplier=1.2,
        depth_multiplier=1.4,
        drop_rate=0.3,
        drop_path_rate=0.3,
        framework="tf",
        crop_pct=0.904,
    )
    return EfficientNet, cfg


@register_model
def efficientnet_b4():
    """EfficientNet-B4. Tensorflow compatible variant."""
    cfg = _efficientnet_cfg(
        name="efficientnet_b4",
        timm_name="tf_efficientnet_b4",
        input_size=(380, 380),
        channel_multiplier=1.4,
        depth_multiplier=1.8,
        drop_rate=0.4,
        drop_path_rate=0.4,
        framework="tf",
        crop_pct=0.922,
    )
    return EfficientNet, cfg


@register_model
def efficientnet_b5():
    """EfficientNet-B5. Tensorflow compatible variant."""
    cfg = _efficientnet_cfg(
        name="efficientnet_b5",
        timm_name="tf_efficientnet_b5",
        input_size=(456, 456),
        channel_multiplier=1.6,
        depth_multiplier=2.2,
        drop_rate=0.4,
        drop_path_rate=0.4,
        framework="tf",
        crop_pct=0.934,
    )
    return EfficientNet, cfg


@register_model
def efficientnet_b6():
    """EfficientNet-B6. Tensorflow compatible variant."""
    cfg = _efficientnet_cfg(
        name="efficientnet_b6",
        timm_name="tf_efficientnet_b6",
        input_size=(528, 528),
        channel_multiplier=1.8,
        depth_multiplier=2.6,
        drop_rate=0.5,
        drop_path_rate=0.5,
        framework="tf",
        crop_pct=0.942,
    )
    return EfficientNet, cfg


@register_model
def efficientnet_b7():
    """EfficientNet-B7. Tensorflow compatible variant."""
    cfg = _efficientnet_cfg(
        name="efficientnet_b7",
        timm_name="tf_efficientnet_b7",
        input_size=(600, 600),
        channel_multiplier=2.0,
        depth_multiplier=3.1,
        drop_rate=0.5,
        drop_path_rate=0.5,
        framework="tf",
        crop_pct=0.949,
    )
    return EfficientNet, cfg


# TODO: Test models B1-B7
