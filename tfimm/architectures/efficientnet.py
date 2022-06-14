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

* MobileNet-V2 models. These models correspond to the ``mobilenetv2_...`` models in
  timm.

  * ``mobilenet_v2_{050, 100, 140}``. These are MobileNet-V2 models with channel
    multiplier set to 0.5, 1.0 and 1.4 respectively.
  * ``mobilenet_v2_{110d, 120d}``. These are MobileNet-V2 models with (channel, depth)
    multipliers set to (1.1, 1.2) and (1.2, 1.4) respectively.

* Original EfficientNet models. These models correspond to the models ``tf_...`` in
  timm.

  * ``efficientnet_{b0, b1, b2, b3, b4, b5, b6, b7, b8}``

* EfficientNet AdvProp models, trained with adversarial examples. These models
  correspond to the ``tf_...`` models in timm.

  * ``efficientnet_{b0, ..., b8}_ap``

* EfficientNet NoisyStudent models, trained via semi-supervised learning. These models
  correspond to the ``tf_...`` models in timm.

  * ``efficientnet_{b0, ..., b7}_ns``
  * ``efficientnet_l2_ns_475``
  * ``efficientnet_l2``

* PyTorch versions of the EfficientNet models. These models use symmetric padding rather
  than "same" padding that is default in TF. They correspond to the ``efficientnet_...``
  models in timm.

  * ``pt_efficientnet_{b0, ..., b4}``

* EfficientNet-EdgeTPU models, optimized for inference on Google's Edge TPU hardware.
  These models correspond to the ``tf_...`` models in timm.

  * ``efficientnet_es``
  * ``efficientnet_em``
  * ``efficientnet_el``

* EfficientNet-Lite models, optimized for inference on mobile devices, CPUs and GPUs.
  These models correspond to the ``tf_...`` models in timm.

  * ``efficientnet_lite0``
  * ``efficientnet_lite1``
  * ``efficientnet_lite2``
  * ``efficientnet_lite3``
  * ``efficientnet_lite4``

* EfficientNet-V2 models. These models correspond to the ``tf_...`` models in timm.

  * ``efficientnet_v2_b0``
  * ``efficientnet_v2_b1``
  * ``efficientnet_v2_b2``
  * ``efficientnet_v2_b3``
  * ``efficientnet_v2_s``
  * ``efficientnet_v2_m``
  * ``efficientnet_v2_l``

* EfficientNet-V2 models, pretrained on ImageNet-21k, fine-tuned on ImageNet-1k. These
  models correspond to the ``tf_...`` models in timm.

  * ``efficientnet_v2_s_in21ft1k``
  * ``efficientnet_v2_m_in21ft1k``
  * ``efficientnet_v2_l_in21ft1k``
  * ``efficientnet_v2_xl_in21ft1k``

* EfficientNet-V2 models, pretrained on ImageNet-21k. These models correspond to the
  ``tf_...`` models in timm.

  * ``efficientnet_v2_s_in21k``
  * ``efficientnet_v2_m_in21k``
  * ``efficientnet_v2_l_in21k``
  * ``efficientnet_v2_xl_in21k``
"""
# Hacked together by / Copyright 2019, Ross Wightman
# Copyright 2022 Marting Bruveris
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import List, Tuple

import tensorflow as tf

from tfimm.architectures.efficientnet_builder import (
    EfficientNetBuilder,
    decode_architecture,
    round_channels,
)
from tfimm.layers import act_layer_factory, norm_layer_factory
from tfimm.models import ModelConfig, keras_serializable, register_model
from tfimm.utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)

from .efficientnet_blocks import create_conv2d

# Model registry will add each entrypoint fn to this
__all__ = ["EfficientNetConfig", "EfficientNet"]

# TODO: Fix list_timm_models with two different model names.


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
        fix_first_last:  Fix first and last block depths when multiplier is applied.
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
    fix_first_last: bool = False
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


@keras_serializable
class EfficientNet(tf.keras.Model):
    """
    Generic EfficientNet implementation supporting depth and width scaling and flexible
    architecture definitions, including

    * EfficientNet B0-B7.

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
            channel_multiplier=cfg.channel_multiplier,
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
            fix_first_last=cfg.fix_first_last,
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


def _mobilenet_v2_cfg(
    name: str,
    timm_name: str,
    channel_multiplier: float = 1.0,
    depth_multiplier: float = 1.0,
    fix_stem_head: bool = False,
    crop_pct: float = 0.875,
    mean: Tuple[float, float, float] = IMAGENET_DEFAULT_MEAN,
    std: Tuple[float, float, float] = IMAGENET_DEFAULT_STD,
):
    """
    Creates the config for a MobileNet-v2 model.

    Ref impl: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py  # noqa: E501
    Paper: https://arxiv.org/abs/1801.04381

    Args:
        name: Model name
        timm_name: Name of model in TIMM
        channel_multiplier: Multiplier for channel scaling.
        depth_multiplier: Multiplier for depth scaling.
        fix_stem_head: Scale stem channels and number of features or not
        crop_pct: Crop percentage for ImageNet evaluation
        mean: Defines preprocessing function.
        std: Defines preprpocessing function.
    """
    round_channels_fn = partial(round_channels, multiplier=channel_multiplier)
    cfg = EfficientNetConfig(
        name=name,
        url="[timm]" + timm_name,
        stem_size=32 if fix_stem_head else round_channels_fn(32),
        architecture=(
            ("ds_r1_k3_s1_c16",),
            ("ir_r2_k3_s2_e6_c24",),
            ("ir_r3_k3_s2_e6_c32",),
            ("ir_r4_k3_s2_e6_c64",),
            ("ir_r3_k3_s1_e6_c96",),
            ("ir_r3_k3_s2_e6_c160",),
            ("ir_r1_k3_s1_e6_c320",),
        ),
        channel_multiplier=channel_multiplier,
        depth_multiplier=depth_multiplier,
        fix_first_last=fix_stem_head,
        nb_features=1280 if fix_stem_head else max(1280, round_channels_fn(1280)),
        norm_layer="batch_norm",
        act_layer="relu6",
        crop_pct=crop_pct,
        mean=mean,
        std=std,
    )
    return cfg


@register_model
def mobilenet_v2_050():
    """MobileNet-V2 with 0.50 channel multiplier"""
    cfg = _mobilenet_v2_cfg(
        name="mobilenet_v2_050",
        timm_name="mobilenetv2_050",
        channel_multiplier=0.50,
    )
    return EfficientNet, cfg


@register_model
def mobilenet_v2_100():
    """MobileNet-V2 with 1.0 channel multiplier"""
    cfg = _mobilenet_v2_cfg(
        name="mobilenet_v2_100",
        timm_name="mobilenetv2_100",
        channel_multiplier=1.0,
    )
    return EfficientNet, cfg


@register_model
def mobilenet_v2_140():
    """MobileNet-V2 with 1.4 channel multiplier"""
    cfg = _mobilenet_v2_cfg(
        name="mobilenet_v2_140",
        timm_name="mobilenetv2_140",
        channel_multiplier=1.4,
    )
    return EfficientNet, cfg


@register_model
def mobilenet_v2_110d():
    """MobileNet-V2 with 1.1 channel and 1.2 depth multiplier"""
    cfg = _mobilenet_v2_cfg(
        name="mobilenet_v2_110d",
        timm_name="mobilenetv2_110d",
        channel_multiplier=1.1,
        depth_multiplier=1.2,
        fix_stem_head=True,
    )
    return EfficientNet, cfg


@register_model
def mobilenet_v2_120d():
    """MobileNet-V2 with 1.2 channel and 1.4 depth multiplier"""
    cfg = _mobilenet_v2_cfg(
        name="mobilenet_v2_120d",
        timm_name="mobilenetv2_120d",
        channel_multiplier=1.2,
        depth_multiplier=1.4,
        fix_stem_head=True,
    )
    return EfficientNet, cfg


def _efficientnet_cfg(
    name: str,
    timm_name: str,
    variant: str,
    input_size: Tuple[int, int],
    framework: str,
    crop_pct: float,
    mean: Tuple[float, float, float] = IMAGENET_DEFAULT_MEAN,
    std: Tuple[float, float, float] = IMAGENET_DEFAULT_STD,
):
    """
    Creates the config for an EfficientNet model.

    Ref impl: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py  # noqa: E501
    Paper: https://arxiv.org/abs/1905.11946

    Args:
        name: Model name
        timm_name: Model name in TIMM
        variant: Model variant, e.g., "b0", etc.
        framework: "tf" or "pytorch" for BN params and padding
        crop_pct: Crop percentage for ImageNet evaluation
    """
    assert framework in {"tf", "pytorch"}

    param_dict = {
        "b0": (1.0, 1.0, 0.2),
        "b1": (1.0, 1.1, 0.2),
        "b2": (1.1, 1.2, 0.3),
        "b3": (1.2, 1.4, 0.3),
        "b4": (1.4, 1.8, 0.4),
        "b5": (1.6, 2.2, 0.4),
        "b6": (1.8, 2.6, 0.5),
        "b7": (2.0, 3.1, 0.5),
        "b8": (2.2, 3.6, 0.5),
        "l2": (4.3, 5.3, 0.5),
    }
    channel_multiplier, depth_multiplier, drop_rate = param_dict[variant]

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
        drop_path_rate=drop_rate,
        norm_layer="batch_norm_tf" if framework == "tf" else "batch_norm",
        act_layer="swish",
        padding="same" if framework == "tf" else "symmetric",
        crop_pct=crop_pct,
        mean=mean,
        std=std,
    )
    return cfg


@register_model
def efficientnet_b0():
    """EfficientNet-B0. Tensorflow compatible variant."""
    cfg = _efficientnet_cfg(
        name="efficientnet_b0",
        timm_name="tf_efficientnet_b0",
        variant="b0",
        input_size=(224, 224),
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
        variant="b1",
        input_size=(240, 240),
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
        variant="b2",
        input_size=(260, 260),
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
        variant="b3",
        input_size=(300, 300),
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
        variant="b4",
        input_size=(380, 380),
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
        variant="b5",
        input_size=(456, 456),
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
        variant="b6",
        input_size=(528, 528),
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
        variant="b7",
        input_size=(600, 600),
        framework="tf",
        crop_pct=0.949,
    )
    return EfficientNet, cfg


@register_model
def efficientnet_b8():
    """EfficientNet-B8. Tensorflow compatible variant."""
    cfg = _efficientnet_cfg(
        name="efficientnet_b8",
        timm_name="tf_efficientnet_b8",
        variant="b8",
        input_size=(672, 672),
        framework="tf",
        crop_pct=0.954,
    )
    return EfficientNet, cfg


@register_model
def efficientnet_b0_ap():
    """EfficientNet-B0 AdvProp. Tensorflow compatible variant."""
    cfg = _efficientnet_cfg(
        name="efficientnet_b0_ap",
        timm_name="tf_efficientnet_b0_ap",
        variant="b0",
        input_size=(224, 224),
        framework="tf",
        crop_pct=0.875,
        mean=IMAGENET_INCEPTION_MEAN,
        std=IMAGENET_INCEPTION_STD,
    )
    return EfficientNet, cfg


@register_model
def efficientnet_b1_ap():
    """EfficientNet-B1 AdvProp. Tensorflow compatible variant."""
    cfg = _efficientnet_cfg(
        name="efficientnet_b1_ap",
        timm_name="tf_efficientnet_b1_ap",
        variant="b1",
        input_size=(240, 240),
        framework="tf",
        crop_pct=0.882,
        mean=IMAGENET_INCEPTION_MEAN,
        std=IMAGENET_INCEPTION_STD,
    )
    return EfficientNet, cfg


@register_model
def efficientnet_b2_ap():
    """EfficientNet-B2 AdvProp. Tensorflow compatible variant."""
    cfg = _efficientnet_cfg(
        name="efficientnet_b2_ap",
        timm_name="tf_efficientnet_b2_ap",
        variant="b2",
        input_size=(260, 260),
        framework="tf",
        crop_pct=0.890,
        mean=IMAGENET_INCEPTION_MEAN,
        std=IMAGENET_INCEPTION_STD,
    )
    return EfficientNet, cfg


@register_model
def efficientnet_b3_ap():
    """EfficientNet-B3 AdvProp. Tensorflow compatible variant."""
    cfg = _efficientnet_cfg(
        name="efficientnet_b3_ap",
        timm_name="tf_efficientnet_b3_ap",
        variant="b3",
        input_size=(300, 300),
        framework="tf",
        crop_pct=0.904,
        mean=IMAGENET_INCEPTION_MEAN,
        std=IMAGENET_INCEPTION_STD,
    )
    return EfficientNet, cfg


@register_model
def efficientnet_b4_ap():
    """EfficientNet-B4 AdvProp. Tensorflow compatible variant."""
    cfg = _efficientnet_cfg(
        name="efficientnet_b4_ap",
        timm_name="tf_efficientnet_b4_ap",
        variant="b4",
        input_size=(380, 380),
        framework="tf",
        crop_pct=0.922,
        mean=IMAGENET_INCEPTION_MEAN,
        std=IMAGENET_INCEPTION_STD,
    )
    return EfficientNet, cfg


@register_model
def efficientnet_b5_ap():
    """EfficientNet-B5 AdvProp. Tensorflow compatible variant."""
    cfg = _efficientnet_cfg(
        name="efficientnet_b5_ap",
        timm_name="tf_efficientnet_b5_ap",
        variant="b5",
        input_size=(456, 456),
        framework="tf",
        crop_pct=0.934,
        mean=IMAGENET_INCEPTION_MEAN,
        std=IMAGENET_INCEPTION_STD,
    )
    return EfficientNet, cfg


@register_model
def efficientnet_b6_ap():
    """EfficientNet-B6 AdvProp. Tensorflow compatible variant."""
    cfg = _efficientnet_cfg(
        name="efficientnet_b6_ap",
        timm_name="tf_efficientnet_b6_ap",
        variant="b6",
        input_size=(528, 528),
        framework="tf",
        crop_pct=0.942,
        mean=IMAGENET_INCEPTION_MEAN,
        std=IMAGENET_INCEPTION_STD,
    )
    return EfficientNet, cfg


@register_model
def efficientnet_b7_ap():
    """EfficientNet-B7 AdvProp. Tensorflow compatible variant."""
    cfg = _efficientnet_cfg(
        name="efficientnet_b7_ap",
        timm_name="tf_efficientnet_b7_ap",
        variant="b7",
        input_size=(600, 600),
        framework="tf",
        crop_pct=0.949,
        mean=IMAGENET_INCEPTION_MEAN,
        std=IMAGENET_INCEPTION_STD,
    )
    return EfficientNet, cfg


@register_model
def efficientnet_b8_ap():
    """EfficientNet-B8 AdvProp. Tensorflow compatible variant."""
    cfg = _efficientnet_cfg(
        name="efficientnet_b8_ap",
        timm_name="tf_efficientnet_b8_ap",
        variant="b8",
        input_size=(672, 672),
        framework="tf",
        crop_pct=0.954,
        mean=IMAGENET_INCEPTION_MEAN,
        std=IMAGENET_INCEPTION_STD,
    )
    return EfficientNet, cfg


@register_model
def efficientnet_b0_ns():
    """EfficientNet-B0 NoisyStudent. Tensorflow compatible variant."""
    cfg = _efficientnet_cfg(
        name="efficientnet_b0_ns",
        timm_name="tf_efficientnet_b0_ns",
        variant="b0",
        input_size=(224, 224),
        framework="tf",
        crop_pct=0.875,
    )
    return EfficientNet, cfg


@register_model
def efficientnet_b1_ns():
    """EfficientNet-B1 NoisyStudent. Tensorflow compatible variant."""
    cfg = _efficientnet_cfg(
        name="efficientnet_b1_ns",
        timm_name="tf_efficientnet_b1_ns",
        variant="b1",
        input_size=(240, 240),
        framework="tf",
        crop_pct=0.882,
    )
    return EfficientNet, cfg


@register_model
def efficientnet_b2_ns():
    """EfficientNet-B2 NoisyStudent. Tensorflow compatible variant."""
    cfg = _efficientnet_cfg(
        name="efficientnet_b2_ns",
        timm_name="tf_efficientnet_b2_ns",
        variant="b2",
        input_size=(260, 260),
        framework="tf",
        crop_pct=0.890,
    )
    return EfficientNet, cfg


@register_model
def efficientnet_b3_ns():
    """EfficientNet-B3 NoisyStudent. Tensorflow compatible variant."""
    cfg = _efficientnet_cfg(
        name="efficientnet_b3_ns",
        timm_name="tf_efficientnet_b3_ns",
        variant="b3",
        input_size=(300, 300),
        framework="tf",
        crop_pct=0.904,
    )
    return EfficientNet, cfg


@register_model
def efficientnet_b4_ns():
    """EfficientNet-B4 NoisyStudent. Tensorflow compatible variant."""
    cfg = _efficientnet_cfg(
        name="efficientnet_b4_ns",
        timm_name="tf_efficientnet_b4_ns",
        variant="b4",
        input_size=(380, 380),
        framework="tf",
        crop_pct=0.922,
    )
    return EfficientNet, cfg


@register_model
def efficientnet_b5_ns():
    """EfficientNet-B5 NoisyStudent. Tensorflow compatible variant."""
    cfg = _efficientnet_cfg(
        name="efficientnet_b5_ns",
        timm_name="tf_efficientnet_b5_ns",
        variant="b5",
        input_size=(456, 456),
        framework="tf",
        crop_pct=0.934,
    )
    return EfficientNet, cfg


@register_model
def efficientnet_b6_ns():
    """EfficientNet-B6 NoisyStudent. Tensorflow compatible variant."""
    cfg = _efficientnet_cfg(
        name="efficientnet_b6_ns",
        timm_name="tf_efficientnet_b6_ns",
        variant="b6",
        input_size=(528, 528),
        framework="tf",
        crop_pct=0.942,
    )
    return EfficientNet, cfg


@register_model
def efficientnet_b7_ns():
    """EfficientNet-B7 NoisyStudent. Tensorflow compatible variant."""
    cfg = _efficientnet_cfg(
        name="efficientnet_b7_ns",
        timm_name="tf_efficientnet_b7_ns",
        variant="b7",
        input_size=(600, 600),
        framework="tf",
        crop_pct=0.949,
    )
    return EfficientNet, cfg


@register_model
def efficientnet_l2_ns_475():
    """EfficientNet-L2 NoisyStudent. Tensorflow compatible variant."""
    cfg = _efficientnet_cfg(
        name="efficientnet_l2_ns_475",
        timm_name="tf_efficientnet_l2_ns_475",
        variant="l2",
        input_size=(475, 475),
        framework="tf",
        crop_pct=0.936,
    )
    return EfficientNet, cfg


@register_model
def efficientnet_l2_ns():
    """EfficientNet-L2 NoisyStudent. Tensorflow compatible variant."""
    cfg = _efficientnet_cfg(
        name="efficientnet_l2_ns",
        timm_name="tf_efficientnet_l2_ns",
        variant="l2",
        input_size=(800, 800),
        framework="tf",
        crop_pct=0.96,
    )
    return EfficientNet, cfg


@register_model
def pt_efficientnet_b0():
    """EfficientNet-B0. Default version from TIMM."""
    cfg = _efficientnet_cfg(
        name="pt_efficientnet_b0",
        timm_name="efficientnet_b0",
        variant="b0",
        input_size=(224, 224),
        framework="pytorch",
        crop_pct=0.875,
    )
    return EfficientNet, cfg


@register_model
def pt_efficientnet_b1():
    """EfficientNet-B1. Default version from TIMM."""
    cfg = _efficientnet_cfg(
        name="pt_efficientnet_b1",
        timm_name="efficientnet_b1",
        variant="b1",
        input_size=(256, 256),
        framework="pytorch",
        crop_pct=1.0,
    )
    return EfficientNet, cfg


@register_model
def pt_efficientnet_b2():
    """EfficientNet-B2. Default version from TIMM."""
    cfg = _efficientnet_cfg(
        name="pt_efficientnet_b2",
        timm_name="efficientnet_b2",
        variant="b2",
        input_size=(256, 256),
        framework="pytorch",
        crop_pct=1.0,
    )
    return EfficientNet, cfg


@register_model
def pt_efficientnet_b3():
    """EfficientNet-B3. Default version from TIMM."""
    cfg = _efficientnet_cfg(
        name="pt_efficientnet_b3",
        timm_name="efficientnet_b3",
        variant="b3",
        input_size=(288, 288),
        framework="pytorch",
        crop_pct=1.0,
    )
    return EfficientNet, cfg


@register_model
def pt_efficientnet_b4():
    """EfficientNet-B4. Default version from TIMM."""
    cfg = _efficientnet_cfg(
        name="pt_efficientnet_b4",
        timm_name="efficientnet_b4",
        variant="b4",
        input_size=(320, 320),
        framework="pytorch",
        crop_pct=1.0,
    )
    return EfficientNet, cfg


def _efficientnet_edge_cfg(
    name: str,
    timm_name: str,
    variant: str,
    input_size: Tuple[int, int],
    framework: str,
    crop_pct: float,
    mean: Tuple[float, float, float] = IMAGENET_DEFAULT_MEAN,
    std: Tuple[float, float, float] = IMAGENET_DEFAULT_STD,
):
    """
    Creates the config for an EfficientNet model.

    Ref impl: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py  # noqa: E501
    Paper: https://arxiv.org/abs/1905.11946

    Args:
        name: Model name
        timm_name: Model name in TIMM
        variant: Model variant, e.g., "b0", etc.
        framework: "tf" or "pytorch" for BN params and padding
        crop_pct: Crop percentage for ImageNet evaluation
    """
    assert framework in {"tf", "pytorch"}

    param_dict = {
        "es": (1.0, 1.0, 0.2),
        "em": (1.0, 1.1, 0.2),
        "el": (1.2, 1.4, 0.3),
    }
    channel_multiplier, depth_multiplier, drop_rate = param_dict[variant]

    cfg = EfficientNetConfig(
        name=name,
        url="[timm]" + timm_name,
        input_size=input_size,
        stem_size=round_channels(32, multiplier=channel_multiplier),
        architecture=(
            # Note: "fc" is present to override a mismatch between stem channels and
            # in channels not present in other models
            ("er_r1_k3_s1_e4_c24_fc24_noskip",),
            ("er_r2_k3_s2_e8_c32",),
            ("er_r4_k3_s2_e8_c48",),
            ("ir_r5_k5_s2_e8_c96",),
            ("ir_r4_k5_s1_e8_c144",),
            ("ir_r2_k5_s2_e8_c192",),
        ),
        channel_multiplier=channel_multiplier,
        depth_multiplier=depth_multiplier,
        nb_features=round_channels(1280, multiplier=channel_multiplier),
        drop_rate=drop_rate,
        drop_path_rate=drop_rate,
        norm_layer="batch_norm_tf" if framework == "tf" else "batch_norm",
        act_layer="relu",
        padding="same" if framework == "tf" else "symmetric",
        crop_pct=crop_pct,
        mean=mean,
        std=std,
    )
    return cfg


@register_model
def efficientnet_es():
    """EfficientNet-Edge Small. Tensorflow compatible variant."""
    cfg = _efficientnet_edge_cfg(
        name="efficientnet_es",
        timm_name="tf_efficientnet_es",
        variant="es",
        input_size=(224, 224),
        framework="tf",
        crop_pct=0.875,
        mean=IMAGENET_INCEPTION_MEAN,
        std=IMAGENET_INCEPTION_STD,
    )
    return EfficientNet, cfg


@register_model
def efficientnet_em():
    """EfficientNet-Edge Medium. Tensorflow compatible variant."""
    cfg = _efficientnet_edge_cfg(
        name="efficientnet_em",
        timm_name="tf_efficientnet_em",
        variant="em",
        input_size=(240, 240),
        framework="tf",
        crop_pct=0.882,
        mean=IMAGENET_INCEPTION_MEAN,
        std=IMAGENET_INCEPTION_STD,
    )
    return EfficientNet, cfg


@register_model
def efficientnet_el():
    """EfficientNet-Edge Large. Tensorflow compatible variant."""
    cfg = _efficientnet_edge_cfg(
        name="efficientnet_el",
        timm_name="tf_efficientnet_el",
        variant="el",
        input_size=(300, 300),
        framework="tf",
        crop_pct=0.904,
        mean=IMAGENET_INCEPTION_MEAN,
        std=IMAGENET_INCEPTION_STD,
    )
    return EfficientNet, cfg


def _efficientnet_v2_base_cfg(
    name: str,
    timm_name: str,
    variant: str,
    input_size: Tuple[int, int],
    crop_pct: float,
):
    """
    Creates the config for an EfficientNet-V2 base model.

    Ref impl: https://github.com/google/automl/tree/master/efficientnetv2
    Paper: `EfficientNetV2: Smaller Models and Faster Training`
    Link: https://arxiv.org/abs/2104.00298
    """
    param_dict = {
        "b0": (1.0, 1.0, 0.2),
        "b1": (1.0, 1.1, 0.2),
        "b2": (1.1, 1.2, 0.3),
        "b3": (1.2, 1.4, 0.3),
    }
    channel_multiplier, depth_multiplier, drop_rate = param_dict[variant]
    round_chs_fn = partial(
        round_channels, multiplier=channel_multiplier, round_limit=0.0
    )
    cfg = EfficientNetConfig(
        name=name,
        url="[timm]" + timm_name,
        input_size=input_size,
        stem_size=round_chs_fn(32),
        architecture=(
            ("cn_r1_k3_s1_e1_c16_skip",),
            ("er_r2_k3_s2_e4_c32",),
            ("er_r2_k3_s2_e4_c48",),
            ("ir_r3_k3_s2_e4_c96_se0.25",),
            ("ir_r5_k3_s1_e6_c112_se0.25",),
            ("ir_r8_k3_s2_e6_c192_se0.25",),
        ),
        channel_multiplier=channel_multiplier,
        depth_multiplier=depth_multiplier,
        nb_features=round_chs_fn(1280),
        drop_rate=drop_rate,
        drop_path_rate=drop_rate,
        norm_layer="batch_norm_tf",
        act_layer="swish",
        padding="same",
        crop_pct=crop_pct,
    )
    return cfg


@register_model
def efficientnet_v2_b0():
    """EfficientNet-V2-B0. Tensorflow compatible variant."""
    cfg = _efficientnet_v2_base_cfg(
        name="efficientnet_v2_b0",
        timm_name="tf_efficientnetv2_b0",
        variant="b0",
        input_size=(192, 192),
        crop_pct=0.875,
    )
    return EfficientNet, cfg


@register_model
def efficientnet_v2_b1():
    """EfficientNet-V2-B1. Tensorflow compatible variant."""
    cfg = _efficientnet_v2_base_cfg(
        name="efficientnet_v2_b1",
        timm_name="tf_efficientnetv2_b1",
        variant="b1",
        input_size=(192, 192),
        crop_pct=0.882,
    )
    return EfficientNet, cfg


@register_model
def efficientnet_v2_b2():
    """EfficientNet-V2-b2. Tensorflow compatible variant."""
    cfg = _efficientnet_v2_base_cfg(
        name="efficientnet_v2_b2",
        timm_name="tf_efficientnetv2_b2",
        variant="b2",
        input_size=(208, 208),
        crop_pct=0.890,
    )
    return EfficientNet, cfg


@register_model
def efficientnet_v2_b3():
    """EfficientNet-V2-b3. Tensorflow compatible variant."""
    cfg = _efficientnet_v2_base_cfg(
        name="efficientnet_v2_b3",
        timm_name="tf_efficientnetv2_b3",
        variant="b3",
        input_size=(240, 240),
        crop_pct=0.904,
    )
    return EfficientNet, cfg


def _efficientnet_v2_s_cfg(
    name: str,
    timm_name: str,
    nb_classes: int = 1000,
):
    """
    Creates the config for an EfficientNet-V2 small model.

    Ref impl: https://github.com/google/automl/tree/master/efficientnetv2
    Paper: `EfficientNetV2: Smaller Models and Faster Training`
    Link: https://arxiv.org/abs/2104.00298
    """
    cfg = EfficientNetConfig(
        name=name,
        url="[timm]" + timm_name,
        nb_classes=nb_classes,
        input_size=(300, 300),
        stem_size=24,
        architecture=(
            ("cn_r2_k3_s1_e1_c24_skip",),
            ("er_r4_k3_s2_e4_c48",),
            ("er_r4_k3_s2_e4_c64",),
            ("ir_r6_k3_s2_e4_c128_se0.25",),
            ("ir_r9_k3_s1_e6_c160_se0.25",),
            ("ir_r15_k3_s2_e6_c256_se0.25",),
        ),
        channel_multiplier=1.0,
        depth_multiplier=1.0,
        nb_features=1280,
        drop_rate=0.3,
        drop_path_rate=0.3,
        norm_layer="batch_norm_tf",
        act_layer="swish",
        padding="same",
        crop_pct=1.0,
        mean=IMAGENET_INCEPTION_MEAN,
        std=IMAGENET_INCEPTION_STD,
    )
    return cfg


@register_model
def efficientnet_v2_s():
    """EfficientNet-V2 Small. Tensorflow compatible variant."""
    cfg = _efficientnet_v2_s_cfg(
        name="efficientnet_v2_s",
        timm_name="tf_efficientnetv2_s",
    )
    return EfficientNet, cfg


@register_model
def efficientnet_v2_s_in21ft1k():
    """
    EfficientNet-V2 Small. Pretrained on ImageNet-21k, fine-tuned on 1k. Tensorflow
    compatible variant.
    """
    cfg = _efficientnet_v2_s_cfg(
        name="efficientnet_v2_s_in21ft1k",
        timm_name="tf_efficientnetv2_s_in21ft1k",
    )
    return EfficientNet, cfg


@register_model
def efficientnet_v2_s_in21k():
    """
    EfficientNet-V2 Small. ImageNet-21k pre-trained weights. Tensorflow compatible
    variant.
    """
    cfg = _efficientnet_v2_s_cfg(
        name="efficientnet_v2_s_in21k",
        timm_name="tf_efficientnetv2_s_in21k",
        nb_classes=21843,
    )
    return EfficientNet, cfg


def _efficientnet_v2_m_cfg(
    name: str,
    timm_name: str,
    nb_classes: int = 1000,
):
    """
    Creates the config for an EfficientNet-V2 medium model.

    Ref impl: https://github.com/google/automl/tree/master/efficientnetv2
    Paper: `EfficientNetV2: Smaller Models and Faster Training`
    Link: https://arxiv.org/abs/2104.00298
    """
    cfg = EfficientNetConfig(
        name=name,
        url="[timm]" + timm_name,
        nb_classes=nb_classes,
        input_size=(384, 384),
        stem_size=24,
        architecture=(
            ("cn_r3_k3_s1_e1_c24_skip",),
            ("er_r5_k3_s2_e4_c48",),
            ("er_r5_k3_s2_e4_c80",),
            ("ir_r7_k3_s2_e4_c160_se0.25",),
            ("ir_r14_k3_s1_e6_c176_se0.25",),
            ("ir_r18_k3_s2_e6_c304_se0.25",),
            ("ir_r5_k3_s1_e6_c512_se0.25",),
        ),
        channel_multiplier=1.0,
        depth_multiplier=1.0,
        nb_features=1280,
        drop_rate=0.4,
        drop_path_rate=0.4,
        norm_layer="batch_norm_tf",
        act_layer="swish",
        padding="same",
        crop_pct=1.0,
        mean=IMAGENET_INCEPTION_MEAN,
        std=IMAGENET_INCEPTION_STD,
    )
    return cfg


@register_model
def efficientnet_v2_m():
    """EfficientNet-V2 Medium. Tensorflow compatible variant."""
    cfg = _efficientnet_v2_m_cfg(
        name="efficientnet_v2_m",
        timm_name="tf_efficientnetv2_m",
    )
    return EfficientNet, cfg


@register_model
def efficientnet_v2_m_in21ft1k():
    """
    EfficientNet-V2 Medium. Pretrained on ImageNet-21k, fine-tuned on 1k. Tensorflow
    compatible variant.
    """
    cfg = _efficientnet_v2_m_cfg(
        name="efficientnet_v2_m_in21ft1k",
        timm_name="tf_efficientnetv2_m_in21ft1k",
    )
    return EfficientNet, cfg


@register_model
def efficientnet_v2_m_in21k():
    """
    EfficientNet-V2 Medium. ImageNet-21k pre-trained weights. Tensorflow compatible
    variant.
    """
    cfg = _efficientnet_v2_m_cfg(
        name="efficientnet_v2_m_in21k",
        timm_name="tf_efficientnetv2_m_in21k",
        nb_classes=21843,
    )
    return EfficientNet, cfg


def _efficientnet_v2_l_cfg(
    name: str,
    timm_name: str,
    nb_classes: int = 1000,
):
    """
    Creates the config for an EfficientNet-V2 large model.

    Ref impl: https://github.com/google/automl/tree/master/efficientnetv2
    Paper: `EfficientNetV2: Smaller Models and Faster Training`
    Link: https://arxiv.org/abs/2104.00298
    """
    cfg = EfficientNetConfig(
        name=name,
        url="[timm]" + timm_name,
        nb_classes=nb_classes,
        input_size=(384, 384),
        stem_size=32,
        architecture=(
            ("cn_r4_k3_s1_e1_c32_skip",),
            ("er_r7_k3_s2_e4_c64",),
            ("er_r7_k3_s2_e4_c96",),
            ("ir_r10_k3_s2_e4_c192_se0.25",),
            ("ir_r19_k3_s1_e6_c224_se0.25",),
            ("ir_r25_k3_s2_e6_c384_se0.25",),
            ("ir_r7_k3_s1_e6_c640_se0.25",),
        ),
        channel_multiplier=1.0,
        depth_multiplier=1.0,
        nb_features=1280,
        drop_rate=0.5,
        drop_path_rate=0.5,
        norm_layer="batch_norm_tf",
        act_layer="swish",
        padding="same",
        crop_pct=1.0,
        mean=IMAGENET_INCEPTION_MEAN,
        std=IMAGENET_INCEPTION_STD,
    )
    return cfg


@register_model
def efficientnet_v2_l():
    """EfficientNet-V2 Large. Tensorflow compatible variant."""
    cfg = _efficientnet_v2_l_cfg(
        name="efficientnet_v2_l",
        timm_name="tf_efficientnetv2_l",
    )
    return EfficientNet, cfg


@register_model
def efficientnet_v2_l_in21ft1k():
    """
    EfficientNet-V2 Large. Pretrained on ImageNet-21k, fine-tuned on 1k. Tensorflow
    compatible variant.
    """
    cfg = _efficientnet_v2_l_cfg(
        name="efficientnet_v2_l_in21ft1k",
        timm_name="tf_efficientnetv2_l_in21ft1k",
    )
    return EfficientNet, cfg


@register_model
def efficientnet_v2_l_in21k():
    """
    EfficientNet-V2 Large. ImageNet-21k pre-trained weights. Tensorflow compatible
    variant.
    """
    cfg = _efficientnet_v2_l_cfg(
        name="efficientnet_v2_l_in21k",
        timm_name="tf_efficientnetv2_l_in21k",
        nb_classes=21843,
    )
    return EfficientNet, cfg


def _efficientnet_v2_xl_cfg(
    name: str,
    timm_name: str,
    nb_classes: int = 1000,
):
    """
    Creates the config for an EfficientNet-V2 XLarge model.

    Ref impl: https://github.com/google/automl/tree/master/efficientnetv2
    Paper: `EfficientNetV2: Smaller Models and Faster Training`
    Link: https://arxiv.org/abs/2104.00298
    """
    cfg = EfficientNetConfig(
        name=name,
        url="[timm]" + timm_name,
        nb_classes=nb_classes,
        input_size=(384, 384),
        stem_size=32,
        architecture=(
            ("cn_r4_k3_s1_e1_c32_skip",),
            ("er_r8_k3_s2_e4_c64",),
            ("er_r8_k3_s2_e4_c96",),
            ("ir_r16_k3_s2_e4_c192_se0.25",),
            ("ir_r24_k3_s1_e6_c256_se0.25",),
            ("ir_r32_k3_s2_e6_c512_se0.25",),
            ("ir_r8_k3_s1_e6_c640_se0.25",),
        ),
        channel_multiplier=1.0,
        depth_multiplier=1.0,
        nb_features=1280,
        drop_rate=0.5,
        drop_path_rate=0.5,
        norm_layer="batch_norm_tf",
        act_layer="swish",
        padding="same",
        crop_pct=1.0,
        mean=IMAGENET_INCEPTION_MEAN,
        std=IMAGENET_INCEPTION_STD,
    )
    return cfg


@register_model
def efficientnet_v2_xl_in21ft1k():
    """
    EfficientNet-V2 XLarge. Pretrained on ImageNet-21k, fine-tuned on 1k. Tensorflow
    compatible variant.
    """
    cfg = _efficientnet_v2_xl_cfg(
        name="efficientnet_v2_xl_in21ft1k",
        timm_name="tf_efficientnetv2_xl_in21ft1k",
    )
    return EfficientNet, cfg


@register_model
def efficientnet_v2_xl_in21k():
    """
    EfficientNet-V2 XLarge. ImageNet-21k pre-trained weights. Tensorflow compatible
    variant.
    """
    cfg = _efficientnet_v2_xl_cfg(
        name="efficientnet_v2_xl_in21k",
        timm_name="tf_efficientnetv2_xl_in21k",
        nb_classes=21843,
    )
    return EfficientNet, cfg


def _efficientnet_lite_cfg(
    name: str,
    timm_name: str,
    variant: str,
    crop_pct: float,
):
    """
    Creates the config for an EfficientNet model.

    Ref impl: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py  # noqa: E501
    Paper: https://arxiv.org/abs/1905.11946

    Args:
        name: Model name
        timm_name: Model name in TIMM
        variant: Model variant, e.g., "lite0", etc.
        crop_pct: Crop percentage for ImageNet evaluation
    """
    param_dict = {
        "lite0": (1.0, 1.0, 224, 0.2),
        "lite1": (1.0, 1.1, 240, 0.2),
        "lite2": (1.1, 1.2, 260, 0.3),
        "lite3": (1.2, 1.4, 280, 0.3),
        "lite4": (1.4, 1.8, 300, 0.3),
    }
    channel_multiplier, depth_multiplier, input_size, drop_rate = param_dict[variant]
    input_size = (input_size, input_size)

    cfg = EfficientNetConfig(
        name=name,
        url="[timm]" + timm_name,
        input_size=input_size,
        stem_size=32,
        architecture=(
            ("ds_r1_k3_s1_e1_c16",),
            ("ir_r2_k3_s2_e6_c24",),
            ("ir_r2_k5_s2_e6_c40",),
            ("ir_r3_k3_s2_e6_c80",),
            ("ir_r3_k5_s1_e6_c112",),
            ("ir_r4_k5_s2_e6_c192",),
            ("ir_r1_k3_s1_e6_c320",),
        ),
        channel_multiplier=channel_multiplier,
        depth_multiplier=depth_multiplier,
        fix_first_last=True,
        nb_features=1280,
        drop_rate=drop_rate,
        drop_path_rate=drop_rate,
        norm_layer="batch_norm_tf",
        act_layer="relu6",
        padding="same",
        crop_pct=crop_pct,
        mean=IMAGENET_INCEPTION_MEAN,
        std=IMAGENET_INCEPTION_STD,
    )
    return cfg


@register_model
def efficientnet_lite0():
    """EfficientNet-Lite0. Tensorflow compatible variant."""
    cfg = _efficientnet_lite_cfg(
        name="efficientnet_lite0",
        timm_name="tf_efficientnet_lite0",
        variant="lite0",
        crop_pct=0.875,
    )
    return EfficientNet, cfg


@register_model
def efficientnet_lite1():
    """EfficientNet-Lite1. Tensorflow compatible variant."""
    cfg = _efficientnet_lite_cfg(
        name="efficientnet_lite1",
        timm_name="tf_efficientnet_lite1",
        variant="lite1",
        crop_pct=0.882,
    )
    return EfficientNet, cfg


@register_model
def efficientnet_lite2():
    """EfficientNet-Lite2. Tensorflow compatible variant."""
    cfg = _efficientnet_lite_cfg(
        name="efficientnet_lite2",
        timm_name="tf_efficientnet_lite2",
        variant="lite2",
        crop_pct=0.890,
    )
    return EfficientNet, cfg


@register_model
def efficientnet_lite3():
    """EfficientNet-Lite3. Tensorflow compatible variant."""
    cfg = _efficientnet_lite_cfg(
        name="efficientnet_lite3",
        timm_name="tf_efficientnet_lite3",
        variant="lite3",
        crop_pct=0.904,
    )
    return EfficientNet, cfg


@register_model
def efficientnet_lite4():
    """EfficientNet-Lite4. Tensorflow compatible variant."""
    cfg = _efficientnet_lite_cfg(
        name="efficientnet_lite4",
        timm_name="tf_efficientnet_lite4",
        variant="lite4",
        crop_pct=0.920,
    )
    return EfficientNet, cfg
