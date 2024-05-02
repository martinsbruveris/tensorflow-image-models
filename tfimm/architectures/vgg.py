"""
We provide an implementation and pretrained weights for the VGG models.

Paper: Very Deep Convolutional Networks For Large-Scale Image Recognition.
`[arXiv:1409.1556] <https://arxiv.org/abs/1409.1556>`_.

This code has been ported from the
`timm <https://github.com/rwightman/pytorch-image-models>`_ implementation.
"""
# Copyright 2023 Marting Bruveris
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Tuple

import tensorflow as tf
import tf_keras

from tfimm.layers import ClassifierHead, act_layer_factory, norm_layer_factory
from tfimm.models import ModelConfig, keras_serializable, register_model
from tfimm.utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# Model registry will add each entrypoint fn to this
__all__ = ["VGGConfig", "VGG"]


@dataclass
class VGGConfig(ModelConfig):
    """
    Configuration class for VGG models.

    Parameters:
        name: Name of the model.
        url: URL for pretrained weights.
        nb_classes: Number of classes for classification head.
        in_channels: Number of input image channels.
        input_size: Input image size (height, width)

        layers: List with number of filters for conv layers and "M" for pooling layers.
        nb_features: Number of features in pre-classification head.
        mlp_ratio: Ratio for expanding nb_features in pre-classification head.
        global_pool: Global pooling layers.

        drop_rate: Dropout rate.

        norm_layer: Normalization layer. See :func:`~norm_layer_factory` for possible
            values.
        act_layer: Activation function. See :func:`~act_layer_factory` for possible
            values.

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

    layers: Tuple = ()
    nb_features: int = 4096
    mlp_ratio: float = 1.0
    global_pool: str = "avg"
    # Regularization
    drop_rate: float = 0.0
    # Other parameters
    norm_layer: str = ""
    act_layer: str = "relu"
    # Parameters for inference
    crop_pct: float = 0.875
    interpolation: str = "bilinear"
    # Preprocessing
    mean: Tuple[float, float, float] = IMAGENET_DEFAULT_MEAN
    std: Tuple[float, float, float] = IMAGENET_DEFAULT_STD
    # Weight transfer
    first_conv: str = "features/0"
    classifier: str = "head/fc"


class ConvMlp(tf_keras.layers.Layer):
    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        mlp_ratio: float,
        drop_rate: float,
        act_layer: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.act_layer = act_layer

        act_layer = act_layer_factory(self.act_layer)

        hidden_dim = int(out_channels * mlp_ratio)
        self.fc1 = tf_keras.layers.Conv2D(
            filters=hidden_dim, kernel_size=kernel_size, use_bias=True, name="fc1"
        )
        self.act1 = act_layer(name="act1")
        self.drop = tf_keras.layers.Dropout(rate=self.drop_rate, name="drop")
        self.fc2 = tf_keras.layers.Conv2D(
            filters=out_channels, kernel_size=1, use_bias=True, name="fc2"
        )
        self.act2 = act_layer(name="act2")

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x, training=training)
        x = self.fc2(x)
        x = self.act2(x)
        return x


@keras_serializable
class VGG(tf_keras.Model):
    """
    Class implementing a VGG network.

    Paper: Very Deep Convolutional Networks For Large-Scale Image Recognition.
    `[arXiv:1409.1556] <https://arxiv.org/abs/1409.1556>`_.

    Parameters:
        cfg: Configuration class for the model.
        **kwargs: Arguments are passed to ``tf_keras.Model``.
    """

    cfg_class = VGGConfig

    def __init__(self, cfg: VGGConfig, **kwargs):
        kwargs["name"] = kwargs.get("name", cfg.name)
        super().__init__(**kwargs)
        self.cfg = cfg

        norm_layer = norm_layer_factory(cfg.norm_layer)
        act_layer = act_layer_factory(cfg.act_layer)

        self.blocks = []  # We can't call it layers, since it conflicts with keras.
        self.block_names = []
        idx = 0
        layer_idx = 0
        for v in self.cfg.layers:
            if v == "M":
                pool = tf_keras.layers.MaxPool2D(
                    pool_size=2, strides=2, name=f"features/{idx}"
                )
                self.blocks.append(pool)
                self.block_names.append(f"layer_{layer_idx}")
                idx += 1
            else:
                conv = tf_keras.layers.Conv2D(
                    filters=v, kernel_size=3, padding="same", name=f"features/{idx}"
                )
                if cfg.norm_layer != "":
                    norm = norm_layer(name=f"features/{idx + 1}")
                    act = act_layer(name=f"features/{idx + 2}")
                    self.blocks.extend([conv, norm, act])
                    self.block_names.extend([None, None, f"layer_{layer_idx}"])
                    idx += 3
                else:
                    act = act_layer(name=f"features/{idx + 1}")
                    self.blocks.extend([conv, act])
                    self.block_names.extend([None, f"layer_{layer_idx}"])
                    idx += 2
            layer_idx += 1

        self.pre_logits = ConvMlp(
            out_channels=self.cfg.nb_features,
            kernel_size=7,
            mlp_ratio=self.cfg.mlp_ratio,
            drop_rate=self.cfg.drop_rate,
            act_layer=self.cfg.act_layer,
            name="pre_logits",
        )
        self.head = ClassifierHead(
            nb_classes=self.cfg.nb_classes,
            pool_type=self.cfg.global_pool,
            drop_rate=self.cfg.drop_rate,
            name="head",
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
        for block, block_name in zip(self.blocks, self.block_names):
            x = block(x, training=training)
            if block_name is not None:
                features[block_name] = x
        x = self.pre_logits(x, training=training)
        features["features"] = x

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

        x = self.head(x, training=training)
        features["logits"] = x
        return (x, features) if return_features else x


@register_model
def vgg11():
    """VGG 11-layer model (configuration "A")."""
    cfg = VGGConfig(
        name="vgg11",
        url="[timm]",
        layers=(64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"),
    )
    return VGG, cfg


@register_model
def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalisation."""
    cfg = VGGConfig(
        name="vgg11_bn",
        url="[timm]",
        layers=(64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"),
        norm_layer="batch_norm",
    )
    return VGG, cfg


@register_model
def vgg13():
    """VGG 13-layer model (configuration "B")."""
    cfg = VGGConfig(
        name="vgg13",
        url="[timm]",
        layers=(
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            "M",
            512,
            512,
            "M",
            512,
            512,
            "M",
        ),
    )
    return VGG, cfg


@register_model
def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalisation."""
    cfg = VGGConfig(
        name="vgg13_bn",
        url="[timm]",
        layers=(
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            "M",
            512,
            512,
            "M",
            512,
            512,
            "M",
        ),
        norm_layer="batch_norm",
    )
    return VGG, cfg


@register_model
def vgg16():
    """VGG 16-layer model (configuration "D")."""
    cfg = VGGConfig(
        name="vgg16",
        url="[timm]",
        layers=(
            # fmt: off
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
            "M",
            # fmt: on
        ),
    )
    return VGG, cfg


@register_model
def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalisation."""
    cfg = VGGConfig(
        name="vgg16_bn",
        url="[timm]",
        layers=(
            # fmt: off
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
            "M",
            # fmt: on
        ),
        norm_layer="batch_norm",
    )
    return VGG, cfg


@register_model
def vgg19():
    """VGG 19-layer model (configuration "E")."""
    cfg = VGGConfig(
        name="vgg19",
        url="[timm]",
        layers=(
            # fmt: off
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
            512,
            "M",
            # fmt: on
        ),
    )
    return VGG, cfg


@register_model
def vgg19_bn():
    """VGG 19-layer model (configuration "E") with batch normalisation."""
    cfg = VGGConfig(
        name="vgg19_bn",
        url="[timm]",
        layers=(
            # fmt: off
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
            512,
            "M",
            # fmt: on
        ),
        norm_layer="batch_norm",
    )
    return VGG, cfg
