"""
We provide an implementation and pretrained weights for the ConvNeXt models.

Paper: A ConvNet for the 2020s.
`[arXiv:2201.03545] <https://arxiv.org/abs/2201.03545>`_.

Original pytorch code and weights from
`Facebook Research <https://github.com/facebookresearch/ConvNeXt>`_.

This code has been ported from the
`timm <https://github.com/rwightman/pytorch-image-models>`_ implementation.

The following models are available.

* Models trained on ImageNet-1k

  * ``convnext_tiny``
  * ``convnext_small``
  * ``convnext_base``
  * ``convnext_large``

* Models trained on ImageNet-22k, fine-tuned on ImageNet-1k

  * ``convnext_tiny_in22ft1k``
  * ``convnext_small_in22ft1k``
  * ``convnext_base_in22ft1k``
  * ``convnext_large_in22ft1k``
  * ``convnext_xlarge_in22ft1k``

* Models trained on ImageNet-22k, fine-tuned on ImageNet-1k at 384 resolution

  * ``convnext_tiny_384_in22ft1k``
  * ``convnext_small_384_in22ft1k``
  * ``convnext_base_384_in22ft1k``
  * ``convnext_large_384_in22ft1k``
  * ``convnext_xlarge_384_in22ft1k``

* Models trained on ImageNet-22k

  * ``convnext_tiny_in22k``
  * ``convnext_small_in22k``
  * ``convnext_base_in22k``
  * ``convnext_large_in22k``
  * ``convnext_xlarge_in22k``
"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the MIT license
# Modifications and additions for timm by / Copyright 2022, Ross Wightman
# Copyright 2022 Marting Bruveris
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from tfimm.layers import MLP, ConvMLP, DropPath, norm_layer_factory
from tfimm.models import ModelConfig, keras_serializable, register_model
from tfimm.utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# Model registry will add each entrypoint fn to this
__all__ = ["ConvNeXtConfig", "ConvNeXt"]


@dataclass
class ConvNeXtConfig(ModelConfig):
    """
    Configuration class for ConvNeXt models.

    Parameters:
        name: Name of the model.
        url: URL for pretrained weights.
        nb_classes: Number of classes for classification head.
        in_channels: Number of input image channels.
        input_size: Input image size (height, width)

        patch_size: Patchifying the image is implemented via a convolutional layer with
            kernel size and stride equal to ``patch_size``.
        embed_dim: Feature dimensions at each stage.
        nb_blocks: Number of blocks at each stage.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim
        conv_mlp_block: There are two equivalent implementations of the ConvNeXt block,
            using either (1) 1x1 convolutions or (2) fully connected layers. In PyTorch
            option (2) also requires permuting channels, which is not needed in
            TensorFlow. We offer both implementations here, because some ``timm`` models
            use (1) while others use (2).

        drop_rate: Dropout rate.
        drop_path_rate: Dropout rate for stochastic depth.

        norm_layer: Normalization layer. See :func:`~norm_layer_factory` for possible
            values.
        act_layer: Activation function. See :func:`~act_layer_factory` for possible
            values.
        init_scale: Inital value for layer scale weights.

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
    patch_size: int = 4
    embed_dim: Tuple = (96, 192, 384, 768)
    nb_blocks: Tuple = (3, 3, 9, 3)
    mlp_ratio: float = 4.0
    conv_mlp_block: bool = False
    # Regularization
    drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    # Other parameters
    norm_layer: str = "layer_norm_eps_1e-6"
    act_layer: str = "gelu"
    init_scale: float = 1e-6
    # Parameters for inference
    crop_pct: float = 0.875
    interpolation: str = "bicubic"
    # Preprocessing
    mean: Tuple[float, float, float] = IMAGENET_DEFAULT_MEAN
    std: Tuple[float, float, float] = IMAGENET_DEFAULT_STD
    # Weight transfer
    first_conv: str = "stem/0"
    classifier: str = "head/fc"


def _weight_initializers(seed=42):
    """Function returns initilializers to be used in the model."""
    kernel_initializer = tf.keras.initializers.TruncatedNormal(
        mean=0.0, stddev=0.02, seed=seed
    )
    bias_initializer = tf.keras.initializers.Zeros()
    return kernel_initializer, bias_initializer


class ConvNeXtBlock(tf.keras.layers.Layer):
    """
    ConvNeXt Block

    There are two equivalent implementations:
      (1) DwConv -> LayerNorm -> 1x1 Conv -> GELU -> 1x1 Conv
      (2) DwConv -> LayerNorm -> Linear -> GELU -> Linear;

    We offer both implementations, selected using the ``conv_mlp_block`` parameter,
    because the ``timm`` implementation also has both and uses (1) for some models
    and (2) for others. There is a slight performance difference in PyTorch, because
    (2) requires permuting axes.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self,
        embed_dim: int,
        mlp_ratio: float,
        conv_mlp_block: bool,
        drop_rate: float,
        drop_path_rate: float,
        norm_layer: str,
        act_layer: str,
        init_scale: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.conv_mlp_block = conv_mlp_block
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        self.init_scale = init_scale

        mlp_layer = ConvMLP if conv_mlp_block else MLP
        norm_layer = norm_layer_factory(norm_layer)
        kernel_initializer, bias_initializer = _weight_initializers()

        self.pad = tf.keras.layers.ZeroPadding2D(padding=3)
        self.conv_dw = tf.keras.layers.DepthwiseConv2D(
            kernel_size=7,
            depthwise_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name="conv_dw",
        )
        self.norm = norm_layer(name="norm")
        self.mlp = mlp_layer(
            hidden_dim=int(mlp_ratio * embed_dim),
            embed_dim=embed_dim,
            drop_rate=drop_rate,
            act_layer=act_layer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name="mlp",
        )
        self.gamma = None
        self.drop_path = DropPath(drop_prob=drop_path_rate)

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(self.embed_dim,),
            initializer=tf.keras.initializers.Constant(value=self.init_scale),
            trainable=True,
            name="gamma",
        )

    def call(self, x, training=False):
        shortcut = x
        x = self.pad(x)
        x = self.conv_dw(x)
        x = self.norm(x, training=training)
        x = self.mlp(x, training=training)
        x = x * self.gamma
        x = self.drop_path(x, training=training)
        x = x + shortcut
        return x


class ConvNeXtStage(tf.keras.layers.Layer):
    """
    One stage of a ConvNeXt network: (optional) downsample layer, followed by a
    sequence of ``ConvNeXtBlox``s.
    """

    def __init__(
        self,
        stride: int,
        embed_dim: int,
        nb_blocks: int,
        mlp_ratio: float,
        conv_mlp_block: bool,
        drop_rate: float,
        drop_path_rate: np.ndarray,
        norm_layer: str,
        act_layer: str,
        init_scale: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.norm_layer = norm_layer

        norm_layer = norm_layer_factory(norm_layer)
        kernel_initializer, bias_initializer = _weight_initializers()

        if stride > 1:
            self.downsample_norm = norm_layer(name="downsample/0")
            self.downsample_conv = tf.keras.layers.Conv2D(
                filters=embed_dim,
                kernel_size=stride,
                strides=stride,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name="downsample/1",
            )
        else:
            self.downsample_norm = None
            self.downsample_conv = None

        self.blocks = [
            ConvNeXtBlock(
                embed_dim=embed_dim,
                mlp_ratio=mlp_ratio,
                conv_mlp_block=conv_mlp_block,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate[idx],
                norm_layer=self.norm_layer,
                act_layer=act_layer,
                init_scale=init_scale,
                name=f"blocks/{idx}",
            )
            for idx in range(nb_blocks)
        ]

    def call(self, x, training=False, return_features=False):
        features = OrderedDict()
        if self.downsample_conv is not None:
            x = self.downsample_norm(x, training=training)
            x = self.downsample_conv(x)
            features["downsample"] = x
        for idx, block in enumerate(self.blocks):
            x = block(x, training=training)
            features[f"block_{idx}"] = x
        return (x, features) if return_features else x


@keras_serializable
class ConvNeXt(tf.keras.Model):
    """
    Class implementing a ConvNeXt network.

    Paper: `A ConvNet for the 2020s <https://arxiv.org/pdf/2201.03545.pdf>`_.

    Parameters:
        cfg: Configuration class for the model.
        **kwargs: Arguments are passed to ``tf.keras.Model``.
    """

    cfg_class = ConvNeXtConfig

    def __init__(self, cfg: ConvNeXtConfig, **kwargs):
        kwargs["name"] = kwargs.get("name", cfg.name)
        super().__init__(**kwargs)
        self.cfg = cfg
        norm_layer = norm_layer_factory(cfg.norm_layer)
        kernel_initializer, bias_initializer = _weight_initializers()

        self.stem_conv = tf.keras.layers.Conv2D(
            filters=cfg.embed_dim[0],
            kernel_size=cfg.patch_size,
            strides=cfg.patch_size,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name="stem/0",
        )
        self.stem_norm = norm_layer(name="stem/1")

        # Stochastic depth
        dpr = np.linspace(0.0, cfg.drop_path_rate, sum(cfg.nb_blocks))
        dpr = np.split(dpr, np.cumsum(cfg.nb_blocks))

        self.stages = []
        nb_stages = len(cfg.nb_blocks)
        for j in range(nb_stages):
            self.stages.append(
                ConvNeXtStage(
                    stride=2 if j > 0 else 1,
                    embed_dim=cfg.embed_dim[j],
                    nb_blocks=cfg.nb_blocks[j],
                    mlp_ratio=cfg.mlp_ratio,
                    conv_mlp_block=cfg.conv_mlp_block,
                    drop_rate=cfg.drop_rate,
                    drop_path_rate=dpr[j],
                    norm_layer=cfg.norm_layer,
                    act_layer=cfg.act_layer,
                    init_scale=cfg.init_scale,
                    name=f"stages/{j}",
                )
            )

        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.norm = norm_layer(name="head/norm")
        self.flatten = tf.keras.layers.Flatten()
        self.drop = tf.keras.layers.Dropout(rate=cfg.drop_rate)
        self.fc = (
            tf.keras.layers.Dense(units=cfg.nb_classes, name="head/fc")
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
        x = self.stem_conv(x)
        x = self.stem_norm(x, training=training)
        features["stem"] = x

        for stage_idx, stage in enumerate(self.stages):
            x = stage(x, training=training, return_features=return_features)
            if return_features:
                x, stage_features = x
                for key, val in stage_features.items():
                    features[f"stage_{stage_idx}/{key}"] = val
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
        x = self.norm(x, training=training)
        x = self.flatten(x)
        features["features"] = x
        x = self.drop(x, training=training)
        x = self.fc(x)
        features["logits"] = x
        return (x, features) if return_features else x


@register_model
def convnext_tiny():
    cfg = ConvNeXtConfig(
        name="convnext_tiny",
        url="[timm]",
        embed_dim=(96, 192, 384, 768),
        nb_blocks=(3, 3, 9, 3),
    )
    return ConvNeXt, cfg


@register_model
def convnext_small():
    cfg = ConvNeXtConfig(
        name="convnext_small",
        url="[timm]",
        embed_dim=(96, 192, 384, 768),
        nb_blocks=(3, 3, 27, 3),
    )
    return ConvNeXt, cfg


@register_model
def convnext_base():
    cfg = ConvNeXtConfig(
        name="convnext_base",
        url="[timm]",
        embed_dim=(128, 256, 512, 1024),
        nb_blocks=(3, 3, 27, 3),
    )
    return ConvNeXt, cfg


@register_model
def convnext_large():
    cfg = ConvNeXtConfig(
        name="convnext_large",
        url="[timm]",
        embed_dim=(192, 384, 768, 1536),
        nb_blocks=(3, 3, 27, 3),
    )
    return ConvNeXt, cfg


@register_model
def convnext_tiny_in22ft1k():
    cfg = ConvNeXtConfig(
        name="convnext_tiny_in22ft1k",
        url="[timm]",
        embed_dim=(96, 192, 384, 768),
        nb_blocks=(3, 3, 9, 3),
    )
    return ConvNeXt, cfg


@register_model
def convnext_small_in22ft1k():
    cfg = ConvNeXtConfig(
        name="convnext_small_in22ft1k",
        url="[timm]",
        embed_dim=(96, 192, 384, 768),
        nb_blocks=(3, 3, 27, 3),
    )
    return ConvNeXt, cfg


@register_model
def convnext_base_in22ft1k():
    cfg = ConvNeXtConfig(
        name="convnext_base_in22ft1k",
        url="[timm]",
        embed_dim=(128, 256, 512, 1024),
        nb_blocks=(3, 3, 27, 3),
    )
    return ConvNeXt, cfg


@register_model
def convnext_large_in22ft1k():
    cfg = ConvNeXtConfig(
        name="convnext_large_in22ft1k",
        url="[timm]",
        embed_dim=(192, 384, 768, 1536),
        nb_blocks=(3, 3, 27, 3),
    )
    return ConvNeXt, cfg


@register_model
def convnext_xlarge_in22ft1k():
    cfg = ConvNeXtConfig(
        name="convnext_xlarge_in22ft1k",
        url="[timm]",
        embed_dim=(256, 512, 1024, 2048),
        nb_blocks=(3, 3, 27, 3),
    )
    return ConvNeXt, cfg


@register_model
def convnext_tiny_384_in22ft1k():
    cfg = ConvNeXtConfig(
        name="convnext_tiny_384_in22ft1k",
        url="[timm]",
        input_size=(384, 384),
        embed_dim=(96, 192, 384, 768),
        nb_blocks=(3, 3, 9, 3),
    )
    return ConvNeXt, cfg


@register_model
def convnext_small_384_in22ft1k():
    cfg = ConvNeXtConfig(
        name="convnext_small_384_in22ft1k",
        url="[timm]",
        input_size=(384, 384),
        embed_dim=(96, 192, 384, 768),
        nb_blocks=(3, 3, 27, 3),
    )
    return ConvNeXt, cfg


@register_model
def convnext_base_384_in22ft1k():
    cfg = ConvNeXtConfig(
        name="convnext_base_384_in22ft1k",
        url="[timm]",
        input_size=(384, 384),
        embed_dim=(128, 256, 512, 1024),
        nb_blocks=(3, 3, 27, 3),
    )
    return ConvNeXt, cfg


@register_model
def convnext_large_384_in22ft1k():
    cfg = ConvNeXtConfig(
        name="convnext_large_384_in22ft1k",
        url="[timm]",
        input_size=(384, 384),
        embed_dim=(192, 384, 768, 1536),
        nb_blocks=(3, 3, 27, 3),
    )
    return ConvNeXt, cfg


@register_model
def convnext_xlarge_384_in22ft1k():
    cfg = ConvNeXtConfig(
        name="convnext_xlarge_384_in22ft1k",
        url="[timm]",
        input_size=(384, 384),
        embed_dim=(256, 512, 1024, 2048),
        nb_blocks=(3, 3, 27, 3),
    )
    return ConvNeXt, cfg


@register_model
def convnext_tiny_in22k():
    cfg = ConvNeXtConfig(
        name="convnext_tiny_in22k",
        url="[timm]",
        nb_classes=21841,
        embed_dim=(96, 192, 384, 768),
        nb_blocks=(3, 3, 9, 3),
    )
    return ConvNeXt, cfg


@register_model
def convnext_small_in22k():
    cfg = ConvNeXtConfig(
        name="convnext_small_in22k",
        url="[timm]",
        nb_classes=21841,
        embed_dim=(96, 192, 384, 768),
        nb_blocks=(3, 3, 27, 3),
    )
    return ConvNeXt, cfg


@register_model
def convnext_base_in22k():
    cfg = ConvNeXtConfig(
        name="convnext_base_in22k",
        url="[timm]",
        nb_classes=21841,
        embed_dim=(128, 256, 512, 1024),
        nb_blocks=(3, 3, 27, 3),
    )
    return ConvNeXt, cfg


@register_model
def convnext_large_in22k():
    cfg = ConvNeXtConfig(
        name="convnext_large_in22k",
        url="[timm]",
        nb_classes=21841,
        embed_dim=(192, 384, 768, 1536),
        nb_blocks=(3, 3, 27, 3),
    )
    return ConvNeXt, cfg


@register_model
def convnext_xlarge_in22k():
    cfg = ConvNeXtConfig(
        name="convnext_xlarge_in22k",
        url="[timm]",
        nb_classes=21841,
        embed_dim=(256, 512, 1024, 2048),
        nb_blocks=(3, 3, 27, 3),
    )
    return ConvNeXt, cfg
