"""
We provide an implementation and pretrained weights for Pooling-based Vision
Transformers (PiT).

Paper: Rethinking Spatial Dimensions of Vision Transformers.
`[arXiv:2103.16302] <https://arxiv.org/abs/2103.16302>`_.

Original pytorch code and weights from
`NAVER AI <https://github.com/naver-ai/pit>`_.

This code has been ported from the
`timm <https://github.com/rwightman/pytorch-image-models>`_ implementation.

The following models are available.

* Models trained on ImageNet-1k

  * ``pit_ti_224``
  * ``pit_xs_224``
  * ``pit_s_224``
  * ``pit_b_224``

* Models trained on ImageNet-1k, using knowledge distillation

  * ``pit_ti_distilled_224``
  * ``pit_xs_distilled_224``
  * ``pit_s_distilled_224``
  * ``pit_b_distilled_224``
"""
# PiT
# Copyright 2021-present NAVER Corp.
# Apache License v2.0
# Modifications for timm by / Copyright 2020 Ross Wightman
# Copyright 2022 Martins Bruveris
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
import tensorflow as tf

from tfimm.architectures.vit import ViTBlock
from tfimm.layers import interpolate_pos_embeddings_grid, norm_layer_factory
from tfimm.models import ModelConfig, keras_serializable, register_model
from tfimm.utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# Model registry will add each entrypoint fn to this
__all__ = ["PoolingVisionTransformer", "PoolingVisionTransformerConfig"]


@dataclass
class PoolingVisionTransformerConfig(ModelConfig):
    """
    Configuration class for ConvNeXt models.

    Parameters:
        name: Name of the model.
        url: URL for pretrained weights.
        nb_classes: Number of classes for classification head.
        in_channels: Number of input image channels.
        input_size: Input image size (height, width)

        patch_size: Patchifying the image is implemented via a convolutional layer with
            kernel size ``patch_size`` and stride given by ``stride``.
        stride: Stride in patch embedding layer.
        embed_dim: Feature dimensions at each stage.
        nb_blocks: Number of blocks at each stage.
        nb_heads: Number of self-attention heads at each stage.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim
        distilled: If ``True``, we add a distillation head in addition to classification
            head.

        drop_rate: Dropout rate.
        attn_drop_rate: Attention dropout rate.
        drop_path_rate: Dropout rate for stochastic depth.

        norm_layer: Normalization layer. See :func:`~norm_layer_factory` for possible
            values.
        act_layer: Activation function. See :func:`~act_layer_factory` for possible
            values.

        interpolate_input: If ``True``, we interpolate position embeddings to given
            input size, so inference can be done for arbitrary input shape. If ``False``
            inference can only be performed at ``input_size``.
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
    patch_size: int = 16
    stride: int = 8
    embed_dim: Tuple = (64, 128, 256)
    nb_blocks: Tuple = (2, 6, 4)
    nb_heads: Tuple = (2, 4, 8)
    mlp_ratio: float = 4.0
    distilled: bool = False
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
    # Preprocessing
    mean: Tuple[float, float, float] = IMAGENET_DEFAULT_MEAN
    std: Tuple[float, float, float] = IMAGENET_DEFAULT_STD
    # Weight transfer
    first_conv: str = "patch_embed/conv"
    classifier: Union[str, Tuple[str, str]] = "head"

    @property
    def nb_tokens(self) -> int:
        """Number of special tokens. Equals 2 if distillation is used, otherwise 1."""
        return 2 if self.distilled else 1

    @property
    def grid_size(self) -> Tuple[int, int]:
        """Grid size for patch embeddings."""
        height = (self.input_size[0] - self.patch_size) // self.stride + 1
        width = (self.input_size[1] - self.patch_size) // self.stride + 1
        return height, width

    @property
    def transform_weights(self):
        """
        Dictionary of functions to transform weights when loading them in models with
        different configs.
        """
        return {"pos_embed": PoolingVisionTransformer.transform_pos_embed}


class ConvHeadPooling(tf.keras.layers.Layer):
    def __init__(
        self,
        nb_tokens: int,
        in_channels: int,
        out_channels: int,
        stride: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.nb_tokens = nb_tokens
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.pad = tf.keras.layers.ZeroPadding2D(padding=stride // 2)
        self.conv = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=stride + 1,
            strides=stride,
            groups=in_channels,
            name="conv",
        )
        self.fc = tf.keras.layers.Dense(units=out_channels, name="fc")

    def call(self, x):
        x, input_size = x
        batch_size, _, nb_channels = tf.unstack(tf.shape(x))

        tokens = x[:, : self.nb_tokens]
        x = x[:, self.nb_tokens :]  # (N, L, C)
        x = tf.reshape(x, (batch_size, *input_size, nb_channels))  # (N, H, W, C)

        x = self.pad(x)
        x = self.conv(x)
        tokens = self.fc(tokens)
        output_size = tf.unstack(tf.shape(x)[1:3])

        x = tf.reshape(x, (batch_size, -1, self.out_channels))
        x = tf.concat([tokens, x], axis=1)

        return x, output_size


@keras_serializable
class PoolingVisionTransformer(tf.keras.Model):
    """
    Class implementing a Pooling-based Vision Transformer (PiT).

    Paper: Rethinking Spatial Dimensions of Vision Transformers.
    `[arXiv:2103.16302] <https://arxiv.org/abs/2103.16302>`_

    Parameters:
        cfg: Configuration class for the model.
        **kwargs: Arguments are passed to ``tf.keras.Model``
    """

    cfg_class = PoolingVisionTransformerConfig

    def __init__(self, cfg: PoolingVisionTransformerConfig, **kwargs):
        kwargs["name"] = kwargs.get("name", cfg.name)
        super().__init__(**kwargs)
        self.cfg = cfg
        norm_layer = norm_layer_factory(cfg.norm_layer)

        self.patch_embed = tf.keras.layers.Conv2D(
            filters=cfg.embed_dim[0],
            kernel_size=cfg.patch_size,
            strides=cfg.stride,
            name="patch_embed/conv",
        )
        self.pos_embed = None
        self.pos_drop = tf.keras.layers.Dropout(rate=cfg.drop_rate)
        self.cls_token = None

        # Stochastic depth
        dpr = np.linspace(0.0, cfg.drop_path_rate, sum(cfg.nb_blocks))
        dpr = np.split(dpr, np.cumsum(cfg.nb_blocks))

        self.blocks = OrderedDict()
        for j in range(len(cfg.nb_blocks)):
            for k in range(cfg.nb_blocks[j]):
                self.blocks[f"stage_{j}/block_{k}"] = ViTBlock(
                    embed_dim=cfg.embed_dim[j],
                    nb_heads=cfg.nb_heads[j],
                    mlp_ratio=cfg.mlp_ratio,
                    qkv_bias=True,
                    drop_rate=cfg.drop_rate,
                    attn_drop_rate=cfg.attn_drop_rate,
                    drop_path_rate=dpr[j][k],
                    norm_layer=cfg.norm_layer,
                    act_layer=cfg.act_layer,
                    name=f"transformers/{j}/blocks/{k}",
                )
            if j < len(cfg.nb_blocks) - 1:
                self.blocks[f"stage_{j}/pool"] = ConvHeadPooling(
                    nb_tokens=cfg.nb_tokens,
                    in_channels=cfg.embed_dim[j],
                    out_channels=cfg.embed_dim[j + 1],
                    stride=2,
                    name=f"transformers/{j}/pool",
                )

        self.norm = norm_layer(name="norm")

        self.head = (
            tf.keras.layers.Dense(units=cfg.nb_classes, name="head")
            if cfg.nb_classes > 0
            else tf.keras.layers.Activation("linear")  # Identity layer
        )
        if cfg.distilled:
            self.head_dist = (
                tf.keras.layers.Dense(units=cfg.nb_classes, name="head_dist")
                if cfg.nb_classes > 0
                else tf.keras.layers.Activation("linear")  # Identity layer
            )
        else:
            self.head_dist = None

    def build(self, input_shape):
        height = (input_shape[1] - self.cfg.patch_size) // self.cfg.stride + 1
        width = (input_shape[2] - self.cfg.patch_size) // self.cfg.stride + 1
        self.pos_embed = self.add_weight(
            # We keep PT style NCHW order to make weight translation easier
            shape=(1, self.cfg.embed_dim[0], height, width),
            initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02),
            trainable=True,
            name="pos_embed",
        )
        self.cls_token = self.add_weight(
            shape=(1, self.cfg.nb_tokens, self.cfg.embed_dim[0]),
            initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02),
            trainable=True,
            name="cls_token",
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

    def transform_pos_embed(
        self, src_weights, target_cfg: PoolingVisionTransformerConfig
    ):
        """
        Transforms the position embedding weights in accordance with `target_cfg` and
        returns them.
        """
        pos_embed = interpolate_pos_embeddings_grid(
            pos_embed=tf.transpose(self.pos_embed, [0, 2, 3, 1]),
            tgt_grid_size=target_cfg.grid_size,
        )
        pos_embed = tf.transpose(pos_embed, [0, 3, 1, 2])
        return pos_embed

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
        x = self.patch_embed(x)
        pos_embed = tf.transpose(self.pos_embed, [0, 2, 3, 1])
        if not self.cfg.interpolate_input:
            x = x + pos_embed
        else:
            grid_size = tf.unstack(tf.shape(x)[1:3])
            pos_embed = interpolate_pos_embeddings_grid(
                pos_embed,
                tgt_grid_size=grid_size,
            )
            x = x + pos_embed
        x = self.pos_drop(x, training=training)

        batch_size, height, width, nb_channels = tf.unstack(tf.shape(x))
        input_size = (height, width)
        cls_token = tf.repeat(self.cls_token, repeats=batch_size, axis=0)
        x = tf.reshape(x, (batch_size, -1, nb_channels))
        x = tf.concat([cls_token, x], axis=1)
        features["patch_embedding"] = x

        for key, block in self.blocks.items():
            if key.endswith("pool"):
                x, input_size = block((x, input_size), training=training)
            else:
                x = block(x, training=training)
            features[key] = x
        features["features_all"] = x

        x = x[:, : self.cfg.nb_tokens]
        x = self.norm(x, training=training)
        x = x if self.cfg.distilled else x[:, 0]
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

        if not self.cfg.distilled:
            x = self.head(x, training=training)
        else:
            y = self.head(x[:, 0], training=training)
            y_dist = self.head_dist(x[:, 1], training=training)
            x = tf.stack((y, y_dist), axis=1)
        features["logits"] = x
        return (x, features) if return_features else x


@register_model
def pit_ti_224():
    cfg = PoolingVisionTransformerConfig(
        name="pit_ti_224",
        url="[timm]",
        patch_size=16,
        stride=8,
        embed_dim=(64, 128, 256),
        nb_blocks=(2, 6, 4),
        nb_heads=(2, 4, 8),
        mlp_ratio=4.0,
    )
    return PoolingVisionTransformer, cfg


@register_model
def pit_xs_224():
    cfg = PoolingVisionTransformerConfig(
        name="pit_xs_224",
        url="[timm]",
        patch_size=16,
        stride=8,
        embed_dim=(96, 192, 384),
        nb_blocks=(2, 6, 4),
        nb_heads=(2, 4, 8),
        mlp_ratio=4.0,
    )
    return PoolingVisionTransformer, cfg


@register_model
def pit_s_224():
    cfg = PoolingVisionTransformerConfig(
        name="pit_s_224",
        url="[timm]",
        patch_size=16,
        stride=8,
        embed_dim=(144, 288, 576),
        nb_blocks=(2, 6, 4),
        nb_heads=(3, 6, 12),
        mlp_ratio=4.0,
    )
    return PoolingVisionTransformer, cfg


@register_model
def pit_b_224():
    cfg = PoolingVisionTransformerConfig(
        name="pit_b_224",
        url="[timm]",
        patch_size=14,
        stride=7,
        embed_dim=(256, 512, 1024),
        nb_blocks=(3, 6, 4),
        nb_heads=(4, 8, 16),
        mlp_ratio=4.0,
    )
    return PoolingVisionTransformer, cfg


@register_model
def pit_ti_distilled_224():
    cfg = PoolingVisionTransformerConfig(
        name="pit_ti_distilled_224",
        url="[timm]",
        patch_size=16,
        stride=8,
        embed_dim=(64, 128, 256),
        nb_blocks=(2, 6, 4),
        nb_heads=(2, 4, 8),
        mlp_ratio=4.0,
        distilled=True,
        classifier=("head", "head_dist"),
    )
    return PoolingVisionTransformer, cfg


@register_model
def pit_xs_distilled_224():
    cfg = PoolingVisionTransformerConfig(
        name="pit_xs_distilled_224",
        url="[timm]",
        patch_size=16,
        stride=8,
        embed_dim=(96, 192, 384),
        nb_blocks=(2, 6, 4),
        nb_heads=(2, 4, 8),
        mlp_ratio=4.0,
        distilled=True,
        classifier=("head", "head_dist"),
    )
    return PoolingVisionTransformer, cfg


@register_model
def pit_s_distilled_224():
    cfg = PoolingVisionTransformerConfig(
        name="pit_s_distilled_224",
        url="[timm]",
        patch_size=16,
        stride=8,
        embed_dim=(144, 288, 576),
        nb_blocks=(2, 6, 4),
        nb_heads=(3, 6, 12),
        mlp_ratio=4.0,
        distilled=True,
        classifier=("head", "head_dist"),
    )
    return PoolingVisionTransformer, cfg


@register_model
def pit_b_distilled_224():
    cfg = PoolingVisionTransformerConfig(
        name="pit_b_distilled_224",
        url="[timm]",
        patch_size=14,
        stride=7,
        embed_dim=(256, 512, 1024),
        nb_blocks=(3, 6, 4),
        nb_heads=(4, 8, 16),
        mlp_ratio=4.0,
        distilled=True,
        classifier=("head", "head_dist"),
    )
    return PoolingVisionTransformer, cfg
