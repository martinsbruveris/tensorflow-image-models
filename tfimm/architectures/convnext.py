"""
We provide an implementation and pretrained weights for the ConvNeXt models.

Paper: A ConvNet for the 2020s.
`[arXiv:2201.03545] <https://arxiv.org/abs/2201.03545>`_.

Paper: ConvNeXt-V2 - Co-designing and Scaling ConvNets with Masked Autoencoders -
`[arXiv:2301.00808] <https://arxiv.org/abs/2301.00808>`_.

Original pytorch code and weights from

  * ConvNeXt: `Facebook Research <https://github.com/facebookresearch/ConvNeXt>`_
  * ConvNeXt-V2: `Facebook Research <https://github.com/facebookresearch/ConvNeXt-V2>`_
  * Models named ``atto``, ``femto``, ``pico``, ``nano`` and with suffixes ``_ols``
    (overlapping stem) and ``_hnf`` (head norm first) are timm originals.

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
# Copyright 2022 Martins Bruveris
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf

from tfimm.layers import MLP, DropPath, GlobalResponseNormMLP, norm_layer_factory
from tfimm.models import (
    ModelConfig,
    keras_serializable,
    register_deprecation,
    register_model,
    register_model_tag,
)
from tfimm.utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    make_divisible,
)

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

        stem_type: Stem type. Can be one of "patch", "overlap" or "overlap_tiered".
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
        use_grn: Use Global Response Norm (ConvNeXt-V2) in MLP.
        head_norm_first: If True, then we use norm -> global pool -> fc ordering in the
            head, like most other nets. Otherwise, pool -> norm -> fc, the default
            ConvNeXt ordering (pretrained FB weights).
        head_hidden_size: Size of MLP hidden layer in head if not None and
            ``head_norm_first=False``.

        drop_rate: Dropout rate.
        drop_path_rate: Dropout rate for stochastic depth.

        norm_layer: Normalization layer. See :func:`~norm_layer_factory` for possible
            values.
        act_layer: Activation function. See :func:`~act_layer_factory` for possible
            values.
        init_scale: Inital value for layer scale weights. If None, layer scale is not
            used.

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
    stem_type: str = "patch"
    patch_size: int = 4
    embed_dim: Tuple = (96, 192, 384, 768)
    nb_blocks: Tuple = (3, 3, 9, 3)
    mlp_ratio: float = 4.0
    conv_mlp_block: bool = False
    use_grn: bool = False
    head_norm_first: bool = False
    head_hidden_size: Optional[int] = None
    # Regularization
    drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    # Other parameters
    norm_layer: str = "layer_norm_eps_1e-6"
    act_layer: str = "gelu"
    init_scale: Optional[float] = 1e-6
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
        use_grn: bool,
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

        mlp_layer = GlobalResponseNormMLP if use_grn else MLP
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
            use_conv=conv_mlp_block,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name="mlp",
        )
        self.gamma = None
        self.drop_path = DropPath(drop_prob=drop_path_rate)

    def build(self, input_shape):
        if self.init_scale is not None:
            self.gamma = self.add_weight(
                shape=(self.embed_dim,),
                initializer=tf.keras.initializers.Constant(value=self.init_scale),
                trainable=True,
                name="gamma",
            )
        super().build(input_shape)

    def call(self, x, training=False):
        shortcut = x
        x = self.pad(x)
        x = self.conv_dw(x)
        x = self.norm(x, training=training)
        x = self.mlp(x, training=training)
        if self.gamma is not None:
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
        use_grn: bool,
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
                use_grn=use_grn,
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

        self.stem = []
        if cfg.stem_type == "patch":
            stem_conv = tf.keras.layers.Conv2D(
                filters=cfg.embed_dim[0],
                kernel_size=cfg.patch_size,
                strides=cfg.patch_size,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name="stem/0",
            )
            stem_norm = norm_layer(name="stem/1")
            self.stem = [stem_conv, stem_norm]
        else:
            mid_channels = (
                make_divisible(cfg.embed_dim[0] // 2, divisor=8)
                if "tiered" in cfg.stem_type
                else cfg.embed_dim[0]
            )
            stem_pad1 = tf.keras.layers.ZeroPadding2D(padding=1, name="stem/0p")
            stem_conv1 = tf.keras.layers.Conv2D(
                filters=mid_channels,
                kernel_size=3,
                strides=2,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name="stem/0",
            )
            stem_pad2 = tf.keras.layers.ZeroPadding2D(padding=1, name="stem/1p")
            stem_conv2 = tf.keras.layers.Conv2D(
                filters=cfg.embed_dim[0],
                kernel_size=3,
                strides=2,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                name="stem/1",
            )
            stem_norm = norm_layer(name="stem/2")
            self.stem = [stem_pad1, stem_conv1, stem_pad2, stem_conv2, stem_norm]

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
                    use_grn=cfg.use_grn,
                    drop_rate=cfg.drop_rate,
                    drop_path_rate=dpr[j],
                    norm_layer=cfg.norm_layer,
                    act_layer=cfg.act_layer,
                    init_scale=cfg.init_scale,
                    name=f"stages/{j}",
                )
            )

        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.norm = norm_layer(
            # For reasons the layer has a different name in timm depending on the order.
            name="norm_pre"
            if cfg.head_norm_first
            else "head/norm"
        )
        self.flatten = tf.keras.layers.Flatten()
        self.drop = tf.keras.layers.Dropout(rate=cfg.drop_rate)
        if cfg.head_hidden_size is not None:
            self.pre_logits = tf.keras.layers.Dense(
                units=cfg.head_hidden_size, activation="gelu", name="head/pre_logits/fc"
            )
        else:
            self.pre_logits = None
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
        for stem_layer in self.stem:
            x = stem_layer(x, training=training)
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

        if self.cfg.head_norm_first:
            x = self.norm(x, training=training)
            x = self.pool(x)
        else:
            x = self.pool(x)
            x = self.norm(x, training=training)
        x = self.flatten(x)
        if self.cfg.head_hidden_size:
            x = self.pre_logits(x)
        features["features"] = x
        x = self.drop(x, training=training)
        x = self.fc(x)
        features["logits"] = x
        return (x, features) if return_features else x


@register_model(default_tag="d2_in1k")
def convnext_atto():
    # Note: Still tweaking depths in timm, will vary between 3-4M param, currently 3.7M.
    cfg = ConvNeXtConfig(
        name="convnext_atto",
        embed_dim=(40, 80, 160, 320),
        nb_blocks=(2, 2, 6, 2),
        conv_mlp_block=True,
    )
    return ConvNeXt, cfg


@register_model(default_tag="a2_in1k")
def convnext_atto_ols():
    # Timm femto variant with overlapping 3x3 conv stem, wider than non-ols femto
    # above, current param count 3.7M.
    cfg = ConvNeXtConfig(
        name="convnext_atto_ols",
        embed_dim=(40, 80, 160, 320),
        nb_blocks=(2, 2, 6, 2),
        conv_mlp_block=True,
        stem_type="overlap_tiered",
    )
    return ConvNeXt, cfg


@register_model(default_tag="d1_in1k")
def convnext_femto():
    # Timm femto variant
    cfg = ConvNeXtConfig(
        name="convnext_femto",
        embed_dim=(48, 96, 192, 384),
        nb_blocks=(2, 2, 6, 2),
        conv_mlp_block=True,
    )
    return ConvNeXt, cfg


@register_model(default_tag="d1_in1k")
def convnext_femto_ols():
    # Timm femto variant
    cfg = ConvNeXtConfig(
        name="convnext_femto_ols",
        embed_dim=(48, 96, 192, 384),
        nb_blocks=(2, 2, 6, 2),
        conv_mlp_block=True,
        stem_type="overlap_tiered",
    )
    return ConvNeXt, cfg


@register_model(default_tag="d1_in1k")
def convnext_pico():
    # Timm pico variant
    cfg = ConvNeXtConfig(
        name="convnext_pico",
        embed_dim=(64, 128, 256, 512),
        nb_blocks=(2, 2, 6, 2),
        conv_mlp_block=True,
    )
    return ConvNeXt, cfg


@register_model(default_tag="d1_in1k")
def convnext_pico_ols():
    # Timm pico variant with overalpping 3x3 conv stem
    cfg = ConvNeXtConfig(
        name="convnext_pico_ols",
        embed_dim=(64, 128, 256, 512),
        nb_blocks=(2, 2, 6, 2),
        conv_mlp_block=True,
        stem_type="overlap_tiered",
    )
    return ConvNeXt, cfg


@register_model(default_tag="in12k_ft_in1k")
def convnext_nano():
    # Timm nano variant with standard stem and head.
    cfg = ConvNeXtConfig(
        name="convnext_nano",
        embed_dim=(80, 160, 320, 640),
        nb_blocks=(2, 2, 8, 2),
        conv_mlp_block=True,
    )
    return ConvNeXt, cfg


@register_model(default_tag="d1h_in1k")
def convnext_nano_ols():
    # Timm nano variant with overlapping 3x3 conv stem.
    cfg = ConvNeXtConfig(
        name="convnext_nano_ols",
        embed_dim=(80, 160, 320, 640),
        nb_blocks=(2, 2, 8, 2),
        conv_mlp_block=True,
        stem_type="overlap",
    )
    return ConvNeXt, cfg


@register_model(default_tag="in12k_ft_in1k")
def convnext_tiny():
    cfg = ConvNeXtConfig(
        name="convnext_tiny",
        embed_dim=(96, 192, 384, 768),
        nb_blocks=(3, 3, 9, 3),
    )
    return ConvNeXt, cfg


@register_model(default_tag="a2h_in1k")
def convnext_tiny_hnf():
    # Timm experimental variant with norm before pooling in head (head norm first)
    cfg = ConvNeXtConfig(
        name="convnext_tiny_hnf",
        embed_dim=(96, 192, 384, 768),
        nb_blocks=(3, 3, 9, 3),
        conv_mlp_block=True,
        head_norm_first=True,
    )
    return ConvNeXt, cfg


@register_model(default_tag="in12k_ft_in1k")
def convnext_small():
    cfg = ConvNeXtConfig(
        name="convnext_small",
        embed_dim=(96, 192, 384, 768),
        nb_blocks=(3, 3, 27, 3),
    )
    return ConvNeXt, cfg


@register_model(default_tag="fb_in22k_ft_in1k")
def convnext_base():
    cfg = ConvNeXtConfig(
        name="convnext_base",
        embed_dim=(128, 256, 512, 1024),
        nb_blocks=(3, 3, 27, 3),
    )
    return ConvNeXt, cfg


@register_model(default_tag="fb_in22k_ft_in1k")
def convnext_large():
    cfg = ConvNeXtConfig(
        name="convnext_large",
        embed_dim=(192, 384, 768, 1536),
        nb_blocks=(3, 3, 27, 3),
    )
    return ConvNeXt, cfg


@register_model(default_tag="clip_laion2b_soup_ft_in12k_in1k_320")
def convnext_large_mlp():
    cfg = ConvNeXtConfig(
        name="convnext_large_mlp",
        embed_dim=(192, 384, 768, 1536),
        nb_blocks=(3, 3, 27, 3),
        head_hidden_size=1536,
    )
    return ConvNeXt, cfg


@register_model(default_tag="fb_in22k_ft_in1k")
def convnext_xlarge():
    cfg = ConvNeXtConfig(
        name="convnext_xlarge",
        embed_dim=(256, 512, 1024, 2048),
        nb_blocks=(3, 3, 27, 3),
    )
    return ConvNeXt, cfg


@register_model(default_tag="")
def convnext_xxlarge():
    cfg = ConvNeXtConfig(
        name="convnext_xxlarge",
        embed_dim=(384, 768, 1536, 3072),
        nb_blocks=(3, 4, 30, 3),
        norm_layer="layer_norm",  # eps=1e-5 for default layer norm
    )
    return ConvNeXt, cfg


@register_model(default_tag="fcmae_ft_in1k")
def convnextv2_atto():
    cfg = ConvNeXtConfig(
        name="convnextv2_atto",
        embed_dim=(40, 80, 160, 320),
        nb_blocks=(2, 2, 6, 2),
        use_grn=True,
        init_scale=None,
        conv_mlp_block=True,
    )
    return ConvNeXt, cfg


@register_model(default_tag="fcmae_ft_in1k")
def convnextv2_femto():
    cfg = ConvNeXtConfig(
        name="convnextv2_femto",
        embed_dim=(48, 96, 192, 384),
        nb_blocks=(2, 2, 6, 2),
        use_grn=True,
        init_scale=None,
        conv_mlp_block=True,
    )
    return ConvNeXt, cfg


@register_model(default_tag="fcmae_ft_in1k")
def convnextv2_pico():
    cfg = ConvNeXtConfig(
        name="convnextv2_pico",
        embed_dim=(64, 128, 256, 512),
        nb_blocks=(2, 2, 6, 2),
        use_grn=True,
        init_scale=None,
        conv_mlp_block=True,
    )
    return ConvNeXt, cfg


@register_model(default_tag="fcmae_ft_in22k_in1k")
def convnextv2_nano():
    cfg = ConvNeXtConfig(
        name="convnextv2_nano",
        embed_dim=(80, 160, 320, 640),
        nb_blocks=(2, 2, 8, 2),
        use_grn=True,
        init_scale=None,
        conv_mlp_block=True,
    )
    return ConvNeXt, cfg


@register_model(default_tag="fcmae_ft_in22k_in1k")
def convnextv2_tiny():
    cfg = ConvNeXtConfig(
        name="convnextv2_tiny",
        embed_dim=(96, 192, 384, 768),
        nb_blocks=(3, 3, 9, 3),
        use_grn=True,
        init_scale=None,
    )
    return ConvNeXt, cfg


@register_model(default_tag="")
def convnextv2_small():
    cfg = ConvNeXtConfig(
        name="convnextv2_small",
        embed_dim=(96, 192, 384, 768),
        nb_blocks=(3, 3, 27, 3),
        use_grn=True,
        init_scale=None,
    )
    return ConvNeXt, cfg


@register_model(default_tag="fcmae_ft_in22k_in1k")
def convnextv2_base():
    cfg = ConvNeXtConfig(
        name="convnextv2_base",
        embed_dim=(128, 256, 512, 1024),
        nb_blocks=(3, 3, 27, 3),
        use_grn=True,
        init_scale=None,
    )
    return ConvNeXt, cfg


@register_model(default_tag="fcmae_ft_in22k_in1k")
def convnextv2_large():
    cfg = ConvNeXtConfig(
        name="convnextv2_large",
        embed_dim=(192, 384, 768, 1536),
        nb_blocks=(3, 3, 27, 3),
        use_grn=True,
        init_scale=None,
    )
    return ConvNeXt, cfg


@register_model(default_tag="fcmae_ft_in22k_in1k_384")
def convnextv2_huge():
    cfg = ConvNeXtConfig(
        name="convnextv2_huge",
        embed_dim=(352, 704, 1408, 2816),
        nb_blocks=(3, 3, 27, 3),
        use_grn=True,
        init_scale=None,
    )
    return ConvNeXt, cfg


def _meta(**kwargs):
    """Default metadata for ConvNeXt models."""
    return {
        "crop_pct": 0.875,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        **kwargs,
    }


def _metav2(**kwargs):
    """Default metadata for ConvNeXt-V2 models."""
    return {
        "crop_pct": 0.875,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "license": "cc-by-nc-4.0",
        "paper_ids": "arXiv:2301.00808",
        "paper_name": "ConvNeXt-V2: Co-designing and Scaling ConvNets with Masked Autoencoders",  # noqa: E501
        "origin_url": "https://github.com/facebookresearch/ConvNeXt-V2",
        **kwargs,
    }


# timm specific variants
register_model_tag(
    model_name="convnext_tiny.in12k_ft_in1k",
    url="[timm]",
    metadata=_meta(crop_pct=0.95, test_input_size=(288, 288), test_crop_pct=1.0),
)
register_model_tag(
    model_name="convnext_small.in12k_ft_in1k",
    url="[timm]",
    metadata=_meta(crop_pct=0.95, test_input_size=(288, 288), test_crop_pct=1.0),
)

register_model_tag(
    model_name="convnext_atto.d2_in1k",
    url="[timm]",
    metadata=_meta(crop_pct=0.95, test_input_size=(288, 288)),
)
register_model_tag(
    model_name="convnext_atto_ols.a2_in1k",
    url="[timm]",
    metadata=_meta(crop_pct=0.95, test_input_size=(288, 288)),
)
register_model_tag(
    model_name="convnext_femto.d1_in1k",
    url="[timm]",
    metadata=_meta(crop_pct=0.95, test_input_size=(288, 288)),
)
register_model_tag(
    model_name="convnext_femto_ols.d1_in1k",
    url="[timm]",
    metadata=_meta(crop_pct=0.95, test_input_size=(288, 288)),
)
register_model_tag(
    model_name="convnext_pico.d1_in1k",
    url="[timm]",
    metadata=_meta(crop_pct=0.95, test_input_size=(288, 288)),
)
register_model_tag(
    model_name="convnext_pico_ols.d1_in1k",
    url="[timm]",
    metadata=_meta(crop_pct=0.95, test_input_size=(288, 288), test_crop_pct=1.0),
)
register_model_tag(
    model_name="convnext_nano.in12k_ft_in1k",
    url="[timm]",
    metadata=_meta(crop_pct=0.95, test_input_size=(288, 288), test_crop_pct=1.0),
)
register_model_tag(
    model_name="convnext_nano.d1h_in1k",
    url="[timm]",
    metadata=_meta(crop_pct=0.95, test_input_size=(288, 288), test_crop_pct=1.0),
)
register_model_tag(
    model_name="convnext_nano_ols.d1h_in1k",
    url="[timm]",
    metadata=_meta(crop_pct=0.95, test_input_size=(288, 288), test_crop_pct=1.0),
)
register_model_tag(
    model_name="convnext_tiny_hnf.a2h_in1k",
    url="[timm]",
    metadata=_meta(crop_pct=0.95, test_input_size=(288, 288), test_crop_pct=1.0),
)

register_model_tag(
    model_name="convnext_tiny.in12k_ft_in1k_384",
    url="[timm]",
    metadata=_meta(crop_pct=1.0, test_input_size=(384, 384), crop_mode="squash"),
)
register_model_tag(
    model_name="convnext_small.in12k_ft_in1k_384",
    url="[timm]",
    metadata=_meta(crop_pct=1.0, test_input_size=(384, 384), crop_mode="squash"),
)

register_model_tag(
    model_name="convnext_nano.in12k",
    url="[timm]",
    cfg=dict(nb_classes=11821),
    metadata=_meta(crop_pct=0.95),
)
register_model_tag(
    model_name="convnext_tiny.in12k",
    url="[timm]",
    cfg=dict(nb_classes=11821),
    metadata=_meta(crop_pct=0.95),
)
register_model_tag(
    model_name="convnext_small.in12k",
    url="[timm]",
    cfg=dict(nb_classes=11821),
    metadata=_meta(crop_pct=0.95),
)

register_model_tag(
    model_name="convnext_tiny.fb_in22k_ft_in1k",
    url="[timm]",
    metadata=_meta(test_input_size=(288, 288), test_crop_pct=1.0),
)
register_model_tag(
    model_name="convnext_small.fb_in22k_ft_in1k",
    url="[timm]",
    metadata=_meta(test_input_size=(288, 288), test_crop_pct=1.0),
)
register_model_tag(
    model_name="convnext_base.fb_in22k_ft_in1k",
    url="[timm]",
    metadata=_meta(test_input_size=(288, 288), test_crop_pct=1.0),
)
register_model_tag(
    model_name="convnext_large.fb_in22k_ft_in1k",
    url="[timm]",
    metadata=_meta(test_input_size=(288, 288), test_crop_pct=1.0),
)
register_model_tag(
    model_name="convnext_xlarge.fb_in22k_ft_in1k",
    url="[timm]",
    metadata=_meta(test_input_size=(288, 288), test_crop_pct=1.0),
)

register_model_tag(
    model_name="convnext_tiny.fb_in1k",
    url="[timm]",
    metadata=_meta(test_input_size=(288, 288), test_crop_pct=1.0),
)
register_model_tag(
    model_name="convnext_small.fb_in1k",
    url="[timm]",
    metadata=_meta(test_input_size=(288, 288), test_crop_pct=1.0),
)
register_model_tag(
    model_name="convnext_base.fb_in1k",
    url="[timm]",
    metadata=_meta(test_input_size=(288, 288), test_crop_pct=1.0),
)
register_model_tag(
    model_name="convnext_large.fb_in1k",
    url="[timm]",
    metadata=_meta(test_input_size=(288, 288), test_crop_pct=1.0),
)

register_model_tag(
    model_name="convnext_tiny.fb_in22k_ft_in1k_384",
    url="[timm]",
    metadata=_meta(test_input_size=(384, 384), test_crop_pct=1.0, crop_mode="squash"),
)
register_model_tag(
    model_name="convnext_small.fb_in22k_ft_in1k_384",
    url="[timm]",
    metadata=_meta(test_input_size=(384, 384), test_crop_pct=1.0, crop_mode="squash"),
)
register_model_tag(
    model_name="convnext_base.fb_in22k_ft_in1k_384",
    url="[timm]",
    metadata=_meta(test_input_size=(384, 384), test_crop_pct=1.0, crop_mode="squash"),
)
register_model_tag(
    model_name="convnext_large.fb_in22k_ft_in1k_384",
    url="[timm]",
    metadata=_meta(test_input_size=(384, 384), test_crop_pct=1.0, crop_mode="squash"),
)
register_model_tag(
    model_name="convnext_xlarge.fb_in22k_ft_in1k_384",
    url="[timm]",
    metadata=_meta(test_input_size=(384, 384), test_crop_pct=1.0, crop_mode="squash"),
)

register_model_tag(
    model_name="convnext_tiny.fb_in22k",
    url="[timm]",
    cfg=dict(nb_classes=21841),
    metadata=_meta(),
)
register_model_tag(
    model_name="convnext_small.fb_in22k",
    url="[timm]",
    cfg=dict(nb_classes=21841),
    metadata=_meta(),
)
register_model_tag(
    model_name="convnext_base.fb_in22k",
    url="[timm]",
    cfg=dict(nb_classes=21841),
    metadata=_meta(),
)
register_model_tag(
    model_name="convnext_large.fb_in22k",
    url="[timm]",
    cfg=dict(nb_classes=21841),
    metadata=_meta(),
)
register_model_tag(
    model_name="convnext_xlarge.fb_in22k",
    url="[timm]",
    cfg=dict(nb_classes=21841),
    metadata=_meta(),
)

register_model_tag(
    model_name="convnextv2_nano.fcmae_ft_in22k_in1k",
    url="[timm]",
    metadata=_metav2(test_input_size=(288, 288), test_crop_pct=1.0),
)
register_model_tag(
    model_name="convnextv2_tiny.fcmae_ft_in22k_in1k",
    url="[timm]",
    metadata=_metav2(test_input_size=(288, 288), test_crop_pct=1.0),
)
register_model_tag(
    model_name="convnextv2_base.fcmae_ft_in22k_in1k",
    url="[timm]",
    metadata=_metav2(test_input_size=(288, 288), test_crop_pct=1.0),
)
register_model_tag(
    model_name="convnextv2_large.fcmae_ft_in22k_in1k",
    url="[timm]",
    metadata=_metav2(test_input_size=(288, 288), test_crop_pct=1.0),
)

register_model_tag(
    model_name="convnextv2_nano.fcmae_ft_in22k_in1k_384",
    url="[timm]",
    metadata=_metav2(test_input_size=(384, 384), crop_pct=1.0, crop_mode="squash"),
)
register_model_tag(
    model_name="convnextv2_tiny.fcmae_ft_in22k_in1k_384",
    url="[timm]",
    metadata=_metav2(test_input_size=(384, 384), crop_pct=1.0, crop_mode="squash"),
)
register_model_tag(
    model_name="convnextv2_base.fcmae_ft_in22k_in1k_384",
    url="[timm]",
    metadata=_metav2(test_input_size=(384, 384), crop_pct=1.0, crop_mode="squash"),
)
register_model_tag(
    model_name="convnextv2_large.fcmae_ft_in22k_in1k_384",
    url="[timm]",
    metadata=_metav2(test_input_size=(384, 384), crop_pct=1.0, crop_mode="squash"),
)
register_model_tag(
    model_name="convnextv2_huge.fcmae_ft_in22k_in1k_384",
    url="[timm]",
    metadata=_metav2(test_input_size=(384, 384), crop_pct=1.0, crop_mode="squash"),
)

register_model_tag(
    model_name="convnextv2_huge.fcmae_ft_in22k_in1k_512",
    url="[timm]",
    metadata=_metav2(test_input_size=(512, 512), crop_pct=1.0, crop_mode="squash"),
)

register_model_tag(
    model_name="convnextv2_atto.fcmae_ft_in1k",
    url="[timm]",
    metadata=_metav2(test_input_size=(288, 288), test_crop_pct=0.95),
)
register_model_tag(
    model_name="convnextv2_femto.fcmae_ft_in1k",
    url="[timm]",
    metadata=_metav2(test_input_size=(288, 288), test_crop_pct=0.95),
)
register_model_tag(
    model_name="convnextv2_pico.fcmae_ft_in1k",
    url="[timm]",
    metadata=_metav2(test_input_size=(288, 288), test_crop_pct=0.95),
)
register_model_tag(
    model_name="convnextv2_nano.fcmae_ft_in1k",
    url="[timm]",
    metadata=_metav2(test_input_size=(288, 288), test_crop_pct=1.0),
)
register_model_tag(
    model_name="convnextv2_tiny.fcmae_ft_in1k",
    url="[timm]",
    metadata=_metav2(test_input_size=(288, 288), test_crop_pct=1.0),
)
register_model_tag(
    model_name="convnextv2_base.fcmae_ft_in1k",
    url="[timm]",
    metadata=_metav2(test_input_size=(288, 288), test_crop_pct=1.0),
)
register_model_tag(
    model_name="convnextv2_large.fcmae_ft_in1k",
    url="[timm]",
    metadata=_metav2(test_input_size=(288, 288), test_crop_pct=1.0),
)
register_model_tag(
    model_name="convnextv2_huge.fcmae_ft_in1k",
    url="[timm]",
    metadata=_metav2(test_input_size=(288, 288), test_crop_pct=1.0),
)

register_model_tag(
    model_name="convnextv2_atto.fcmae",
    url="[timm]",
    cfg=dict(nb_classes=0),
    metadata=_metav2(),
)
register_model_tag(
    model_name="convnextv2_femto.fcmae",
    url="[timm]",
    cfg=dict(nb_classes=0),
    metadata=_metav2(),
)
register_model_tag(
    model_name="convnextv2_pico.fcmae",
    url="[timm]",
    cfg=dict(nb_classes=0),
    metadata=_metav2(),
)
register_model_tag(
    model_name="convnextv2_nano.fcmae",
    url="[timm]",
    cfg=dict(nb_classes=0),
    metadata=_metav2(),
)
register_model_tag(
    model_name="convnextv2_tiny.fcmae",
    url="[timm]",
    cfg=dict(nb_classes=0),
    metadata=_metav2(),
)
register_model_tag(
    model_name="convnextv2_base.fcmae",
    url="[timm]",
    cfg=dict(nb_classes=0),
    metadata=_metav2(),
)
register_model_tag(
    model_name="convnextv2_large.fcmae",
    url="[timm]",
    cfg=dict(nb_classes=0),
    metadata=_metav2(),
)
register_model_tag(
    model_name="convnextv2_huge.fcmae",
    url="[timm]",
    cfg=dict(nb_classes=0),
    metadata=_metav2(),
)

# CLIP weights, fine-tuned on in1k or in12k + in1k
register_model_tag(
    model_name="convnext_base.clip_laion2b_augreg_ft_in12k_in1k",
    url="[timm]",
    metadata=_meta(
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, input_size=(256, 256), crop_pct=1.0
    ),
)
register_model_tag(
    model_name="convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384",
    url="[timm]",
    metadata=_meta(
        mean=OPENAI_CLIP_MEAN,
        std=OPENAI_CLIP_STD,
        input_size=(384, 384),
        crop_pct=1.0,
        crop_mode="squash",
    ),
)
register_model_tag(
    model_name="convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_320",
    url="[timm]",
    metadata=_meta(
        mean=OPENAI_CLIP_MEAN,
        std=OPENAI_CLIP_STD,
        input_size=(320, 320),
        crop_pct=1.0,
    ),
)
register_model_tag(
    model_name="convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384",
    url="[timm]",
    metadata=_meta(
        mean=OPENAI_CLIP_MEAN,
        std=OPENAI_CLIP_STD,
        input_size=(384, 384),
        crop_pct=1.0,
        crop_mode="squash",
    ),
)

register_model_tag(
    model_name="convnext_base.clip_laion2b_augreg_ft_in1k",
    url="[timm]",
    metadata=_meta(
        mean=OPENAI_CLIP_MEAN,
        std=OPENAI_CLIP_STD,
        input_size=(256, 256),
        crop_pct=1.0,
    ),
)
register_model_tag(
    model_name="convnext_base.clip_laiona_augreg_ft_in1k_384",
    url="[timm]",
    metadata=_meta(
        mean=OPENAI_CLIP_MEAN,
        std=OPENAI_CLIP_STD,
        input_size=(384, 384),
        crop_pct=1.0,
    ),
)
register_model_tag(
    model_name="convnext_large_mlp.clip_laion2b_augreg_ft_in1k",
    url="[timm]",
    metadata=_meta(
        mean=OPENAI_CLIP_MEAN,
        std=OPENAI_CLIP_STD,
        input_size=(256, 256),
        crop_pct=1.0,
    ),
)
register_model_tag(
    model_name="convnext_large_mlp.clip_laion2b_augreg_ft_in1k_384",
    url="[timm]",
    metadata=_meta(
        mean=OPENAI_CLIP_MEAN,
        std=OPENAI_CLIP_STD,
        input_size=(384, 384),
        crop_pct=1.0,
        crop_mode="squash",
    ),
)
register_model_tag(
    model_name="convnext_xxlarge.clip_laion2b_soup_ft_in1k",
    url="[timm]",
    metadata=_meta(
        mean=OPENAI_CLIP_MEAN,
        std=OPENAI_CLIP_STD,
        input_size=(256, 256),
        crop_pct=1.0,
    ),
)

register_deprecation("convnext_tiny_in22ft1k", "convnext_tiny.fb_in22k_ft_in1k")
register_deprecation("convnext_small_in22ft1k", "convnext_small.fb_in22k_ft_in1k")
register_deprecation("convnext_base_in22ft1k", "convnext_base.fb_in22k_ft_in1k")
register_deprecation("convnext_large_in22ft1k", "convnext_large.fb_in22k_ft_in1k")
register_deprecation("convnext_xlarge_in22ft1k", "convnext_xlarge.fb_in22k_ft_in1k")

# fmt: off
register_deprecation("convnext_tiny_384_in22ft1k", "convnext_tiny.fb_in22k_ft_in1k_384")
register_deprecation("convnext_small_384_in22ft1k", "convnext_small.fb_in22k_ft_in1k_384")  # noqa: E501
register_deprecation("convnext_base_384_in22ft1k", "convnext_base.fb_in22k_ft_in1k_384")
register_deprecation("convnext_large_384_in22ft1k", "convnext_large.fb_in22k_ft_in1k_384")  # noqa: E501
register_deprecation("convnext_xlarge_384_in22ft1k", "convnext_xlarge.fb_in22k_ft_in1k_384")  # noqa: E501
# fmt: on

register_deprecation("convnext_tiny_in22k", "convnext_tiny.fb_in22k")
register_deprecation("convnext_small_in22k", "convnext_small.fb_in22k")
register_deprecation("convnext_base_in22k", "convnext_base.fb_in22k")
register_deprecation("convnext_large_in22k", "convnext_large.fb_in22k")
register_deprecation("convnext_xlarge_in22k", "convnext_xlarge.fb_in22k")
