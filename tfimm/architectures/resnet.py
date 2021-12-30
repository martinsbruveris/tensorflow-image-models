"""
TensorFlow implementation of ResNets

Based on timm/models/resnet.py by Ross Wightman.

It includes the following models:
- Resnets from the PyTorch model hub
- ResNets trained by Ross Wightman
- Models pretrained on weakly-supervised data, finetuned on ImageNet
  Paper: Exploring the Limits of Weakly Supervised Pretraining
  Link: https://arxiv.org/abs/1805.00932
- Models pretrained in semi-supervised way on YFCC100M, finetuned on ImageNet
  Paper: Billion-scale Semi-Supervised Learning for Image Classification
  Link: https://arxiv.org/abs/1905.00546
- ResNets with ECA layers
  Paper: ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
  Link: https://arxiv.org/pdf/1910.03151.pdf
- ResNets with anti-aliasing layers
  Paper: Making Convolutional Networks Shift-Invariant Again
  Link: https://arxiv.org/pdf/1904.11486.pdf
- Models from the "Revisiting ResNets" paper
  Paper: Revisiting ResNets
  Link: https://arxiv.org/abs/2103.07579
- ResNets with a squeeze-and-excitation layer
  Paper: Squeeze-and-Excitation Networks
  Link: https://arxiv.org/abs/1709.01507

Copyright 2021 Martins Bruveris
Copyright 2021 Ross Wightman
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import tensorflow as tf

from tfimm.layers import (
    BlurPool2D,
    ClassifierHead,
    DropPath,
    act_layer_factory,
    attn_layer_factory,
    norm_layer_factory,
)
from tfimm.models import ModelConfig, keras_serializable, register_model
from tfimm.utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# model_registry will add each entrypoint fn to this
__all__ = ["ResNet", "ResNetConfig", "BasicBlock"]


# TODO: Implement DropBlock, drop_block_rate in timm
#       See: https://arxiv.org/pdf/1810.12890.pdf)
@dataclass
class ResNetConfig(ModelConfig):
    nb_classes: int = 1000
    in_channels: int = 3
    input_size: Tuple[int, int] = (224, 224)
    # Residual blocks
    block: str = "basic_block"
    nb_blocks: Tuple = (2, 2, 2, 2)
    nb_channels: Tuple = (64, 128, 256, 512)
    cardinality: int = 1  # Number of groups in bottleneck conv
    base_width: int = 64  # Determines number of channels in block
    downsample_mode: str = "conv"
    zero_init_last_bn: bool = True
    # Stem
    stem_width: int = 64
    stem_type: str = ""
    replace_stem_pool: bool = False
    # Other params
    block_reduce_first: int = 1
    down_kernel_size: int = 1
    act_layer: str = "relu"
    norm_layer: str = "batch_norm"
    aa_layer: str = ""
    attn_layer: str = ""
    se_ratio: float = 0.0625
    # Regularization
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    # Head
    global_pool: str = "avg"
    # Parameters for inference
    test_input_size: Optional[Tuple[int, int]] = None
    pool_size: int = 7  # For test-time pooling (not implemented yet)
    crop_pct: float = 0.875
    interpolation: str = "bilinear"
    # Preprocessing
    mean: Tuple[float, float, float] = IMAGENET_DEFAULT_MEAN
    std: Tuple[float, float, float] = IMAGENET_DEFAULT_STD
    # Weight transfer
    first_conv: str = "conv1"
    classifier: str = "fc"

    def __post_init__(self):
        if self.test_input_size is None:
            self.test_input_size = self.input_size


class BasicBlock(tf.keras.layers.Layer):
    expansion = 1

    def __init__(
        self,
        cfg: ResNetConfig,
        nb_channels: int,
        stride: int,
        drop_path_rate: float,
        downsample_layer,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert cfg.cardinality == 1, "BasicBlock only supports cardinality of 1"
        assert cfg.base_width == 64, "BasicBlock does not support changing base width"

        self.cfg = cfg
        self.downsample_layer = downsample_layer
        self.act_layer = act_layer_factory(cfg.act_layer)
        self.norm_layer = norm_layer_factory(cfg.norm_layer)
        self.attn_layer = attn_layer_factory(cfg.attn_layer)

        # Num channels after first conv
        first_planes = nb_channels // cfg.block_reduce_first
        out_planes = nb_channels * self.expansion  # Num channels after second conv
        use_aa = cfg.aa_layer and stride == 2

        self.pad1 = tf.keras.layers.ZeroPadding2D(padding=1)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=first_planes,
            kernel_size=3,
            # If we use anti-aliasing, the anti-aliasing layer takes care of strides
            strides=1 if use_aa else stride,
            use_bias=False,
            name="conv1",
        )
        self.bn1 = self.norm_layer(name="bn1")
        self.act1 = self.act_layer()
        self.aa = BlurPool2D(stride=stride) if use_aa else None

        self.pad2 = tf.keras.layers.ZeroPadding2D(padding=1)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=out_planes,
            kernel_size=3,
            use_bias=False,
            name="conv2",
        )
        initializer = "zeros" if cfg.zero_init_last_bn else "ones"
        if cfg.norm_layer == "batch_norm":
            # Only batch norm layer has moving_variance_initializer parameter
            self.bn2 = self.norm_layer(
                gamma_initializer=initializer,
                moving_variance_initializer=initializer,
                name="bn2",
            )
        else:
            self.bn2 = self.norm_layer(gamma_initializer=initializer, name="bn2")
        if cfg.attn_layer == "se":
            self.se = self.attn_layer(rd_ratio=cfg.se_ratio, name="se")
        else:
            self.se = self.attn_layer(name="se")
        self.drop_path = DropPath(drop_prob=drop_path_rate)
        self.act2 = self.act_layer()

    def call(self, x, training=False):
        shortcut = x

        x = self.pad1(x)
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        if self.aa is not None:
            x = self.aa(x, training=training)

        x = self.pad2(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x = self.se(x, training=training)

        x = self.drop_path(x, training=training)

        if self.downsample_layer is not None:
            shortcut = self.downsample_layer(shortcut, training=training)
        x += shortcut
        x = self.act2(x)

        return x


class Bottleneck(tf.keras.layers.Layer):
    expansion = 4

    def __init__(
        self,
        cfg: ResNetConfig,
        nb_channels: int,
        stride: int,
        drop_path_rate: float,
        downsample_layer,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.cfg = cfg
        self.downsample_layer = downsample_layer
        self.act_layer = act_layer_factory(cfg.act_layer)
        self.norm_layer = norm_layer_factory(cfg.norm_layer)
        self.attn_layer = attn_layer_factory(cfg.attn_layer)

        # Number of channels after second convolution
        width = int(math.floor(nb_channels * (cfg.base_width / 64)) * cfg.cardinality)
        # Number of channels after first convolution
        first_planes = width // cfg.block_reduce_first
        # Number of channels after third convolution
        out_planes = nb_channels * self.expansion
        use_aa = cfg.aa_layer and stride == 2

        self.conv1 = tf.keras.layers.Conv2D(
            filters=first_planes,
            kernel_size=1,
            use_bias=False,
            name="conv1",
        )
        self.bn1 = self.norm_layer(name="bn1")
        self.act1 = self.act_layer()

        self.pad2 = tf.keras.layers.ZeroPadding2D(padding=1)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=width,
            kernel_size=3,
            # If we use anti-aliasing, the anti-aliasing layer takes care of strides
            strides=1 if use_aa else stride,
            groups=cfg.cardinality,
            use_bias=False,
            name="conv2",
        )
        self.bn2 = self.norm_layer(name="bn2")
        self.act2 = self.act_layer()
        self.aa = BlurPool2D(stride=stride) if use_aa else None

        self.conv3 = tf.keras.layers.Conv2D(
            filters=out_planes,
            kernel_size=1,
            use_bias=False,
            name="conv3",
        )
        initializer = "zeros" if cfg.zero_init_last_bn else "ones"
        if cfg.norm_layer == "batch_norm":
            # Only batch norm layer has moving_variance_initializer parameter
            self.bn3 = self.norm_layer(
                gamma_initializer=initializer,
                moving_variance_initializer=initializer,
                name="bn3",
            )
        else:
            self.bn3 = self.norm_layer(gamma_initializer=initializer, name="bn3")
        if cfg.attn_layer == "se":
            self.se = self.attn_layer(rd_ratio=cfg.se_ratio, name="se")
        else:
            self.se = self.attn_layer(name="se")
        self.drop_path = DropPath(drop_prob=drop_path_rate)
        self.act3 = self.act_layer()

    def call(self, x, training=False):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)

        x = self.pad2(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)
        if self.aa is not None:
            x = self.aa(x, training=training)

        x = self.conv3(x)
        x = self.bn3(x, training=training)

        x = self.se(x, training=training)

        x = self.drop_path(x, training=training)

        if self.downsample_layer is not None:
            shortcut = self.downsample_layer(shortcut, training=training)
        x += shortcut
        x = self.act3(x)

        return x


def downsample_avg(cfg: ResNetConfig, out_channels: int, stride: int, name: str):
    norm_layer = norm_layer_factory(cfg.norm_layer)

    if stride != 1:
        pool = tf.keras.layers.AveragePooling2D(
            pool_size=2, strides=stride, padding="same"
        )
    else:
        pool = tf.keras.layers.Activation("linear")
    conv = tf.keras.layers.Conv2D(
        filters=out_channels,
        kernel_size=1,
        strides=1,
        use_bias=False,
        name=name + "/downsample/1",
    )
    bn = norm_layer(name=name + "/downsample/2")
    return tf.keras.Sequential([pool, conv, bn])


def downsample_conv(cfg: ResNetConfig, out_channels: int, stride: int, name: str):
    norm_layer = norm_layer_factory(cfg.norm_layer)

    # This layer is part of the conv layer in pytorch and so is not being tracked here
    p = (stride + cfg.down_kernel_size) // 2 - 1
    pad = tf.keras.layers.ZeroPadding2D(padding=p)

    conv = tf.keras.layers.Conv2D(
        filters=out_channels,
        kernel_size=cfg.down_kernel_size,
        strides=stride,
        use_bias=False,
        name=name + "/downsample/0",
    )
    bn = norm_layer(name=name + "/downsample/1")
    return tf.keras.Sequential([pad, conv, bn])


def make_stage(
    idx: int,
    in_channels: int,
    cfg: ResNetConfig,
    name: str,
):
    stage_name = f"layer{idx + 1}"  # Weight compatibility requires this name

    assert cfg.block in {"basic_block", "bottleneck"}
    block_cls = BasicBlock if cfg.block == "basic_block" else Bottleneck
    nb_blocks = cfg.nb_blocks[idx]
    nb_channels = cfg.nb_channels[idx]
    # The actual number of channels after the block. Not the same as nb_channels,
    # because Bottleneck blocks have an expansion factor = 4.
    out_channels = nb_channels * block_cls.expansion

    assert cfg.downsample_mode in {"avg", "conv"}
    downsample_fn = downsample_avg if cfg.downsample_mode == "avg" else downsample_conv

    # We need to know the absolute number of blocks to set stochastic depth decay
    total_nb_blocks = sum(cfg.nb_blocks)
    total_block_idx = sum(cfg.nb_blocks[:idx])

    blocks = []
    for block_idx in range(nb_blocks):
        stride = 1 if idx == 0 or block_idx > 0 else 2
        if (block_idx == 0) and (stride != 1 or in_channels != out_channels):
            downsample_layer = downsample_fn(
                cfg, out_channels, stride, name=f"{name}/{stage_name}/0"
            )
        else:
            downsample_layer = None

        # Stochastic depth linear decay rule
        block_dpr = cfg.drop_path_rate * total_block_idx / (total_nb_blocks - 1)

        blocks.append(
            block_cls(
                cfg,
                nb_channels=nb_channels,
                stride=stride,
                downsample_layer=downsample_layer,
                drop_path_rate=block_dpr,
                name=f"{stage_name}/{block_idx}",
            )
        )

        in_channels = nb_channels
        total_block_idx += 1
    return blocks, in_channels


@keras_serializable
class ResNet(tf.keras.Model):
    """
    ResNet / ResNeXt / SE-ResNeXt / SE-Net

    This class implements various ResNet versions.

    Parameters
    ----------
    nb_classes : int, default 1000
        Number of classification classes.
    in_channels : int, default 3
        Number of input (color) channels.
    input_size: Tuple[int, int], default (224, 224)
        Input image size
    block : str
        Which residual block to use. One of {"basic_block", "bottleneck"}
    nb_blocks : Tuple of int
        Numbers of blocks in each stage
    nb_channels: Tuple of int, default (64, 128, 256, 512)
        Number of channels in each stage
    cardinality : int, default 1
        Number of convolution groups for 3x3 conv in Bottleneck.
    base_width : int, default 64
        Factor determining bottleneck channels, via formula
            `channels * base_width / 64 * cardinality`
    downsample_mode: str, default "conv"
        How to downsample in the skip connection between stages. One of {"conv", "avg"}
    zero_init_last_bn: bool, default True
        Should last BN layer in each residual block be zero-initialized
    stem_width : int, default 64
        Number of channels in stem convolutions
    stem_type : str, default ""
        The type of stem:
          - "", default - a single 7x7 conv with a width of stem_width
          - "deep" - three 3x3 convolution layers of widths stem_width, stem_width,
              stem_width * 2
          - "deep_tiered" - three 3x3 conv layers of widths stem_width // 4 * 3,
              stem_width, stem_width * 2
    replace_stem_pool: bool, default False
        Replace maxpooling in stem by conv+bn+act layers
    block_reduce_first: int, default 1
        Reduction factor for first convolution output width of residual blocks,
        1 for all archs at the moment (it is to for a SE-Net in timm that is not
        implemented here).
    down_kernel_size: int, default 1
        Kernel size of residual block downsampling path, 1x1 for all archs (it is 3x3
        for a SE-Net in timm that is not implemented here)
    act_layer : str, default "relu"
        Activation layer, see `act_layer_factory` for allowed values
    norm_layer : str, default "batch_norm"
        Normalization layer, see `norm_layer_factory` for allowed values
    aa_layer : str, default ""
        Anti-aliasing layer. Can be "" (no AA-layer) or "blur_pool".
    attn_layer: str, default ""
        Attention layer, see `attn_layer_factory` for allowed values.
    se_ratio: float, default 1. / 16 = 0.0625
        Reduction ratio in SE blocks. Ignored, if attn_layer != "se".
    drop_rate : float, default 0.
        Dropout probability before classifier, for training.
    drop_path_rate: float, default 0.
        Dropout probability for DropPath layers, for training.
    global_pool : str, default "avg"
        Global pooling type. One of "avg", "max"
    """

    cfg_class = ResNetConfig

    # The blur_kernel parameter in BlurPool2D layers is non-trainable and the value
    # is set during layer construction. The pycharm weights are not stored, hence we
    # cannot load them
    keys_to_ignore_on_load_missing = ["blur_kernel"]

    def __init__(self, cfg: ResNetConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg

        self.act_layer = act_layer_factory(cfg.act_layer)
        self.norm_layer = norm_layer_factory(cfg.norm_layer)

        if cfg.stem_type in {"deep", "deep_tiered"}:
            in_channels = cfg.stem_width * 2
            if cfg.stem_type == "deep_tiered":
                stem_chns = (3 * (cfg.stem_width // 4), cfg.stem_width)
            else:
                stem_chns = (cfg.stem_width, cfg.stem_width)
            self.pad1 = tf.keras.layers.ZeroPadding2D(padding=1)
            conv1_0 = tf.keras.layers.Conv2D(
                filters=stem_chns[0],
                kernel_size=3,
                strides=2,
                use_bias=False,
                name=f"{self.name}/conv1/0",
            )
            bn1_0 = self.norm_layer(name=f"{self.name}/conv1/1")
            act1_0 = self.act_layer()
            conv1_1 = tf.keras.layers.Conv2D(
                filters=stem_chns[1],
                kernel_size=3,
                padding="same",
                use_bias=False,
                name=f"{self.name}/conv1/3",
            )
            bn1_1 = self.norm_layer(name=f"{self.name}/conv1/4")
            act1_1 = self.act_layer()
            conv1_2 = tf.keras.layers.Conv2D(
                filters=in_channels,
                kernel_size=3,
                padding="same",
                use_bias=False,
                name=f"{self.name}/conv1/6",
            )
            self.conv1 = tf.keras.Sequential(
                [conv1_0, bn1_0, act1_0, conv1_1, bn1_1, act1_1, conv1_2]
            )
        else:
            in_channels = 64
            # In TF "same" padding with strides != 1 is not the same as (3, 3) padding
            # in pytorch, hence the need for an explicit padding layer
            self.pad1 = tf.keras.layers.ZeroPadding2D(padding=3)
            self.conv1 = tf.keras.layers.Conv2D(
                filters=in_channels,
                kernel_size=7,
                strides=2,
                use_bias=False,
                name="conv1",
            )
        self.bn1 = self.norm_layer(name="bn1")
        self.act1 = self.act_layer()

        # Stem Pooling
        if cfg.replace_stem_pool:
            # Note that if replace_stem_pool=True, we are ignoring the aa_layer
            # None of the timm models use both.
            pad = tf.keras.layers.ZeroPadding2D(padding=1)
            conv = tf.keras.layers.Conv2D(
                filters=in_channels,
                kernel_size=3,
                strides=2,
                use_bias=False,
                name=f"{self.name}/maxpool/0",
            )
            bn = self.norm_layer(name=f"{self.name}/maxpool/1")
            act = self.act_layer()
            self.maxpool = tf.keras.Sequential([pad, conv, bn, act])
        else:
            if cfg.aa_layer:
                pad = tf.keras.layers.ZeroPadding2D(padding=1)
                pool = tf.keras.layers.MaxPool2D(pool_size=3, strides=1)
                aa = BlurPool2D(stride=2, name=f"{self.name}/maxpool/2")
                self.maxpool = tf.keras.Sequential([pad, pool, aa])
            else:
                pad = tf.keras.layers.ZeroPadding2D(padding=1)
                pool = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)
                self.maxpool = tf.keras.Sequential([pad, pool])

        self.blocks = []
        for idx in range(4):
            stage_blocks, in_channels = make_stage(
                idx=idx, in_channels=in_channels, cfg=cfg, name=self.name
            )
            self.blocks.extend(stage_blocks)

        # Head (pooling and classifier)
        self.head = ClassifierHead(
            nb_classes=cfg.nb_classes,
            pool_type=cfg.global_pool,
            drop_rate=cfg.drop_rate,
            use_conv=False,
            name="remove",
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
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.maxpool(x)
        features["stem"] = x

        for j, block in enumerate(self.blocks):
            x = block(x, training=training)
            features[f"block_{j}"] = x
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
def resnet18():
    """Constructs a ResNet-18 model."""
    cfg = ResNetConfig(
        name="resnet18", url="", block="basic_block", nb_blocks=(2, 2, 2, 2)
    )
    return ResNet, cfg


@register_model
def resnet18d():
    """Constructs a ResNet-18-D model."""
    cfg = ResNetConfig(
        name="resnet18d",
        url="",
        block="basic_block",
        nb_blocks=(2, 2, 2, 2),
        stem_width=32,
        stem_type="deep",
        downsample_mode="avg",
        interpolation="bicubic",
        first_conv="conv1/0",
    )
    return ResNet, cfg


@register_model
def resnet26():
    """Constructs a ResNet-26 model."""
    cfg = ResNetConfig(
        name="resnet26",
        url="",
        block="bottleneck",
        nb_blocks=(2, 2, 2, 2),
        interpolation="bicubic",
    )
    return ResNet, cfg


@register_model
def resnet26d():
    """Constructs a ResNet-26-D model."""
    cfg = ResNetConfig(
        name="resnet26d",
        url="",
        block="bottleneck",
        nb_blocks=(2, 2, 2, 2),
        stem_width=32,
        stem_type="deep",
        downsample_mode="avg",
        interpolation="bicubic",
        first_conv="conv1/0",
    )
    return ResNet, cfg


@register_model
def resnet26t():
    """Constructs a ResNet-26-T model."""
    cfg = ResNetConfig(
        name="resnet26t",
        url="",
        input_size=(256, 256),
        block="bottleneck",
        nb_blocks=(2, 2, 2, 2),
        pool_size=8,
        stem_width=32,
        stem_type="deep_tiered",
        downsample_mode="avg",
        crop_pct=0.94,
        interpolation="bicubic",
        first_conv="conv1/0",
    )
    return ResNet, cfg


@register_model
def resnet34():
    """Constructs a ResNet-34 model."""
    cfg = ResNetConfig(
        name="resnet34", url="", block="basic_block", nb_blocks=(3, 4, 6, 3)
    )
    return ResNet, cfg


@register_model
def resnet34d():
    """Constructs a ResNet-34-D model."""
    cfg = ResNetConfig(
        name="resnet34d",
        url="",
        block="basic_block",
        nb_blocks=(3, 4, 6, 3),
        stem_width=32,
        stem_type="deep",
        downsample_mode="avg",
        interpolation="bicubic",
        first_conv="conv1/0",
    )
    return ResNet, cfg


@register_model
def resnet50():
    """Constructs a ResNet-50 model."""
    cfg = ResNetConfig(
        name="resnet50",
        url="",
        block="bottleneck",
        nb_blocks=(3, 4, 6, 3),
        interpolation="bicubic",
        crop_pct=0.95,
    )
    return ResNet, cfg


@register_model
def resnet50d():
    """Constructs a ResNet-50-D model."""
    cfg = ResNetConfig(
        name="resnet50d",
        url="",
        block="bottleneck",
        nb_blocks=(3, 4, 6, 3),
        stem_width=32,
        stem_type="deep",
        downsample_mode="avg",
        interpolation="bicubic",
        first_conv="conv1/0",
    )
    return ResNet, cfg


@register_model
def resnet101():
    """Constructs a ResNet-101 model."""
    cfg = ResNetConfig(
        name="resnet101",
        url="",
        block="bottleneck",
        nb_blocks=(3, 4, 23, 3),
        interpolation="bicubic",
        crop_pct=0.95,
    )
    return ResNet, cfg


@register_model
def resnet101d():
    """Constructs a ResNet-101-D model."""
    cfg = ResNetConfig(
        name="resnet101d",
        url="",
        input_size=(256, 256),
        block="bottleneck",
        nb_blocks=(3, 4, 23, 3),
        pool_size=8,
        stem_width=32,
        stem_type="deep",
        downsample_mode="avg",
        test_input_size=(320, 320),
        interpolation="bicubic",
        crop_pct=1.0,
        first_conv="conv1/0",
    )
    return ResNet, cfg


@register_model
def resnet152():
    """Constructs a ResNet-152 model."""
    cfg = ResNetConfig(
        name="resnet152",
        url="",
        block="bottleneck",
        nb_blocks=(3, 8, 36, 3),
        interpolation="bicubic",
        crop_pct=0.95,
    )
    return ResNet, cfg


@register_model
def resnet152d():
    """Constructs a ResNet-152-D model."""
    cfg = ResNetConfig(
        name="resnet152d",
        url="",
        input_size=(256, 256),
        block="bottleneck",
        nb_blocks=(3, 8, 36, 3),
        pool_size=8,
        stem_width=32,
        stem_type="deep",
        downsample_mode="avg",
        test_input_size=(320, 320),
        interpolation="bicubic",
        crop_pct=1.0,
        first_conv="conv1/0",
    )
    return ResNet, cfg


@register_model
def resnet200d():
    """Constructs a ResNet-200-D model."""
    cfg = ResNetConfig(
        name="resnet200d",
        url="",
        input_size=(256, 256),
        block="bottleneck",
        nb_blocks=(3, 24, 36, 3),
        pool_size=8,
        stem_width=32,
        stem_type="deep",
        downsample_mode="avg",
        test_input_size=(320, 320),
        interpolation="bicubic",
        crop_pct=1.0,
        first_conv="conv1/0",
    )
    return ResNet, cfg


@register_model
def tv_resnet34():
    """Constructs a ResNet-34 model with original Torchvision weights."""
    cfg = ResNetConfig(
        name="tv_resnet34", url="", block="basic_block", nb_blocks=(3, 4, 6, 3)
    )
    return ResNet, cfg


@register_model
def tv_resnet50():
    """Constructs a ResNet-50 model with original Torchvision weights."""
    cfg = ResNetConfig(
        name="tv_resnet50", url="", block="bottleneck", nb_blocks=(3, 4, 6, 3)
    )
    return ResNet, cfg


@register_model
def tv_resnet101():
    """Constructs a ResNet-101 model w/ Torchvision pretrained weights."""
    cfg = ResNetConfig(
        name="tv_resnet101", url="", block="bottleneck", nb_blocks=(3, 4, 23, 3)
    )
    return ResNet, cfg


@register_model
def tv_resnet152():
    """Constructs a ResNet-152 model w/ Torchvision pretrained weights."""
    cfg = ResNetConfig(
        name="tv_resnet152", url="", block="bottleneck", nb_blocks=(3, 8, 36, 3)
    )
    return ResNet, cfg


@register_model
def wide_resnet50_2():
    """Constructs a Wide ResNet-50-2 model.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    cfg = ResNetConfig(
        name="wide_resnet50_2",
        url="",
        block="bottleneck",
        nb_blocks=(3, 4, 6, 3),
        base_width=128,
        interpolation="bicubic",
    )
    return ResNet, cfg


@register_model
def wide_resnet101_2():
    """Constructs a Wide ResNet-101-2 model.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same.
    """
    cfg = ResNetConfig(
        name="wide_resnet101_2",
        url="",
        block="bottleneck",
        nb_blocks=(3, 4, 23, 3),
        base_width=128,
    )
    return ResNet, cfg


@register_model
def resnet50_gn():
    """Constructs a ResNet-50 model w/ GroupNorm"""
    cfg = ResNetConfig(
        name="resnet50_gn",
        url="",
        block="bottleneck",
        nb_blocks=(3, 4, 6, 3),
        norm_layer="group_norm",
        crop_pct=0.94,
        interpolation="bicubic",
    )
    return ResNet, cfg


@register_model
def resnext50_32x4d():
    """Constructs a ResNeXt50-32x4d model."""
    cfg = ResNetConfig(
        name="resnext50_32x4d",
        url="",
        block="bottleneck",
        nb_blocks=(3, 4, 6, 3),
        cardinality=32,
        base_width=4,
        crop_pct=0.95,
        interpolation="bicubic",
    )
    return ResNet, cfg


@register_model
def resnext50d_32x4d():
    """
    Constructs a ResNeXt50d-32x4d model. ResNext50 w/ deep stem & avg pool downsample
    """
    cfg = ResNetConfig(
        name="resnext50d_32x4d",
        url="",
        block="bottleneck",
        nb_blocks=(3, 4, 6, 3),
        cardinality=32,
        base_width=4,
        stem_width=32,
        stem_type="deep",
        downsample_mode="avg",
        interpolation="bicubic",
        first_conv="conv1/0",
    )
    return ResNet, cfg


@register_model
def resnext101_32x8d():
    """Constructs a ResNeXt-101 32x8d model."""
    cfg = ResNetConfig(
        name="resnext101_32x8d",
        url="",
        block="bottleneck",
        nb_blocks=(3, 4, 23, 3),
        cardinality=32,
        base_width=8,
    )
    return ResNet, cfg


@register_model
def tv_resnext50_32x4d():
    """Constructs a ResNeXt50-32x4d model with original Torchvision weights."""
    cfg = ResNetConfig(
        name="tv_resnext50_32x4d",
        url="",
        block="bottleneck",
        nb_blocks=(3, 4, 6, 3),
        cardinality=32,
        base_width=4,
    )
    return ResNet, cfg


@register_model
def ig_resnext101_32x8d():
    """Constructs a ResNeXt-101 32x8 model pretrained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining"
    <https://arxiv.org/abs/1805.00932>`_
    Weights from https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/
    """
    cfg = ResNetConfig(
        name="ig_resnext101_32x8d",
        url="",
        block="bottleneck",
        nb_blocks=(3, 4, 23, 3),
        cardinality=32,
        base_width=8,
    )
    return ResNet, cfg


@register_model
def ig_resnext101_32x16d():
    """Constructs a ResNeXt-101 32x16 model pretrained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining"
    <https://arxiv.org/abs/1805.00932>`_
    Weights from https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/
    """
    cfg = ResNetConfig(
        name="ig_resnext101_32x16d",
        url="",
        block="bottleneck",
        nb_blocks=(3, 4, 23, 3),
        cardinality=32,
        base_width=16,
    )
    return ResNet, cfg


@register_model
def ig_resnext101_32x32d():
    """Constructs a ResNeXt-101 32x32 model pretrained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining"
    <https://arxiv.org/abs/1805.00932>`_
    Weights from https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/
    """
    cfg = ResNetConfig(
        name="ig_resnext101_32x32d",
        url="",
        block="bottleneck",
        nb_blocks=(3, 4, 23, 3),
        cardinality=32,
        base_width=32,
    )
    return ResNet, cfg


@register_model
def ig_resnext101_32x48d():
    """Constructs a ResNeXt-101 32x48 model pretrained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining"
    <https://arxiv.org/abs/1805.00932>`_
    Weights from https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/
    """
    cfg = ResNetConfig(
        name="ig_resnext101_32x48d",
        url="",
        block="bottleneck",
        nb_blocks=(3, 4, 23, 3),
        cardinality=32,
        base_width=48,
    )
    return ResNet, cfg


@register_model
def ssl_resnet18():
    """Constructs a semi-supervised ResNet-18 model pre-trained on YFCC100M dataset and
    finetuned on ImageNet
    `"Billion-scale Semi-Supervised Learning for Image Classification"
    <https://arxiv.org/abs/1905.00546>`_
    Weights from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    cfg = ResNetConfig(
        name="ssl_resnet18",
        url="",
        block="basic_block",
        nb_blocks=(2, 2, 2, 2),
    )
    return ResNet, cfg


@register_model
def ssl_resnet50():
    """Constructs a semi-supervised ResNet-50 model pre-trained on YFCC100M dataset and
    finetuned on ImageNet
    `"Billion-scale Semi-Supervised Learning for Image Classification"
    <https://arxiv.org/abs/1905.00546>`_
    Weights from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    cfg = ResNetConfig(
        name="ssl_resnet50",
        url="",
        block="bottleneck",
        nb_blocks=(3, 4, 6, 3),
    )
    return ResNet, cfg


@register_model
def ssl_resnext50_32x4d():
    """
    Constructs a semi-supervised ResNeXt-50 32x4 model pre-trained on YFCC100M dataset
    and finetuned on ImageNet
    `"Billion-scale Semi-Supervised Learning for Image Classification"
    <https://arxiv.org/abs/1905.00546>`_
    Weights from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    cfg = ResNetConfig(
        name="ssl_resnext50_32x4d",
        url="",
        block="bottleneck",
        nb_blocks=(3, 4, 6, 3),
        cardinality=32,
        base_width=4,
    )
    return ResNet, cfg


@register_model
def ssl_resnext101_32x4d():
    """
    Constructs a semi-supervised ResNeXt-101 32x4 model pre-trained on YFCC100M dataset
    and finetuned on ImageNet
    `"Billion-scale Semi-Supervised Learning for Image Classification"
    <https://arxiv.org/abs/1905.00546>`_
    Weights from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    cfg = ResNetConfig(
        name="ssl_resnext101_32x4d",
        url="",
        block="bottleneck",
        nb_blocks=(3, 4, 23, 3),
        cardinality=32,
        base_width=4,
    )
    return ResNet, cfg


@register_model
def ssl_resnext101_32x8d():
    """
    Constructs a semi-supervised ResNeXt-101 32x8 model pre-trained on YFCC100M dataset
    and finetuned on ImageNet
    `"Billion-scale Semi-Supervised Learning for Image Classification"
    <https://arxiv.org/abs/1905.00546>`_
    Weights from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    cfg = ResNetConfig(
        name="ssl_resnext101_32x8d",
        url="",
        block="bottleneck",
        nb_blocks=(3, 4, 23, 3),
        cardinality=32,
        base_width=8,
    )
    return ResNet, cfg


@register_model
def ssl_resnext101_32x16d():
    """
    Constructs a semi-supervised ResNeXt-101 32x16 model pre-trained on YFCC100M dataset
    and finetuned on ImageNet
    `"Billion-scale Semi-Supervised Learning for Image Classification"
    <https://arxiv.org/abs/1905.00546>`_
    Weights from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    cfg = ResNetConfig(
        name="ssl_resnext101_32x16d",
        url="",
        block="bottleneck",
        nb_blocks=(3, 4, 23, 3),
        cardinality=32,
        base_width=16,
    )
    return ResNet, cfg


@register_model
def swsl_resnet18():
    """
    Constructs a semi-weakly supervised Resnet-18 model pre-trained on 1B weakly
    supervised image dataset and finetuned on ImageNet.
    `"Billion-scale Semi-Supervised Learning for Image Classification"
    <https://arxiv.org/abs/1905.00546>`_
    Weights from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    cfg = ResNetConfig(
        name="swsl_resnet18",
        url="",
        block="basic_block",
        nb_blocks=(2, 2, 2, 2),
    )
    return ResNet, cfg


@register_model
def swsl_resnet50():
    """
    Constructs a semi-weakly supervised ResNet-50 model pre-trained on 1B weakly
    supervised image dataset and finetuned on ImageNet.
    `"Billion-scale Semi-Supervised Learning for Image Classification"
    <https://arxiv.org/abs/1905.00546>`_
    Weights from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    cfg = ResNetConfig(
        name="swsl_resnet50",
        url="",
        block="bottleneck",
        nb_blocks=(3, 4, 6, 3),
    )
    return ResNet, cfg


@register_model
def swsl_resnext50_32x4d():
    """
    Constructs a semi-weakly supervised ResNeXt-50 32x4 model pre-trained on 1B weakly
    supervised image dataset and finetuned on ImageNet.
    `"Billion-scale Semi-Supervised Learning for Image Classification"
    <https://arxiv.org/abs/1905.00546>`_
    Weights from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    cfg = ResNetConfig(
        name="swsl_resnext50_32x4d",
        url="",
        block="bottleneck",
        nb_blocks=(3, 4, 6, 3),
        cardinality=32,
        base_width=4,
    )
    return ResNet, cfg


@register_model
def swsl_resnext101_32x4d():
    """
    Constructs a semi-weakly supervised ResNeXt-101 32x4 model pre-trained on 1B weakly
    supervised image dataset and finetuned on ImageNet.
    `"Billion-scale Semi-Supervised Learning for Image Classification"
    <https://arxiv.org/abs/1905.00546>`_
    Weights from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    cfg = ResNetConfig(
        name="swsl_resnext101_32x4d",
        url="",
        block="bottleneck",
        nb_blocks=(3, 4, 23, 3),
        cardinality=32,
        base_width=4,
    )
    return ResNet, cfg


@register_model
def swsl_resnext101_32x8d():
    """
    Constructs a semi-weakly supervised ResNeXt-101 32x8 model pre-trained on 1B weakly
    supervised image dataset and finetuned on ImageNet.
    `"Billion-scale Semi-Supervised Learning for Image Classification"
    <https://arxiv.org/abs/1905.00546>`_
    Weights from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    cfg = ResNetConfig(
        name="swsl_resnext101_32x8d",
        url="",
        block="bottleneck",
        nb_blocks=(3, 4, 23, 3),
        cardinality=32,
        base_width=8,
    )
    return ResNet, cfg


@register_model
def swsl_resnext101_32x16d():
    """
    Constructs a semi-weakly supervised ResNeXt-101 32x16 model pre-trained on 1B
    weakly supervised image dataset and finetuned on ImageNet.
    `"Billion-scale Semi-Supervised Learning for Image Classification"
    <https://arxiv.org/abs/1905.00546>`_
    Weights from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/
    """
    cfg = ResNetConfig(
        name="swsl_resnext101_32x16d",
        url="",
        block="bottleneck",
        nb_blocks=(3, 4, 23, 3),
        cardinality=32,
        base_width=16,
    )
    return ResNet, cfg


@register_model
def ecaresnet26t():
    """
    Constructs an ECA-ResNeXt-26-T model.
    This is technically a 28 layer ResNet, like a 'D' bag-of-tricks model but with
    tiered 24, 32, 64 channels in the deep stem and ECA attn.
    """
    cfg = ResNetConfig(
        name="ecaresnet26t",
        url="",
        block="bottleneck",
        input_size=(256, 256),
        nb_blocks=(2, 2, 2, 2),
        stem_type="deep_tiered",
        stem_width=32,
        downsample_mode="avg",
        attn_layer="eca",
        test_input_size=(320, 320),
        pool_size=8,
        crop_pct=0.95,
        interpolation="bicubic",
        first_conv="conv1/0",
    )
    return ResNet, cfg


@register_model
def ecaresnet50d():
    """Constructs a ResNet-50-D model with eca."""
    cfg = ResNetConfig(
        name="ecaresnet50d",
        url="",
        block="bottleneck",
        nb_blocks=(3, 4, 6, 3),
        stem_type="deep",
        stem_width=32,
        downsample_mode="avg",
        attn_layer="eca",
        interpolation="bicubic",
        first_conv="conv1/0",
    )
    return ResNet, cfg


@register_model
def ecaresnet50t():
    """
    Constructs an ECA-ResNet-50-T model.
    Like a 'D' bag-of-tricks model but with tiered 24, 32, 64 channels in the deep stem
    and ECA attn.
    """
    cfg = ResNetConfig(
        name="ecaresnet50t",
        url="",
        block="bottleneck",
        nb_blocks=(3, 4, 6, 3),
        stem_type="deep_tiered",
        stem_width=32,
        downsample_mode="avg",
        attn_layer="eca",
        test_input_size=(320, 320),
        pool_size=8,
        crop_pct=0.95,
        interpolation="bicubic",
        first_conv="conv1/0",
    )
    return ResNet, cfg


@register_model
def ecaresnetlight():
    """Constructs a ResNet-50-D light model with eca."""
    cfg = ResNetConfig(
        name="ecaresnetlight",
        url="",
        block="bottleneck",
        nb_blocks=(1, 1, 11, 3),
        stem_width=32,
        downsample_mode="avg",
        attn_layer="eca",
        interpolation="bicubic",
    )
    return ResNet, cfg


@register_model
def ecaresnet101d():
    """Constructs a ResNet-101-D model with eca."""
    cfg = ResNetConfig(
        name="ecaresnet101d",
        url="",
        block="bottleneck",
        nb_blocks=(3, 4, 23, 3),
        stem_type="deep",
        stem_width=32,
        downsample_mode="avg",
        attn_layer="eca",
        interpolation="bicubic",
        first_conv="conv1/0",
    )
    return ResNet, cfg


@register_model
def ecaresnet269d():
    """Constructs a ResNet-269-D model with ECA."""
    cfg = ResNetConfig(
        name="ecaresnet269d",
        url="",
        input_size=(320, 320),
        block="bottleneck",
        nb_blocks=(3, 30, 48, 8),
        stem_type="deep",
        stem_width=32,
        downsample_mode="avg",
        attn_layer="eca",
        test_input_size=(352, 352),
        pool_size=10,
        crop_pct=1.0,
        interpolation="bicubic",
        first_conv="conv1/0",
    )
    return ResNet, cfg


@register_model
def resnetblur50():
    """Constructs a ResNet-50 model with blur anti-aliasing"""
    cfg = ResNetConfig(
        name="resnetblur50",
        url="",
        block="bottleneck",
        nb_blocks=(3, 4, 6, 3),
        aa_layer="blur_pool",
        interpolation="bicubic",
    )
    return ResNet, cfg


@register_model
def resnetrs50():
    """Constructs a ResNet-RS-50 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from
    https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    cfg = ResNetConfig(
        name="resnetrs50",
        url="",
        input_size=(160, 160),
        block="bottleneck",
        nb_blocks=(3, 4, 6, 3),
        stem_type="deep",
        stem_width=32,
        replace_stem_pool=True,
        downsample_mode="avg",
        attn_layer="se",
        se_ratio=0.25,
        test_input_size=(224, 224),
        pool_size=5,
        crop_pct=0.91,
        interpolation="bicubic",
        first_conv="conv1/0",
    )
    return ResNet, cfg


@register_model
def resnetrs101():
    """Constructs a ResNet-RS-101 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from
    https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    cfg = ResNetConfig(
        name="resnetrs101",
        url="",
        input_size=(192, 192),
        block="bottleneck",
        nb_blocks=(3, 4, 23, 3),
        stem_type="deep",
        stem_width=32,
        replace_stem_pool=True,
        downsample_mode="avg",
        attn_layer="se",
        se_ratio=0.25,
        test_input_size=(288, 288),
        pool_size=6,
        crop_pct=0.94,
        interpolation="bicubic",
        first_conv="conv1/0",
    )
    return ResNet, cfg


@register_model
def resnetrs152():
    """Constructs a ResNet-RS-152 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from
    https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    cfg = ResNetConfig(
        name="resnetrs152",
        url="",
        input_size=(256, 256),
        block="bottleneck",
        nb_blocks=(3, 8, 36, 3),
        stem_type="deep",
        stem_width=32,
        replace_stem_pool=True,
        downsample_mode="avg",
        attn_layer="se",
        se_ratio=0.25,
        test_input_size=(320, 320),
        pool_size=8,
        crop_pct=1.0,
        interpolation="bicubic",
        first_conv="conv1/0",
    )
    return ResNet, cfg


@register_model
def resnetrs200():
    """Constructs a ResNet-RS-200 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from
    https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    cfg = ResNetConfig(
        name="resnetrs200",
        url="",
        input_size=(256, 256),
        block="bottleneck",
        nb_blocks=(3, 24, 36, 3),
        stem_type="deep",
        stem_width=32,
        replace_stem_pool=True,
        downsample_mode="avg",
        attn_layer="se",
        se_ratio=0.25,
        test_input_size=(320, 320),
        pool_size=8,
        crop_pct=1.0,
        interpolation="bicubic",
        first_conv="conv1/0",
    )
    return ResNet, cfg


@register_model
def resnetrs270():
    """Constructs a ResNet-RS-270 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from
    https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    cfg = ResNetConfig(
        name="resnetrs270",
        url="",
        input_size=(256, 256),
        block="bottleneck",
        nb_blocks=(4, 29, 53, 4),
        stem_type="deep",
        stem_width=32,
        replace_stem_pool=True,
        downsample_mode="avg",
        attn_layer="se",
        se_ratio=0.25,
        test_input_size=(352, 352),
        pool_size=8,
        crop_pct=1.0,
        interpolation="bicubic",
        first_conv="conv1/0",
    )
    return ResNet, cfg


@register_model
def resnetrs350():
    """Constructs a ResNet-RS-350 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from
    https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    cfg = ResNetConfig(
        name="resnetrs350",
        url="",
        input_size=(288, 288),
        block="bottleneck",
        nb_blocks=(4, 36, 72, 4),
        stem_type="deep",
        stem_width=32,
        replace_stem_pool=True,
        downsample_mode="avg",
        attn_layer="se",
        se_ratio=0.25,
        test_input_size=(384, 384),
        pool_size=9,
        crop_pct=1.0,
        interpolation="bicubic",
        first_conv="conv1/0",
    )
    return ResNet, cfg


@register_model
def resnetrs420():
    """Constructs a ResNet-RS-420 model
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from
    https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    cfg = ResNetConfig(
        name="resnetrs420",
        url="",
        input_size=(320, 320),
        block="bottleneck",
        nb_blocks=(4, 44, 87, 4),
        stem_type="deep",
        stem_width=32,
        replace_stem_pool=True,
        downsample_mode="avg",
        attn_layer="se",
        se_ratio=0.25,
        test_input_size=(416, 416),
        pool_size=10,
        crop_pct=1.0,
        interpolation="bicubic",
        first_conv="conv1/0",
    )
    return ResNet, cfg


@register_model
def seresnet50():
    cfg = ResNetConfig(
        name="seresnet50",
        url="",
        block="bottleneck",
        nb_blocks=(3, 4, 6, 3),
        attn_layer="se",
        interpolation="bicubic",
    )
    return ResNet, cfg


@register_model
def seresnet152d():
    cfg = ResNetConfig(
        name="seresnet152d",
        url="",
        input_size=(256, 256),
        block="bottleneck",
        nb_blocks=(3, 8, 36, 3),
        stem_type="deep",
        stem_width=32,
        downsample_mode="avg",
        attn_layer="se",
        test_input_size=(320, 320),
        pool_size=8,
        crop_pct=1.0,
        interpolation="bicubic",
        first_conv="conv1/0",
    )
    return ResNet, cfg


@register_model
def seresnext26d_32x4d():
    """Constructs a SE-ResNeXt-26-D model.`
    This is technically a 28 layer ResNet, using the 'D' modifier from Gluon /
    bag-of-tricks for combination of deep stem and avg_pool in downsample.
    """
    cfg = ResNetConfig(
        name="seresnext26d_32x4d",
        url="",
        block="bottleneck",
        nb_blocks=(2, 2, 2, 2),
        cardinality=32,
        base_width=4,
        stem_type="deep",
        stem_width=32,
        downsample_mode="avg",
        attn_layer="se",
        interpolation="bicubic",
        first_conv="conv1/0",
    )
    return ResNet, cfg


@register_model
def seresnext26t_32x4d():
    """Constructs a SE-ResNet-26-T model.
    This is technically a 28 layer ResNet, like a 'D' bag-of-tricks model but with
    tiered 24, 32, 64 channels in the deep stem.
    """
    cfg = ResNetConfig(
        name="seresnext26t_32x4d",
        url="",
        block="bottleneck",
        nb_blocks=(2, 2, 2, 2),
        cardinality=32,
        base_width=4,
        stem_type="deep_tiered",
        stem_width=32,
        downsample_mode="avg",
        attn_layer="se",
        interpolation="bicubic",
        first_conv="conv1/0",
    )
    return ResNet, cfg


@register_model
def seresnext50_32x4d():
    cfg = ResNetConfig(
        name="seresnext50_32x4d",
        url="",
        block="bottleneck",
        nb_blocks=(3, 4, 6, 3),
        cardinality=32,
        base_width=4,
        attn_layer="se",
        interpolation="bicubic",
    )
    return ResNet, cfg
