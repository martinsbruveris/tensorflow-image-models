"""
TensorFlow implementation of ResNets

Based on timm/models/resnet.py by Ross Wightman.

Copyright 2021 Martins Bruveris
"""

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import tensorflow as tf

from tfimm.layers import ClassifierHead, act_layer_factory, norm_layer_factory
from tfimm.models import ModelConfig, keras_serializable, register_model
from tfimm.utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# model_registry will add each entrypoint fn to this
__all__ = ["ResNet", "ResNetConfig", "BasicBlock"]


@dataclass
class ResNetConfig(ModelConfig):
    block: str
    nb_blocks: List
    nb_classes: int = 1000
    in_chans: int = 3
    input_size: Tuple[int, int] = (224, 224)
    pool_size: Tuple[int, int] = (7, 7)
    cardinality: int = 1
    base_width: int = 64
    stem_width: int = 64
    stem_type: str = ""
    replace_stem_pool: bool = False
    output_stride: int = 32
    block_reduce_first: int = 1
    down_kernel_size: int = 1
    avg_down: bool = False
    act_layer: str = "relu"
    norm_layer: str = "batch_norm"
    aa_layer: Any = None  # TODO: Not implemented
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    drop_block_rate: float = 0.0
    global_pool: str = "avg"
    block_args: Any = None  # TODO: What is this?
    zero_init_last_bn: bool = True
    # Parameters for inference
    test_input_size: Optional[Tuple[int, int]] = None
    crop_pct: float = 0.875
    interpolation: str = "bilinear"
    mean: float = IMAGENET_DEFAULT_MEAN
    std: float = IMAGENET_DEFAULT_STD
    first_conv: str = "conv1"
    classifier: str = "fc"

    def __post_init__(self):
        if self.test_input_size is None:
            self.test_input_size = self.input_size


class BasicBlock(tf.keras.layers.Layer):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        cardinality=1,
        base_width=64,
        reduce_first=1,
        dilation=1,
        first_dilation=None,
        act_layer="relu",
        norm_layer="batch_norm",
        attn_layer=None,
        aa_layer=None,
        drop_block=None,
        drop_path=None,
        zero_init_last_bn=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert cardinality == 1, "BasicBlock only supports cardinality of 1"
        assert base_width == 64, "BasicBlock does not support changing base width"

        self.planes = planes
        self.stride = stride
        self.downsample = downsample
        self.reduce_first = reduce_first
        self.dilation = dilation
        self.first_dilation = first_dilation
        self.act_layer = act_layer_factory(act_layer)
        self.norm_layer = norm_layer_factory(norm_layer)
        self.attn_layer = attn_layer
        self.aa_layer = aa_layer
        self.drop_block = drop_block
        self.drop_path = drop_path
        self.zero_init_last_bn = zero_init_last_bn

    def build(self, input_shape: tf.TensorShape):
        first_planes = self.planes // self.reduce_first
        out_planes = self.planes * self.expansion
        first_dilation = self.first_dilation or self.dilation
        use_aa = self.aa_layer is not None and (
            self.stride == 2 or first_dilation != self.dilation
        )
        if use_aa:
            raise NotImplementedError("use_aa=True not implemented yet.")
        if self.dilation != 1:
            raise NotImplementedError("dilation!=1 not implemented yet.")
        if self.first_dilation != 1:
            raise NotImplementedError("first_dilation!=1 not implemented yet.")

        self.pad1 = tf.keras.layers.ZeroPadding2D(padding=first_dilation)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=first_planes,
            kernel_size=3,
            strides=1 if use_aa else self.stride,
            use_bias=False,
            name="conv1",
        )
        self.bn1 = self.norm_layer(name="bn1")
        self.act1 = self.act_layer()
        self.aa = (
            aa_layer(channels=first_planes, stride=self.stride) if use_aa else None
        )

        self.conv2 = tf.keras.layers.Conv2D(
            filters=out_planes,
            kernel_size=3,
            padding="same",  # TODO: Only for dilation=1 (pytorch: padding=dilation)
            dilation_rate=self.dilation,
            use_bias=False,
            name="conv2",
        )
        self.bn2 = self.norm_layer(
            gamma_initializer="zeros" if self.zero_init_last_bn else "ones",
            moving_variance_initializer="zeros" if self.zero_init_last_bn else "ones",
            name="bn2",
        )
        self.se = create_attn(self.attn_layer, out_planes)
        self.act2 = self.act_layer()

    def call(self, x, training=False):
        shortcut = x

        x = self.pad1(x)
        x = self.conv1(x)
        x = self.bn1(x, training)
        if self.drop_block is not None:
            raise NotImplementedError("drop_block!=None not implemented yet...")
            # x = self.drop_block(x)
        x = self.act1(x)
        if self.aa is not None:
            raise NotImplementedError("aa!=None not implemented yet...")
            # x = self.aa(x)

        x = self.conv2(x)
        x = self.bn2(x, training)
        if self.drop_block is not None:
            raise NotImplementedError("drop_block!=None not implemented yet...")
            # x = self.drop_block(x)

        if self.se is not None:
            raise NotImplementedError("se!=None not implemented yet...")
            # x = self.se(x)

        if self.drop_path is not None:
            raise NotImplementedError("drop_path!=None not implemented yet...")
            # x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut, training)
        x += shortcut
        x = self.act2(x)

        return x


class Bottleneck(tf.keras.layers.Layer):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        cardinality=1,
        base_width=64,
        reduce_first=1,
        dilation=1,
        first_dilation=None,
        act_layer="relu",
        norm_layer="batch_norm",
        attn_layer=None,
        aa_layer=None,
        drop_block=None,
        drop_path=None,
        zero_init_last_bn=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.planes = planes
        self.stride = stride
        self.downsample = downsample
        self.cardinality = cardinality
        self.base_width = base_width
        self.reduce_first = reduce_first
        self.dilation = dilation
        self.first_dilation = first_dilation
        self.act_layer = act_layer_factory(act_layer)
        self.norm_layer = norm_layer_factory(norm_layer)
        self.attn_layer = attn_layer
        self.aa_layer = aa_layer
        self.drop_block = drop_block
        self.drop_path = drop_path
        self.zero_init_last_bn = zero_init_last_bn

    def build(self, inpu_shape: tf.TensorShape):
        width = int(math.floor(self.planes * (self.base_width / 64)) * self.cardinality)
        first_planes = width // self.reduce_first
        outplanes = self.planes * self.expansion
        first_dilation = self.first_dilation or self.dilation
        use_aa = self.aa_layer is not None and (
            self.stride == 2 or first_dilation != self.dilation
        )
        if use_aa:
            raise NotImplementedError("use_aa=True not implemented yet.")

        self.conv1 = tf.keras.layers.Conv2D(
            filters=first_planes,
            kernel_size=1,
            use_bias=False,
            name="conv1",
        )
        self.bn1 = self.norm_layer(name="bn1")
        self.act1 = self.act_layer()

        self.pad2 = tf.keras.layers.ZeroPadding2D(padding=first_dilation)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=width,
            kernel_size=3,
            strides=1 if use_aa else self.stride,
            dilation_rate=first_dilation,
            groups=self.cardinality,
            use_bias=False,
            name="conv2",
        )
        self.bn2 = self.norm_layer(name="bn2")
        self.act2 = self.act_layer()

        self.aa = aa_layer(channels=width, stride=self.stride) if use_aa else None

        self.conv3 = tf.keras.layers.Conv2D(
            filters=outplanes,
            kernel_size=1,
            use_bias=False,
            name="conv3",
        )
        self.bn3 = self.norm_layer(
            gamma_initializer="zeros" if self.zero_init_last_bn else "ones",
            moving_variance_initializer="zeros" if self.zero_init_last_bn else "ones",
            name="bn3",
        )
        self.se = create_attn(self.attn_layer, outplanes)
        self.act3 = self.act_layer()

    def call(self, x, training=False):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x, training)
        if self.drop_block is not None:
            raise NotImplementedError("drop_block!=None not implemented yet...")
            # x = self.drop_block(x)
        x = self.act1(x)

        x = self.pad2(x)
        x = self.conv2(x)
        x = self.bn2(x, training)
        if self.drop_block is not None:
            raise NotImplementedError("drop_block!=None not implemented yet...")
            # x = self.drop_block(x)
        x = self.act2(x)
        if self.aa is not None:
            raise NotImplementedError("aa!=None not implemented yet...")
            # x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x, training)
        if self.drop_block is not None:
            raise NotImplementedError("drop_block!=None not implemented yet...")
            # x = self.drop_block(x)

        if self.se is not None:
            raise NotImplementedError("se!=None not implemented yet...")
            # x  = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut, training)
        x += shortcut
        x = self.act3(x)

        return x


def aa_layer(channels, stride):
    raise NotImplementedError("aa_layer not implemented.")


def create_attn(attn_layer, outplanes):
    if attn_layer is None:
        return None
    else:
        raise NotImplementedError("create_attn not implemented.")


def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def downsample_avg(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    first_dilation=None,
    norm_layer=None,
    name="",
):
    norm_layer = norm_layer_factory(norm_layer)
    avg_stride = stride if dilation == 1 else 1

    layers = []
    if stride != 1 or dilation != 1:
        pool = tf.keras.layers.AveragePooling2D(
            pool_size=2, strides=avg_stride, padding="same"
        )
        layers.append(pool)
    conv = tf.keras.layers.Conv2D(
        filters=out_channels,
        kernel_size=1,
        strides=1,
        use_bias=False,
        name=name + "/downsample/1",
    )
    bn = norm_layer(name=name + "/downsample/2")
    layers.extend([conv, bn])
    return tf.keras.Sequential(layers)


def downsample_conv(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    first_dilation=None,
    norm_layer="batch_norm",
    name="",
):
    norm_layer = norm_layer_factory(norm_layer)
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    # This layer is part of the conv layer in pytorch and so is not being tracked here
    pad = tf.keras.layers.ZeroPadding2D(padding=p)
    conv = tf.keras.layers.Conv2D(
        filters=out_channels,
        kernel_size=kernel_size,
        strides=stride,
        dilation_rate=first_dilation,
        use_bias=False,
        name=name + "/downsample/0",
    )
    bn = norm_layer(name=name + "/downsample/1")
    return tf.keras.Sequential([pad, conv, bn])


def drop_blocks(drop_block_rate=0.0):
    if drop_block_rate:
        raise NotImplementedError("drop_block_rate>0 not implemented.")
    return [None, None, None, None]


def make_blocks(
    block_fn,
    channels,
    block_repeats,
    inplanes,
    reduce_first=1,
    output_stride=32,
    down_kernel_size=1,
    avg_down=False,
    drop_block_rate=0.0,
    drop_path_rate=0.0,
    norm_layer="batch_norm",
    **kwargs,
):
    stages = []
    feature_info = []
    net_num_blocks = sum(block_repeats)
    net_block_idx = 0
    net_stride = 4
    dilation = prev_dilation = 1
    for stage_idx, (planes, num_blocks, db) in enumerate(
        zip(channels, block_repeats, drop_blocks(drop_block_rate))
    ):
        # never liked this name, but weight compat requires it
        stage_name = f"layer{stage_idx + 1}"
        stride = 1 if stage_idx == 0 else 2
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride

        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            down_kwargs = dict(
                in_channels=inplanes,
                out_channels=planes * block_fn.expansion,
                kernel_size=down_kernel_size,
                stride=stride,
                dilation=dilation,
                first_dilation=prev_dilation,
                norm_layer=norm_layer,
                name=f"res_net/{stage_name}/0",
            )
            if avg_down:
                downsample = downsample_avg(**down_kwargs)
            else:
                downsample = downsample_conv(**down_kwargs)

        block_kwargs = dict(
            reduce_first=reduce_first, dilation=dilation, drop_block=db, **kwargs
        )
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            # stochastic depth linear decay rule
            block_dpr = drop_path_rate * net_block_idx / (net_num_blocks - 1)
            if block_dpr > 0.0:
                raise NotImplementedError("block_dpr>0. not implemented yet.")
            blocks.append(
                block_fn(
                    inplanes,
                    planes,
                    stride,
                    downsample,
                    first_dilation=prev_dilation,
                    drop_path=None,
                    **block_kwargs,
                    # TODO: sort out name scope
                    name=f"res_net/{stage_name}/{block_idx}",
                )
            )
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1

        stages.append(tf.keras.Sequential(blocks))

    return stages, feature_info


@keras_serializable
class ResNet(tf.keras.Model):
    """ResNet / ResNeXt / SE-ResNeXt / SE-Net

    This class implements all variants of ResNet, ResNeXt, SE-ResNeXt, and SENet that
      * have > 1 stride in the 3x3 conv layer of bottleneck
      * have conv-bn-act ordering

    This ResNet impl supports a number of stem and downsample options based on the v1c,
    v1d, v1e, and v1s variants included in the MXNet Gluon ResNetV1b model. The C and D
    variants are also discussed in the 'Bag of Tricks' paper:
    https://arxiv.org/pdf/1812.01187. The B variant is equivalent to torchvision
    default.

    ResNet variants (the same modifications can be used in SE/ResNeXt models as well):
      * normal, b - 7x7 stem, stem_width = 64, same as torchvision ResNet, NVIDIA ResNet
          'v1.5', Gluon v1b
      * c - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64)
      * d - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64), average pool in
          downsample
      * e - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128), average pool in
          downsample
      * s - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128)
      * t - 3 layer deep 3x3 stem, stem width = 32 (24, 48, 64), average pool in
          downsample
      * tn - 3 layer deep 3x3 stem, stem width = 32 (24, 32, 64), average pool in
          downsample

    ResNeXt
      * normal - 7x7 stem, stem_width = 64, standard cardinality and base widths
      * same c,d, e, s variants as ResNet can be enabled

    SE-ResNeXt
      * normal - 7x7 stem, stem_width = 64
      * same c, d, e, s variants as ResNet can be enabled

    SENet-154 - 3 layer deep 3x3 stem (same as v1c-v1s), stem_width = 64,
        cardinality=64, reduction by 2 on width of first bottleneck convolution, 3x3
        downsample convs after first block

    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockGl, BottleneckGl.
    nb_blocks : list of int
        Numbers of blocks in each stage
    nb_classes : int, default 1000
        Number of classification classes.
    in_chans : int, default 3
        Number of input (color) channels.
    cardinality : int, default 1
        Number of convolution groups for 3x3 conv in Bottleneck.
    base_width : int, default 64
        Factor determining bottleneck channels. `planes * base_width / 64 * cardinality`
    stem_width : int, default 64
        Number of channels in stem convolutions
    stem_type : str, default ''
        The type of stem:
          * '', default - a single 7x7 conv with a width of stem_width
          * 'deep' - three 3x3 convolution layers of widths stem_width, stem_width,
              stem_width * 2
          * 'deep_tiered' - three 3x3 conv layers of widths stem_width//4 * 3,
              stem_width, stem_width * 2
    block_reduce_first: int, default 1
        Reduction factor for first convolution output width of residual blocks,
        1 for all archs except senets, where 2
    down_kernel_size: int, default 1
        Kernel size of residual block downsampling path, 1x1 for most archs, 3x3 for
        senets
    avg_down : bool, default False
        Whether to use average pooling for projection skip connection between
        stages/downsample.
    output_stride : int, default 32
        Set the output stride of the network, 32, 16, or 8. Typically used in
        segmentation.
    act_layer : nn.Module, activation layer
    norm_layer : nn.Module, normalization layer
    aa_layer : nn.Module, anti-aliasing layer
    drop_rate : float, default 0.
        Dropout probability before classifier, for training
    global_pool : str, default 'avg'
        Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax'
    """

    cfg_class = ResNetConfig

    def __init__(self, cfg: ResNetConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg

        self.block = eval(cfg.block)
        self.nb_blocks = cfg.nb_blocks
        self.nb_classes = cfg.nb_classes
        self.in_chans = cfg.in_chans
        self.input_size = cfg.input_size
        self.cardinality = cfg.cardinality
        self.base_width = cfg.base_width
        self.stem_width = cfg.stem_width
        self.stem_type = cfg.stem_type
        self.output_stride = cfg.output_stride
        self.block_reduce_first = cfg.block_reduce_first
        self.down_kernel_size = cfg.down_kernel_size
        self.avg_down = cfg.avg_down
        self.replace_stem_pool = cfg.replace_stem_pool
        self.act_layer_str = cfg.act_layer
        self.act_layer = act_layer_factory(cfg.act_layer)
        self.norm_layer_str = cfg.norm_layer
        self.norm_layer = norm_layer_factory(cfg.norm_layer)
        self.aa_layer = cfg.aa_layer
        self.drop_rate = cfg.drop_rate
        self.drop_path_rate = cfg.drop_path_rate
        self.drop_block_rate = cfg.drop_block_rate
        self.global_pool = cfg.global_pool
        # TODO: Always true
        self.zero_init_last_bn = cfg.zero_init_last_bn
        # TODO: Only used to set attn_layer
        self.block_args = cfg.block_args or dict()

        # TODO: Add feature info for feature pyramid extraction

    @property
    def dummy_inputs(self) -> tf.Tensor:
        return tf.zeros((1, *self.input_size, self.in_chans))

    def build(self, input_shape: tf.TensorShape):
        if "deep" in self.stem_type:
            in_planes = self.stem_width * 2
            stem_chns = (self.stem_width, self.stem_width)
            if "tiered" in self.stem_type:
                stem_chns = (3 * (self.stem_width // 4), self.stem_width)
            self.pad1 = tf.keras.layers.ZeroPadding2D(padding=1)
            conv1_0 = tf.keras.layers.Conv2D(
                filters=stem_chns[0],
                kernel_size=3,
                strides=2,
                use_bias=False,
                name=f"{self.name}/conv1/0",
            )
            bn1_0 = self.norm_layer(name="resnet/conv1/1")
            act1_0 = self.act_layer()
            conv1_1 = tf.keras.layers.Conv2D(
                filters=stem_chns[1],
                kernel_size=3,
                padding="same",
                use_bias=False,
                name=f"{self.name}/conv1/3",
            )
            bn1_1 = self.norm_layer(name="resnet/conv1/4")
            act1_1 = self.act_layer()
            conv1_2 = tf.keras.layers.Conv2D(
                filters=in_planes,
                kernel_size=3,
                padding="same",
                use_bias=False,
                name=f"{self.name}/conv1/6",
            )
            self.conv1 = tf.keras.Sequential(
                [conv1_0, bn1_0, act1_0, conv1_1, bn1_1, act1_1, conv1_2]
            )
        else:
            in_planes = 64
            # In TF "same" paddding with strides != 1 is not the same as (3, 3) padding
            # in pytorch, hence the need for an explicit padding layer
            self.pad1 = tf.keras.layers.ZeroPadding2D(padding=3)
            self.conv1 = tf.keras.layers.Conv2D(
                filters=in_planes,
                kernel_size=7,
                strides=2,
                use_bias=False,
                name="conv1",
            )
        self.bn1 = self.norm_layer(name="bn1")
        self.act1 = self.act_layer()

        # Stem Pooling
        if self.replace_stem_pool:
            raise NotImplementedError("replace_stem_pool=True not implemented")
        else:
            if self.aa_layer is not None:
                raise NotImplementedError("aa_layer!=None not implemented.")
            else:
                self.maxpool = tf.keras.Sequential(
                    [
                        tf.keras.layers.ZeroPadding2D(padding=1),
                        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                    ]
                )

        # Feature Blocks
        channels = [64, 128, 256, 512]
        stage_modules, stage_feature_info = make_blocks(
            self.block,
            channels,
            self.nb_blocks,
            in_planes,
            cardinality=self.cardinality,
            base_width=self.base_width,
            output_stride=self.output_stride,
            reduce_first=self.block_reduce_first,
            avg_down=self.avg_down,
            down_kernel_size=self.down_kernel_size,
            act_layer=self.act_layer_str,
            norm_layer=self.norm_layer_str,
            aa_layer=self.aa_layer,
            drop_block_rate=self.drop_block_rate,
            drop_path_rate=self.drop_path_rate,
            **self.block_args,
        )

        self.layer1 = stage_modules[0]
        self.layer2 = stage_modules[1]
        self.layer3 = stage_modules[2]
        self.layer4 = stage_modules[3]

        # Head (pooling and classifier))
        self.head = ClassifierHead(
            nb_classes=self.nb_classes,
            pool_type=self.global_pool,
            drop_rate=self.drop_rate,
            use_conv=False,
            name="remove",
        )

    def forward_features(self, x, training=False):
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.bn1(x, training)
        x = self.act1(x)
        x = self.maxpool(x)

        x = self.layer1(x, training)
        x = self.layer2(x, training)
        x = self.layer3(x, training)
        x = self.layer4(x, training)
        return x

    def call(self, x, training=False):
        x = self.forward_features(x, training)
        # noinspection PyCallingNonCallable
        x = self.head(x)
        return x


@register_model
def resnet18():
    """Constructs a ResNet-18 model."""
    cfg = ResNetConfig(
        name="resnet18", url="", block="BasicBlock", nb_blocks=[2, 2, 2, 2]
    )
    return ResNet, cfg


@register_model
def resnet18d():
    """Constructs a ResNet-18-D model."""
    cfg = ResNetConfig(
        name="resnet18d",
        url="",
        block="BasicBlock",
        nb_blocks=[2, 2, 2, 2],
        stem_width=32,
        stem_type="deep",
        avg_down=True,
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
        block="Bottleneck",
        nb_blocks=[2, 2, 2, 2],
        interpolation="bicubic",
    )
    return ResNet, cfg


@register_model
def resnet26d():
    """Constructs a ResNet-26-D model."""
    cfg = ResNetConfig(
        name="resnet26d",
        url="",
        block="Bottleneck",
        nb_blocks=[2, 2, 2, 2],
        stem_width=32,
        stem_type="deep",
        avg_down=True,
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
        block="Bottleneck",
        nb_blocks=[2, 2, 2, 2],
        pool_size=(8, 8),
        stem_width=32,
        stem_type="deep_tiered",
        avg_down=True,
        crop_pct=0.94,
        interpolation="bicubic",
        first_conv="conv1/0",
    )
    return ResNet, cfg


@register_model
def resnet34():
    """Constructs a ResNet-34 model."""
    cfg = ResNetConfig(
        name="resnet34", url="", block="BasicBlock", nb_blocks=[3, 4, 6, 3]
    )
    return ResNet, cfg


@register_model
def resnet34d():
    """Constructs a ResNet-34-D model."""
    cfg = ResNetConfig(
        name="resnet34d",
        url="",
        block="BasicBlock",
        nb_blocks=[3, 4, 6, 3],
        stem_width=32,
        stem_type="deep",
        avg_down=True,
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
        block="Bottleneck",
        nb_blocks=[3, 4, 6, 3],
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
        block="Bottleneck",
        nb_blocks=[3, 4, 6, 3],
        stem_width=32,
        stem_type="deep",
        avg_down=True,
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
        block="Bottleneck",
        nb_blocks=[3, 4, 23, 3],
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
        block="Bottleneck",
        nb_blocks=[3, 4, 23, 3],
        pool_size=(8, 8),
        stem_width=32,
        stem_type="deep",
        avg_down=True,
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
        block="Bottleneck",
        nb_blocks=[3, 8, 36, 3],
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
        block="Bottleneck",
        nb_blocks=[3, 8, 36, 3],
        pool_size=(8, 8),
        stem_width=32,
        stem_type="deep",
        avg_down=True,
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
        block="Bottleneck",
        nb_blocks=[3, 24, 36, 3],
        pool_size=(8, 8),
        stem_width=32,
        stem_type="deep",
        avg_down=True,
        test_input_size=(320, 320),
        interpolation="bicubic",
        crop_pct=1.0,
        first_conv="conv1/0",
    )
    return ResNet, cfg
