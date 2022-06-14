import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import tensorflow as tf

from tfimm.layers import (
    DropPath,
    FanoutInitializer,
    PadConv2D,
    PadDepthwiseConv2D,
    act_layer_factory,
    norm_layer_factory,
)
from tfimm.utils import make_divisible


def create_conv2d(
    kernel_size: Union[int, Tuple[int, int], List],
    nb_experts: Optional[int] = None,
    nb_groups: int = 1,
    depthwise: bool = False,
    **kwargs,
):
    """
    Selects a 2D convolution implementation based on arguments. Creates and returns one
    of Conv2D, DepthwiseConv2D.

    Used extensively by EfficientNet, MobileNetV3 and related networks.
    """
    # We change the default value for use_bias to False.
    kwargs["use_bias"] = kwargs.get("use_bias", False)

    if isinstance(kernel_size, list):
        assert nb_experts is None  # MixNet + CondConv combo not supported currently
        assert nb_groups is None  # MixedConv groups are defined by kernel list
        # We're going to use only lists for defining the MixedConv2d kernel groups,
        # ints, tuples, other iterables will continue to pass to normal conv and specify
        # non-square kernels
        raise NotImplementedError("MixedConv2D not implemented yet...")
        # m = MixedConv2d(in_channels, out_channels, kernel_size, **kwargs)
    else:
        if nb_experts is not None:
            raise NotImplementedError("ConvConv2D not implemented yet...")
            # m = CondConv2d(
            #     in_channels, out_channels, kernel_size, groups=groups, **kwargs
            # )
        elif depthwise:
            # Depthwise convolution
            conv = PadDepthwiseConv2D(
                kernel_size=kernel_size,
                depthwise_initializer=FanoutInitializer(depthwise=True),
                **kwargs,
            )
        else:
            # Regular (group-wise) convolution
            conv = PadConv2D(
                kernel_size=kernel_size,
                groups=nb_groups,
                kernel_initializer=FanoutInitializer(nb_groups=nb_groups),
                **kwargs,
            )
    return conv


@dataclass
class BlockArgs:
    """
    Class holding the arguments for a block (DC/IR/etc.). Arguments come from two
    sources. Some are decoded from the block definition string and others (e.g.,
    padding) are set by the global context. Some parameters, such as number of
    filters are updated by the global multipliers.

    Contains a function to decode arguments from the string notation. E.g.::

        ir_r2_k3_s2_e1_i32_o16_se0.25_noskip

    All args can exist in any order except for the leading string which indicates the
    block type.

    * Block type (ir = InvertedResidual, ds = DepthwiseSeparable,
      dsa = DepthwiseSeparable with pointwise activation, cn = ConvBnAct, etc.)
    * r - Number of repeat blocks
    * k - Kernel size
    * s - Stride (int, no tuple)
    * e - Expansion ratio
    * c - Output channels
    * se - Squeuze & excitation ratio
    * n - Activation function ("re", "r6", "hs", "sw", etc.)
    """

    block_type: str
    nb_repeats: int
    nb_experts: Optional[int]
    filters: int
    # Used to deal with in_channels issue in EdgeResidual blocks
    force_in_channels: Optional[int]
    exp_kernel_size: Tuple[int, int]
    dw_kernel_size: Tuple[int, int]
    pw_kernel_size: Tuple[int, int]
    stride: int
    padding: Optional[str]
    dilation_rate: int
    group_size: Optional[int]
    exp_ratio: float
    pw_act: bool
    use_se: bool
    se_ratio: float
    norm_layer: Optional[str]
    act_layer: Optional[str]
    skip_connection: bool
    drop_path_rate: float

    @staticmethod
    def decode(block_string):
        """Gets a block from a string notation of arguments."""
        assert isinstance(block_string, str)
        ops = block_string.split("_")
        options = {"block_type": ops[0]}
        for op in ops[1:]:
            if op == "noskip":
                options["skip"] = False
            elif op == "skip":
                options["skip"] = True
            elif op.startswith("n"):
                act_dict = {
                    "re": "relu",
                    "r6": "relu6",
                    "hs": "hard_swish",
                    "sw": "swish",
                    "mi": "mish",
                }
                options["n"] = act_dict[op[1:]]
            else:
                splits = re.split(r"(\d.*)", op)
                if len(splits) >= 2:
                    key, value = splits[:2]
                    options[key] = value

        skip = False if options["block_type"] == "dsa" else options.get("skip", True)
        if options["block_type"] != "er":
            exp_kernel_size = BlockArgs._parse_ksize(options.get("a", "1"))
            dw_kernel_size = BlockArgs._parse_ksize(options.get("k"))
        else:
            exp_kernel_size = BlockArgs._parse_ksize(options.get("k"))
            dw_kernel_size = (1, 1)

        return BlockArgs(
            block_type=options["block_type"],
            nb_repeats=int(options.get("r")),
            nb_experts=int(options.get("cc", 0)) or None,
            filters=int(options.get("c")),
            force_in_channels=int(options.get("fc", 0)) or None,
            exp_kernel_size=exp_kernel_size,
            dw_kernel_size=dw_kernel_size,
            pw_kernel_size=BlockArgs._parse_ksize(options.get("p", "1")),
            stride=int(options.get("s")),
            padding=None,
            dilation_rate=1,
            group_size=int(options.get("gs")) if "gs" in options else None,
            exp_ratio=float(options.get("e", 1.0)),
            pw_act=options["block_type"] == "dsa",
            use_se=True,  # Which network doesn't use this?
            se_ratio=float(options.get("se", 0.0)),
            norm_layer=None,
            act_layer=options.get("n", None),
            skip_connection=skip,
            drop_path_rate=0.0,
        )

    @staticmethod
    def _parse_ksize(ss) -> Tuple[int, int]:
        if ss.isdigit():
            return int(ss), int(ss)
        else:
            ss = ss.split(".")
            return int(ss[0]), int(ss[1])

    @property
    def nb_groups(self):
        if not self.group_size:  # 0 or None
            return 1  # normal conv with 1 group
        else:
            # Note: group_size == 1 -> depthwise conv
            assert self.filters % self.group_size == 0
            return self.filters // self.group_size


class SqueezeExcite(tf.keras.layers.Layer):
    """
    Squeeze-and-Excitation layer with specific features for the EfficientNet family.

    Args:
        rd_ratio: Ratio of squeeze reduction
        rd_channels: Sets number of reduction channels explicitely. Overrides the
            rd_ratio paramter.
        act_layer: Activation function post reduction convolution
        gate_layer: Activation function for the attention gate
        force_act_layer: Overrides value for act_layer.
    """

    def __init__(
        self,
        rd_ratio: float = 0.25,
        rd_channels: Optional[int] = None,
        act_layer: str = "relu",
        gate_layer: str = "sigmoid",
        force_act_layer: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rd_ratio = rd_ratio
        self.rd_channels = rd_channels

        act_layer = act_layer_factory(force_act_layer or act_layer)
        gate_layer = act_layer_factory(gate_layer)

        self.conv_reduce = None
        self.act1 = act_layer(name="act1")
        self.conv_expand = None
        self.gate = gate_layer(name="gate")

    def build(self, input_shape):
        channels = input_shape[-1]
        rd_channels = self.rd_channels or round(channels * self.rd_ratio)
        self.conv_reduce = tf.keras.layers.Conv2D(
            filters=rd_channels,
            kernel_size=1,
            use_bias=True,
            kernel_initializer=FanoutInitializer(),
            name="conv_reduce",
        )
        self.conv_expand = tf.keras.layers.Conv2D(
            filters=channels,
            kernel_size=1,
            use_bias=True,
            kernel_initializer=FanoutInitializer(),
            name="conv_expand",
        )

    def call(self, x):
        x_se = tf.reduce_mean(x, axis=(1, 2), keepdims=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x_se = self.gate(x_se)
        x = x * x_se
        return x


class ConvBnAct(tf.keras.layers.Layer):
    """Conv + Norm Layer + Activation with optional skip connection."""

    def __init__(self, cfg: BlockArgs, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        # This depends on number of input channels and is set during build phase
        self.skip_connection = None

        norm_layer = norm_layer_factory(cfg.norm_layer)
        act_layer = act_layer_factory(cfg.act_layer)

        self.conv = create_conv2d(
            filters=cfg.filters,
            kernel_size=cfg.dw_kernel_size,
            strides=cfg.stride,
            padding=cfg.padding,
            dilation_rate=cfg.dilation_rate,
            name="conv",
        )
        self.bn1 = norm_layer(name="bn1")
        self.act1 = act_layer()
        self.drop_path = None

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.skip_connection = (
            self.cfg.stride == 1
            and self.cfg.filters == in_channels
            and self.cfg.skip_connection
        )
        if self.skip_connection:
            self.drop_path = DropPath(drop_prob=self.cfg.drop_path_rate)

    def call(self, x, training: bool = False):
        shortcut = x
        x = self.conv(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        if self.skip_connection:
            x = self.drop_path(x, training=training)
            x = x + shortcut
        return x


class DepthwiseSeparableConv(tf.keras.layers.Layer):
    """
    Depthwise separable block. Used for DS convolutions in MobileNet-V1 and in the place
    of IR blocks that have no expansion convolution. This is an alternative to an IR
    block with an optional first pw convolution.
    """

    def __init__(self, cfg: BlockArgs, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        # This depends on number of input channels and is set during build phase
        self.skip_connection = None

        norm_layer = norm_layer_factory(cfg.norm_layer)
        act_layer = act_layer_factory(cfg.act_layer)

        self.conv_dw = create_conv2d(
            kernel_size=cfg.dw_kernel_size,
            strides=cfg.stride,
            padding=cfg.padding,
            dilation_rate=cfg.dilation_rate,
            depthwise=True,
            name="conv_dw",
        )
        self.bn1 = norm_layer(name="bn1")
        self.act1 = act_layer()
        self.se = (
            SqueezeExcite(rd_ratio=cfg.se_ratio, act_layer=cfg.act_layer, name="se")
            if cfg.use_se and cfg.se_ratio > 0.0
            else None
        )
        self.conv_pw = create_conv2d(
            filters=cfg.filters,
            kernel_size=cfg.pw_kernel_size,
            padding=cfg.padding,
            nb_groups=cfg.nb_groups,
            name="conv_pw",
        )
        self.bn2 = norm_layer(name="bn2")
        self.act2 = act_layer() if cfg.pw_act else None
        self.drop_path = None

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.skip_connection = (
            self.cfg.stride == 1
            and self.cfg.filters == in_channels
            and self.cfg.skip_connection
        )
        if self.skip_connection:
            self.drop_path = DropPath(drop_prob=self.cfg.drop_path_rate)

    def call(self, x, training: bool = False):
        shortcut = x
        x = self.conv_dw(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        if self.se is not None:
            x = self.se(x, training=training)
        x = self.conv_pw(x)
        x = self.bn2(x, training=training)
        if self.act2 is not None:
            x = self.act2(x)
        if self.skip_connection:
            x = self.drop_path(x, training=training)
            x = x + shortcut
        return x


class InvertedResidual(tf.keras.layers.Layer):
    """
    Inverted residual block with optional SE.

    Originally used in MobileNet-V2, this layer is often referred to as "MBConv" (Mobile
    inverted bottleneck convolution) and is also used in

    * MNasNet - https://arxiv.org/abs/1807.11626
    * EfficientNet - https://arxiv.org/abs/1905.11946
    * MobileNet-V2 - https://arxiv.org/abs/1801.04381
    * MobileNet-V3 - https://arxiv.org/abs/1905.02244
    """

    def __init__(self, cfg: BlockArgs, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        # This depends on number of input channels and is set during build phase
        self.skip_connection = None

        norm_layer = norm_layer_factory(cfg.norm_layer)
        act_layer = act_layer_factory(cfg.act_layer)

        # Point-wise expansion
        self.conv_pw = None
        self.bn1 = norm_layer(name="bn1")
        self.act1 = act_layer()

        # Depth-wise convolution
        self.conv_dw = create_conv2d(
            kernel_size=cfg.dw_kernel_size,
            strides=cfg.stride,
            padding=cfg.padding,
            dilation_rate=cfg.dilation_rate,
            depthwise=True,
            name="conv_dw",
        )
        self.bn2 = norm_layer(name="bn2")
        self.act2 = act_layer()

        # Squeeze-and-excitation
        self.se = (
            SqueezeExcite(rd_ratio=cfg.se_ratio, act_layer=cfg.act_layer, name="se")
            if cfg.use_se and cfg.se_ratio > 0.0
            else None
        )

        # Point-wise linear projection
        self.conv_pwl = create_conv2d(
            filters=cfg.filters,
            kernel_size=cfg.pw_kernel_size,
            padding=cfg.padding,
            name="conv_pwl",
        )
        self.bn3 = norm_layer(name="bn3")
        self.drop_path = None

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.skip_connection = (
            self.cfg.stride == 1
            and self.cfg.filters == in_channels
            and self.cfg.skip_connection
        )
        self.conv_pw = create_conv2d(
            filters=make_divisible(in_channels * self.cfg.exp_ratio, 8),
            kernel_size=self.cfg.exp_kernel_size,
            padding=self.cfg.padding,
            nb_groups=self.cfg.nb_groups,
            name="conv_pw",
        )
        if self.skip_connection:
            self.drop_path = DropPath(drop_prob=self.cfg.drop_path_rate)

    def call(self, x, training: bool = False):
        shortcut = x
        x = self.conv_pw(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.conv_dw(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)
        if self.se is not None:
            x = self.se(x, training=training)
        x = self.conv_pwl(x)
        x = self.bn3(x, training=training)
        if self.skip_connection:
            x = self.drop_path(x, training=training)
            x = x + shortcut
        return x


class EdgeResidual(tf.keras.layers.Layer):
    """
    Residual block with expansion convolution followed by pointwise-linear w/ stride

    Originally introduced in `EfficientNet-EdgeTPU: Creating Accelerator-Optimized
    Neural Networks with AutoML`
    * https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html

    This layer is also called FusedMBConv in the MobileDet, EfficientNet-X, and
    EfficientNet-V2 papers
    * MobileDet - https://arxiv.org/abs/2004.14525
    * EfficientNet-X - https://arxiv.org/abs/2102.05610
    * EfficientNet-V2 - https://arxiv.org/abs/2104.00298
    """

    def __init__(self, cfg: BlockArgs, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        # This depends on number of input channels and is set during build phase
        self.skip_connection = None

        norm_layer = norm_layer_factory(cfg.norm_layer)
        act_layer = act_layer_factory(cfg.act_layer)

        # Point-wise expansion
        self.conv_exp = None
        self.bn1 = norm_layer(name="bn1")
        self.act1 = act_layer()

        # Squeeze-and-excitation
        self.se = (
            SqueezeExcite(rd_ratio=cfg.se_ratio, act_layer=cfg.act_layer, name="se")
            if cfg.use_se and cfg.se_ratio > 0.0
            else None
        )

        # Point-wise linear projection
        self.conv_pwl = create_conv2d(
            filters=cfg.filters,
            kernel_size=cfg.pw_kernel_size,
            padding=cfg.padding,
            name="conv_pwl",
        )
        self.bn2 = norm_layer(name="bn2")
        self.drop_path = None

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.skip_connection = (
            self.cfg.stride == 1
            and self.cfg.filters == in_channels
            and self.cfg.skip_connection
        )
        # If force_in_channels is set, we use that value for the purpose of calculating
        # the number of filters in the convolution
        force_in_channels = self.cfg.force_in_channels or in_channels
        self.conv_exp = create_conv2d(
            filters=make_divisible(force_in_channels * self.cfg.exp_ratio, 8),
            kernel_size=self.cfg.exp_kernel_size,
            strides=self.cfg.stride,
            padding=self.cfg.padding,
            nb_groups=self.cfg.nb_groups,
            name="conv_exp",
        )
        if self.skip_connection:
            self.drop_path = DropPath(drop_prob=self.cfg.drop_path_rate)

    def call(self, x, training: bool = False):
        shortcut = x
        x = self.conv_exp(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        if self.se is not None:
            x = self.se(x, training=training)
        x = self.conv_pwl(x)
        x = self.bn2(x, training=training)
        if self.skip_connection:
            x = self.drop_path(x, training=training)
            x = x + shortcut
        return x
