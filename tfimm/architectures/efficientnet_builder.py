""" EfficientNet, MobileNetV3, etc Builder

Assembles EfficieNet and related network feature blocks from string definitions.
Handles stride, dilation calculations, and selects feature extraction points.

Hacked together by / Copyright 2019, Ross Wightman
"""
import math
from collections import OrderedDict
from copy import deepcopy
from typing import List, Optional, Tuple, Union

from tfimm.utils import make_divisible

from .efficientnet_blocks import (
    BlockArgs,
    ConvBnAct,
    DepthwiseSeparableConv,
    EdgeResidual,
    InvertedResidual,
)

_DEBUG_BUILDER = False


def _log(msg):
    if _DEBUG_BUILDER:
        print(msg)


def round_channels(
    channels: int,
    multiplier: float = 1.0,
    divisor: int = 8,
    min_channels: Optional[int] = None,
    round_limit: float = 0.9,
):
    """Round number of filters based on depth multiplier."""
    return make_divisible(
        value=channels * multiplier,
        divisor=divisor,
        min_value=min_channels,
        round_limit=round_limit,
    )


def _scale_stage_depth(
    stack_args: List[BlockArgs],
    depth_multiplier=1.0,
    depth_trunc="ceil",
) -> List[BlockArgs]:
    """
    Per-stage depth scaling.

    Scales the block repeats in each stage. This depth scaling impl maintains
    compatibility with the EfficientNet scaling method, while allowing sensible
    scaling for other models that may have multiple block arg definitions in each stage.

    Returns a list of BlockArgs already repeated the correct number of times.
    """
    repeats = [ba.nb_repeats for ba in stack_args]

    # We scale the total repeat count for each stage, there may be multiple block arg
    # defs per stage so we need to sum.
    nb_repeats = sum(repeats)
    if depth_trunc == "round":
        # Truncating to int by rounding allows stages with few repeats to remain
        # proportionally smaller for longer. This is a good choice when stage
        # definitions include single repeat stages that we'd prefer to keep that way as
        # long as possible.
        nb_repeats_scaled = max(1, round(nb_repeats * depth_multiplier))
    else:
        # The default for EfficientNet truncates repeats to int via "ceil". Any
        # multiplier > 1.0 will result in an increased depth for every stage.
        nb_repeats_scaled = int(math.ceil(nb_repeats * depth_multiplier))

    # Proportionally distribute repeat count scaling to each block definition in the
    # stage. Allocation is done in reverse as it results in the first block being less
    # likely to be scaled. The first block makes less sense to repeat in most of the
    # arch definitions.
    repeats_scaled = []
    for r in repeats[::-1]:
        rs = max(1, round((r / nb_repeats * nb_repeats_scaled)))
        repeats_scaled.append(rs)
        nb_repeats -= r
        nb_repeats_scaled -= rs
    repeats_scaled = repeats_scaled[::-1]

    # Apply the calculated scaling to each block arg in the stage
    sa_scaled = []
    for ba, rep in zip(stack_args, repeats_scaled):
        sa_scaled.extend([deepcopy(ba) for _ in range(rep)])
    return sa_scaled


def decode_architecture(
    architecture: Tuple[Tuple[str, ...], ...],
    depth_multiplier: Union[float, Tuple[float, ...]] = 1.0,
    depth_truncation: str = "ceil",
    experts_multiplier: int = 1,
    fix_first_last: bool = False,
    group_size: Optional[int] = None,
) -> List[List[BlockArgs]]:
    """
    Decode block architecture definition strings into block kwargs.

    Args:
        architecture: Architecture definition strings, tuple of a tuple of strings
        depth_multiplier: Network depth multiplier
        depth_truncation: Networ depth truncation mode when applying multiplier
        experts_multiplier: CondConv experts multiplier
        fix_first_last: Fix first and last block depths when multiplier is applied
        group_size: Group size override for all blocks that weren't explicitly set in
            architecture string

    Returns:
        List of lists of block args
    """
    if isinstance(depth_multiplier, tuple):
        assert len(depth_multiplier) == len(architecture)
    else:
        depth_multiplier = (depth_multiplier,) * len(architecture)

    arch_args = []
    for stack_idx, (block_strings, multiplier) in enumerate(
        zip(architecture, depth_multiplier)
    ):
        assert isinstance(block_strings, tuple)
        stack_args = []
        for block_str in block_strings:
            assert isinstance(block_str, str)
            block_args = BlockArgs.decode(block_str)
            if block_args.nb_experts is not None:
                block_args.nb_experts *= experts_multiplier
            if group_size is not None:
                block_args.group_size = group_size
            stack_args.append(block_args)

        fix_depths = fix_first_last and stack_idx in {0, len(architecture) - 1}
        mod_multiplier = 1.0 if fix_depths else multiplier
        stack_args = _scale_stage_depth(stack_args, mod_multiplier, depth_truncation)

        arch_args.append(stack_args)
    return arch_args


class EfficientNetBuilder:
    """
    Build Trunk Blocks

    Adapted from timm.
    """

    def __init__(
        self,
        output_stride=32,
        channel_multiplier: float = 1.0,
        padding="",
        se_from_exp=False,  # ???
        act_layer=None,
        norm_layer=None,
        drop_path_rate=0.0,
    ):
        self.output_stride = output_stride
        self.channel_multiplier = channel_multiplier
        self.padding = padding
        # Calculate se channel reduction from expanded (mid) chs
        self.se_from_exp = se_from_exp
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        self.drop_path_rate = drop_path_rate

    def _make_block(
        self,
        block_args: BlockArgs,
        stage_idx: int,
        block_idx: int,
        total_idx: int,
        nb_blocks: int,
    ):
        block_name = f"blocks.{stage_idx}.{block_idx}"
        # Stochastic depth
        drop_path_rate = self.drop_path_rate * total_idx / nb_blocks

        block_type = block_args.block_type
        block_args.filters = round_channels(block_args.filters, self.channel_multiplier)
        if block_args.force_in_channels is not None:
            block_args.force_in_channels = round_channels(
                block_args.force_in_channels, self.channel_multiplier
            )
        block_args.padding = self.padding
        block_args.norm_layer = self.norm_layer
        # block act fn overrides the model default
        block_args.act_layer = block_args.act_layer or self.act_layer
        assert block_args.act_layer is not None

        block_args.drop_path_rate = drop_path_rate
        if block_type != "cn":
            # TODO: Add parameter se_from_exp (used in Mobilenet v3) which does not
            #       adjust se_ratio.
            block_args.se_ratio /= block_args.exp_ratio

        if block_type == "ir":
            if block_args.nb_experts is not None:
                # TODO: Not implemented yet
                _log(f"  CondConvResidual {block_idx}, Args: {str(block_args)}")
                block = CondConvResidual(cfg=block_args, name=block_name)  # noqa: F821
            else:
                _log(f"  InvertedResidual {block_idx}, Args: {str(block_args)}")
                block = InvertedResidual(cfg=block_args, name=block_name)
        elif block_type in {"ds", "dsa"}:
            _log(f"  DepthwiseSeparable {block_idx}, Args: {str(block_args)}")
            block = DepthwiseSeparableConv(cfg=block_args, name=block_name)
        elif block_type == "er":
            _log(f"  EdgeResidual {block_idx}, Args: {str(block_args)}")
            block = EdgeResidual(cfg=block_args, name=block_name)
        elif block_type == "cn":
            _log(f"  ConvBnAct {block_idx}, Args: {str(block_args)}")
            block = ConvBnAct(cfg=block_args, name=block_name)  # noqa: F821
        else:
            raise ValueError(f"Unknown block type {block_type} while building model.")

        return block

    def __call__(self, architecture: List[List[BlockArgs]]) -> OrderedDict:
        """
        Build the blocks

        Args:
            architecture: A list of lists, outer list defines stages, inner list
                contains block configuration(s).

        Return:
             OrderedDict of block layers.
        """
        _log(f"Building model trunk with {len(architecture)} stages...")
        total_block_count = sum([len(x) for x in architecture])
        total_block_idx = 0
        current_stride = 2
        current_dilation = 1
        blocks = OrderedDict()

        # outer list of block_args defines the stacks
        for stack_idx, stack_args in enumerate(architecture):
            _log(f"Stack: {stack_idx}")
            assert isinstance(stack_args, list)

            # Each stack (stage of blocks) contains a list of block arguments
            for block_idx, block_args in enumerate(stack_args):
                _log(f" Block: {block_idx}")

                assert block_args.stride in {1, 2}
                # Only the first block in any stack can have a stride > 1
                if block_idx >= 1:
                    block_args.stride = 1

                next_dilation = current_dilation
                if block_args.stride > 1:
                    next_output_stride = current_stride * block_args.stride
                    if next_output_stride > self.output_stride:
                        next_dilation = current_dilation * block_args.stride
                        block_args.stride = 1
                        _log(
                            f"  Converting stride to dilation to maintain output "
                            f"stride of{self.output_stride}."
                        )
                    else:
                        current_stride = next_output_stride
                block_args.dilation_rate = current_dilation
                if next_dilation != current_dilation:
                    current_dilation = next_dilation

                block = self._make_block(
                    block_args,
                    stage_idx=stack_idx,
                    block_idx=block_idx,
                    total_idx=total_block_idx,
                    nb_blocks=total_block_count,
                )
                blocks[f"stage_{stack_idx}/block_{block_idx}"] = block

                total_block_idx += 1  # incr global block idx (across all stacks)
        return blocks
