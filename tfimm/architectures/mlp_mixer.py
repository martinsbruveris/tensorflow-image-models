"""
MLP-Mixer, ResMLP, and gMLP models

This implementation is ported from the timm, which is based on the original
implementation from the MLP-Mixer paper.

Official JAX impl:
https://github.com/google-research/vision_transformer/blob/linen/vit_jax/models_mixer.py

Paper: MLP-Mixer: An all-MLP Architecture for Vision
Arxiv: https://arxiv.org/abs/2105.01601

Also supporting ResMLP, and a preliminary implementation of gMLP

Code: https://github.com/facebookresearch/deit
Paper: ResMLP: Feedforward networks for image classification...
Arxiv: https://arxiv.org/abs/2105.03404

Paper: Pay Attention to MLPs
Arxiv: https://arxiv.org/abs/2105.08050

A thank you to paper authors for releasing code and weights.

Copyright 2021 Martins Bruveris
Copyright 2021 Ross Wightman
"""
from dataclasses import dataclass
from typing import Tuple

import tensorflow as tf

from tfimm.layers import DropPath, GluMLP, MLP, PatchEmbeddings, act_layer_factory, norm_layer_factory
from tfimm.models import ModelConfig, keras_serializable, register_model
from tfimm.utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# Model registry will add each entrypoint function to this
__all__ = ["MLPMixer", "MLPMixerConfig"]


@dataclass
class MLPMixerConfig(ModelConfig):
    nb_classes: int = 1000
    in_chans: int = 3
    input_size: Tuple[int, int] = (224, 224)
    patch_size: int = 16
    embed_dim: int = 512
    nb_blocks: int = 16
    mlp_ratio: Tuple[float, float] = (0.5, 4.0)
    block_layer: str = "mixer_block"
    mlp_layer: str = "mlp"
    # Regularization
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    # Other parameters
    norm_layer: str = "layer_norm_eps_1e-6"
    act_layer: str = "gelu"
    init_values: float = 1e-4  # Initialisation for ResBlocks
    nlhb: bool = False  # Negative logarithmic head bias
    stem_norm: bool = False
    # Parameters for inference
    crop_pct: float = 0.875
    interpolation: str = "bicubic"
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    # Weight transfer
    first_conv: str = "stem/proj"
    classifier: str = "head"

    @property
    def nb_patches(self) -> int:
        return (self.input_size[0] // self.patch_size) * (self.input_size[1] // self.patch_size)

#
# def _cfg(url='', **kwargs):
#     return {
#         'url': url,
#         'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
#         'crop_pct': 0.875, 'interpolation': 'bicubic', 'fixed_input_size': True,
#         'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
#         'first_conv': 'stem.proj', 'classifier': 'head',
#         **kwargs
#     }
#
#
# default_cfgs = dict(
#
#     resmlp_12_224=_cfg(
#         url='https://dl.fbaipublicfiles.com/deit/resmlp_12_no_dist.pth',
#         mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
#     resmlp_24_224=_cfg(
#         url='https://dl.fbaipublicfiles.com/deit/resmlp_24_no_dist.pth',
#         #url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resmlp_24_224_raa-a8256759.pth',
#         mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
#     resmlp_36_224=_cfg(
#         url='https://dl.fbaipublicfiles.com/deit/resmlp_36_no_dist.pth',
#         mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
#     resmlp_big_24_224=_cfg(
#         url='https://dl.fbaipublicfiles.com/deit/resmlpB_24_no_dist.pth',
#         mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
#
#     resmlp_12_distilled_224=_cfg(
#         url='https://dl.fbaipublicfiles.com/deit/resmlp_12_dist.pth',
#         mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
#     resmlp_24_distilled_224=_cfg(
#         url='https://dl.fbaipublicfiles.com/deit/resmlp_24_dist.pth',
#         mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
#     resmlp_36_distilled_224=_cfg(
#         url='https://dl.fbaipublicfiles.com/deit/resmlp_36_dist.pth',
#         mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
#     resmlp_big_24_distilled_224=_cfg(
#         url='https://dl.fbaipublicfiles.com/deit/resmlpB_24_dist.pth',
#         mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
#
#     resmlp_big_24_224_in22ft1k=_cfg(
#         url='https://dl.fbaipublicfiles.com/deit/resmlpB_24_22k.pth',
#         mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
#
#     resmlp_12_224_dino=_cfg(
#         url='https://dl.fbaipublicfiles.com/deit/resmlp_12_dino.pth',
#         mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
#     resmlp_24_224_dino=_cfg(
#         url='https://dl.fbaipublicfiles.com/deit/resmlp_24_dino.pth',
#         mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
#
#     gmlp_ti16_224=_cfg(),
#     gmlp_s16_224=_cfg(
#         url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gmlp_s16_224_raa-10536d42.pth',
#     ),
#     gmlp_b16_224=_cfg(),
# )


class MixerBlock(tf.keras.layers.Layer):
    """
    Residual Block w/ token mixing and channel MLPs
    Based on: "MLP-Mixer: An all-MLP Architecture for Vision"
    """
    def __init__(self, cfg: MLPMixerConfig, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg

        norm_layer = norm_layer_factory(cfg.norm_layer)
        mlp_layer = MLP_LAYER_DICT[cfg.mlp_layer]
        tokens_dim, channels_dim = [int(x * cfg.embed_dim) for x in cfg.mlp_ratio]

        self.norm1 = norm_layer(name="norm1")
        self.mlp_tokens = mlp_layer(
            hidden_dim=tokens_dim,
            embed_dim=cfg.nb_patches,
            drop_rate=cfg.drop_rate,
            act_layer=cfg.act_layer,
            name="mlp_tokens",
        )
        self.drop_path = (
            DropPath(drop_prob=cfg.drop_path_rate)
            if cfg.drop_path_rate > 0
            else tf.keras.layers.Activation("linear")
        )
        self.norm2 = norm_layer(name="norm2")
        self.mlp_channels = mlp_layer(
            hidden_dim=channels_dim,
            embed_dim=cfg.embed_dim,
            drop_rate=cfg.drop_rate,
            act_layer=cfg.act_layer,
            name="mlp_channels",
        )

    def call(self, x, training=False):
        shortcut = x
        x = self.norm1(x, training=training)
        x = tf.transpose(x, perm=(0, 2, 1))
        x = self.mlp_tokens(x, training=training)
        x = tf.transpose(x, perm=(0, 2, 1))
        x = self.drop_path(x, training=training)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x, training=training)
        x = self.mlp_channels(x, training=training)
        x = self.drop_path(x, training=training)
        x = x + shortcut

        return x


class ResBlock(tf.keras.layers.Layer):
    """
    Residual MLP block with LayerScale

    Based on: ResMLP: Feedforward networks for image classification...
    """
    def __init__(self, cfg: MLPMixerConfig, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg

        norm_layer = norm_layer_factory(cfg.norm_layer)
        mlp_layer = MLP_LAYER_DICT[cfg.mlp_layer]
        self.norm1 = norm_layer(name="norm1")
        self.linear_tokens = tf.keras.layers.Dense(
            units=cfg.nb_patches,
            name="linear_tokens",
        )
        self.drop_path = (
            DropPath(drop_prob=cfg.drop_path_rate)
            if cfg.drop_path_rate > 0
            else tf.keras.layers.Activation("linear")
        )
        self.norm2 = norm_layer(name="norm2")
        self.mlp_channels = mlp_layer(
            hidden_dim=int(cfg.embed_dim * cfg.mlp_ratio[1]),
            embed_dim=cfg.embed_dim,
            drop_rate=cfg.drop_rate,
            act_layer=cfg.act_layer,
            name="mlp_channels",
        )

    def build(self, input_shape):
        self.ls1 = self.add_weight(
            shape=(self.cfg.embed_dim,),
            initializer=tf.keras.initializers.Constant(self.cfg.init_values),
            trainable=True,
            name="ls1",
        )
        self.ls2 = self.add_weight(
            shape=(self.cfg.embed_dim,),
            initializer=tf.keras.initializers.Constant(self.cfg.init_values),
            trainable=True,
            name="ls2",
        )

    def call(self, x, training=False):
        shortcut = x
        x = self.norm1(x, training=training)
        x = tf.transpose(x, perm=(0, 2, 1))
        x = self.linear_tokens(x, training=training)
        x = tf.transpose(x, perm=(0, 2, 1))
        x = self.ls1 * x
        x = self.drop_path(x, training=training)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x, training=training)
        x = self.mlp_channels(x, training=training)
        x = self.ls2 * x
        x = self.drop_path(x, training=training)
        x = x + shortcut
        return x


# class SpatialGatingUnit(nn.Module):
#     """ Spatial Gating Unit
#
#     Based on: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
#     """
#     def __init__(self, dim, seq_len, norm_layer=nn.LayerNorm):
#         super().__init__()
#         gate_dim = dim // 2
#         self.norm = norm_layer(gate_dim)
#         self.proj = nn.Linear(seq_len, seq_len)
#
#     def init_weights(self):
#         # special init for the projection gate, called as override by base model init
#         nn.init.normal_(self.proj.weight, std=1e-6)
#         nn.init.ones_(self.proj.bias)
#
#     def forward(self, x):
#         u, v = x.chunk(2, dim=-1)
#         v = self.norm(v)
#         v = self.proj(v.transpose(-1, -2))
#         return u * v.transpose(-1, -2)
#
#
# class SpatialGatingBlock(nn.Module):
#     """ Residual Block w/ Spatial Gating
#
#     Based on: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
#     """
#     def __init__(
#             self, dim, seq_len, mlp_ratio=4, mlp_layer=GatedMlp,
#             norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, drop=0., drop_path=0.):
#         super().__init__()
#         channel_dim = int(dim * mlp_ratio)
#         self.norm = norm_layer(dim)
#         sgu = partial(SpatialGatingUnit, seq_len=seq_len)
#         self.mlp_channels = mlp_layer(dim, channel_dim, act_layer=act_layer, gate_layer=sgu, drop=drop)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#
#     def forward(self, x):
#         x = x + self.drop_path(self.mlp_channels(self.norm(x)))
#         return x


BLOCK_LAYER_DICT = {
    "mixer_block": MixerBlock,
    "res_block": ResBlock,
    # "spatial_gating_block": SpatialGatingBlock,
}

MLP_LAYER_DICT = {
    "mlp": MLP,
    "glu_mlp": GluMLP,
    # "gated_mlp": GatedMLP,
}


@keras_serializable
class MLPMixer(tf.keras.Model):
    cfg_class = MLPMixerConfig

    def __init__(self, cfg: MLPMixerConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.nb_features = cfg.embed_dim

        norm_layer = norm_layer_factory(cfg.norm_layer)
        block_layer = BLOCK_LAYER_DICT[cfg.block_layer]

        self.stem = PatchEmbeddings(
            patch_size=cfg.patch_size,
            embed_dim=cfg.embed_dim,
            norm_layer=cfg.norm_layer if cfg.stem_norm else "",
            name="stem",
        )
        self.blocks = [
            block_layer(cfg=cfg, name=f"blocks/{j}") for j in range(cfg.nb_blocks)
        ]
        self.norm = norm_layer(name="norm")
        self.head = (
            tf.keras.layers.Dense(units=cfg.nb_classes, name="head")
            if cfg.nb_classes > 0
            else tf.keras.layers.Activation("linear")
        )

    @property
    def dummy_inputs(self) -> tf.Tensor:
        return tf.zeros((1, *self.cfg.input_size, self.cfg.in_chans))

    def forward_features(self, x, training=False):
        x = self.stem(x, training=training)
        for block in self.blocks:
            x = block(x, training=training)
        x = self.norm(x, training=training)
        x = tf.reduce_mean(x, axis=1)
        return x

    def call(self, x, training=False):
        x = self.forward_features(x, training=training)
        x = self.head(x)
        return x


# class MlpMixer(nn.Module):
#
#     def __init__(
#             self,
#             num_classes=1000,
#             img_size=224,
#             in_chans=3,
#             patch_size=16,
#             num_blocks=8,
#             embed_dim=512,
#             mlp_ratio=(0.5, 4.0),
#             block_layer=MixerBlock,
#             mlp_layer=Mlp,
#             norm_layer=partial(nn.LayerNorm, eps=1e-6),
#             act_layer=nn.GELU,
#             drop_rate=0.,
#             drop_path_rate=0.,
#             nlhb=False,
#             stem_norm=False,
#     ):
#         super().__init__()
#         self.num_classes = num_classes
#         self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
#
#         self.stem = PatchEmbed(
#             img_size=img_size, patch_size=patch_size, in_chans=in_chans,
#             embed_dim=embed_dim, norm_layer=norm_layer if stem_norm else None)
#         # FIXME drop_path (stochastic depth scaling rule or all the same?)
#         self.blocks = nn.Sequential(*[
#             block_layer(
#                 embed_dim, self.stem.num_patches, mlp_ratio, mlp_layer=mlp_layer, norm_layer=norm_layer,
#                 act_layer=act_layer, drop=drop_rate, drop_path=drop_path_rate)
#             for _ in range(num_blocks)])
#         self.norm = norm_layer(embed_dim)
#         self.head = nn.Linear(embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
#
#         self.init_weights(nlhb=nlhb)
#
#     def init_weights(self, nlhb=False):
#         head_bias = -math.log(self.num_classes) if nlhb else 0.
#         named_apply(partial(_init_weights, head_bias=head_bias), module=self)  # depth-first
#
#     def get_classifier(self):
#         return self.head
#
#     def reset_classifier(self, num_classes, global_pool=''):
#         self.num_classes = num_classes
#         self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
#
#     def forward_features(self, x):
#         x = self.stem(x)
#         x = self.blocks(x)
#         x = self.norm(x)
#         x = x.mean(dim=1)
#         return x
#
#     def forward(self, x):
#         x = self.forward_features(x)
#         x = self.head(x)
#         return x
#
#
# def _init_weights(module: nn.Module, name: str, head_bias: float = 0., flax=False):
#     """ Mixer weight initialization (trying to match Flax defaults)
#     """
#     if isinstance(module, nn.Linear):
#         if name.startswith('head'):
#             nn.init.zeros_(module.weight)
#             nn.init.constant_(module.bias, head_bias)
#         else:
#             if flax:
#                 # Flax defaults
#                 lecun_normal_(module.weight)
#                 if module.bias is not None:
#                     nn.init.zeros_(module.bias)
#             else:
#                 # like MLP init in vit (my original init)
#                 nn.init.xavier_uniform_(module.weight)
#                 if module.bias is not None:
#                     if 'mlp' in name:
#                         nn.init.normal_(module.bias, std=1e-6)
#                     else:
#                         nn.init.zeros_(module.bias)
#     elif isinstance(module, nn.Conv2d):
#         lecun_normal_(module.weight)
#         if module.bias is not None:
#             nn.init.zeros_(module.bias)
#     elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
#         nn.init.ones_(module.weight)
#         nn.init.zeros_(module.bias)
#     elif hasattr(module, 'init_weights'):
#         # NOTE if a parent module contains init_weights method, it can override the init of the
#         # child modules as this will be called in depth-first order.
#         module.init_weights()
#
#
# def checkpoint_filter_fn(state_dict, model):
#     """ Remap checkpoints if needed """
#     if 'patch_embed.proj.weight' in state_dict:
#         # Remap FB ResMlp models -> timm
#         out_dict = {}
#         for k, v in state_dict.items():
#             k = k.replace('patch_embed.', 'stem.')
#             k = k.replace('attn.', 'linear_tokens.')
#             k = k.replace('mlp.', 'mlp_channels.')
#             k = k.replace('gamma_', 'ls')
#             if k.endswith('.alpha') or k.endswith('.beta'):
#                 v = v.reshape(1, 1, -1)
#             out_dict[k] = v
#         return out_dict
#     return state_dict


@register_model
def mixer_s32_224():
    """
    Mixer-S/32 224x224
    Paper: MLP-Mixer: An all-MLP Architecture for Vision
    Link: https://arxiv.org/abs/2105.01601
    """
    cfg = MLPMixerConfig(
        name="mixer_s32_224",
        url="",
        patch_size=32,
        embed_dim=512,
        nb_blocks=8,
    )
    return MLPMixer, cfg


@register_model
def mixer_s16_224():
    """
    Mixer-S/16 224x224
    Paper: MLP-Mixer: An all-MLP Architecture for Vision
    Link: https://arxiv.org/abs/2105.01601
    """
    cfg = MLPMixerConfig(
        name="mixer_s16_224",
        url="",
        patch_size=16,
        embed_dim=512,
        nb_blocks=8,
    )
    return MLPMixer, cfg


@register_model
def mixer_b32_224():
    """
    Mixer-B/32 224x224
    Paper: MLP-Mixer: An all-MLP Architecture for Vision
    Link: https://arxiv.org/abs/2105.01601
    """
    cfg = MLPMixerConfig(
        name="mixer_b32_224",
        url="",
        patch_size=32,
        embed_dim=768,
        nb_blocks=12,
    )
    return MLPMixer, cfg


@register_model
def mixer_b16_224():
    """ Mixer-B/16 224x224. ImageNet-1k pretrained weights.
    Paper: MLP-Mixer: An all-MLP Architecture for Vision
    Link: https://arxiv.org/abs/2105.01601
    """
    cfg = MLPMixerConfig(
        name="mixer_b16_224",
        url="",
        patch_size=16,
        embed_dim=768,
        nb_blocks=12,
    )
    return MLPMixer, cfg


@register_model
def mixer_b16_224_in21k():
    """
    Mixer-B/16 224x224. ImageNet-21k pretrained weights.
    Paper: MLP-Mixer: An all-MLP Architecture for Vision
    Link: https://arxiv.org/abs/2105.01601
    """
    cfg = MLPMixerConfig(
        name="mixer_b16_224_in21k",
        url="",
        nb_classes=21843,
        patch_size=16,
        embed_dim=768,
        nb_blocks=12,
    )
    return MLPMixer, cfg


@register_model
def mixer_l32_224():
    """
    Mixer-L/32 224x224.
    Paper: MLP-Mixer: An all-MLP Architecture for Vision
    Link: https://arxiv.org/abs/2105.01601
    """
    cfg = MLPMixerConfig(
        name="mixer_l32_224",
        url="",
        patch_size=32,
        embed_dim=1024,
        nb_blocks=24,
    )
    return MLPMixer, cfg


@register_model
def mixer_l16_224():
    """
    Mixer-L/16 224x224. ImageNet-1k pretrained weights.
    Paper: MLP-Mixer: An all-MLP Architecture for Vision
    Link: https://arxiv.org/abs/2105.01601
    """
    cfg = MLPMixerConfig(
        name="mixer_l16_224",
        url="",
        patch_size=16,
        embed_dim=1024,
        nb_blocks=24,
    )
    return MLPMixer, cfg


@register_model
def mixer_l16_224_in21k():
    """
    Mixer-L/16 224x224. ImageNet-21k pretrained weights.
    Paper: MLP-Mixer: An all-MLP Architecture for Vision
    Link: https://arxiv.org/abs/2105.01601
    """
    cfg = MLPMixerConfig(
        name="mixer_l16_224_in21k",
        url="",
        nb_classes=21843,
        patch_size=16,
        embed_dim=1024,
        nb_blocks=24,
    )
    return MLPMixer, cfg


@register_model
def mixer_b16_224_miil():
    """
    Mixer-B/16 224x224. ImageNet-21k pretrained weights.
    Paper: MLP-Mixer: An all-MLP Architecture for Vision
    Link: https://arxiv.org/abs/2105.01601
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    cfg = MLPMixerConfig(
        name="mixer_b16_224_miil",
        url="",
        patch_size=16,
        embed_dim=768,
        nb_blocks=12,
        crop_pct=0.875,
        interpolation="bilinear",
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )
    return MLPMixer, cfg


@register_model
def mixer_b16_224_miil_in21k():
    """
    Mixer-B/16 224x224. ImageNet-1k pretrained weights.
    Paper: MLP-Mixer: An all-MLP Architecture for Vision
    Link: https://arxiv.org/abs/2105.01601
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    cfg = MLPMixerConfig(
        name="mixer_b16_224_miil_in21k",
        url="",
        nb_classes=11221,
        patch_size=16,
        embed_dim=768,
        nb_blocks=12,
        crop_pct=0.875,
        interpolation="bilinear",
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )
    return MLPMixer, cfg


@register_model
def gmixer_12_224():
    """
    Glu-Mixer-12 224x224
    Experiment by Ross Wightman, adding (Si)GLU to MLP-Mixer
    """
    cfg = MLPMixerConfig(
        name="gmixer_12_224",
        url="",
        patch_size=16,
        embed_dim=384,
        nb_blocks=12,
        mlp_ratio=(1.0, 4.0),
        mlp_layer="glu_mlp",
        act_layer="swish",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )
    return MLPMixer, cfg


@register_model
def gmixer_24_224():
    """
    Glu-Mixer-24 224x224
    Experiment by Ross Wightman, adding (Si)GLU to MLP-Mixer
    """
    cfg = MLPMixerConfig(
        name="gmixer_24_224",
        url="",
        patch_size=16,
        embed_dim=384,
        nb_blocks=24,
        mlp_ratio=(1.0, 4.0),
        mlp_layer="glu_mlp",
        act_layer="swish",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )
    return MLPMixer, cfg


@register_model
def resmlp_12_224():
    """
    ResMLP-12
    Paper: ResMLP: Feedforward networks for image classification...
    Link: https://arxiv.org/abs/2105.03404
    """
    cfg = MLPMixerConfig(
        name="resmlp_12_224",
        url="",
        patch_size=16,
        embed_dim=384,
        nb_blocks=12,
        mlp_ratio=(4.0, 4.0),
        block_layer="res_block",
        norm_layer="affine",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )
    return MLPMixer, cfg


@register_model
def resmlp_24_224():
    """
    ResMLP-24
    Paper: ResMLP: Feedforward networks for image classification...
    Link: https://arxiv.org/abs/2105.03404
    """
    cfg = MLPMixerConfig(
        name="resmlp_24_224",
        url="",
        patch_size=16,
        embed_dim=384,
        nb_blocks=24,
        mlp_ratio=(4.0, 4.0),
        block_layer="res_block",
        init_values=1e-5,
        norm_layer="affine",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )
    return MLPMixer, cfg


@register_model
def resmlp_36_224():
    """
    ResMLP-36
    Paper: ResMLP: Feedforward networks for image classification...
    Link: https://arxiv.org/abs/2105.03404
    """
    cfg = MLPMixerConfig(
        name="resmlp_36_224",
        url="",
        patch_size=16,
        embed_dim=384,
        nb_blocks=36,
        mlp_ratio=(4.0, 4.0),
        block_layer="res_block",
        init_values=1e-6,
        norm_layer="affine",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )
    return MLPMixer, cfg


@register_model
def resmlp_big_24_224():
    """
    ResMLP-B-24
    Paper: ResMLP: Feedforward networks for image classification...
    Link: https://arxiv.org/abs/2105.03404
    """
    cfg = MLPMixerConfig(
        name="resmlp_big_24_224",
        url="",
        patch_size=8,
        embed_dim=768,
        nb_blocks=24,
        mlp_ratio=(4.0, 4.0),
        block_layer="res_block",
        init_values=1e-6,
        norm_layer="affine",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )
    return MLPMixer, cfg


@register_model
def resmlp_12_distilled_224():
    """
    ResMLP-12
    Paper: ResMLP: Feedforward networks for image classification...
    Link: https://arxiv.org/abs/2105.03404
    """
    cfg = MLPMixerConfig(
        name="resmlp_12_distilled_224",
        url="",
        patch_size=16,
        embed_dim=384,
        nb_blocks=16,
        mlp_ratio=(4.0, 4.0),
        block_layer="res_block",
        norm_layer="affine",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )
    return MLPMixer, cfg


@register_model
def resmlp_24_distilled_224():
    """
    ResMLP-24
    Paper: ResMLP: Feedforward networks for image classification...
    Link: https://arxiv.org/abs/2105.03404
    """
    cfg = MLPMixerConfig(
        name="resmlp_24_distilled_224",
        url="",
        patch_size=16,
        embed_dim=384,
        nb_blocks=24,
        mlp_ratio=(4.0, 4.0),
        block_layer="res_block",
        init_values=1e-5,
        norm_layer="affine",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )
    return MLPMixer, cfg


@register_model
def resmlp_36_distilled_224():
    """
    ResMLP-36
    Paper: ResMLP: Feedforward networks for image classification...
    Link: https://arxiv.org/abs/2105.03404
    """
    cfg = MLPMixerConfig(
        name="resmlp_36_distilled_224",
        url="",
        patch_size=16,
        embed_dim=384,
        nb_blocks=36,
        mlp_ratio=(4.0, 4.0),
        block_layer="res_block",
        init_values=1e-6,
        norm_layer="affine",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )
    return MLPMixer, cfg


@register_model
def resmlp_big_24_distilled_224():
    """
    ResMLP-B-24
    Paper: ResMLP: Feedforward networks for image classification...
    Link: https://arxiv.org/abs/2105.03404
    """
    cfg = MLPMixerConfig(
        name="resmlp_big_24_distilled_224",
        url="",
        patch_size=8,
        embed_dim=768,
        nb_blocks=24,
        mlp_ratio=(4.0, 4.0),
        block_layer="res_block",
        init_values=1e-6,
        norm_layer="affine",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )
    return MLPMixer, cfg


@register_model
def resmlp_big_24_224_in22ft1k():
    """
    ResMLP-B-24
    Paper: ResMLP: Feedforward networks for image classification...
    Link: https://arxiv.org/abs/2105.03404
    """
    cfg = MLPMixerConfig(
        name="resmlp_big_24_224_in22ft1k",
        url="",
        patch_size=8,
        embed_dim=768,
        nb_blocks=24,
        mlp_ratio=(4.0, 4.0),
        block_layer="res_block",
        init_values=1e-6,
        norm_layer="affine",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )
    return MLPMixer, cfg


@register_model
def resmlp_12_224_dino():
    """
    ResMLP-12
    Paper: ResMLP: Feedforward networks for image classification...
    Link: https://arxiv.org/abs/2105.03404
    Model pretrained via DINO (self-supervised) - https://arxiv.org/abs/2104.14294
    """
    cfg = MLPMixerConfig(
        name="resmlp_12_224_dino",
        url="",
        patch_size=16,
        embed_dim=384,
        nb_blocks=12,
        mlp_ratio=(4.0, 4.0),
        block_layer="res_block",
        norm_layer="affine",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )
    return MLPMixer, cfg


@register_model
def resmlp_24_224_dino():
    """
    ResMLP-24
    Paper: ResMLP: Feedforward networks for image classification...
    Link: https://arxiv.org/abs/2105.03404
    Model pretrained via DINO (self-supervised) - https://arxiv.org/abs/2104.14294
    """
    cfg = MLPMixerConfig(
        name="resmlp_24_224_dino",
        url="",
        patch_size=16,
        embed_dim=384,
        nb_blocks=24,
        mlp_ratio=(4.0, 4.0),
        block_layer="res_block",
        init_values=1e-5,
        norm_layer="affine",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )
    return MLPMixer, cfg


# @register_model
# def gmlp_ti16_224(pretrained=False, **kwargs):
#     """ gMLP-Tiny
#     Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
#     """
#     model_args = dict(
#         patch_size=16, num_blocks=30, embed_dim=128, mlp_ratio=6, block_layer=SpatialGatingBlock,
#         mlp_layer=GatedMlp, **kwargs)
#     model = _create_mixer('gmlp_ti16_224', pretrained=pretrained, **model_args)
#     return model
#
#
# @register_model
# def gmlp_s16_224(pretrained=False, **kwargs):
#     """ gMLP-Small
#     Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
#     """
#     model_args = dict(
#         patch_size=16, num_blocks=30, embed_dim=256, mlp_ratio=6, block_layer=SpatialGatingBlock,
#         mlp_layer=GatedMlp, **kwargs)
#     model = _create_mixer('gmlp_s16_224', pretrained=pretrained, **model_args)
#     return model
#
#
# @register_model
# def gmlp_b16_224(pretrained=False, **kwargs):
#     """ gMLP-Base
#     Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
#     """
#     model_args = dict(
#         patch_size=16, num_blocks=30, embed_dim=512, mlp_ratio=6, block_layer=SpatialGatingBlock,
#         mlp_layer=GatedMlp, **kwargs)
#     model = _create_mixer('gmlp_b16_224', pretrained=pretrained, **model_args)
#     return model
