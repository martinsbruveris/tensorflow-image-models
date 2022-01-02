"""
TensorFlow implementation of the Visual Transformer

Based on timm/models/visual_transformer.py by Ross Wightman.
Based on transformers/models/vit by HuggingFace

Copyright 2021 Martins Bruveris
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import tensorflow as tf

from tfimm.layers import (
    MLP,
    DropPath,
    PatchEmbeddings,
    interpolate_pos_embeddings,
    norm_layer_factory,
)
from tfimm.models import ModelConfig, keras_serializable, register_model
from tfimm.utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)

from .resnetv2 import ResNetV2, ResNetV2Config, ResNetV2Stem

# model_registry will add each entrypoint fn to this
__all__ = ["ViT", "ViTConfig"]


@dataclass
class ViTConfig(ModelConfig):
    nb_classes: int = 1000
    in_channels: int = 3
    input_size: Tuple[int, int] = (224, 224)
    patch_layer: str = "patch_embeddings"
    patch_nb_blocks: tuple = ()
    patch_size: int = 16
    embed_dim: int = 768
    nb_blocks: int = 12
    nb_heads: int = 12
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    representation_size: Optional[int] = None
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
    crop_pct: float = 0.875
    interpolation: str = "bicubic"
    mean: Tuple[float, float, float] = IMAGENET_INCEPTION_MEAN
    std: Tuple[float, float, float] = IMAGENET_INCEPTION_STD
    first_conv: str = "patch_embed/proj"
    # DeiT models have two classifier heads, one for distillation
    classifier: Union[str, Tuple[str, str]] = "head"

    """
    Args:
        nb_classes: Number of classes for classification head
        in_channels: Number of input channels
        input_size: Input image size
        patch_layer: Layer used to transform image to patches. Possible values are
            `patch_embeddings` and `hybrid_embeddings`.
        patch_nb_blocks: When `patch_layer="hybrid_embeddings`, this is the number of
            residual blocks in each stage. Set to `()` to use only the stem.
        patch_size: Patch size; Image size must be multiple of patch size. For hybrid
            embedding layer, this patch size is applied after the convolutional layers.
        embed_dim: Embedding dimension
        nb_blocks: Depth of transformer (number of encoder blocks)
        nb_heads: Number of self-attention heads
        mlp_ratio: Ratio of mlp hidden dim to embedding dim
        qkv_bias: Enable bias for qkv if True
        representation_size: Enable and set representation layer (pre-logits) to this
            value if set
        distilled: Model includes a distillation token and head as in DeiT models
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        drop_path_rate: Dropout rate for stochastic depth
        norm_layer: Normalization layer
        act_layer: Activation function
    """

    @property
    def nb_tokens(self) -> int:
        """Number of special tokens"""
        return 2 if self.distilled else 1

    @property
    def grid_size(self) -> Tuple[int, int]:
        grid_size = (
            self.input_size[0] // self.patch_size,
            self.input_size[1] // self.patch_size,
        )
        if self.patch_layer == "hybrid_embeddings":
            # 2 reductions in the stem, 1 reduction in each stage except the first one
            reductions = 2 + max(len(self.patch_nb_blocks) - 1, 0)
            stride = 2 ** reductions
            grid_size = (grid_size[0] // stride, grid_size[1] // stride)
        return grid_size

    @property
    def nb_patches(self) -> int:
        """Number of patches without class and distillation tokens."""
        return self.grid_size[0] * self.grid_size[1]

    @property
    def transform_weights(self):
        return {"pos_embed": ViT.transform_pos_embed}


class ViTMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, cfg: ViTConfig, **kwargs):
        super().__init__(**kwargs)
        head_dim = cfg.embed_dim // cfg.nb_heads
        self.scale = head_dim ** -0.5
        self.cfg = cfg

        self.qkv = tf.keras.layers.Dense(
            units=3 * cfg.embed_dim, use_bias=cfg.qkv_bias, name="qkv"
        )
        self.attn_drop = tf.keras.layers.Dropout(rate=cfg.attn_drop_rate)
        self.proj = tf.keras.layers.Dense(units=cfg.embed_dim, name="proj")
        self.proj_drop = tf.keras.layers.Dropout(rate=cfg.drop_rate)

    def call(self, x, training=False):
        # B (batch size), N (sequence length), D (embedding dimension),
        # H (number of heads)
        batch_size, seq_length = tf.unstack(tf.shape(x)[:2])
        qkv = self.qkv(x)  # (B, N, 3*D)
        qkv = tf.reshape(qkv, (batch_size, seq_length, 3, self.cfg.nb_heads, -1))
        qkv = tf.transpose(qkv, (2, 0, 3, 1, 4))  # (3, B, H, N, D/H)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = self.scale * tf.linalg.matmul(q, k, transpose_b=True)  # (B, H, N, N)
        attn = tf.nn.softmax(attn, axis=-1)  # (B, H, N, N)
        attn = self.attn_drop(attn, training=training)

        x = tf.linalg.matmul(attn, v)  # (B, H, N, D/H)
        x = tf.transpose(x, (0, 2, 1, 3))  # (B, N, H, D/H)
        x = tf.reshape(x, (batch_size, seq_length, -1))  # (B, N, D)

        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        return x


class Block(tf.keras.layers.Layer):
    def __init__(self, cfg: ViTConfig, **kwargs):
        super().__init__(**kwargs)
        self.norm_layer = norm_layer_factory(cfg.norm_layer)

        self.norm1 = self.norm_layer(name="norm1")
        self.attn = ViTMultiHeadAttention(cfg, name="attn")
        self.drop_path = DropPath(drop_prob=cfg.drop_path_rate)
        self.norm2 = self.norm_layer(name="norm2")
        self.mlp = MLP(
            hidden_dim=int(cfg.embed_dim * cfg.mlp_ratio),
            embed_dim=cfg.embed_dim,
            drop_rate=cfg.drop_rate,
            act_layer=cfg.act_layer,
            name="mlp",
        )

    def call(self, x, training=False):
        shortcut = x
        x = self.norm1(x, training=training)
        x = self.attn(x, training=training)
        x = self.drop_path(x, training=training)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x, training=training)
        x = self.mlp(x, training=training)
        x = self.drop_path(x, training=training)
        x = x + shortcut
        return x


class HybridEmbeddings(tf.keras.layers.Layer):
    """
    CNN feature map embedding

    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(
        self,
        in_channels: int,
        input_size: tuple,
        nb_blocks: tuple,
        patch_size: int,
        embed_dim: int,
        drop_path_rate: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if nb_blocks == ():
            self.backbone = ResNetV2Stem(
                stem_type="same",
                stem_width=64,
                conv_padding="same",
                preact=False,
                act_layer="relu",
                norm_layer="group_norm",
                name="backbone",
            )
        else:
            backbone_cfg = ResNetV2Config(
                nb_classes=0,
                in_channels=in_channels,
                input_size=input_size,
                nb_blocks=nb_blocks,
                preact=False,
                stem_type="same",
                global_pool="",
                conv_padding="same",
                drop_path_rate=drop_path_rate,
            )
            self.backbone = ResNetV2(backbone_cfg, name="backbone")

        self.projection = tf.keras.layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            use_bias=True,
            name="proj",
        )

    def call(self, x, training=False, return_shape=False):
        x = self.backbone(x, training=training)
        x = self.projection(x)

        # Change the 2D spatial dimensions to a single temporal dimension.
        batch_size, height, width = tf.unstack(tf.shape(x)[:3])
        x = tf.reshape(tensor=x, shape=(batch_size, height * width, -1))
        return (x, (height, width)) if return_shape else x


@keras_serializable
class ViT(tf.keras.Model):
    cfg_class = ViTConfig

    def __init__(self, cfg: ViTConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nb_features = cfg.embed_dim  # For consistency with other models
        self.norm_layer = norm_layer_factory(cfg.norm_layer)
        self.cfg = cfg

        if cfg.patch_layer == "patch_embeddings":
            self.patch_embed = PatchEmbeddings(
                patch_size=cfg.patch_size,
                embed_dim=cfg.embed_dim,
                norm_layer="",  # ViT does not use normalization in patch embeddings
                name="patch_embed",
            )
        elif cfg.patch_layer == "hybrid_embeddings":
            self.patch_embed = HybridEmbeddings(
                in_channels=cfg.in_channels,
                input_size=cfg.input_size,
                nb_blocks=cfg.patch_nb_blocks,
                patch_size=cfg.patch_size,
                embed_dim=cfg.embed_dim,
                drop_path_rate=cfg.drop_path_rate,
                name="patch_embed",
            )
        else:
            raise ValueError(f"Unknown patch layer: {cfg.patch_layer}.")
        self.cls_token = None
        self.dist_token = None
        self.pos_embed = None
        self.pos_drop = tf.keras.layers.Dropout(rate=cfg.drop_rate)

        self.blocks = [Block(cfg, name=f"blocks/{j}") for j in range(cfg.nb_blocks)]
        self.norm = self.norm_layer(name="norm")

        # Some models have a representation layer on top of cls token
        if cfg.representation_size:
            if cfg.distilled:
                raise ValueError(
                    "Cannot combine distillation token and a representation layer."
                )
            self.nb_features = cfg.representation_size
            self.pre_logits = tf.keras.layers.Dense(
                units=cfg.representation_size, activation="tanh", name="pre_logits/fc"
            )
        else:
            self.pre_logits = None

        # Classifier head(s)
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
        self.cls_token = self.add_weight(
            shape=(1, 1, self.cfg.embed_dim),
            initializer="zeros",
            trainable=True,
            name="cls_token",
        )
        self.dist_token = (
            self.add_weight(
                shape=(1, 1, self.cfg.embed_dim),
                initializer="zeros",
                trainable=True,
                name="dist_token",
            )
            if self.cfg.distilled
            else None
        )
        self.pos_embed = self.add_weight(
            shape=(1, self.cfg.nb_patches + self.cfg.nb_tokens, self.cfg.embed_dim),
            initializer="zeros",
            trainable=True,
            name="pos_embed",
        )

    @property
    def dummy_inputs(self) -> tf.Tensor:
        return tf.zeros((1, *self.cfg.input_size, self.cfg.in_channels))

    @property
    def feature_names(self) -> List[str]:
        return (
            ["patch_embedding"]
            + [f"block_{j}" for j in range(self.cfg.nb_blocks)]
            + ["features_all", "features", "logits"]
        )

    def transform_pos_embed(self, target_cfg: ViTConfig):
        return interpolate_pos_embeddings(
            pos_embed=self.pos_embed,
            src_grid_size=self.cfg.grid_size,
            tgt_grid_size=target_cfg.grid_size,
            nb_tokens=self.cfg.nb_tokens,
        )

    def forward_features(self, x, training=False, return_features=False):
        features = {}
        batch_size = tf.shape(x)[0]

        x, grid_size = self.patch_embed(x, return_shape=True)
        cls_token = tf.repeat(self.cls_token, repeats=batch_size, axis=0)
        if not self.cfg.distilled:
            x = tf.concat((cls_token, x), axis=1)
        else:
            dist_token = tf.repeat(self.dist_token, repeats=batch_size, axis=0)
            x = tf.concat((cls_token, dist_token, x), axis=1)
        if not self.cfg.interpolate_input:
            x = x + self.pos_embed
        else:
            pos_embed = interpolate_pos_embeddings(
                self.pos_embed,
                src_grid_size=self.cfg.grid_size,
                tgt_grid_size=grid_size,
                nb_tokens=self.cfg.nb_tokens,
            )
            x = x + pos_embed
        x = self.pos_drop(x, training=training)
        features["patch_embedding"] = x

        for j, block in enumerate(self.blocks):
            x = block(x, training=training)
            features[f"block_{j}"] = x
        x = self.norm(x, training=training)
        features["features_all"] = x

        if self.cfg.distilled:
            # Here we diverge from timm and return both outputs as one tensor. That way
            # all models always have one output by default
            x = x[:, :2]
        elif self.cfg.representation_size:
            x = self.pre_logits(x[:, 0])
        else:
            x = x[:, 0]
        features["features"] = x
        return (x, features) if return_features else x

    def call(self, x, training=False, return_features=False):
        features = {}
        x = self.forward_features(x, training, return_features)
        if return_features:
            x, features = x
        if not self.cfg.distilled:
            x = self.head(x)
        else:
            y = self.head(x[:, 0])
            y_dist = self.head_dist(x[:, 1])
            x = tf.stack((y, y_dist), axis=1)
        features["logits"] = x
        return (x, features) if return_features else x


@register_model
def vit_tiny_patch16_224():
    """ViT-Tiny (Vit-Ti/16)"""
    cfg = ViTConfig(
        name="vit_tiny_patch16_224",
        url="",
        patch_size=16,
        embed_dim=192,
        nb_blocks=12,
        nb_heads=3,
    )
    return ViT, cfg


@register_model
def vit_tiny_patch16_384():
    """ViT-Tiny (Vit-Ti/16) @ 384x384."""
    cfg = ViTConfig(
        name="vit_tiny_patch16_384",
        url="",
        input_size=(384, 384),
        patch_size=16,
        embed_dim=192,
        nb_blocks=12,
        nb_heads=3,
        crop_pct=1.0,
    )
    return ViT, cfg


@register_model
def vit_small_patch32_224():
    """ViT-Small (ViT-S/32)"""
    cfg = ViTConfig(
        name="vit_small_patch32_224",
        url="",
        patch_size=32,
        embed_dim=384,
        nb_blocks=12,
        nb_heads=6,
    )
    return ViT, cfg


@register_model
def vit_small_patch32_384():
    """ViT-Small (ViT-S/32) at 384x384."""
    cfg = ViTConfig(
        name="vit_small_patch32_384",
        url="",
        input_size=(384, 384),
        patch_size=32,
        embed_dim=384,
        nb_blocks=12,
        nb_heads=6,
        crop_pct=1.0,
    )
    return ViT, cfg


@register_model
def vit_small_patch16_224():
    """ViT-Small (ViT-S/16)"""
    cfg = ViTConfig(
        name="vit_small_patch16_224",
        url="",
        patch_size=16,
        embed_dim=384,
        nb_blocks=12,
        nb_heads=6,
    )
    return ViT, cfg


@register_model
def vit_small_patch16_384():
    """ViT-Small (ViT-S/16)"""
    cfg = ViTConfig(
        name="vit_small_patch16_384",
        url="",
        input_size=(384, 384),
        patch_size=16,
        embed_dim=384,
        nb_blocks=12,
        nb_heads=6,
        crop_pct=1.0,
    )
    return ViT, cfg


@register_model
def vit_base_patch32_224():
    """
    ViT-Base (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k, source
    https://github.com/google-research/vision_transformer.
    """
    cfg = ViTConfig(
        name="vit_base_patch32_224",
        url="",
        patch_size=32,
        embed_dim=768,
        nb_blocks=12,
        nb_heads=12,
    )
    return ViT, cfg


@register_model
def vit_base_patch32_384():
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source
    https://github.com/google-research/vision_transformer.
    """
    cfg = ViTConfig(
        name="vit_base_patch32_384",
        url="",
        input_size=(384, 384),
        patch_size=32,
        embed_dim=768,
        nb_blocks=12,
        nb_heads=12,
        crop_pct=1.0,
    )
    return ViT, cfg


@register_model
def vit_base_patch16_224():
    """
    ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source
    https://github.com/google-research/vision_transformer.
    """
    cfg = ViTConfig(
        name="vit_base_patch16_224",
        url="",
        patch_size=16,
        embed_dim=768,
        nb_blocks=12,
        nb_heads=12,
    )
    return ViT, cfg


@register_model
def vit_base_patch16_384():
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source
    https://github.com/google-research/vision_transformer.
    """
    cfg = ViTConfig(
        name="vit_base_patch16_384",
        url="",
        input_size=(384, 384),
        patch_size=16,
        embed_dim=768,
        nb_blocks=12,
        nb_heads=12,
        crop_pct=1.0,
    )
    return ViT, cfg


@register_model
def vit_base_patch8_224():
    """
    ViT-Base (ViT-B/8) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source
    https://github.com/google-research/vision_transformer.
    """
    cfg = ViTConfig(
        name="vit_base_patch8_224",
        url="",
        patch_size=8,
        embed_dim=768,
        nb_blocks=12,
        nb_heads=12,
    )
    return ViT, cfg


@register_model
def vit_large_patch32_224():
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    No pretrained weights.
    """
    cfg = ViTConfig(
        name="vit_large_patch32_224",
        url="",
        patch_size=32,
        embed_dim=1024,
        nb_blocks=24,
        nb_heads=16,
    )
    return ViT, cfg


@register_model
def vit_large_patch32_384():
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source
    https://github.com/google-research/vision_transformer.
    """
    cfg = ViTConfig(
        name="vit_large_patch32_384",
        url="",
        input_size=(384, 384),
        patch_size=32,
        embed_dim=1024,
        nb_blocks=24,
        nb_heads=16,
        crop_pct=1.0,
    )
    return ViT, cfg


@register_model
def vit_large_patch16_224():
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source
    https://github.com/google-research/vision_transformer.
    """
    cfg = ViTConfig(
        name="vit_large_patch16_224",
        url="",
        patch_size=16,
        embed_dim=1024,
        nb_blocks=24,
        nb_heads=16,
    )
    return ViT, cfg


@register_model
def vit_large_patch16_384():
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source
    https://github.com/google-research/vision_transformer.
    """
    cfg = ViTConfig(
        name="vit_large_patch16_384",
        url="",
        input_size=(384, 384),
        patch_size=16,
        embed_dim=1024,
        nb_blocks=24,
        nb_heads=16,
        crop_pct=1.0,
    )
    return ViT, cfg


@register_model
def vit_base_patch32_sam_224():
    """
    ViT-Base (ViT-B/32) w/ SAM pretrained weights.
    Paper: https://arxiv.org/abs/2106.01548
    """
    cfg = ViTConfig(
        name="vit_base_patch32_sam_224",
        url="",
        patch_size=32,
        embed_dim=768,
        nb_blocks=12,
        nb_heads=12,
    )
    return ViT, cfg


@register_model
def vit_base_patch16_sam_224():
    """
    ViT-Base (ViT-B/16) w/ SAM pretrained weights.
    Paper: https://arxiv.org/abs/2106.01548
    """
    cfg = ViTConfig(
        name="vit_base_patch16_sam_224",
        url="",
        patch_size=16,
        embed_dim=768,
        nb_blocks=12,
        nb_heads=12,
    )
    return ViT, cfg


@register_model
def vit_tiny_patch16_224_in21k():
    """
    ViT-Tiny (Vit-Ti/16). ImageNet-21k weights @ 224x224, source
    https://github.com/google-research/vision_transformer.
    Note: This model has a valid 21k classifier head and no representation layer.
    """
    cfg = ViTConfig(
        name="vit_tiny_patch16_224_in21k",
        url="",
        nb_classes=21843,
        patch_size=16,
        embed_dim=192,
        nb_blocks=12,
        nb_heads=3,
    )
    return ViT, cfg


@register_model
def vit_small_patch32_224_in21k():
    """
    ViT-Small (ViT-S/16) ImageNet-21k weights @ 224x224, source
    https://github.com/google-research/vision_transformer.
    Note: This model has a valid 21k classifier head and no representation layer.
    """
    cfg = ViTConfig(
        name="vit_small_patch32_224_in21k",
        url="",
        nb_classes=21843,
        patch_size=32,
        embed_dim=384,
        nb_blocks=12,
        nb_heads=6,
    )
    return ViT, cfg


@register_model
def vit_small_patch16_224_in21k():
    """
    ViT-Small (ViT-S/16) ImageNet-21k weights @ 224x224, source
    https://github.com/google-research/vision_transformer.
    Note: This model has a valid 21k classifier head and no representation layer.
    """
    cfg = ViTConfig(
        name="vit_small_patch16_224_in21k",
        url="",
        nb_classes=21843,
        patch_size=16,
        embed_dim=384,
        nb_blocks=12,
        nb_heads=6,
    )
    return ViT, cfg


@register_model
def vit_base_patch32_224_in21k():
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source
    https://github.com/google-research/vision_transformer.
    Note: This model has a valid 21k classifier head and no representation layer.
    """
    cfg = ViTConfig(
        name="vit_base_patch32_224_in21k",
        url="",
        nb_classes=21843,
        patch_size=32,
        embed_dim=768,
        nb_blocks=12,
        nb_heads=12,
    )
    return ViT, cfg


@register_model
def vit_base_patch16_224_in21k():
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source
    https://github.com/google-research/vision_transformer.
    Note: This model has a valid 21k classifier head and no representation layer.
    """
    cfg = ViTConfig(
        name="vit_base_patch16_224_in21k",
        url="",
        nb_classes=21843,
        patch_size=16,
        embed_dim=768,
        nb_blocks=12,
        nb_heads=12,
    )
    return ViT, cfg


@register_model
def vit_base_patch8_224_in21k():
    """ViT-Base model (ViT-B/8) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source
    https://github.com/google-research/vision_transformer.
    Note: This model has a valid 21k classifier head and no representation layer.
    """
    cfg = ViTConfig(
        name="vit_base_patch8_224_in21k",
        url="",
        nb_classes=21843,
        patch_size=8,
        embed_dim=768,
        nb_blocks=12,
        nb_heads=12,
    )
    return ViT, cfg


@register_model
def vit_large_patch32_224_in21k():
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source
    https://github.com/google-research/vision_transformer.
    Note: This model has a representation layer but the 21k classifier head is zero'd
    out in original weights.
    """
    cfg = ViTConfig(
        name="vit_large_patch32_224_in21k",
        url="",
        nb_classes=21843,
        patch_size=32,
        embed_dim=1024,
        nb_blocks=24,
        nb_heads=16,
        representation_size=1024,
    )
    return ViT, cfg


@register_model
def vit_large_patch16_224_in21k():
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source
    https://github.com/google-research/vision_transformer.
    Note: This model has a valid 21k classifier head and no representation layer.
    """
    cfg = ViTConfig(
        name="vit_large_patch16_224_in21k",
        url="",
        nb_classes=21843,
        patch_size=16,
        embed_dim=1024,
        nb_blocks=24,
        nb_heads=16,
    )
    return ViT, cfg


@register_model
def vit_huge_patch14_224_in21k():
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source
    https://github.com/google-research/vision_transformer.
    Note: This model has a representation layer but the 21k classifier head is zero'd
    out in original weights.
    """
    cfg = ViTConfig(
        name="vit_huge_patch14_224_in21k",
        url="",
        nb_classes=21843,
        patch_size=14,
        embed_dim=1280,
        nb_blocks=32,
        nb_heads=16,
        representation_size=1280,
    )
    return ViT, cfg


@register_model
def deit_tiny_patch16_224():
    """
    DeiT-tiny model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    cfg = ViTConfig(
        name="deit_tiny_patch16_224",
        url="",
        patch_size=16,
        embed_dim=192,
        nb_blocks=12,
        nb_heads=3,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )
    return ViT, cfg


@register_model
def deit_small_patch16_224():
    """
    DeiT-small model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    cfg = ViTConfig(
        name="deit_small_patch16_224",
        url="",
        patch_size=16,
        embed_dim=384,
        nb_blocks=12,
        nb_heads=6,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )
    return ViT, cfg


@register_model
def deit_base_patch16_224():
    """
    DeiT base model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    cfg = ViTConfig(
        name="deit_base_patch16_224",
        url="",
        patch_size=16,
        embed_dim=768,
        nb_blocks=12,
        nb_heads=12,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )
    return ViT, cfg


@register_model
def deit_base_patch16_384():
    """
    DeiT base model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    cfg = ViTConfig(
        name="deit_base_patch16_384",
        url="",
        input_size=(384, 384),
        patch_size=16,
        embed_dim=768,
        nb_blocks=12,
        nb_heads=12,
        crop_pct=1.0,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )
    return ViT, cfg


@register_model
def deit_tiny_distilled_patch16_224():
    """
    DeiT-tiny distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    cfg = ViTConfig(
        name="deit_tiny_distilled_patch16_224",
        url="",
        patch_size=16,
        embed_dim=192,
        nb_blocks=12,
        nb_heads=3,
        distilled=True,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        classifier=("head", "head_dist"),
    )
    return ViT, cfg


@register_model
def deit_small_distilled_patch16_224():
    """
    DeiT-small distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    cfg = ViTConfig(
        name="deit_small_distilled_patch16_224",
        url="",
        patch_size=16,
        embed_dim=384,
        nb_blocks=12,
        nb_heads=6,
        distilled=True,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        classifier=("head", "head_dist"),
    )
    return ViT, cfg


@register_model
def deit_base_distilled_patch16_224():
    """
    DeiT-base distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    cfg = ViTConfig(
        name="deit_base_distilled_patch16_224",
        url="",
        patch_size=16,
        embed_dim=768,
        nb_blocks=12,
        nb_heads=12,
        distilled=True,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        classifier=("head", "head_dist"),
    )
    return ViT, cfg


@register_model
def deit_base_distilled_patch16_384():
    """
    DeiT-base distilled model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    cfg = ViTConfig(
        name="deit_base_distilled_patch16_384",
        url="",
        input_size=(384, 384),
        patch_size=16,
        embed_dim=768,
        nb_blocks=12,
        nb_heads=12,
        distilled=True,
        crop_pct=1.0,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        classifier=("head", "head_dist"),
    )
    return ViT, cfg


@register_model
def vit_base_patch16_224_miil_in21k():
    """ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    See paper: https://arxiv.org/pdf/2104.10972v4.pdf
    """
    cfg = ViTConfig(
        name="vit_base_patch16_224_miil_in21k",
        url="",
        nb_classes=11221,
        input_size=(224, 224),
        patch_size=16,
        embed_dim=768,
        nb_blocks=12,
        nb_heads=12,
        qkv_bias=False,
        crop_pct=0.875,
        interpolation="bilinear",
        mean=(0.0, 0.0, 0.0),
        std=(0.0, 0.0, 0.0),
    )
    return ViT, cfg


@register_model
def vit_base_patch16_224_miil():
    """
    ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    See paper: https://arxiv.org/pdf/2104.10972v4.pdf
    """
    cfg = ViTConfig(
        name="vit_base_patch16_224_miil",
        url="",
        input_size=(224, 224),
        patch_size=16,
        embed_dim=768,
        nb_blocks=12,
        nb_heads=12,
        qkv_bias=False,
        crop_pct=0.875,
        interpolation="bilinear",
        mean=(0.0, 0.0, 0.0),
        std=(0.0, 0.0, 0.0),
    )
    return ViT, cfg
