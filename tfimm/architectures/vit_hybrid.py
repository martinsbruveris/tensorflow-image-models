"""
Hybrid Vision Transformer (ViT) in TensorFlow

A TensorFlow implement of the Hybrid Vision Transformers as described in:

'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929

`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270

These hybrid model definitions depend on code in vision_transformer.py.
They were moved here to keep file sizes sane.

Copyright 2021 Martins Bruveris
Copyright 2021 Ross Wightman
"""

from tfimm.models import register_model

from .vit import ViT, ViTConfig

# Model_registry will add each entrypoint function to this
__all__ = []


@register_model
def vit_tiny_r_s16_p8_224():
    """R+ViT-Ti/S16 w/ 8x8 patch hybrid @ 224 x 224."""
    cfg = ViTConfig(
        name="vit_tiny_r_s16_p8_224",
        input_size=(224, 224),
        patch_layer="hybrid_embeddings",
        patch_nb_blocks=(),
        patch_size=8,
        embed_dim=192,
        nb_blocks=12,
        nb_heads=3,
        crop_pct=0.9,
        first_conv="patch_embed/backbone/conv",
    )
    return ViT, cfg


@register_model
def vit_tiny_r_s16_p8_384():
    """R+ViT-Ti/S16 w/ 8x8 patch hybrid @ 384 x 384."""
    cfg = ViTConfig(
        name="vit_tiny_r_s16_p8_384",
        input_size=(384, 384),
        patch_layer="hybrid_embeddings",
        patch_nb_blocks=(),
        patch_size=8,
        embed_dim=192,
        nb_blocks=12,
        nb_heads=3,
        crop_pct=1.0,
        first_conv="patch_embed/backbone/conv",
    )
    return ViT, cfg


@register_model
def vit_small_r26_s32_224():
    """R26+ViT-S/S32 hybrid."""
    cfg = ViTConfig(
        name="vit_small_r26_s32_224",
        input_size=(224, 224),
        patch_layer="hybrid_embeddings",
        patch_nb_blocks=(2, 2, 2, 2),
        patch_size=1,
        embed_dim=384,
        nb_blocks=12,
        nb_heads=6,
        crop_pct=0.9,
        first_conv="patch_embed/backbone/stem/conv",
    )
    return ViT, cfg


@register_model
def vit_small_r26_s32_384():
    """R26+ViT-S/S32 hybrid."""
    cfg = ViTConfig(
        name="vit_small_r26_s32_384",
        input_size=(384, 384),
        patch_layer="hybrid_embeddings",
        patch_nb_blocks=(2, 2, 2, 2),
        patch_size=1,
        embed_dim=384,
        nb_blocks=12,
        nb_heads=6,
        crop_pct=1.0,
        first_conv="patch_embed/backbone/stem/conv",
    )
    return ViT, cfg


@register_model
def vit_base_r50_s16_384():
    """
    R50+ViT-B/16 hybrid from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source
    https://github.com/google-research/vision_transformer.
    """
    cfg = ViTConfig(
        name="vit_base_r50_s16_384",
        input_size=(384, 384),
        patch_layer="hybrid_embeddings",
        patch_nb_blocks=(3, 4, 9),
        patch_size=1,
        embed_dim=768,
        nb_blocks=12,
        nb_heads=12,
        crop_pct=1.0,
        first_conv="patch_embed/backbone/stem/conv",
    )
    return ViT, cfg


@register_model
def vit_large_r50_s32_224():
    """R50+ViT-L/S32 hybrid."""
    cfg = ViTConfig(
        name="vit_large_r50_s32_224",
        input_size=(224, 224),
        patch_layer="hybrid_embeddings",
        patch_nb_blocks=(3, 4, 6, 3),
        patch_size=1,
        embed_dim=1024,
        nb_blocks=24,
        nb_heads=16,
        crop_pct=0.9,
        first_conv="patch_embed/backbone/stem/conv",
    )
    return ViT, cfg


@register_model
def vit_large_r50_s32_384():
    """R50+ViT-L/S32 hybrid."""
    cfg = ViTConfig(
        name="vit_large_r50_s32_384",
        input_size=(384, 384),
        patch_layer="hybrid_embeddings",
        patch_nb_blocks=(3, 4, 6, 3),
        patch_size=1,
        embed_dim=1024,
        nb_blocks=24,
        nb_heads=16,
        crop_pct=1.0,
        first_conv="patch_embed/backbone/stem/conv",
    )
    return ViT, cfg


@register_model
def vit_tiny_r_s16_p8_224_in21k():
    """R+ViT-Ti/S16 w/ 8x8 patch hybrid.  ImageNet-21k."""
    cfg = ViTConfig(
        name="vit_tiny_r_s16_p8_224_in21k",
        nb_classes=21843,
        input_size=(224, 224),
        patch_layer="hybrid_embeddings",
        patch_nb_blocks=(),
        patch_size=8,
        embed_dim=192,
        nb_blocks=12,
        nb_heads=3,
        crop_pct=0.9,
        first_conv="patch_embed/backbone/conv",
    )
    return ViT, cfg


@register_model
def vit_small_r26_s32_224_in21k():
    """R26+ViT-S/S32 hybrid. ImageNet-21k."""
    cfg = ViTConfig(
        name="vit_small_r26_s32_224_in21k",
        nb_classes=21843,
        input_size=(224, 224),
        patch_layer="hybrid_embeddings",
        patch_nb_blocks=(2, 2, 2, 2),
        patch_size=1,
        embed_dim=384,
        nb_blocks=12,
        nb_heads=6,
        crop_pct=0.9,
        first_conv="patch_embed/backbone/stem/conv",
    )
    return ViT, cfg


@register_model
def vit_base_r50_s16_224_in21k():
    """
    R50+ViT-B/16 hybrid model from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source
    https://github.com/google-research/vision_transformer.
    """
    cfg = ViTConfig(
        name="vit_base_r50_s16_224_in21k",
        nb_classes=21843,
        input_size=(224, 224),
        patch_layer="hybrid_embeddings",
        patch_nb_blocks=(3, 4, 9),
        patch_size=1,
        embed_dim=768,
        nb_blocks=12,
        nb_heads=12,
        representation_size=768,
        crop_pct=0.9,
        first_conv="patch_embed/backbone/stem/conv",
    )
    return ViT, cfg


@register_model
def vit_large_r50_s32_224_in21k():
    """R50+ViT-L/S32 hybrid. ImageNet-21k."""
    cfg = ViTConfig(
        name="vit_large_r50_s32_224_in21k",
        nb_classes=21843,
        input_size=(224, 224),
        patch_layer="hybrid_embeddings",
        patch_nb_blocks=(3, 4, 6, 3),
        patch_size=1,
        embed_dim=1024,
        nb_blocks=24,
        nb_heads=16,
        crop_pct=0.9,
        first_conv="patch_embed/backbone/stem/conv",
    )
    return ViT, cfg
