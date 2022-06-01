from tfimm.architectures import (
    CaiT,
    CaiTConfig,
    ConvMixer,
    ConvMixerConfig,
    ConvNeXt,
    ConvNeXtConfig,
    EfficientNet,
    EfficientNetConfig,
    MLPMixer,
    MLPMixerConfig,
    PoolFormer,
    PoolFormerConfig,
    PoolingVisionTransformer,
    PoolingVisionTransformerConfig,
    PyramidVisionTransformer,
    PyramidVisionTransformerConfig,
    PyramidVisionTransformerV2,
    PyramidVisionTransformerV2Config,
    ResNet,
    ResNetConfig,
    ResNetV2,
    ResNetV2Config,
    SwinTransformer,
    SwinTransformerConfig,
    ViT,
    ViTConfig,
)
from tfimm.models import register_model

TEST_ARCHITECTURES = [
    "cait_test_model",  # cait.py
    "convmixer_test_model",  # convmixer.py
    "convnext_test_model",  # convnext.py
    "efficientnet_test_model",  # efficientnet.py
    "mixer_test_model",  # mlp_mixer.py
    "resmlp_test_model",
    "gmlp_test_model",
    "pit_test_model",  # pit.py
    "pit_distilled_test_model",
    "poolformer_test_model",  # poolformer.py
    "pvt_test_model",  # pvt.py
    "pvt_v2_test_model",  # pvt_v2.py
    "resnet_test_model_1",  # resnet.py
    "resnet_test_model_2",
    "resnetv2_test_model",  # resnetv2.py
    "swin_test_model",  # swin.py
    "vit_test_model",  # vit.py
    "deit_test_model",
    "vit_r_test_model_1",  # vit_hybrid.py
    "vit_r_test_model_2",
]


@register_model
def cait_test_model():
    cfg = CaiTConfig(
        name="cait_test_model",
        nb_classes=12,
        input_size=(32, 32),
        patch_size=8,
        embed_dim=4,
        nb_blocks=2,
        nb_heads=2,
    )
    return CaiT, cfg


@register_model
def convmixer_test_model():
    cfg = ConvMixerConfig(
        name="convmixer_test_model",
        nb_classes=12,
        input_size=(32, 32),
        patch_size=(8, 8),
        embed_dim=4,
        depth=2,
        kernel_size=3,
        act_layer="relu",
    )
    return ConvMixer, cfg


@register_model
def convnext_test_model():
    cfg = ConvNeXtConfig(
        name="convnext_test_model",
        nb_classes=12,
        input_size=(32, 32),
        embed_dim=(4, 4, 4, 4),
        nb_blocks=(1, 1, 1, 1),
    )
    return ConvNeXt, cfg


@register_model
def efficientnet_test_model():
    cfg = EfficientNetConfig(
        name="efficientnet_test_model",
        input_size=(32, 32),
        architecture=(
            ("ds_r1_k3_s1_e1_c16_se0.25",),
            ("ir_r2_k3_s2_e6_c24_se0.25",),
        ),
        nb_features=32,
    )
    return EfficientNet, cfg


@register_model
def mixer_test_model():
    cfg = MLPMixerConfig(
        name="mixer_test_model",
        nb_classes=12,
        input_size=(32, 32),
        patch_size=8,
        embed_dim=4,
        nb_blocks=2,
    )
    return MLPMixer, cfg


@register_model
def resmlp_test_model():
    cfg = MLPMixerConfig(
        name="resmlp_test_model",
        nb_classes=12,
        input_size=(32, 32),
        patch_size=8,
        embed_dim=4,
        nb_blocks=2,
        mlp_ratio=(4.0, 4.0),
        block_layer="res_block",
        norm_layer="affine",
    )
    return MLPMixer, cfg


@register_model
def gmlp_test_model():
    cfg = MLPMixerConfig(
        name="gmlp_test_model",
        nb_classes=12,
        input_size=(32, 32),
        patch_size=8,
        embed_dim=4,
        nb_blocks=2,
        mlp_ratio=(6.0, 6.0),
        block_layer="spatial_gating_block",
        mlp_layer="gated_mlp",
    )
    return MLPMixer, cfg


@register_model
def pit_test_model():
    cfg = PoolingVisionTransformerConfig(
        name="pit_test_model",
        nb_classes=12,
        input_size=(32, 32),
        patch_size=4,
        stride=2,
        embed_dim=(2, 4, 4),
        nb_blocks=(1, 1, 1),
        nb_heads=(1, 1, 1),
        mlp_ratio=2.0,
    )
    return PoolingVisionTransformer, cfg


@register_model
def pit_distilled_test_model():
    cfg = PoolingVisionTransformerConfig(
        name="pit_distilled_test_model",
        nb_classes=12,
        input_size=(32, 32),
        patch_size=4,
        stride=2,
        embed_dim=(2, 4, 4),
        nb_blocks=(1, 1, 1),
        nb_heads=(1, 1, 1),
        mlp_ratio=2.0,
        distilled=True,
        classifier=("head", "head_dist"),
    )
    return PoolingVisionTransformer, cfg


@register_model
def poolformer_test_model():
    cfg = PoolFormerConfig(
        name="poolformer_test_model",
        nb_classes=12,
        input_size=(32, 32),
        embed_dim=(2, 4, 6, 8),
        nb_blocks=(1, 1, 1, 1),
    )
    return PoolFormer, cfg


@register_model
def pvt_test_model():
    cfg = PyramidVisionTransformerConfig(
        name="pvt_test_model",
        nb_classes=12,
        input_size=(32, 32),
        embed_dim=(2, 4, 6, 8),
        nb_blocks=(1, 1, 1, 1),
        nb_heads=(1, 2, 3, 4),
    )
    return PyramidVisionTransformer, cfg


@register_model
def pvt_v2_test_model():
    cfg = PyramidVisionTransformerV2Config(
        name="pvt_v2_test_model",
        nb_classes=12,
        input_size=(32, 32),
        embed_dim=(2, 4, 6, 8),
        nb_blocks=(1, 1, 1, 1),
        nb_heads=(1, 2, 3, 4),
    )
    return PyramidVisionTransformerV2, cfg


@register_model
def resnet_test_model_1():
    cfg = ResNetConfig(
        name="resnet_test_model_1",
        nb_classes=12,
        input_size=(32, 32),
        block="basic_block",
        nb_blocks=(1, 1, 1, 1),
        nb_channels=(2, 4, 6, 8),
    )
    return ResNet, cfg


@register_model
def resnet_test_model_2():
    cfg = ResNetConfig(
        name="resnet_test_model_2",
        nb_classes=12,
        input_size=(32, 32),
        stem_type="deep",
        block="bottleneck",
        nb_blocks=(1, 1, 1, 1),
        nb_channels=(2, 4, 6, 8),
        first_conv="conv1/0",
    )
    return ResNet, cfg


@register_model
def resnetv2_test_model():
    cfg = ResNetV2Config(
        name="resnetv2_test_model",
        input_size=(32, 32),
        nb_blocks=(1, 1, 1, 1),
        nb_channels=(2, 4, 6, 8),
        width_factor=1,
        norm_layer="group_norm_1grp",
    )
    return ResNetV2, cfg


@register_model
def swin_test_model():
    cfg = SwinTransformerConfig(
        name="swin_test_model",
        nb_classes=12,
        input_size=(64, 64),
        patch_size=2,
        embed_dim=4,
        nb_blocks=(1, 1, 1, 1),
        nb_heads=(1, 1, 1, 1),
        window_size=4,
    )
    return SwinTransformer, cfg


@register_model
def vit_test_model():
    cfg = ViTConfig(
        name="vit_test_model",
        nb_classes=12,
        input_size=(32, 32),
        patch_size=8,
        embed_dim=4,
        nb_blocks=2,
        nb_heads=2,
    )
    return ViT, cfg


@register_model
def deit_test_model():
    cfg = ViTConfig(
        name="deit_test_model",
        nb_classes=12,
        input_size=(32, 32),
        patch_size=8,
        embed_dim=4,
        nb_blocks=2,
        nb_heads=2,
        distilled=True,
        classifier=("head", "head_dist"),
    )
    return ViT, cfg


@register_model
def vit_r_test_model_1():
    cfg = ViTConfig(
        name="vit_r_test_model_1",
        nb_classes=12,
        input_size=(32, 32),
        patch_layer="hybrid_embeddings",
        patch_nb_blocks=(),
        patch_size=8,
        embed_dim=4,
        nb_blocks=2,
        nb_heads=2,
        first_conv="patch_embed/backbone/conv",
    )
    return ViT, cfg


@register_model
def vit_r_test_model_2():
    cfg = ViTConfig(
        name="vit_r_test_model_2",
        nb_classes=12,
        input_size=(32, 32),
        patch_layer="hybrid_embeddings",
        patch_nb_blocks=(1, 1, 1, 1),
        patch_size=1,
        embed_dim=4,
        nb_blocks=2,
        nb_heads=2,
        first_conv="patch_embed/backbone/stem/conv",
    )
    return ViT, cfg
