from tfimm.architectures import ResNet, ResNetConfig
from tfimm.models import register_model


@register_model
def test_model():
    cfg = ResNetConfig(
        name="test_model",
        nb_classes=12,
        input_size=(32, 32),
        block="basic_block",
        nb_blocks=(1, 1, 1, 1),
        nb_channels=(2, 4, 6, 8),
    )
    return ResNet, cfg
