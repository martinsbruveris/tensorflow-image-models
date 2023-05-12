from dataclasses import dataclass

from tfimm.architectures.convnext import ConvNeXt, ConvNeXtConfig
from tfimm.models import keras_serializable

from .factory import lora_non_trainable_weights, lora_trainable_weights
from .layers import convert_to_lora_layer
from .registry import register_lora_architecture

__all__ = ["LoRAConvNeXt", "LoRAConvNeXtConfig"]


@dataclass
class LoRAConvNeXtConfig(ConvNeXtConfig):
    lora_rank: int = 4
    lora_alpha: float = 1.0
    lora_train_bias: str = "none"
    lora_train_classifer: bool = True
    # TODO: lora_dropout


@keras_serializable
@register_lora_architecture
class LoRAConvNeXt(ConvNeXt):
    cfg_class = LoRAConvNeXtConfig

    def __init__(self, cfg: LoRAConvNeXtConfig, **kwargs):
        # We first create the original model
        super().__init__(cfg, **kwargs)
        self.cfg = cfg

        # Then we replace all the layers we want to replace. Here we only replace the
        # 1x1 convolutions in MLP blocks.
        lora_cfg = {"lora_rank": cfg.lora_rank, "lora_alpha": cfg.lora_alpha}
        for stage in self.stages:
            for block in stage.blocks:
                block.mlp.fc1 = convert_to_lora_layer(block.mlp.fc1, **lora_cfg)
                block.mlp.fc2 = convert_to_lora_layer(block.mlp.fc2, **lora_cfg)

    @property
    def trainable_weights(self):
        classifier = [self.cfg.classifier] if self.cfg.lora_train_classifer else []
        return lora_trainable_weights(
            self,
            train_bias=self.cfg.lora_train_bias,
            trainable_layers=classifier,
        )

    @property
    def non_trainable_weights(self):
        classifier = [self.cfg.classifier] if self.cfg.lora_train_classifer else []
        return lora_non_trainable_weights(
            self,
            train_bias=self.cfg.lora_train_bias,
            trainable_layers=classifier,
        )
