from dataclasses import dataclass

from tfimm.architectures.convnext import ConvNeXt, ConvNeXtConfig
from tfimm.architectures.lora.layers import LoRADense
from tfimm.models import keras_serializable

from .factory import lora_non_trainable_weights, lora_trainable_weights
from .registry import register_lora_architecture

__all__ = ["LoRAConvNeXt", "LoRAConvNeXtConfig"]


@dataclass
class LoRAConvNeXtConfig(ConvNeXtConfig):
    lora_rank: int = 4
    lora_alpha: float = 1.0
    lora_train_bias: str = "none"
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
                layer_config = block.mlp.fc1.get_config()
                layer_config.update(lora_cfg)
                block.mlp.fc1 = LoRADense.from_config(layer_config)
                layer_config = block.mlp.fc2.get_config()
                layer_config.update(lora_cfg)
                block.mlp.fc2 = LoRADense.from_config(layer_config)

    @property
    def trainable_weights(self):
        return lora_trainable_weights(self, train_bias=self.cfg.lora_train_bias)

    @property
    def non_trainable_weights(self):
        return lora_non_trainable_weights(self, train_bias=self.cfg.lora_train_bias)
