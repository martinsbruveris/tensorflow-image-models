from dataclasses import dataclass
from tfimm.architectures.lora.layers import LoRADense
from tfimm.architectures.convnext import ConvNeXtConfig, ConvNeXt
from tfimm.models import keras_serializable

from .registry import register_lora_architecture


__all__ = ["LoRAConvNeXt", "LoRAConvNeXtConfig"]


@dataclass
class LoRAConvNeXtConfig(ConvNeXtConfig):
    lora_rank: int = 4
    lora_alpha: float = 1.0
    # TODO: lora_dropout


@keras_serializable
@register_lora_architecture
class LoRAConvNeXt(ConvNeXt):
    cfg_class = LoRAConvNeXtConfig
    keys_to_ignore_on_load_missing = ["kernel_lora"]

    def __init__(self, cfg: LoRAConvNeXtConfig, **kwargs):
        # We first create the original model
        super().__init__(cfg, **kwargs)

        lora_cfg = {"lora_rank": cfg.lora_rank, "lora_alpha": cfg.lora_alpha}

        # Then we replace all the layers we want to replace
        for stage in self.stages:
            for block in stage.blocks:
                layer_config = block.mlp.fc1.get_config()
                layer_config.update(lora_cfg)
                block.mlp.fc1 = LoRADense.from_config(layer_config)
                layer_config = block.mlp.fc2.get_config()
                layer_config.update(lora_cfg)
                block.mlp.fc2 = LoRADense.from_config(layer_config)

    def merge_lora_weights(self):
        for layer in self._flatten_layers(recursive=True, include_self=False):
            if hasattr(layer, "merge_lora_weights"):
                layer.merge_lora_weights()

    def unmerge_lora_weights(self):
        for layer in self._flatten_layers(recursive=True, include_self=False):
            if hasattr(layer, "unmerge_lora_weights"):
                layer.unmerge_lora_weights()
