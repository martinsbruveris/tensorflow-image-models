from dataclasses import dataclass
from tfimm.architectures import convnext
from tfimm.architectures.lora.layers import LoRADense


__all__ = ["LoRAConvNeXt", "LoRAConvNeXtConfig"]


@dataclass
class LoRAConvNeXtConfig(convnext.ConvNeXtConfig):
    lora_rank: int = 4
    lora_alpha: float = 1
    # TODO: lora_dropout


class LoRAConvNeXt(convnext.ConvNeXt):
    keys_to_ignore_on_load_missing = ["kernel_lora"]

    def __init__(self, cfg: LoRAConvNeXtConfig, **kwargs):
        # We first create the original model
        super().__init__(cfg, **kwargs)
        self.scaling = cfg.lora_alpha / cfg.lora_rank

        lora_cfg = {"lora_rank": self.lora_rank, "lora_alpha": self.lora_alpha}

        # Then we replace all the layers we want to replace
        for stage in self.stages:
            for block in stage.blocks:
                layer_config = block.mlp.fc1.get_config()
                layer_config.update(lora_cfg)
                block.mlp.fc1 = LoRADense.from_config(layer_config)
                layer_config = block.mlp.fc2.get_config()
                layer_config.update(lora_cfg)
                block.mlp.fc2 = LoRADense.from_config(layer_config)

        # Note that we are doing this before the model is built, so weights have
        # not been created yet, etc.
