from dataclasses import dataclass

import tensorflow as tf

from tfimm.architectures.convnext import ConvNeXt, ConvNeXtConfig
from tfimm.models import keras_serializable
from .registry import register_lora_architecture


# TODO: This is temporary...
class LoraDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_weight = None

    def build(self, input_shape):
        self.lora_weight = self.add_weight(
            name="lora_weight",
            shape=(1,),
            initializer=tf.keras.initializers.constant(0.0),
            trainable=True,
        )

    def call(self, x):
        return super().call(x + self.lora_weight)


@dataclass
class LoRAConvNeXtConfig(ConvNeXtConfig):
    lora_rank: int = 2
    lora_alpha: float = 1.0  # TODO: Find better default value



@keras_serializable
@register_lora_architecture
class LoRAConvNeXt(ConvNeXt):

    cfg_class = LoRAConvNeXtConfig

    def __init__(self, cfg: LoRAConvNeXtConfig, **kwargs):
        super().__init__(cfg, **kwargs)

        for stage in self.stages:
            for block in stage.blocks:
                # TODO: Will need to adapt this to take into account LoRA parameters...
                block.mlp.fc1 = LoraDense.from_config(block.mlp.fc1.get_config())
                block.mlp.fc2 = LoraDense.from_config(block.mlp.fc2.get_config())
