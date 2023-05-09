import tensorflow as tf
from dataclasses import dataclass
from tfimm.architectures import convnext


class LoraDense(tf.keras.layers.Dense):
    def __init__(self, *args, lora_rank: int = 4, lora_alpha: float = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / lora_rank

    def build(self, input_shape):
        super().build(input_shape)
        last_dim = self.kernel.shape[0]
        self.kernel_0 = self.add_weight(
            "kernel",
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=False,
        )
        self.kernel_lora_a = self.add_weight(
            "kernel_lora_a",
            shape=[last_dim, self.lora_rank],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )
        self.kernel_lora_b = self.add_weight(
            "kernel_lora_b",
            shape=[self.lora_rank, self.units],
            initializer=tf.keras.initializers.Zeros(),
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )

        self.kernel = (
            self.kernel_0 + self.kernel_lora_a @ self.kernel_lora_b * self.scaling
        )


@dataclass
class LoraConfig:
    """Shared base class for LoRA configurations."""

    lora_rank: int = 4
    lora_alpha: float = 1
    # TODO: lora_dropout

    def apply(self, config):
        config["lora_rank"] = self.lora_rank
        config["lora_alpha"] = self.lora_alpha


@dataclass
class LoRAConvNeXtConfig(convnext.ConvNeXtConfig, LoraConfig):
    pass


class LoRAConvNeXt(convnext.ConvNeXt):
    keys_to_ignore_on_load_missing = ["kernel_lora"]

    def __init__(self, cfg: LoRAConvNeXtConfig, **kwargs):
        # We first create the original model
        super().__init__(cfg, **kwargs)

        # Then we replace all the layers we want to replace
        for stage in self.stages:
            for block in stage.blocks:
                layer_config = block.mlp.fc1.get_config()
                cfg.apply(layer_config)
                block.mlp.fc1 = LoraDense.from_config(layer_config)

        # Note that we are doing this before the model is built, so weights have
        # not been created yet, etc.


def lora_convnext_tiny():
    cfg = LoRAConvNeXtConfig(
        name="convnext_tiny",
        url="[timm]",
        embed_dim=(96, 192, 384, 768),
        nb_blocks=(3, 3, 9, 3),
    )
    return LoRAConvNeXt, cfg
