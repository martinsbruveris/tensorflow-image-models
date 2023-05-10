import dataclasses

import tensorflow as tf

from tfimm.models import (
    create_model as create_full_model,
    model_class,
    model_config,
    transfer_weights,
)

from .registry import lora_architecture, lora_config


def create_model(
    model_name: str,
    pretrained: bool = False,
    model_path: str = "",
    *,
    train_bias: str = "none",
    **kwargs,
) -> tf.keras.Model:
    """
    Creates a LoRA model. This model will have all layers, except the LoRA layers set
    to ``trainable=False``. For LoRA layers, only the low-rank adaptation weights will
    be trainable. The only exception are bias weights, controlled by the behaviour of
    ``train_bias``.

    Args:
        model_name: Name of model to instantiate.
        pretrained: If ``True``, load pretrained weights as specified by the ``url``
            field in config. If ``url`` is ``[timm]``, the weights will be downloaded
            from ``timm`` and converted to TensorFlow. See
            :py:func:`tfimm.models.create_model` for details.
        model_path: Path of model weights to load after model is initialized. This takes
            over ``pretrained``.
        train_bias: If "none" or "all", no or all bias weights are trainable
            respectively. If "lora_only", only the bias weights of LoRA layers are set
            to trainable.
        **kwargs: LoRA parameters, such as ``lora_rank`` and ``lora_alpha`` need to be
            passed as kwargs and will be added to the model config.

    Returns:
        The created model.
    """
    cls = model_class(model_name)
    cfg = model_config(model_name)
    lora_cls = lora_architecture(cls)
    lora_cfg_cls = lora_config(cls)

    # We split kwargs into LoRA-specific kwargs and all other ones. We only pass
    # non-LoRA kwargs to create the original model.
    full_model_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("lora_")}
    lora_kwargs = {k: v for k, v in kwargs.items() if k.startswith("lora_")}

    full_model = create_full_model(
        model_name=model_name,
        pretrained=pretrained,
        model_path=model_path,
        **full_model_kwargs,
    )

    # Build LoRA model config by combining the original model config with LoRA kwargs.
    lora_cfg = lora_cfg_cls(**dataclasses.asdict(cfg), **lora_kwargs)

    # Instantiate LoRA model and build it
    model = lora_cls(cfg=lora_cfg)
    model(model.dummy_inputs)

    # Transfer weights to LoRA model
    transfer_weights(
        src_model=full_model, dst_model=model, weights_to_ignore=["lora_weight"]
    )

    # Set all non-LoRA layers to non-trainable
    mark_only_lora_as_trainable(model, train_bias=train_bias)

    return model


def mark_only_lora_as_trainable(model: tf.keras.Model, train_bias: str = "none"):
    if train_bias not in {"none", "all", "lora_only"}:
        raise ValueError(f"Unknown value for train_bias: {train_bias}.")

    # We first set everything to non-trainable
    model.trainable = False

    # Then we mark LoRA and (optionally) bias layers as trainable
    for layer in model._flatten_layers(recursive=True, include_self=False):
        if hasattr(layer, "mark_only_lora_as_trainable"):
            layer.mark_only_lora_as_trainable(train_bias in {"all", "lora_only"})
        elif train_bias in {"all"}:
            _mark_only_bias_as_trainable(layer)


def _mark_only_bias_as_trainable(layer: tf.keras.layers.Layer):
    # TODO: Implement setting biases only to be trainable.
    raise NotImplementedError("Need to implement this...")
