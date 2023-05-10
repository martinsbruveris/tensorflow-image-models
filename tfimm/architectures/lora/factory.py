import dataclasses

import tensorflow as tf

from tfimm.models import (
    create_model as create_full_model,
    model_class,
    model_config,
    transfer_weights,
)

from .registry import lora_architecture, lora_base_architecture, lora_config

# List of patterns to match LoRA weights, so they can be excluded from weight transfers
# between a model and its LoRA version.
LORA_WEIGHT_NAMES = ["kernel_lora_a", "kernel_lora_b"]


def create_model(
    model_name: str,
    pretrained: bool = False,
    model_path: str = "",
    **kwargs,
) -> tf.keras.Model:
    """
    Creates a LoRA model.

    Args:
        model_name: Name of model to instantiate.
        pretrained: If ``True``, load pretrained weights as specified by the ``url``
            field in config. If ``url`` is ``[timm]``, the weights will be downloaded
            from ``timm`` and converted to TensorFlow. See
            :py:func:`tfimm.models.create_model` for details.
        model_path: Path of model weights to load after model is initialized. This takes
            over ``pretrained``.
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
        src_model=full_model, dst_model=model, weights_to_ignore=LORA_WEIGHT_NAMES
    )

    return model


def convert_to_lora_model(model: tf.keras.Model, **kwargs) -> tf.keras.Model:
    """
    Creates a LoRA version of a model.

    Args:
        model: Source model. Has to be an instance of a class that has a corresponding
            LoRA architecture registered.
        **kwargs: LoRA parameters, such as ``lora_rank`` and ``lora_alpha`` need to be
            passed as kwargs and will be added to the model config.

    Returns:
        LoRA model.
    """
    lora_cls = lora_architecture(type(model))
    lora_cfg_cls = lora_config(type(model))

    # Build LoRA model config by combining the original model config with LoRA kwargs.
    cfg = model.cfg
    lora_cfg = lora_cfg_cls(**dataclasses.asdict(cfg), **kwargs)

    # Instantiate LoRA model and build it
    lora_model = lora_cls(cfg=lora_cfg)
    lora_model(lora_model.dummy_inputs)

    # Transfer weights to LoRA model
    transfer_weights(
        src_model=model, dst_model=lora_model, weights_to_ignore=LORA_WEIGHT_NAMES
    )

    return model


def convert_to_regular_model(model: tf.keras.Model) -> tf.keras.Model:
    """
    Converts a LoRA model to a regular model.

    Args:
        model: LoRA model to be converted.

    Returns:
        The converted model.
    """
    base_cls = lora_base_architecture(type(model))
    base_cfg_cls = base_cls.cfg_class

    # Convert config to base config class. This effectively removes the LoRA config
    # parameters from the model config. We do assume here that the LoRA model config
    # is a superset of the base model config. And also, that config classes are
    # dataclasses.
    cfg = model.cfg
    base_cfg_fields = {f.name for f in dataclasses.fields(base_cfg_cls)}
    base_cfg = base_cfg_cls(
        **{k: v for k, v in dataclasses.asdict(cfg) if k in base_cfg_fields}
    )

    # Then create the base model, build it and transfer weights to it.
    base_model = base_cls(cfg=base_cfg)
    base_model(base_model.dummy_inputs)

    # Before we can transfer weights we need to merge them. We keep track of which
    # layers had to be merged, so we can unmerge them (and only them) afterwards
    merge_indices = set()
    for idx, layer in enumerate(
        model._flatten_layers(recursive=True, include_self=False)
    ):
        if getattr(layer, "is_lora_layer", False) and not layer.merged:
            layer.merge_weights()
            merge_indices.add(idx)

    # Drum roll... Weight transfer happening here.
    transfer_weights(src_model=model, dst_model=base_model)

    # Now unmerge the previously merged layers again
    for idx, layer in enumerate(
        model._flatten_layers(recursive=True, include_self=False)
    ):
        if idx in merge_indices:
            assert getattr(layer, "is_lora_layer", False)
            layer.unmerge_weights()

    return base_model


def set_only_lora_layers_trainable(model: tf.keras.Model, train_bias: str = "none"):
    """
    Marks only LoRA layers in the model as trainable. This model will have all layers,
    except LoRA layers set to ``trainable=False``. For LoRA layers, only the low-rank
    adaptation weights will be trainable. The only exception are bias weights,
    controlled by the value of ``train_bias``.

    Args:
        model: Model to be modified.
        train_bias: If "none" or "all", no or all bias weights are trainable
            respectively. If "lora_only", only the bias weights of LoRA layers are set
            to trainable.

    Returns:
        Nothing. The model is modified in place.
    """
    if train_bias not in {"none", "all", "lora_only"}:
        raise ValueError(f"Unknown value for train_bias: {train_bias}.")

    # We first set everything to non-trainable
    model.trainable = False

    # Then we mark LoRA and (optionally) bias layers as trainable
    for layer in model._flatten_layers(recursive=True, include_self=False):
        if getattr(layer, "is_lora_layer", False):
            layer.set_only_lora_weights_trainable(train_bias in {"all", "lora_only"})
        elif train_bias in {"all"}:
            _set_bias_weights_trainable(layer)


def _set_bias_weights_trainable(layer: tf.keras.layers.Layer):
    # TODO: Implement setting biases only to be trainable.
    raise NotImplementedError("Need to implement this...")
