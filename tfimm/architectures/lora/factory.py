import dataclasses
from typing import List, Optional

import tensorflow as tf
import tf_keras

from tfimm.models import (
    create_model as create_full_model,
    model_class,
    transfer_weights,
)
from tfimm.models.factory import _get_layer_name

from .layers import LORA_WEIGHT_NAMES
from .registry import lora_architecture, lora_base_architecture, lora_config


def create_model(
    model_name: str,
    pretrained: bool = False,
    model_path: str = "",
    **kwargs,
) -> tf_keras.Model:
    """
    Creates a LoRA model from a ``tfimm`` model name.

    Args:
        model_name: Name of model to instantiate.
        pretrained: If ``True``, load pretrained weights as specified by the ``url``
            field in config. If ``url`` is ``[timm]``, the weights will be downloaded
            from ``timm`` and converted to TensorFlow. See
            :py:func:`tfimm.create_model` for details.
        model_path: Path of model weights to load after model is initialized. This takes
            precedence over ``pretrained``.
        **kwargs: LoRA parameters, such as ``lora_rank`` and ``lora_alpha`` need to be
            passed as kwargs and will be added to the model config.

    Returns:
        The created model.
    """
    cls = model_class(model_name)
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

    # Build LoRA model config by combining the full model config with LoRA kwargs.
    lora_cfg = lora_cfg_cls(**dataclasses.asdict(full_model.cfg), **lora_kwargs)

    # Instantiate LoRA model and build it
    model = lora_cls(cfg=lora_cfg)
    model(model.dummy_inputs)

    # Transfer weights to LoRA model
    transfer_weights(
        src_model=full_model, dst_model=model, weights_to_ignore=LORA_WEIGHT_NAMES
    )

    return model


def convert_to_lora_model(model: tf_keras.Model, **kwargs) -> tf_keras.Model:
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
    cfg_dict = dataclasses.asdict(model.cfg)
    cfg_dict.update(kwargs)
    lora_cfg = lora_cfg_cls(**cfg_dict)

    # Instantiate LoRA model and build it
    lora_model = lora_cls(cfg=lora_cfg)
    lora_model(lora_model.dummy_inputs)

    # Transfer weights to LoRA model
    transfer_weights(
        src_model=model, dst_model=lora_model, weights_to_ignore=LORA_WEIGHT_NAMES
    )

    return lora_model


def convert_to_regular_model(model: tf_keras.Model) -> tf_keras.Model:
    """
    Converts a LoRA model to a regular model.

    Args:
        model: LoRA model to be converted. Has be be a class that has been registered
            as a LoRA architecture.

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
    base_cfg_dict = {
        key: value
        for key, value in dataclasses.asdict(cfg).items()
        if not key.startswith("lora_")
    }
    base_cfg = base_cfg_cls(**base_cfg_dict)

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


def merge_lora_weights(model: tf_keras.Model):
    """
    Recursively merge weights in all LoRA layers in the given model. The model is
    modified in place.

    Args:
        model: Model for merging weights.
    """
    for layer in model._flatten_layers(recursive=True, include_self=True):
        if getattr(layer, "is_lora_layer", False) and not layer.merged:
            layer.merge_weights()


def lora_trainable_weights(
    model: tf_keras.Model,
    train_bias: str = "none",
    trainable_layers: Optional[List[str]] = None,
) -> List[tf.Variable]:
    """
    Returns a list of variables to be used instead of ``model.trainable_weights`` when
    doing LoRA training.

    Args:
        model: A keras model.
        train_bias: If ``"none"`` or ``"all"``, no or all bias weights are trainable
            respectively. If ``"lora_only"``, only the bias weights of LoRA layers are
            set to trainable.
        trainable_layers: A list of layer names that should be kept trainable.

    Returns:
        List of LoRA trainable weights.
    """
    if train_bias not in {"none", "all", "lora_only"}:
        raise ValueError(f"Unknown value for train_bias: {train_bias}.")

    trainable_ids = set()
    trainable_weights = []
    for layer in model._flatten_layers(recursive=True, include_self=True):
        if getattr(layer, "is_lora_layer", False):
            weights = layer.lora_trainable_weights(train_bias in {"all", "lora_only"})
        elif train_bias in {"all"}:
            weights = _bias_variables(layer)
        else:
            weights = []
        trainable_ids.union(set(id(w) for w in weights))
        trainable_weights.extend(weights)

    trainable_layers = trainable_layers or []
    for weight in model.weights:
        if (
            _get_layer_name(weight.name) in trainable_layers
            and id(weight) not in trainable_ids
        ):
            trainable_ids.add(id(weight))
            trainable_weights.append(weight)

    return trainable_weights


def lora_non_trainable_weights(
    model: tf_keras.Model,
    train_bias: str = "none",
    trainable_layers: Optional[List[str]] = None,
) -> List[tf.Variable]:
    """
    Returns a list of non-trainable weights for the LoRA model. This function
    complements :py:func:`lora_trainable_weights`.

    Args:
        model: A keras model.
        train_bias: If ``"none"`` or ``"all"``, no or all bias weights are trainable
            respectively. If ``"lora_only"``, only the bias weights of LoRA layers are
            set to trainable.
        trainable_layers: A list of layer names that should be kept trainable.

    Returns:
        List of LoRA non-trainable weights.
    """
    # We start with a list of all weights and remove the trainable weights. We
    # explicitly call `(non_)trainable_weights` from tf_keras.Model, because we expect
    # our model to overwrite the `(non_)trainable_weights` properties with the result
    # of this function.
    weights = tf_keras.Model.trainable_weights.fget(
        model
    ) + tf_keras.Model.non_trainable_weights.fget(model)
    trainable_weights = lora_trainable_weights(
        model,
        train_bias=train_bias,
        trainable_layers=trainable_layers,
    )
    trainable_ids = set(id(w) for w in trainable_weights)  # Variables are not hashable
    non_trainable_weights = [w for w in weights if id(w) not in trainable_ids]
    return non_trainable_weights


def _bias_variables(layer: tf_keras.layers.Layer) -> List[tf.Variable]:
    """Return bias variables of a given layer as a list."""
    # There is no fundamental concept of a "bias-variable", rather the notion of bias
    # is a convention. And so we have to proceed layer-by-layer and specify for each
    # layer, which are the bias variables (and if they are used).
    if (
        type(layer)
        in {
            tf_keras.layers.Conv1D,
            tf_keras.layers.Conv2D,
            tf_keras.layers.Conv3D,
            tf_keras.layers.Dense,
            tf_keras.layers.DepthwiseConv1D,
            tf_keras.layers.DepthwiseConv2D,
        }
        and layer.use_bias
    ):
        return [layer.bias]
    elif (
        type(layer)
        in {
            tf_keras.layers.BatchNormalization,
            tf_keras.layers.LayerNormalization,
        }
        and layer.center
    ):
        return [layer.beta]
    else:
        return []
