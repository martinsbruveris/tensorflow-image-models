import tensorflow as tf

from tfimm.layers.norm import Affine, GroupNormalization


def act_layer_factory(act_layer: str):
    """Returns a function that creates the required activation layer."""
    if act_layer in {"linear", "swish", "relu", "gelu", "sigmoid"}:
        return lambda: tf.keras.layers.Activation(act_layer)
    else:
        raise ValueError(f"Unknown activation: {act_layer}.")


def norm_layer_factory(norm_layer: str):
    """Returns a function that creates a normalization layer"""
    if norm_layer == "":
        return lambda **kwargs: tf.keras.layers.Activation("linear")

    elif norm_layer == "batch_norm":
        bn_class = tf.keras.layers.BatchNormalization
        bn_args = {
            "momentum": 0.9,  # We use PyTorch default args here
            "epsilon": 1e-5,
        }
        return lambda **kwargs: bn_class(**bn_args, **kwargs)

    elif norm_layer == "layer_norm":
        bn_class = tf.keras.layers.LayerNormalization
        bn_args = {"epsilon": 1e-5}  # We use PyTorch default args here
        return lambda **kwargs: bn_class(**bn_args, **kwargs)

    elif norm_layer == "layer_norm_eps_1e-6":
        bn_class = tf.keras.layers.LayerNormalization
        bn_args = {"epsilon": 1e-6}
        return lambda **kwargs: bn_class(**bn_args, **kwargs)

    elif norm_layer == "affine":
        return Affine

    elif norm_layer == "group_norm":
        return GroupNormalization

    else:
        raise ValueError(f"Unknown normalization layer: {norm_layer}")
