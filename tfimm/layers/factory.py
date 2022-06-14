import tensorflow as tf

from tfimm.layers.norm import Affine, GroupNormalization


def act_layer_factory(act_layer: str):
    """Returns a function that creates the required activation layer."""
    if act_layer in {"linear", "swish", "relu", "gelu", "sigmoid"}:
        return lambda **kwargs: tf.keras.layers.Activation(act_layer, **kwargs)
    if act_layer == "relu6":
        return lambda **kwargs: tf.keras.layers.ReLU(max_value=6, **kwargs)
    else:
        raise ValueError(f"Unknown activation: {act_layer}.")


def norm_layer_factory(norm_layer: str):
    """Returns a function that creates a normalization layer"""
    if norm_layer == "":
        return lambda **kwargs: tf.keras.layers.Activation("linear", **kwargs)

    elif norm_layer == "batch_norm":
        bn_class = tf.keras.layers.BatchNormalization
        bn_args = {
            "momentum": 0.9,  # We use PyTorch default args here
            "epsilon": 1e-5,
        }
        return lambda **kwargs: bn_class(**bn_args, **kwargs)

    elif norm_layer == "batch_norm_tf":  # Batch norm with TF default for epsilon
        bn_class = tf.keras.layers.BatchNormalization
        bn_args = {
            "momentum": 0.9,
            "epsilon": 1e-3,
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

    elif norm_layer == "group_norm_1grp":
        # Group normalization with one group. Used by PoolFormer.
        bn_class = GroupNormalization
        bn_args = {"nb_groups": 1}
        return lambda **kwargs: bn_class(**bn_args, **kwargs)

    else:
        raise ValueError(f"Unknown normalization layer: {norm_layer}")
