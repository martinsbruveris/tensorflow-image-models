import tensorflow as tf


def act_layer_factory(act_layer: str):
    """Returns a function that creates the required activation layer."""
    if act_layer == "swish":
        return lambda: tf.keras.layers.Activation("swish")
    elif act_layer == "relu":
        return lambda: tf.keras.layers.Activation("relu")
    else:
        raise ValueError(f"Unknown activation: {act_layer}.")


def norm_layer_factory(norm_layer: str):
    """Returns a function that creates a normalization layer"""
    if norm_layer == "none":
        return lambda **kwargs: lambda x, training=True: x
    elif norm_layer == "batch_norm":
        bn_class = tf.keras.layers.BatchNormalization
        bn_args = {
            "momentum": 0.9,  # We use PyTorch default args here
            "epsilon": 1e-5,
        }
        return lambda **kwargs: bn_class(**bn_args, **kwargs)
    else:
        raise ValueError(f"Unknown normalization layer: {norm_layer}")