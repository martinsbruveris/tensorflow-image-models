from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2_as_graph,
)


def get_flops(model: tf.keras.Model, input_shape: tuple) -> int:
    """
    Calculate FLOPS for a `tf.keras.Model`. Ignore operations used in only training
    mode such as Initialization. Use `tf.profiler` of tensorflow v1 api.

    `input_shape` should be the full input, including batch size.
    """
    if not isinstance(model, tf.keras.Model):
        raise KeyError("`model` argument must be `tf.keras.Model` instance.")

    try:
        # Convert `tf.keras model` into frozen graph
        inputs = tf.TensorSpec(list(input_shape), "float32")
        real_model = tf.function(model).get_concrete_function(inputs)
        frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(
            graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
        )
        # For some reason `total_float_ops` returns twice the number of FLOPS when
        # compared with results in papers. This difference is consistent. Hence, we
        # divide by 2 to adjust for it, even though we don't understand the reasons...
        flops = flops.total_float_ops // 2
    except ValueError:
        # For big models we can run out of memory...
        flops = 0

    return flops


def get_parameters(model: tf.keras.Model) -> int:
    trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    non_trainable_params = np.sum(
        [np.prod(v.get_shape()) for v in model.non_trainable_weights]
    )
    params = trainable_params + non_trainable_params
    return params
