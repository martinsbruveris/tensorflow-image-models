import math
from time import time
from typing import Optional, Tuple

import tensorflow as tf

from tfimm.models import registry
from tfimm.models.factory import create_model
from tfimm.utils import to_2tuple


def _below_resolution(
    lower: int,
    upper: int,
    resolution_abs: int,
    resolution_rel: Optional[float],
):
    """We check if (upper - lower) <= resolution."""
    if resolution_rel is not None:
        if abs(upper - lower) <= upper * resolution_rel:
            return True

    # Absolute resolution is always at least 1
    if abs(upper - lower) <= resolution_abs:
        return True

    return False


def _time_function(fun, img, nb_batches, verbose):
    """Helper function to time the execution of `fun(img)`."""
    # We ignore the first run because graph compilation takes time. And some memory.
    fun(img)

    # Now we start counting batches
    start = time()
    for j in range(nb_batches):
        fun(img)
        if verbose:
            print(f"Batch {j}: {(time() - start) / (j + 1):.3f}sec.")
    duration = time() - start
    return duration


def time_model(
    model_name,
    target,
    input_size,
    nb_classes,
    batch_size,
    float_policy,
    nb_batches,
    verbose=False,
):
    """
    Time backpropagation speed of model. The loss is simply the mean of all model
    outputs.

    Args:
        model_name: Model to be timed, will be created using `create_model`.
        target: One of "inference" or "backprop"
        input_size: Model input size
        nb_classes: Number of classes
        batch_size: Batch size to be used for testing.
        float_policy: Can be "float32" or "mixed_float16"
        nb_batches: Backpropagation time is averages over `nb_batches` calls.
        verbose: If `True`, we print duration of each batch

    Returns:
        Backpropagation throughput in img/sec.
    """
    assert float_policy in {"float32", "mixed_float16"}

    tf.keras.backend.clear_session()  # Release GPU memory
    # Need to set policy before creating model
    tf.keras.mixed_precision.set_global_policy(float_policy)
    dtype = "float32" if float_policy == "float32" else "float16"

    input_size = to_2tuple(input_size) if input_size is not None else input_size
    model = create_model(model_name, input_size=input_size, nb_classes=nb_classes)
    img = tf.ones(
        (batch_size, *model.cfg.input_size, model.cfg.in_channels),
        dtype=dtype,
    )

    if target == "inference":

        @tf.function(experimental_relax_shapes=True, jit_compile=True)
        def _fun(x):
            return model(x, training=False)

    elif target == "backprop":
        optimizer = tf.optimizers.SGD(learning_rate=0.0001)

        @tf.function(experimental_relax_shapes=True)
        def _fun(x):
            with tf.GradientTape() as tape:
                output = model(x, training=True)
                # The loss is always computed in float32 in order to not lose precision
                # Here we simulate it to make profiling more accurate
                output = tf.cast(output, "float32")
                loss = tf.reduce_mean(output)
                grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

    else:
        raise ValueError(f"Unknown target: {target}.")

    duration = _time_function(_fun, img, nb_batches, verbose)
    img_per_sec = batch_size * nb_batches / duration
    return img_per_sec


def find_max_batch_size(
    model_name: str,
    target: str = "inference",
    input_size: Optional[int] = None,
    nb_classes: Optional[int] = None,
    float_policy: str = "float32",
    nb_batches: int = 3,
    start_batch_size: int = 256,
    resolution_abs: int = 1,
    resolution_rel: Optional[float] = 0.1,
    verbose: bool = False,
) -> Tuple[int, float]:
    """
    Searches for largest batch size that fits in memory.

    Args:
        model_name: Model to validate
        target: Can be "inference" or "backprop"
        float_policy: Can be "float32" or "mixed_float16"
        nb_batches: For how many batches to run the test
        start_batch_size: First batch size to try
        resolution_abs: We stop, if upper-lower <= resolution_abs
        resolution_rel: We stop, if (upper-lower) <= upper * resolution_rel
        verbose: If True, we print information about search progress
    Returns:
        Maximum batch size that does not lead to OOM errors.
        Inference time in img/sec with that batch size
    """
    upper_limit = None
    lower_limit = 0

    # Find hard batch size cap depending on model input size. The whole batch should
    # be <0.5 GB of memory.
    cfg = registry.model_config(model_name)
    img_size = 4 * cfg.input_size[0] * cfg.input_size[1] * cfg.in_channels
    max_memory = 5 * 10 ** 8
    # We want max batch size to be a power of 2
    max_batch_size = 2 ** math.floor(math.log2(max_memory / img_size))

    continue_search = True
    next_batch_size = min(start_batch_size, max_batch_size)
    img_per_sec = 0.0
    while continue_search:
        batch_size = next_batch_size
        if verbose:
            print(f"Trying: {batch_size}. Range: ({lower_limit}, {upper_limit})")
        try:
            img_per_sec = time_model(
                model_name=model_name,
                target=target,
                input_size=input_size,
                nb_classes=nb_classes,
                batch_size=batch_size,
                float_policy=float_policy,
                nb_batches=nb_batches,
            )
            success = True
            lower_limit = batch_size

            if batch_size >= max_batch_size:
                continue_search = False
            elif upper_limit is None:
                next_batch_size = 2 * batch_size
                next_batch_size = min(next_batch_size, max_batch_size)
            elif _below_resolution(
                lower_limit, upper_limit, resolution_abs, resolution_rel
            ):
                continue_search = False
            else:
                next_batch_size = (upper_limit + batch_size) // 2

        except (
            tf.errors.InternalError,
            # The next one catches creating models with invalid parameters
            tf.errors.InvalidArgumentError,
            tf.errors.ResourceExhaustedError,
            tf.errors.UnknownError,
        ):
            success = False
            upper_limit = batch_size
            if _below_resolution(
                lower_limit, upper_limit, resolution_abs, resolution_rel
            ):
                continue_search = False
            else:
                next_batch_size = (batch_size + lower_limit) // 2

        finally:
            if verbose:
                print(f"Batch size {batch_size}: {'valid' if success else 'oom'}")

    return lower_limit, img_per_sec
