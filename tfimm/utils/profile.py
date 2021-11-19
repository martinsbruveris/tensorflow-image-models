from time import time
from typing import Optional

import tensorflow as tf

from tfimm.models.factory import create_model


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
    _ = fun(img)

    # Now we start counting batches
    start = time()
    for j in range(nb_batches):
        _ = fun(img)
        if verbose:
            print(f"Batch {j}: {(time() - start) / (j + 1):.3f}sec.")
    duration = time() - start
    return duration


def time_inference(model_name, batch_size, nb_batches=1, verbose=False):
    """
    Time inference of model.

    Args:
        model_name: Model to be timed, will be created using `create_model`.
        batch_size: Batch size to be used for inference. Usually determined first
            by `find_max_batch_size`.
        nb_batches: Inference time is averages over `nb_batches` calls.
        verbose: If `True`, we print duration of each batch

    Returns:
        Inference throughput in img/sec.
    """
    model = create_model(model_name)
    img = tf.ones(
        (batch_size, *model.cfg.input_size, model.cfg.in_chans),
        dtype="float32",
    )

    @tf.function(experimental_relax_shapes=True)
    def _fun(x):
        return model(x, training=False)

    duration = _time_function(_fun, img, nb_batches, verbose)
    img_per_sec = batch_size * nb_batches / duration
    tf.keras.backend.clear_session()  # Release GPU memory
    return img_per_sec


def time_backprop(model_name, batch_size, nb_batches=1, verbose=False):
    """
    Time backpropagation speed of model. The loss is simply the mean of all model
    outputs.

    Args:
        model_name: Model to be timed, will be created using `create_model`.
        batch_size: Batch size to be used for inference. Usually determined first
            by `find_max_batch_size`.
        nb_batches: Backpropagation time is averages over `nb_batches` calls.
        verbose: If `True`, we print duration of each batch

    Returns:
        Backpropagation throughput in img/sec.
    """
    model = create_model(model_name)
    img = tf.ones(
        (batch_size, *model.cfg.input_size, model.cfg.in_chans),
        dtype="float32",
    )
    optimizer = tf.optimizers.SGD(learning_rate=0.0001)

    @tf.function(experimental_relax_shapes=True)
    def _fun(x):
        with tf.GradientTape() as tape:
            output = model(x, training=True)
            loss = tf.reduce_mean(output)
            grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    duration = _time_function(_fun, img, nb_batches, verbose)
    img_per_sec = batch_size * nb_batches / duration
    tf.keras.backend.clear_session()  # Release GPU memory
    return img_per_sec


def find_max_batch_size(
    model_name: str,
    test_target: str = "inference",
    start_batch_size: int = 256,
    resolution_abs: int = 1,
    resolution_rel: Optional[float] = 0.1,
    verbose: bool = False,
) -> int:
    """
    Searches for largest batch size that fits in memory.

    Args:
        model_name: Model to validate
        test_target: Can be "inference" or "backprop"
        start_batch_size: First batch size to try
        resolution_abs: We stop, if upper-lower <= resolution_abs
        resolution_rel: We stop, if (upper-lower) <= upper * resolution_rel
        verbose: If True, we print information about search progress
    Returns:
        Maximum batch size that does not lead to OOM errors.
    """
    assert test_target in {"inference", "backprop"}

    upper_limit = None
    lower_limit = 0

    continue_search = True
    next_batch_size = start_batch_size
    while continue_search:
        batch_size = next_batch_size
        if verbose:
            print(f"Trying: {batch_size}. Range: ({lower_limit}, {upper_limit})")
        try:
            if test_target == "inference":
                time_inference(model_name, batch_size)
            else:
                time_backprop(model_name, batch_size)

            success = True
            lower_limit = batch_size

            if upper_limit is None:
                next_batch_size = 2 * batch_size
            elif _below_resolution(
                lower_limit, upper_limit, resolution_abs, resolution_rel
            ):
                continue_search = False
            else:
                next_batch_size = (upper_limit + batch_size) // 2

        except (tf.errors.ResourceExhaustedError, tf.errors.UnknownError):
            # Clear GPU memory
            tf.keras.backend.clear_session()

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

    return lower_limit
