import pytest
import tensorflow as tf

from tfimm.train import OptimizerConfig, OptimizerFactory, Timekeeping


@pytest.mark.parametrize("optimizer", ["sgd", "adam", "rmsprop"])
@pytest.mark.parametrize(
    "lr_schedule", ["const", "multisteps", "cosine_decay", "exponential_decay"]
)
@pytest.mark.parametrize("lr_warmup", [-1, 2])
@pytest.mark.parametrize("mixed_precision", [True, False])
def test_optimizer(optimizer, lr_schedule, lr_warmup, mixed_precision):
    timekeeping = Timekeeping(
        nb_epochs=5,
        batch_size=256,
        nb_samples_per_epoch=1000,
    )
    cfg = OptimizerConfig(
        lr=0.001,
        optimizer=optimizer,
        lr_schedule=lr_schedule,
        lr_warmup=lr_warmup,
        lr_decay_rate=0.8,
        lr_decay_frequency=1,
        lr_boundaries=(2, 3),
        lr_values=(0.001, 0.0001, 0.00001),
    )
    optimizer = OptimizerFactory(cfg, timekeeping, mixed_precision)()

    var = tf.Variable(3.0, dtype="float32")

    # `loss` is a callable that takes no argument and returns the value to minimize.
    def loss():
        return 3.0 * var

    # We test if we can use the optimizer
    for _ in range(10):
        optimizer.minimize(loss, var_list=[var])
