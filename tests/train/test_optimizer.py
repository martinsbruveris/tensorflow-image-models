import pytest
import tensorflow as tf

from tfimm.train import (
    LRConstConfig,
    LRCosineDecayConfig,
    LRExpDecayConfig,
    LRMultiStepsConfig,
    OptimizerConfig,
    OptimizerFactory,
    Timekeeping,
)


def get_schedule(
    name: str,
    lr: float,
    lr_decay_frequency: int,
    lr_decay_rate: float,
    lr_boundaries: tuple,
    lr_values: tuple,
):
    if name == "const":
        return LRConstConfig(lr), "LRConstFactory"
    elif name == "multisteps":
        return (
            LRMultiStepsConfig(lr_boundaries=lr_boundaries, lr_values=lr_values),
            "LRMultiStepsFactory",
        )
    elif name == "cosine_decay":
        return LRCosineDecayConfig(lr=lr), "LRCosineDecayFactory"
    elif name == "exponential_decay":
        return (
            LRExpDecayConfig(
                lr=lr,
                lr_decay_rate=lr_decay_rate,
                lr_decay_frequency=lr_decay_frequency,
            ),
            "LRExponentialDecayFactory",
        )
    else:
        raise NameError(f"Unknown schedule: {name}")


@pytest.mark.parametrize(
    "optimizer", ["sgd", "adam", "rmsprop", "adamax", "adadelta", "adagrad"]
)
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

    schedule, schedule_class = get_schedule(
        lr_schedule,
        lr=0.001,
        lr_decay_frequency=1,
        lr_decay_rate=0.8,
        lr_boundaries=(2, 3),
        lr_values=(0.001, 0.0001, 0.00001),
    )

    cfg = OptimizerConfig(
        optimizer=optimizer,
        lr_schedule=schedule,
        lr_schedule_class=schedule_class,
        lr_warmup=lr_warmup,
    )
    optimizer = OptimizerFactory(cfg, timekeeping, mixed_precision)()

    var = tf.Variable(3.0, dtype="float32")

    # `loss` is a callable that takes no argument and returns the value to minimize.
    def loss():
        return 3.0 * var

    # We test if we can use the optimizer
    for _ in range(10):
        optimizer.minimize(loss, var_list=[var])
