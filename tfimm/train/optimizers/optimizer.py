from dataclasses import dataclass
from typing import Any

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

from ..registry import cfg_serializable, get_class


@dataclass
class OptimizerConfig:
    # Lr schedule. Currently, supports `const`, `multisteps`, `cosine_decay` and
    # `exponential_decay`. Different schedules will use different sets of parameters
    # from below.
    lr_schedule: Any
    lr_schedule_class: str

    # Linear lr warmup over `lr_warmup` number of epochs from 0 to the value specified
    # by the schedule.
    lr_warmup: int = -1

    # Which optimizer to use. Currently supports `sgd`, `adam`, `rmsprop`, `adadelta`,
    # `adamax` and `adagrad`.
    optimizer: str = "sgd"
    # Parameters for the optimizer. Usually momentum values. Not all optimizers need
    # both betas, e.g., `sgd` only uses `betas[0]` for its momentum. But, for
    # consistency, we always need to pass a tuple for `betas`.
    betas: tuple = (0.9, 0.999)
    # Gradient clipping parameters. `clipnorm` clips gradients using the l2-norm (on
    # each weight), while `clipvalue` clips each entry independently. Internally TF
    # applies `tf.clip_by_norm` or `tf.clip_by_value` on each weight.
    clipnorm: float = -1.0
    clipvalue: float = -1.0
    # Epsilon parameter for Adam and RMSProp
    epsilon: float = 1e-7
    # Rho parameter representing the decay rate for Adadelta
    rho: float = 0.95
    # Initial accumulator value for Adagrad. Starting value for the accumulators
    # (per-parameter momentum values). Must be non-negative.
    initial_accumulator_value: float = 0.1


@cfg_serializable
class OptimizerFactory:
    cfg_class = OptimizerConfig

    def __init__(self, cfg: OptimizerConfig, timekeeping, mixed_precision: bool):
        self.cfg = cfg
        self.timekeeping = timekeeping
        self.mixed_precision = mixed_precision

    def lr_schedule(self):
        """Create learning rate schedule as defined by config."""

        lr = get_class(self.cfg.lr_schedule_class)(
            cfg=self.cfg.lr_schedule,
            timekeeping=self.timekeeping,
        )()

        if self.cfg.lr_warmup != -1:
            lr = WarmupWrapper(
                lr_schedule=lr,
                warmup_steps=self.cfg.lr_warmup * self.timekeeping.nb_steps_per_epoch,
            )

        return lr

    def optimizer(self, lr):
        cfg = self.cfg

        if cfg.clipnorm != -1.0 and cfg.clipvalue != -1.0:
            raise ValueError(
                "`clipnorm` and `clipvalue` cannot both be used simultaneously."
            )

        # We cannot use `None` in the config class, because we want to adhere to typing
        # to make parsing of configs easier. But TF expects `None`.
        clipnorm = cfg.clipnorm if cfg.clipnorm != -1.0 else None
        clipvalue = cfg.clipvalue if cfg.clipvalue != -1.0 else None

        if cfg.optimizer == "sgd":
            opt = tf.keras.optimizers.SGD(
                learning_rate=lr,
                momentum=cfg.betas[0],
                clipnorm=clipnorm,
                clipvalue=clipvalue,
            )
        elif cfg.optimizer == "adam":
            opt = tf.keras.optimizers.Adam(
                learning_rate=lr,
                beta_1=cfg.betas[0],
                beta_2=cfg.betas[1],
                clipnorm=clipnorm,
                clipvalue=clipvalue,
                epsilon=cfg.epsilon,
            )
        elif cfg.optimizer == "rmsprop":
            opt = tf.keras.optimizers.RMSprop(
                learning_rate=lr,
                rho=cfg.betas[0],
                momentum=cfg.betas[1],
                clipnorm=clipnorm,
                clipvalue=clipvalue,
                epsilon=cfg.epsilon,
            )
        elif cfg.optimizer == "adamax":
            opt = tf.keras.optimizers.Adamax(
                learning_rate=lr,
                beta_1=cfg.betas[0],
                beta_2=cfg.betas[1],
                clipnorm=clipnorm,
                clipvalue=clipvalue,
                epsilon=cfg.epsilon,
            )
        elif cfg.optimizer == "adadelta":
            opt = tf.keras.optimizers.Adadelta(
                learning_rate=lr,
                rho=cfg.rho,
                clipnorm=clipnorm,
                clipvalue=clipvalue,
                epsilon=cfg.epsilon,
            )
        elif cfg.optimizer == "adagrad":
            opt = tf.keras.optimizers.Adagrad(
                learning_rate=lr,
                initial_accumulator_value=cfg.initial_accumulator_value,
                clipnorm=clipnorm,
                clipvalue=clipvalue,
                epsilon=cfg.epsilon,
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}.")

        if self.mixed_precision:
            opt = tf.keras.mixed_precision.LossScaleOptimizer(opt, dynamic=True)

        return opt

    def __call__(self):
        lr = self.lr_schedule()
        opt = self.optimizer(lr)
        return opt


class WarmupWrapper(LearningRateSchedule):
    """Implements linear learning rate warmup."""

    def __init__(self, lr_schedule: LearningRateSchedule, warmup_steps: int):
        self.lr_schedule = lr_schedule
        self.warmup_steps = warmup_steps
        # Warmup will linearly increase the lr from 0 to the value we would have if
        # we followed the lr schedule.
        self.target_lr = lr_schedule(warmup_steps)

    def __call__(self, step):
        warmup_lr = self.target_lr * tf.cast(step / self.warmup_steps, tf.float32)
        # We can't use an if/else statement, because the code is compiled in graph mode.
        lr = tf.cond(
            step < self.warmup_steps,
            true_fn=lambda: warmup_lr,
            false_fn=lambda: self.lr_schedule(step),
        )
        return lr

    def get_config(self):
        return {
            "warmup_steps": self.warmup_steps,
            "lr_schedule": tf.keras.optimizers.schedules.serialize(self.lr_schedule),
        }

    @classmethod
    def from_config(cls, config):
        return WarmupWrapper(
            lr_schedule=tf.keras.optimizers.schedules.deserialize(
                config["lr_schedule"]
            ),
            warmup_steps=config["warmup_steps"],
        )
