from abc import ABC, abstractclassmethod
from dataclasses import dataclass
from typing import Any

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

from ..registry import cfg_serializable


class BaseLRSchedule(ABC):
    def __init__(self, cfg: Any, timekeeping: Any):
        self.cfg = cfg
        self.timekeeping = timekeeping

    @abstractclassmethod
    def __call__(self) -> LearningRateSchedule:
        """Return a tensorflow learning rate schedule"""


@dataclass
class LRConstConfig:
    # Learning rate value
    lr: float


@cfg_serializable
class LRConstFactory(BaseLRSchedule):
    cfg_class = LRConstConfig

    def __init__(self, cfg: LRConstConfig, timekeeping: Any):
        super().__init__(cfg, timekeeping)

    def __call__(self):
        # We simulate a constant lr using a keras `LearningRateSchedule` so we can
        # wrap it using the `WarmupWrapper` below.
        return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=[0], values=[self.cfg.lr, self.cfg.lr]
        )


@dataclass
class LRMultiStepsConfig:
    # Used by `multisteps`. At the epochs given by `lr_boundaries` we change the lr to
    # the values given by `lr_values`. Note that we need
    # `len(lr_values) = len(lr_boundaries) + 1`.
    lr_boundaries: tuple
    lr_values: tuple


@cfg_serializable
class LRMultiStepsFactory(BaseLRSchedule):
    cfg_class = LRMultiStepsConfig

    def __init__(self, cfg: LRMultiStepsConfig, timekeeping: Any):
        super().__init__(cfg, timekeeping)

    def __call__(self):
        # We convert `lr_boundaries` from epochs to steps.
        boundaries = [
            val * self.timekeeping.nb_steps_per_epoch for val in self.cfg.lr_boundaries
        ]
        return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=boundaries, values=self.cfg.lr_values
        )


@dataclass
class LRCosineDecayConfig:
    # Learning rate value
    lr: float
    alpha: float = 0.0


@cfg_serializable
class LRCosineDecayFactory(BaseLRSchedule):
    cfg_class = LRCosineDecayConfig

    def __init__(self, cfg: LRCosineDecayConfig, timekeeping: Any):
        super().__init__(cfg, timekeeping)

    def __call__(self):
        return tf.keras.optimizers.schedules.CosineDecay(
            self.cfg.lr, decay_steps=self.timekeeping.nb_steps, alpha=self.cfg.alpha
        )


@dataclass
class LRExpDecayConfig:
    # Learning rate value
    lr: float
    # Used by `exponential_decay`. We decay the lr every `lr_decay_frequency` epochs
    # by the facto `lr_decay_rate`.
    lr_decay_rate: float
    lr_decay_frequency: int
    staircase: bool = True


@cfg_serializable
class LRExponentialDecayFactory(BaseLRSchedule):
    cfg_class = LRExpDecayConfig

    def __init__(self, cfg: LRExpDecayConfig, timekeeping: Any):
        super().__init__(cfg, timekeeping)

    def __call__(self):
        return tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.cfg.lr,
            decay_steps=self.cfg.lr_decay_frequency
            * self.timekeeping.nb_steps_per_epoch,
            decay_rate=self.cfg.lr_decay_rate,
            staircase=self.cfg.staircase,
        )
