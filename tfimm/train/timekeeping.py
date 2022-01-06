from dataclasses import dataclass

from .registry import cfg_serializable


@cfg_serializable
@dataclass
class Timekeeping:
    """
    The purpose of this class is to coordinate "time" across the trainer class, the
    problem class and the optimizer. By "time" we mean:
     - How long are we going to train for (`nb_epochs`)?
     - How quickly is time passing (`batch_size`)?
     - How long is one epoch (`nb_samples_per_epoch`)?
    This class allows the user to define these fundamental objects in one place in the
    config.
    """

    # How long should the training last?
    nb_epochs: int
    # How many samples do we see per training step, i.e., per call to `train_step()`?
    # We use this value only for time keeping, the dataset is free to return batches
    # of arbitrary shape.
    batch_size: int
    # The trainer class will increase the epoch counter when (a) the `train_ds`
    # iterator is exhausted or (b) we have seen `nb_samples_per_epoch. When
    # `train_ds` is a finite dataset and `nb_samples_per_epoch > 0`, then both
    # conditions will trigger a new epoch. This can lead to epochs of uneven length.
    # Use at your own risk.
    #
    # On the other hand, `nb_samples_per_epoch` will be used by various learning
    # rate schedulers to know when to change the learning rate.
    # If `nb_samples_per_epoch = -1`, most learning rate schedulers will not work
    #
    # The recommendation is to use infinite datasets, e.g., via `ds.repeat()` combined
    # with a value for `nb_samples_per_epoch`.
    nb_samples_per_epoch: int = -1

    @property
    def nb_steps(self) -> int:
        """Total number of steps for the training."""
        if self.nb_samples_per_epoch == -1:
            raise ValueError(
                "Cannot compute `nb_steps`, if `nb_samples_per_epoch` is not set."
            )
        return self.nb_epochs * self.nb_steps_per_epoch

    @property
    def nb_steps_per_epoch(self) -> int:
        """Number of training steps in each epoch."""
        if self.nb_samples_per_epoch == -1:
            raise ValueError(
                "Cannot compute `nb_steps_per_epoch`, if `nb_samples_per_epoch` is not "
                "set."
            )
        return self.nb_samples_per_epoch // self.batch_size
