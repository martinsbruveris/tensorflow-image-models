from abc import ABC, abstractmethod


class ProblemBase(ABC):
    """Problem base class describing the interface expected from each problem class."""

    @property
    def ckpt_variables(self) -> dict:
        """Return dictionary with all variables that need to be added to checkpoint."""
        return {}

    def start_epoch(self):
        """Called at the beginning of an epoch. Can be used to reset Keras metrics."""
        pass

    @abstractmethod
    def train_step(self, data, it: int):
        """Perform one step of training."""
        raise NotImplementedError

    def validation(self, dataset) -> dict:
        """
        Function performs validation on a dataset and returns a dictionary of metrics.
        """
        return {}
