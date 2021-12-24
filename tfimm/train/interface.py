from abc import ABC, abstractmethod


class ProblemBase(ABC):
    """Problem base class describing the interface expected from each problem class."""

    @abstractmethod
    def train_step(self, data, it: int) -> (float, dict):
        """
        Perform one step of training. Each problem class has to implement at least
        this method. The function should return a loss, used to display training
        progress and a dictionary of metrics that will be logged to W&B.
        """
        raise NotImplementedError

    def start_epoch(self):
        """Called at the beginning of an epoch. Can be used to reset Keras metrics."""
        pass

    def ckpt_variables(self, model_only: bool = False) -> dict:
        """
        Return dictionary with all variables that need to be added to checkpoint.

        If `model_only=True`, return only the model variables, i.e., those variables
        that we would want to transfer across training scripts. This should exclude
        optimizer state or running averages.
        """
        return {}

    def validation(self, dataset) -> dict:
        """
        Function performs validation on a dataset and returns a dictionary of metrics.
        """
        return {}

    def save_model(self, save_dir):
        """
        The trainer class saves models as checkpoints. We may want to save them in a
        more usable form, e.g., as Keras models or saved_models. This saving has to
        be problem specific.
        """
        pass
