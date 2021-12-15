from dataclasses import dataclass

from .registry import cfg_serializable


# Each class with hyperparameters should have an associated config dataclass with
# parameters. The typehints will be used to create the command line parser (not
# implemented yet).
@dataclass
class TrainerConfig:
    nb_epochs: int
    input_shape: tuple
    # Validation
    skip_first_val: bool = False
    validation_every_it: int = -1
    # Checkpointing
    ckpt_every_it: int = -1
    ckpt_to_keep: int = 3
    resume_from_ckpt: bool = True
    # Display
    display_loss_every_it: int = 1000
    verbose: bool = True
    debug_mode: bool = False


# Each class with hyperparameters needs to be decorated with the `@cfg_serializable`
# decorator, which will associate the class and the corresponding config dataclass
# via the `cfg_class` field.
@cfg_serializable
class BasicTrainer:
    cfg_class = TrainerConfig

    # The constructor will in general take only the config object.
    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg
        pass
