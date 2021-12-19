import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Any

try:
    import wandb
except ImportError:
    wandb = None
    logging.info("Could not import `wand`. Logging to W&B not possible.")

import tfimm.train.config as config
from tfimm.models.factory import create_model, create_preprocessing
from tfimm.models.registry import model_config
from tfimm.train.classification import ClassificationConfig
from tfimm.train.datasets import TFDSConfig
from tfimm.train.registry import cfg_serializable, get_class
from tfimm.train.trainer import TrainerConfig


@dataclass
class ModelConfig:
    model_name: str
    pretrained: str
    input_size: tuple
    in_chans: int
    nb_classes: int
    drop_rate: float = 0.0


@cfg_serializable
class ModelFactory:
    cfg_class = ModelConfig

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    def __call__(self):
        model = create_model(
            self.cfg.model_name,
            pretrained=self.cfg.pretrained,
            input_size=self.cfg.input_size,
            in_chans=self.cfg.in_chans,
            nb_classes=self.cfg.nb_classes,
            drop_rate=self.cfg.drop_rate,
        )
        preprocessing = create_preprocessing(self.cfg.model_name)
        return model, preprocessing

    @property
    def tf_input_shape(self):
        """
        Returns the input shape to be used for keras.Input layers. Some models, e.g.,
        ResNets, can operate on arbitrary sized inputs, others, e.g., transformers,
        need fixed-sized inputs.
        """
        cfg = model_config(self.cfg.model_name)
        if getattr(cfg, "fixed_input_size", False):
            input_shape = *self.cfg.input_size, self.cfg.in_chans
        else:
            input_shape = (None, None, self.cfg.in_chans)
        return input_shape


@dataclass
class ExperimentConfig:
    trainer: Any
    trainer_class: str
    problem: Any
    problem_class: str
    # Data
    train_dataset: Any
    train_dataset_class: str
    val_dataset: Any
    val_dataset_class: str
    # One of 0, 10, 20, 30, 40, 50: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
    logging_level: int = 30
    # W&B parameters
    log_wandb: bool = False
    experiment_name: str = "default"  # Experiment name in W&B
    project_name: str = "default"  # Project name in W&B
    entity: str = "default"  # Entity in W&B
    sweep: bool = False

    """
    Args:
        experiment_name: Experiment name; used in W&B
        project_name: Project name; used in W&B
        sweep: Is this run part of a W&B hyper-parameter sweep. If True, we are adding
            a suffix to the name and save directory, because all runs in the sweep
            will have the same name.
    """


def setup_logging(logging_level):
    """
    Creates a logger that logs to stdout. Sets this logger as the global default.
    Logging format is
        2020-12-05 21:44:09,908: Message.

    Returns:
        Doesn't return anything. Modifies global logger.
    """
    logging.basicConfig(level=logging_level)
    fmt = logging.Formatter("%(asctime)s: %(message)s", datefmt="%y-%b-%d %H:%M:%S")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # We want to change the format of the root logger, which is the first one
    # in the logger.handlers list. A bit hacky, but there we go.
    root = logger.handlers[0]
    root.setFormatter(fmt)


def run(cfg):
    """Runs experiment defined by config"""

    # Configure logging level
    setup_logging(cfg.logging_level)

    # Print config to stdout
    logging.info("Running with arguments")
    config.pprint(cfg)

    # Prepare W&B
    if cfg.log_wandb:
        wandb.init(
            dir=tempfile.gettempdir(),
            config=config.deep_to_flat(config.to_dict_format(cfg)),
            entity=cfg.entity,
            project=cfg.project_name,
            name=cfg.experiment_name,
            resume=False,
        )

    # In case we are using W&B sweep we need to adjust project name etc. to make them
    # specific for each run in the sweep.
    if cfg.sweep:
        ckpt_dir = getattr(cfg.trainer, "ckpt_dir", "")
        if ckpt_dir:
            setattr(cfg.trainer, "ckpt_dir", os.path.join(ckpt_dir, wandb.run.id))
        wandb.run.name = wandb.run.name + f"{wandb.run.id}"
        wandb.run.save()

    # Construct datasets
    train_ds = get_class(cfg.train_dataset_class)(cfg=cfg.train_dataset)
    # TODO: Make val dataset optional
    val_ds = get_class(cfg.val_dataset_class)(cfg=cfg.val_dataset)

    problem = get_class(cfg.problem_class)(cfg=cfg.problem)

    trainer = get_class(cfg.trainer_class)(
        problem=problem,
        train_ds=train_ds,
        val_ds=val_ds,
        log_wandb=cfg.log_wandb,
        cfg=cfg.trainer,
    )

    trainer.train()


def main():
    cfg = ExperimentConfig(
        trainer=TrainerConfig(
            nb_epochs=3,
            nb_samples_per_epoch=640,
            display_loss_every_it=5,
            ckpt_dir="tmp/exp_2",
            init_ckpt="tmp/exp_1/ckpt-3",
        ),
        trainer_class="SingleGPUTrainer",
        problem=ClassificationConfig(
            model=ModelConfig(
                model_name="resnet18",
                pretrained="",
                input_size=(64, 64),
                in_chans=3,
                nb_classes=10,
            ),
            model_class="ModelFactory",
            binary_loss=False,
            weight_decay=0.01,
            lr=0.01,
            mixed_precision=False,
        ),
        problem_class="ClassificationProblem",
        train_dataset=TFDSConfig(
            dataset_name="cifar10",
            split="train",
            input_size=(64, 64),
            batch_size=32,
            repeat=True,
            shuffle=True,
            nb_samples=-1,
            dtype="float32",
        ),
        train_dataset_class="TFDSWrapper",
        val_dataset=TFDSConfig(
            dataset_name="cifar10",
            split="test",
            input_size=(64, 64),
            batch_size=32,
            repeat=False,
            shuffle=False,
            nb_samples=320,
            dtype="float32",
        ),
        val_dataset_class="TFDSWrapper",
        log_wandb=False,
    )

    run(cfg)


if __name__ == "__main__":
    main()
