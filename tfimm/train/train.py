import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import wandb
except ImportError:
    wandb = None
    logging.info("Could not import `wand`. Logging to W&B not possible.")

import tfimm.train.config as config
from tfimm.train.classification import ClassificationConfig
from tfimm.train.datasets import TFDSConfig
from tfimm.train.model import ModelConfig
from tfimm.train.registry import get_class
from tfimm.train.trainer import TrainerConfig
from tfimm.train.utils import setup_logging


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
    # If this run is part of a W&B hyperparameter sweep, we need to add suffixes to
    # the run names and checkpoint directories, because otherwise all runs in the sweep
    # will have the same name and the checkpoints will overwrite each other.
    sweep: bool = False


def run(cfg: ExperimentConfig):
    """Runs experiment defined by `cfg`."""

    # Configure logging level
    setup_logging(cfg.logging_level)

    # Print config to stdout
    logging.info("Running with arguments")
    config.pprint(cfg)

    # Save config to file
    ckpt_dir = getattr(cfg.trainer, "ckpt_dir", "")
    if ckpt_dir:
        config.dump_config(cfg, Path(ckpt_dir) / "config.yaml")

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
                nb_channels=3,
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
