import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

try:
    import wandb
except ImportError:
    wandb = None
    logging.info("Could not import `wand`. Logging to W&B not possible.")

import tfimm.train.config as config
from tfimm.train.registry import get_class
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
    val_dataset: Any = None
    val_dataset_class: str = ""
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
    # Configuration file
    cfg_file: str = ""


def run(cfg: Union[ExperimentConfig, dict], parse_args: bool = True):
    """
    Runs experiment defined by `cfg`.

    If `parse_args=True`, we will parse command line arguments. We will always parse
    the cfg_file.
    """
    # If we want to parse command line args, we provide None to `parse_args`, which
    # will default to using argv[1:]. By providing the empty list it will not look
    # for other arguments to parse.
    args = None if parse_args else []
    cfg = config.parse_args(cfg, cfg_class=ExperimentConfig, args=args)

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

    # Construct constituent objects
    train_ds = get_class(cfg.train_dataset_class)(cfg=cfg.train_dataset)
    val_ds = (
        get_class(cfg.val_dataset_class)(cfg=cfg.val_dataset)
        if cfg.val_dataset_class
        else None
    )
    problem = get_class(cfg.problem_class)(cfg=cfg.problem)
    trainer = get_class(cfg.trainer_class)(
        problem=problem,
        train_ds=train_ds,
        val_ds=val_ds,
        log_wandb=cfg.log_wandb,
        cfg=cfg.trainer,
    )

    # Do the training!
    trainer.train()
