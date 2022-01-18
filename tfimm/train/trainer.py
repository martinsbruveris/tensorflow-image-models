import logging
import time
from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf

try:
    import wandb
except ImportError:
    wandb = None
    logging.info("Could not import `wandb`. Logging to W&B not possible.")

from .registry import cfg_serializable


@dataclass
class TrainerConfig:
    """
    Configuration class for :class:`SingleGPUTrainer`.

    Attributes:
        validation_before_training: If ``True``, we perform validation before the first
            training step. Particularly useful if continuing to train an already
            partially trained model.
        validation_every_it: We always perform validation at the end of each epoch.
            If `validation_every_it` is set, we also perform validation during the
            epoch every given number of steps.

        ckpt_dir: Checkpoints and models are only saved if ``ckpt_dir`` is a non-empty
            string.
        init_ckpt_dir: If ``init_ckpt_dir`` is set, we load the model state from this
            checkpoint at the beginning of training. This parameter can be used to
            manually continue fine-tuning from the end state of a previous training
            run. This can be either the path to a single checkpoint or a directory.
            If ``init_ckpt_dir`` is a directory, we will select the latest checkpoint.
        resume_from_ckpt: This parameter is used to recover from crashed or restarting
            compute nodes. If ``True``, we try to load the latest checkpoint from
            ``ckpt_dir`` at the start of training. Note, that this will override any
            state loaded from `init_ckpt_dir``.
        ckpt_every_it: We always save a checkpoint at the end of each epoch. If
            ``ckpt_every_it`` is set, we also save a checkpoint every given number of
            steps.
        ckpt_to_keep: This parameter controls how many checkpoints we want to store. In
            addition to the last ``ckpt_to_keep`` number of checkpoints, we will also
            keep a checkpoint every 12 hours. The latter is useful for long-running
            training jobs.

        display_loss_every_it: If ``verbose=True``, we log to the training loss every
            given number of iterations to ``logging.info``.
        verbose: If ``True`` we log progress information to `logging.info``. If
            ``False`` we suppress logging outputs.
    """

    # Validation
    validation_before_training: bool = True
    validation_every_it: int = -1

    # Checkpointing
    ckpt_dir: str = ""
    init_ckpt: str = ""
    resume_from_ckpt: bool = True
    ckpt_every_it: int = -1
    ckpt_to_keep: int = 3

    # Display
    display_loss_every_it: int = 1000
    verbose: bool = True


@cfg_serializable
class SingleGPUTrainer:
    """
    Trainer class for single GPU training.
    """

    cfg_class = TrainerConfig

    def __init__(
        self,
        problem,
        train_ds,
        val_ds,
        timekeeping,
        log_wandb: bool,
        cfg: TrainerConfig,
    ):
        """Initialize the trainer object"""
        self.problem = problem
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.timekeeping = timekeeping
        self.log_wandb = log_wandb
        self.cfg = cfg

        # Check that we will be able to log to W&B if needed
        if self.log_wandb and not wandb:
            raise ValueError("Cannot log to W&B as `wandb` could not be imported.")

        # Other training-related ops
        self.epoch = tf.Variable(0)
        self.batch_size = self.timekeeping.batch_size

        # Checkpoint related variables
        self.ckpt = None
        self.ckpt_variables = None
        self.ckpt_manager = None
        self.init_ckpt_saver()

        # Restore model state
        self.load_ckpt()

    ###
    # Training
    ###
    def train(self):
        """Training loop"""
        if self.cfg.verbose:
            logging.info("Starting training...")

        # Run validation first
        if self.cfg.validation_before_training:
            self.validation()

        # Save checkpoint before starting training
        self.save_ckpt()

        duration = tf.keras.metrics.Mean(dtype=tf.float32)  # Time tracker
        first_epoch = int(self.epoch.numpy())

        ds_iter = iter(self.train_ds)
        for epoch_idx in range(first_epoch, self.timekeeping.nb_epochs):
            self.problem.start_epoch()

            it = 0
            samples_seen = 0
            last_time = time.time()
            while True:
                # Start a new epoch, if we have seen enough samples
                if (
                    self.timekeeping.nb_samples_per_epoch != -1
                    and samples_seen >= self.timekeeping.nb_samples_per_epoch
                ):
                    break

                try:
                    data = next(ds_iter)  # Fetch next piece of data
                except StopIteration:
                    ds_iter = iter(self.train_ds)  # Reset iterator for next epoch
                    break

                loss, logs = self.problem.train_step(data, it=it)
                samples_seen += self.batch_size
                duration.update_state(time.time() - last_time)
                last_time = time.time()

                # Print training progress
                if it == 0 or (
                    self.cfg.display_loss_every_it > 0
                    and it % self.cfg.display_loss_every_it == 0
                ):
                    self.print_training_progress(it, loss, duration)
                    duration.reset_states()

                # Run validation
                if (
                    it > 0
                    and self.cfg.validation_every_it > 0
                    and it % self.cfg.validation_every_it == 0
                ):
                    self.validation()

                # Save checkpoint
                if (
                    it > 0
                    and self.cfg.ckpt_every_it > 0
                    and it % self.cfg.ckpt_every_it == 0
                ):
                    self.save_ckpt()

                # Log metrics
                if self.log_wandb:
                    logs.update({"train/epoch": epoch_idx})
                    wandb.log(logs, commit=True)

                it += 1

            self.validation()
            # We increase the epoch counter before saving the checkpoint, because when
            # load the checkpoint we want to continue training from the next epoch
            # onwards.
            self.epoch.assign_add(1)
            self.save_ckpt()

        if self.cfg.verbose:
            logging.info("... done training.")

    ###
    # Validation
    ###
    def validation(self):
        if self.val_ds is None:
            return
        if self.cfg.verbose:
            logging.info("Validation...")

        start_time = time.time()
        metric, logs = self.problem.validation(self.val_ds)
        duration = time.time() - start_time

        self.print_val_progress(metric, duration)

        if self.log_wandb:
            wandb.log(logs, commit=False)

    ###
    # Print progress
    ###
    def print_training_progress(self, it, loss, duration):
        sec_per_step = duration.result().numpy()
        samples_per_step = self.batch_size
        samples_per_sec = samples_per_step / sec_per_step
        status = (
            "Train: "
            + f"Epoch {int(self.epoch.numpy())} : "
            + f"Iter {it} "
            + f"samples/sec={samples_per_sec:.1f}, "
            + f"sec/step={sec_per_step:.3f}, "
            + f"Global loss={loss:.5f}"
        )
        if self.cfg.verbose:
            logging.info(status)

    def print_val_progress(self, metric, duration):
        status = (
            "Val: "
            + f"Epoch {int(self.epoch.numpy())} : "
            + f"duration={duration:.1f}s, "
            + f"Metric={metric:.4f}"
        )
        if self.cfg.verbose:
            logging.info(status)

    ###
    # Checkpoint management
    ###
    def init_ckpt_saver(self):
        """Creates a checkpoint and a checkpoint manager"""
        self.ckpt_variables = {"epoch": self.epoch}
        self.ckpt_variables.update(self.problem.ckpt_variables(model_only=False))
        self.ckpt = tf.train.Checkpoint(**self.ckpt_variables)

        if self.cfg.ckpt_dir:
            self.ckpt_manager = tf.train.CheckpointManager(
                checkpoint=self.ckpt,
                directory=self.cfg.ckpt_dir,
                max_to_keep=self.cfg.ckpt_to_keep,
                keep_checkpoint_every_n_hours=12,
            )

    def save_ckpt(self):
        """Save a model checkpoint"""
        if not self.ckpt_manager:
            return

        if self.cfg.verbose:
            logging.info("Saving checkpoint...")

        ckpt_path = self.ckpt_manager.save()
        if self.cfg.verbose:
            if ckpt_path:
                logging.info(f"... checkpoint saved in {ckpt_path}")
            else:
                logging.info("... checkpoint wasn't saved -- Not sure why.")

        # The problem class might want to save models in some other, more readily
        # usable format.
        self.problem.save_model(self.cfg.ckpt_dir)

        if self.cfg.verbose:
            logging.info("... saving finished.")

    def load_ckpt(self):
        """
        Restore state from a checkpoint. If `resume_from_ckpt=True` and a checkppint
        is present, restore that checkpoint. Otherwise, if `init_ckpt` is set, load
        model weights only from `init_ckpt`. If neither is set, don't restore anything.
        """
        if (
            self.cfg.resume_from_ckpt
            and self.ckpt_manager
            and self.ckpt_manager.latest_checkpoint
        ):
            ckpt_path = self.ckpt_manager.latest_checkpoint
            if self.cfg.verbose:
                logging.info(f"Restoring from checkpoint {ckpt_path}...")
            self.ckpt.restore(ckpt_path).expect_partial()
        elif self.cfg.init_ckpt:
            init_ckpt = Path(self.cfg.init_ckpt)
            if init_ckpt.is_dir():
                ckpt_path = tf.train.latest_checkpoint(str(init_ckpt))
            else:
                ckpt_path = init_ckpt
            if self.cfg.verbose:
                logging.info(f"Restoring model only from checkpoint {ckpt_path}...")

            # We create a separate checkpoint object that restores only the
            # model variables, nothing else
            ckpt_variables = self.problem.ckpt_variables(model_only=True)
            ckpt = tf.train.Checkpoint(**ckpt_variables)
            ckpt.restore(str(ckpt_path)).expect_partial()
        else:
            if self.cfg.verbose:
                logging.info("No weights/ckpt loaded, keep model as is...")

        if self.cfg.verbose:
            logging.info("...model loaded.")
