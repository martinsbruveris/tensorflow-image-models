import logging
import time
from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf

try:
    import wandb
except ImportError:
    logging.info("Could not import `wand`. Logging to W&B not possible.")

from tfimm.train.registry import cfg_serializable


@dataclass
class TrainerConfig:
    # A new epoch is started when (a) the `train_ds` iterator is exhausted or (b) we
    # have seen `nb_samples_per_epoch. When `train_ds` is a finite dataset and
    # `nb_samples_per_epoch > 0`, then both conditions will trigger a new epoch. This
    # can lead to epochs of uneven length. Use at your own risk.
    nb_epochs: int
    nb_samples_per_epoch: int = -1

    # Validation
    # If `True`, we perform validation before starting training
    validation_before_training: bool = True
    # We always perform validation at the end of each epoch. If `validation_every_it`
    # is set, we also perform validation during the epoch every given number of steps.
    validation_every_it: int = -1

    # Checkpointing
    # TODO: Add saving functionality
    ckpt_every_it: int = -1
    ckpt_to_keep: int = 3
    resume_from_ckpt: bool = True
    exp_dir: str = ""

    # Display
    display_loss_every_it: int = 1000
    # If `verbose=False`, we supress any logging.info outputs.
    verbose: bool = True


@cfg_serializable
class SingleGPUTrainer:
    cfg_class = TrainerConfig

    def __init__(self, problem, train_ds, val_ds, log_wandb: bool, cfg: TrainerConfig):
        """Initialize the trainer object"""
        self.problem = problem
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.log_wandb = log_wandb
        self.cfg = cfg

        # Determine batch size by looking at a sample from the dataset
        data = next(iter(self.train_ds))
        data = data[0] if isinstance(data, tuple) else data
        self.batch_size = len(data)

        # Other training-related ops
        self.epoch = tf.Variable(0)

        self.ckpt = None
        self.ckpt_variables = None
        self.ckpt_manager = None
        self.init_ckpt_saver()

        # Restore model state
        if self.cfg.resume_from_ckpt:
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
        for epoch_idx in range(first_epoch, self.cfg.nb_epochs):
            self.problem.start_epoch()

            it = 0
            samples_seen = 0
            last_time = time.time()
            while True:
                # Start a new epoch, if we have seen enough samples
                if (
                    self.cfg.nb_samples_per_epoch != -1
                    and samples_seen >= self.cfg.nb_samples_per_epoch
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
                    logs.update({"epoch": epoch_idx})
                    wandb.log(logs, commit=True)

                it += 1

            self.validation()
            self.save_ckpt()
            self.epoch.assign_add(1)  # End of epoch

        if self.cfg.verbose:
            logging.info("... done training.")

    ###
    # Validation
    ###
    def validation(self):
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
            + f"duration={duration:.1f}, "
            + f"Metric={metric:.4f}"
        )
        if self.cfg.verbose:
            logging.info(status)

    ###
    # Saving
    ###
    def save_models(self, save_path):
        for model in self.model.optimizers:
            if len(self.model.optimizers) == 1:
                model_path = Path(save_path) / Path("model")
            else:
                model_path = Path(save_path) / Path(model["name"])
            # TODO: not sure this is safe
            # if model_path.is_dir():
            #     shutil.rmtree(model_path)
            tf.saved_model.save(model["model"], str(model_path))

    ###
    # Checkpoint management
    ###
    def init_ckpt_saver(self):
        """Creates a checkpoint and a checkpoint manager"""
        self.ckpt_variables = {"epoch": self.epoch}
        self.ckpt_variables.update(self.problem.ckpt_variables)
        self.ckpt = tf.train.Checkpoint(**self.ckpt_variables)

        if self.cfg.exp_dir:
            self.ckpt_manager = tf.train.CheckpointManager(
                checkpoint=self.ckpt,
                directory=self.exp_dir,
                max_to_keep=self.ckpt_to_keep,
                keep_checkpoint_every_n_hours=12,
            )

    def save_ckpt(self):
        """Save a model checkpoint"""
        if not self.ckpt_manager:
            return

        if self.cfg.verbose:
            logging.info("Saving checkpoint...")

        ckpt_path = self.ckpt_manager.save()
        if self.verbose:
            if not ckpt_path:
                logging.info("... checkpoint wasn't saved -- Not sure why.")
            else:
                logging.info(f"... checkpoint saved in {ckpt_path}")

        # Save model in saved_model format
        self.save_models(self.exp_dir)

        # Save weights in h5 format
        save_path = Path(self.exp_dir)
        for model in self.model.optimizers:
            model["model"].save_weights(str(save_path / Path(f"/{model['name']}.h5")))

        if self.verbose:
            logging.info("... saving finished.")

    def load_ckpt(self):
        """Load a model checkpoint
        If model weights are given, load weights. Otherwise, try to load model
        from checkpoint. If no checkpoint is present, do nothing.
        """
        # Run initializers
        self.ckpt.restore(None)

        if self.ckpt_manager and self.ckpt_manager.latest_checkpoint:
            if self.verbose:
                logging.info(
                    f"Restoring model from checkpoint "
                    f"{self.ckpt_manager.latest_checkpoint}..."
                )

            self.ckpt.restore(self.ckpt_manager.latest_checkpoint).expect_partial()

        else:
            if self.cfg.verbose:
                logging.info("No weights/ckpt loaded, keep model as is...")

        if self.cfg.verbose:
            logging.info("...model loaded.")
