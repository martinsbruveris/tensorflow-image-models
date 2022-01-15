import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tensorflow as tf

from ..interface import ProblemBase
from ..registry import cfg_serializable, get_class


@dataclass
class DistillationConfig:
    student: Any
    student_class: str
    teacher: Any
    teacher_class: str
    optimizer: Any
    optimizer_class: str = "OptimizerFactory"
    # If `True`, we normalize both teacher and student embeddings with respect to the
    # L2-norm, before computing the loss.
    normalize_embeddings: bool = True
    # We apply weight decay by summing the squares of all trainable variables and
    # multiplying them by `weight_decay`. We are ignoring Keras weight regularizers
    # and the automatically generated model losses.
    weight_decay: float = 0.0
    mixed_precision: bool = False


@cfg_serializable
class DistillationProblem(ProblemBase):
    cfg_class = DistillationConfig

    def __init__(self, cfg: DistillationConfig, timekeeping):
        self.cfg = cfg
        self.timekeeping = timekeeping

        # Setting global state before building model
        if cfg.mixed_precision:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")

        # Building the model
        student, preprocess_student = get_class(cfg.student_class)(cfg=cfg.student)()
        teacher, preprocess_teacher = get_class(cfg.teacher_class)(cfg=cfg.teacher)()
        self.student = student
        self.preprocess_student = preprocess_student
        self.teacher = teacher
        self.preprocess_teacher = preprocess_teacher

        # Training metrics
        self.avg_l2_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.avg_reg_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.avg_loss = tf.keras.metrics.Mean(dtype=tf.float32)

        # Optimizer
        self.optimizer = get_class(cfg.optimizer_class)(
            cfg=cfg.optimizer,
            timekeeping=timekeeping,
            mixed_precision=cfg.mixed_precision,
        )()

    def ckpt_variables(self, model_only: bool = False):
        """Return dictionary with all variables that need to be added to checkpoint."""
        variables = {"model": self.student}
        if not model_only:
            variables["avg_l2_loss"] = self.avg_l2_loss
            variables["avg_reg_loss"] = self.avg_reg_loss
            variables["avg_loss"] = self.avg_loss
            variables["optimizer"] = self.optimizer
        return variables

    def start_epoch(self):
        """Called at the beginning of an epoch. Used to reset moving averages."""
        self.avg_l2_loss.reset_states()
        self.avg_reg_loss.reset_states()
        self.avg_loss.reset_states()

    def train_step(self, img, it):
        """Perform one step of training."""
        l2_loss, reg_loss = self.train_step_inner(img)

        self.avg_l2_loss.update_state(l2_loss)
        self.avg_reg_loss.update_state(reg_loss)
        self.avg_loss.update_state(reg_loss + l2_loss)

        logs = {
            "train/l2_loss": l2_loss,
            "train/reg_loss": reg_loss,
            "train/loss": l2_loss + reg_loss,
            "train/lr": self.optimizer.learning_rate(self.optimizer.iterations),
            "train/steps": self.optimizer.iterations,
        }
        return self.avg_loss.result().numpy(), logs

    @tf.function
    def train_step_inner(self, img):
        img_student = self.preprocess_student(img)
        img_teacher = self.preprocess_teacher(img)
        emb_teacher = self.teacher(img_teacher, training=False)
        emb_teacher = tf.cast(emb_teacher, tf.float32)
        # Axes over which to normalize embeddings
        emb_axes = tf.range(1, tf.rank(emb_teacher))
        if self.cfg.normalize_embeddings:
            emb_teacher = tf.math.l2_normalize(emb_teacher, axis=emb_axes)

        with tf.GradientTape() as tape:
            emb_student = self.student(img_student, training=True)
            emb_student = tf.cast(emb_student, tf.float32)
            if self.cfg.normalize_embeddings:
                emb_student = tf.math.l2_normalize(emb_student, axis=emb_axes)

            # First we compute the l2-norm for each embedding
            l2_loss = tf.square(emb_student - emb_teacher)
            l2_loss = tf.reduce_sum(l2_loss, axis=emb_axes)
            # And then we average over the batch dimension
            l2_loss = tf.reduce_mean(l2_loss)

            # Weight decay
            reg_loss = 0.0
            nb_variables = 0
            for weight in self.student.trainable_variables:
                reg_loss += tf.reduce_sum(tf.square(weight))
                nb_variables += tf.size(weight)
            reg_loss /= tf.cast(nb_variables, tf.float32)
            reg_loss *= self.cfg.weight_decay

            # Total loss
            loss = l2_loss + reg_loss
            if self.cfg.mixed_precision:
                loss = self.optimizer.get_scaled_loss(loss)

        grads = tape.gradient(loss, self.student.trainable_variables)
        if self.cfg.mixed_precision:
            grads = self.optimizer.get_unscaled_gradients(grads)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))

        return l2_loss, reg_loss

    def validation(self, dataset):
        if hasattr(dataset, "validation"):
            # Validation is done by the dataset. We need to construct a callable model
            # and pass it to the dataset, which will do everything for us.
            def model(img):
                img = self.preprocess_student(img)
                emb = self.student(img, training=False)
                emb = tf.cast(emb, tf.float32)
                return emb

            return dataset.validation(model)
        else:
            # We don't have any default validation implemented at the moment.
            return 0.0, {}

    def save_model(self, save_dir):
        """Save models ready for inference."""
        save_dir = Path(save_dir)

        # We need to set policy to float32 for saving, otherwise we save models that
        # perform inference with float16, which is extremely slow on CPUs
        old_policy = tf.keras.mixed_precision.global_policy()
        tf.keras.mixed_precision.set_global_policy("float32")
        # After changing the policy, we need to create a new model using the policy
        model_factory = get_class(self.cfg.student_class)(cfg=self.cfg.student)
        save_model, _ = model_factory()
        save_model.set_weights(self.student.get_weights())

        model_dir = save_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            # If `save_dir` points to a network file system or S3FS, sometimes TF saving
            # can be very slow. It is faster to save to a temporary directory first and
            # copying data across.
            local_dir = Path(tmpdir) / "model"
            save_model.save(str(local_dir))
            # TODO: Add support for S3 paths here...
            try:
                shutil.copytree(str(local_dir), str(model_dir), dirs_exist_ok=True)
            except shutil.Error:
                logging.info("Unexpected error when copying model from tmpdir.")

        # Restore original float policy
        tf.keras.mixed_precision.set_global_policy(old_policy)
