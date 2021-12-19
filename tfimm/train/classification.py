from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf

from tfimm.train.interface import ProblemBase
from tfimm.train.registry import cfg_serializable, get_class


@dataclass
class ClassificationConfig:
    model: Any
    model_class: str
    # We treat binary classification problems as a special case, because for binary
    # problems the model can return just one logit, which is the logit for class 1.
    # The logit for class 0 is implicitly set to 0.0.
    binary_loss: bool = False
    # We apply weight decay by summing the squares of all trainable variables and
    # multiplying them by `weight_decay`. We are ignoring Keras weight regularizers
    # and the automatically generated model losses.
    weight_decay: float = 0.0
    lr: float = 0.01  # Learning rate
    mixed_precision: bool = False


@cfg_serializable
class ClassificationProblem(ProblemBase):
    cfg_class = ClassificationConfig

    def __init__(self, cfg: ClassificationConfig):
        self.cfg = cfg

        # Setting global state before building model
        if cfg.mixed_precision:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")

        # Building the model
        model, preprocess = get_class(cfg.model_class)(cfg=cfg.model)()
        self.model = model
        self.preprocess = preprocess
        self._ckpt_variables = {"model": self.model}

        # Training metrics
        self.avg_ce_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.avg_reg_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.avg_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.avg_acc = tf.keras.metrics.Accuracy(dtype=tf.float32)

        # Optimizer
        # TODO: Allow optimizer customization and learning rate schedules...
        # TODO: Add learning rate warmup...
        self.optimizer = tf.optimizers.SGD(self.cfg.lr, momentum=0.9)

    @property
    def ckpt_variables(self):
        """Return dictionary with all variables that need to be added to checkpoint."""
        return self._ckpt_variables

    def start_epoch(self):
        """Called at the beginning of an epoch. Used to reset moving averages."""
        self.avg_ce_loss.reset_states()
        self.avg_reg_loss.reset_states()
        self.avg_loss.reset_states()
        self.avg_acc.reset_states()

    def train_step(self, data, it):
        """Perform one step of training."""
        img, labels = data
        ce_loss, reg_loss, preds = self.train_step_inner(img, labels)

        self.avg_ce_loss.update_state(ce_loss)
        self.avg_reg_loss.update_state(reg_loss)
        self.avg_loss.update_state(reg_loss + ce_loss)
        self.avg_acc.update_state(preds, labels)

        logs = {
            "train/ce_loss": self.avg_ce_loss.result().numpy(),
            "train/reg_loss": self.avg_reg_loss.result().numpy(),
            "train/loss": self.avg_loss.result().numpy(),
            "train/acc": self.avg_acc.result().numpy(),
        }
        return logs["train/loss"], logs

    @tf.function
    def train_step_inner(self, img, labels):
        img = self.preprocess(img)

        with tf.GradientTape() as tape:
            logits = self.logits(img, training=True)
            # Regardless of mixed precision or not, we compute the loss in float32
            logits = tf.cast(logits, tf.float32)
            ce_loss = self.softmax_loss(logits, labels)

            # Weight decay
            # TODO: Exclude certain variables from weight decay based on model cfg
            reg_loss = 0.0
            for weight in self.model.trainable_variables:
                reg_loss += tf.reduce_sum(tf.square(weight))
            reg_loss *= self.cfg.weight_decay

            # Total loss
            loss = ce_loss + reg_loss
            if self.cfg.mixed_precision:
                loss = self.optimizer.get_scaled_loss(loss)

        grads = tape.gradient(loss, self.model.trainable_variables)
        if self.cfg.mixed_precision:
            grads = self.optimizer.get_unscaled_gradients(grads)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        preds = self.predict(logits)
        return ce_loss, reg_loss, preds

    def logits(self, img, training):
        logits = self.model(img, training=training)
        logits = tf.cast(logits, tf.float32)
        return logits

    def predict(self, logits):
        if self.cfg.binary_loss:
            logits = tf.concat([tf.zeros_like(logits), logits], axis=-1)
        preds = tf.argmax(logits, axis=-1)
        return preds

    def softmax_loss(self, logits, labels):
        if self.cfg.binary_loss:
            loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.cast(labels, tf.float32), logits=logits[:, 0]
            )
        else:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits
            )
        loss = tf.reduce_mean(loss)
        return loss

    def logits_and_labels(self, dataset):
        """
        Returns logits and labels from dataset. The dataset must be finite.

        Returns:
            np.ndarrays for logits and labels of shape (N, nb_classes) and (N,)
        """

        def _inference(_img):
            _img = self.preprocess(_img)
            _logits = self.model(_img, training=False)
            _logits = tf.cast(_logits, tf.float32)
            if self.cfg.binary_loss:
                # In the binary case the model can return logits only for class 1.
                # The logit for class 0 is assumed to be 0.0.
                _logits = _logits[..., 0]
                _logits = tf.stack([tf.zeros_like(_logits), _logits], axis=-1)
            else:
                # Normalize logits to have sum=0. While not necessary to compute
                # accuracy, this becomes important if we want to compare decision
                # thresholds across epochs and training runs.
                _logits = _logits - tf.reduce_mean(_logits, axis=-1, keepdims=True)
            return _logits

        labels, logits = [], []
        for img_batch, labels_batch in dataset:
            logits_batch = _inference(img_batch)
            logits.append(logits_batch.numpy())
            labels.append(labels_batch.numpy())
        labels = np.concatenate(labels, axis=0)
        logits = np.concatenate(logits, axis=0)
        return logits, labels

    def validation(self, dataset):
        """
        Function performs validation on a dataset and returns a dictionary of metrics.
        """
        logits, labels = self.logits_and_labels(dataset)
        preds = tf.argmax(logits, axis=-1).numpy()

        # This is work in progress. For now, we measure only accuracy. Later, we should
        # add top-5 accuracy, etc.
        acc = np.sum(preds == labels) / len(labels)

        logs = {"val/acc": acc}
        return logs["val/acc"], logs
