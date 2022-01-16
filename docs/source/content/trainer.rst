Trainer Class
=============

The trainer class performs high-level orchestration of the training process.
Specifically, it takes care of the following:

* The main training loop, keeping count of training steps and epochs.
* Checkpoint saving and loading existing checkpoints at the start of training.
* Calling validation.
* Logging to `Weights & Biases`_ (W&B).

The trainer class is only the orchestrator, it relies on the problem class to implement
these operations. Each problem class is a subclass of ``ProblemBase``.

.. py:module:: tfimm.train.trainer

.. autoclass:: TrainerConfig
.. autoclass:: SingleGPUTrainer


.. _Weights & Biases: https://wandb.ai/site

