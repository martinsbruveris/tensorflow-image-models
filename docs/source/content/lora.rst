LoRA
====

.. py:module:: tfimm.architectures.lora

*Low-Rank Adaption (LoRA)* is a parameter-efficient fine-tuning method developed
originally for large language models, but adapted here for vision models. This module
contains TensorFLow code for LoRA layers and their integration with ``tfimm`` models.
For details on LoRA see the paper

**LoRA: Low-Rank Adaptation of Large Language Models.**
*Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang,
Lu Wang, Weizhu Chen.*
Paper: `[arXiv:2106.09685] <https://arxiv.org/abs/2106.09685>`_

Usage
-----

For supported architectures we can use :func:`tfimm.architectures.lora.create_model`
instead of :func:`tfimm.create_model` to create a LoRA model.

.. code-block:: pycon

    >>> from tfimm.architectures import lora
    >>> model = lora.create_model(
    ...    model_name="convnext_tiny", pretrained=True, lora_rank=2
    ... )

When we look at the model summary, we see that most model parameters are non-trainable
and only the low-rank weight updates are trainable.

.. code-block:: pycon

    >>> model.summary()
    ...
    =================================================================
    Total params: 28,721,608
    Trainable params: 132,480
    Non-trainable params: 28,589,128
    _________________________________________________________________

LoRA models can be converted back to regular models.

.. code-block:: pycon

    >>> type(model)
    <class 'tfimm.architectures.lora.convnext.LoRAConvNeXt'>
    >>> regular_model = lora.convert_to_regular_model(model)
    >>> type(regular_model)
    <class 'tfimm.architectures.convnext.ConvNeXt'>

Supported architectures
-----------------------

Currently we support the following architectures

* ConvNeXt

And the following layers

* Dense

Implementation
--------------

In order to perform LoRA training, the first task is to convert a regular model to its
LoRA version. For ``tfimm`` architectures we do this by subclassing and modifying layers
in ``__init__``. E.g., ``LoRAConvNeXt`` is subclassed from
``ConvNeXt`` and we replace the dense layers in each MLP block by their LoRA
counterparts.

We use a registry system to track model classes and their LoRA counterparts. A
simplified example:

.. code-block:: python

    from tfimm.architectures import lora

    @dataclass
    class ResNetConfig:
        nb_blocks = (3, 4, 6, 3)

    class ResNet(tf_keras.Model):
        cfg_class: ResNetConfig

        def __init__(self, cfg, **kwargs):
            ...

    class LoRAResNetConfig(ResNetConfig):
        lora_rank = 2

    @lora.register_lora_architecture
    class LoRAResNet(ResNet):
        cfg_class: LoRAResNetConfig

        def __init__(self, cfg, **kwargs):
            super().__init__(cfg, **kwargs)  # Create the original model
            ... # Then replace layers with LoRA versions

We make the following assumptions

* Model parameters are specified via a configuration dataclass and the configuration
  class of each model is defined via the ``cfg_class`` class attribute.
* The configuration of the LoRA model is a superset of the configuration of the base
  model.

Under these assumptions we can use the
:func:`register_lora_architecture` decorator to
associate ``LoRAResNet`` as the LoRA variant of the ``ResNet`` class.

Now, given an instance of ``ResNet``, we can use
:func:`convert_to_lora_model` to convert
it to a ``LoRAResNet`` instance *and* transfer all weights.

.. code-block:: python

    model = ResNet(cfg=ResNetConfig())
    ... # Build model or load pre-trained weights

    lora_model = lora.convert_to_lora_model(model, lora_rank=2)

The ``lora_model.trainable_weights`` property correctly returns only the LoRA trainable
weights, i.e., the low-rank updates. We additionally have the option to train the
biases as well, either only for LoRA layers or for all layers. This can be specified
by passing the values ``"none"``, ``"lora_only"`` or  ``"all"`` for the
``lora_train_bias`` parameter.

.. code-block:: python

    lora_model = convert_to_lora_model(
        model, lora_rank=2, lora_train_bias="lora_only"
    )

Sequential and functional models
--------------------------------

The current implementation focusses on models created by subclassing, which is the case
for all ``tfimm`` models. In particular, the registry system works only for subclassed
models. However, some of the functionality also works for functional models.

* LoRA layers are the basic building blocks for both subclassed as well as
  functional models.
* Transferring weights works for all models, regardless of type, provided the regular
  model and LoRA variant have the same architecture with the exception of LoRA layers.
  Use the :func:`tfimm.models.transfer_weights` function to tranfer weights to LoRA.

  .. code-block:: python

      from tfimm.architectures import lora
      from tfimm.models import transfer_weights

      # Transfer weights into the LoRA model
      transfer_weights(
          regular_model, lora_model, weights_to_ignore=lora.LORA_WEIGHT_NAMES
      )

* After training, we need to manually merge weights and then transfer them back to the
  regular model.

  .. code-block:: python

      lora.merge_weights(lora_model)
      transfer_weights(lora_model, regular_model)

* The functions
  :func:`lora_trainable_weights` and
  :func:`lora_non_trainable_weights` work for all
  models, regardless of type and return a list of weights to be used for LoRA training
  (or all other weights).

Interface
---------

All functions are available under ``tfimm.architectures.lora``.

Factory
~~~~~~~

.. autofunction:: convert_to_lora_model
.. autofunction:: convert_to_regular_model
.. autofunction:: create_model
.. autofunction:: merge_lora_weights
.. autofunction:: lora_non_trainable_weights
.. autofunction:: lora_trainable_weights

Layers
~~~~~~

.. autoclass:: LoRAConv2D
.. autoclass:: LoRADense
.. autofunction:: convert_to_lora_layer

Registry
~~~~~~~~

.. autofunction:: lora_architecture
.. autofunction:: lora_base_architecture
.. autofunction:: lora_config
.. autofunction:: register_lora_architecture
