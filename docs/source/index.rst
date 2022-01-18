TensorFlow Image Models (tfimm)
=================================

TensorFlow Image Models (``tfimm``) is a collection of image models with pretrained
weights, obtained by porting architectures from timm_ to TensorFlow. It contains
vision transformers (ViT, DeiT, CaiT, PVT and Swin Transformers), MLP-Mixer models
(MLP-Mixer, ResMLP, gMLP and ConvMixer) and various ResNet flavours (ResNet, ResNeXt,
ECA-ResNet, SE-ResNet).

.. _timm: https://github.com/rwightman/pytorch-image-models

Contents
--------

:doc:`content/transformers`
    Configuration and parameters specific to transformers
:doc:`content/trainer`
    Trainer class

.. Hidden TOCs

.. toctree::
   :maxdepth: 2
   :caption: Models

   content/transformers
   content/convnext

.. toctree::
   :maxdepth: 2
   :caption: Training

   content/trainer