# Change Log

## v0.2.13 - 2023-05-03

- Adding VGG models.
- Fixing bugs in SAM model inference.

## v0.2.12 - 2023-04-17

- Fixing small bugs in SAM.

## v0.2.11 - 2023-04-17

- Added Segment Aything models.

## v0.2.10 - 2023-02-27

- Added post-pooling features for ConvNeXt in feature dictionary.

## v0.2.9 - 2022-10-28

- Exposed attention map in ViT models.

## v0.2.8 - 2022-09-05

- `tfimm` now supports python 3.10.

## v0.2.7 - 2022-06-14

- Added EfficinentNet and MobileNet-V2 models.

## v0.2.6 - 2022-05-13

- Added tiny and small ConvNeXt models.

## v0.2.5 - 2022-02-21

- Preprocessing works for abritrary number of `in_channels`.
- Removed temporary version restriction on `libclang``

## v0.2.4 - 2022-01-31

- Adding PiT models
- Simplified `pretrained` parameter in `create_function`.
- Added model-specific cache
- Added adaptability of `in_channels`

## v0.2.3 - 2022-01-20

- Added ConvNeXt models
- Added PoolFormer models
- Improved LR schedulers for training framework

## v0.2.2. - 2022-01-17

- Improvements to the training framework.

## v0.2.1 - 2022-01-07

- Small changes to the training framework.

## v0.2.0 - 2022-01-02

- Added hybrid Vision Transformers (`vit_hybrid`).
- Added `resnetv2` module, which inlcudes Big Transfer (BiT) resnets.
- Added Pyramid Vision Transformer models
- Added first version of training framework (`tfimm/train`). Still work in progress. 
  Possibly buggy.

## v0.1.5 - 2021-12-12

- Added option for models to return intermediate features via `return_features` 
  parameter
- Added `DropPath` regularization to `vit` module (stochastic depth)
- Added ability to load saved models from a local cache

## v0.1.4 - 2021-12-08

- Fixed bug with dropout in Classifier layer

## v0.1.3 - 2021-12-07

- Added CaiT models
- Added MLP-Mixer, ResMLP and gMLP models
- Added ResNet models
- Fixed bug with Swin Transformer and mixed precision

## v0.1.2 - 2021-11-25

- Reduced TF version requirement from 2.5 to 2.4.
- Added ConvMixer models
- Added Swin Transformer models

## v0.1.1 - 2021-11-22

- Refactored code in `resnet.py`.
- Added `create_preprocessing` function to generate model-specific preprocessing.
- Added profiling results (max batch size and throughput for inference and 
  backpropagation) for K80 and V100 (`float32` and mixed precision) GPUs.
- Fixed bug with ViT models and mixed precision.

## v0.1.0 - 2021-11-17

- First release.