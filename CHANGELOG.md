# Change Log

## Unpublished

- Added CaiT models
- Added MLP-Mixer, ResMLP and gMLP models

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