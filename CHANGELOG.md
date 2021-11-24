# Change Log

## Unpublished

- Added Swin Transformer models (mixed precision untested)

## v0.1.1 - 2021-11-22

- Refactored code in `resnet.py`.
- Added `create_preprocessing` function to generate model-specific preprocessing.
- Added profiling results (max batch size and throughput for inference and 
  backpropagation) for K80 and V100 (`float32` and mixed precision) GPUs.
- Fixed bug with ViT models and mixed precision.

## v0.1.0 - 2021-11-17

- First release.