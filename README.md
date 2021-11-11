# TensorFlow-Image-Models

This is a port of 
[PyTorch Image Models](https://github.com/rwightman/pytorch-image-models) to Tensorflow.

Work in progress...

## Datasets and evaluation

### ImageNet validation split

To be able to use the ImageNet validation split, we need to manually download and
copy `ILSVRC2012_img_val.tar` to `~/tensorflow_datasets/downloads/manual/` and then run
```python
import tensorflow_datasets as tfds

tfds.builder("imagenet2012_real").download_and_prepare()
```
Then we have access to both original and reassessed (ReaL) labels under the keys
`original_label` and `real_label`. We use `imagenet2012_real`, because to process the 
`imagenet2012` dataset via `tensorflow-datasets` requires us to download validation
_and_ train splits, even though we don't intend to use train (for now).

Having downloaded `ILSVRC2012_img_val.tar`, we can also access `imagenet2012_corrupted`
via `tensorflow-datasets`.

Other ImageNet variants, such as
- ImageNet-A (`imagenet_a`)
- ImageNet-R (`imagenet_r`) and
- ImageNet-v2 (`imagenet_v2`, defaults to matched-frequency)
can be downloaded automatically via `tensorflow-datasets`.

## To do before 0.1.0

- [x] Create git repo and upload to GitHub
- [ ] Convert all other resnet models from timm
- [ ] Convert vision transformer models from timm
- [ ] Ability to alter number of classes for finetuning
- [x] pyproj.toml + python environment
- [ ] Unit tests
- [ ] Refactor resnet.py code
- [ ] Refactor utils/timm.py file
- [ ] Remove num_batches_tracked variables warning from conversion
- [x] Black formatting
- [x] Fix Flake8 warnings
- [x] Fix isort warnings
- [x] Makefile with style checks
- [ ] Complete README file
- [ ] Run unit tests on GitHub CI
- [ ] Add licence file + licence headers to python files
- [ ] Publish package on PyPi

### ResNet model tracker

| Name | Total | Converted | Documented |
|---|:---:|:---:|---|
|ecaresnet | 6 + 2 pruned | 
|ig_resnext | 4 |
|resnet | 14 | ✅ |
|resnet_blur | 1 |
|resnetrs | 7 |
|resnext | 3 |
|seresne{x}t | 5 |
|ssl_resne{x}t | 6 |
|swsl_resne{x}t | 6 |
|tv_resne{x}t | 5 |
|wide_resnet | 2 |



## Optional

- [ ] CPU profiling (#params, FLOPS, single-image inference time, memory consumption)
- [ ] GPU profiling (max batch size, batch inference time)
      See tf.errors.ResourceExhaustedError for OOM errors
- [ ] Run validation on imagenet to measure accuracy of converted models
- [ ] Ability to alter number of input channels (check first_conv parameter)
- [ ] Host converted models on Google Drive + download functionality
- [ ] Apply weight decay (either during build process or afterwards)