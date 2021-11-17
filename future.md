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

Development setup

- [x] (must) Create git repo and upload to GitHub
- [x] (must) pyproj.toml + python environment
- [x] (must) Black formatting
- [x] (must) Fix Flake8 warnings
- [x] (must) Fix isort warnings
- [x] (must) Makefile with style checks

Testing
 
- [x] (must) Add first tests
- [x] (must) Test creation of each model
- [x] (optional) Run unit tests on GitHub CI

ResNet models

| Name           | Total | Converted | Documented |
|----------------|:-----:|:---------:|------------|
| ecaresnet      |   6   |           |            |
| ig_resnext     |   4   |           |            |
| resnet         |  14   |     ✅    |            |
| resnet_blur    |   1   |           |            |
| resnetrs       |   7   |           |            |
| resnext        |   3   |           |            |
| seresne{x}t    |   5   |           |            |
| ssl_resne{x}t  |   6   |           |            |
| swsl_resne{x}t |   6   |           |            |
| tv_resne{x}t   |   5   |           |            |
| wide_resnet    |   2   |           |            |

Vision Transformer models

| Name           | Total | Converted | Documented |
|----------------|:-----:|:---------:|------------|
| vit            |  14   |    ✅     |            |
| vit_in21k      |   8   |    ✅     |            |
| vit_sam        |   2   |    ✅     |            |
| deit           |   8   |    ✅     |            |
| vit_miil       |   2   |    ✅     |            |

Codebase

- [ ] (optional) Refactor resnet.py code
- [ ] (optional) Refactor utils/timm.py file
- [ ] (optional) Remove num_batches_tracked variables warning from conversion

Evaluation

- [ ] (must) Evaluate model on ImageNet
- [ ] (must) Run evaluation on GPU, e.g., Google Colab
- [ ] (optional) Evaluate model on ImageNet-ReaL
- [ ] (optional) Evaluate model on ImageNet-v2
- [ ] (optional) Evaluate model on ImageNet-A
- [ ] (optional) Evaluate model on ImageNet-R
- [ ] (optional) Evaluate model on ImageNet-C
- [ ] (must) Profiling single image inference time on CPU
- [ ] (must) Profiling max batch size and batch inference time on GPU
      See tf.errors.ResourceExhaustedError for OOM errors
- [ ] (optional) Profiling #params, FLOPS
- [ ] (optional) Profiling memory consumption on CPU

Finetuning

- [x] (must) Load weights from file to model with `nb_classes`
- [ ] (must) Set weight decay in loaded model
- [ ] (must) Fine-tune pretrained model on CIFAR-100
- [ ] (must) Evaluate model on CIFAR-100
- [ ] (optional) Load weights from file to model with `in_chans != 3`
- [ ] (optional) Fine-tune model on MNIST (and variants)
- [ ] (optional) Evaluate model on MNIST (and variants)

Release

- [ ] (must) Complete README file
- [ ] (must) Add licence file + licence headers to python files
- [ ] (must) Publish package on PyPi
- [ ] (optional) Host converted models on Google Drive + download functionality

Future

- [ ] (optional) Add pretrained [DINO models](https://github.com/facebookresearch/dino)
- [ ] (optional) Add [T2T-ViT models](https://github.com/yitu-opensource/T2T-ViT)
- [ ] (optional) Check compatibility with [tf-explain](https://github.com/sicara/tf-explain)
