# TensorFlow-Image-Models

- [Introduction](#introduction)
- [Usage](#usage)
- [Models](#models)
- [Profiling](#profiling)
- [License](#license)

## Introduction

TensorfFlow-Image-Models (`tfimm`) is a collection of image models with pretrained
weights, obtained by porting architectures from 
[timm](https://github.com/rwightman/pytorch-image-models) to TensorFlow. The hope is
that the number of available architectures will grow over time. For now, it contains
vision transformers (ViT, DeiT, CaiT and Swin Transformers), MLP-Mixer models 
(MLP-Mixer, ResMLP, gMLP and ConvMixer) and ResNets.

This work would not have been possible wihout Ross Wightman's `timm` library and the
work on PyTorch/TensorFlow interoperability in HuggingFace's `transformer` repository.
I tried to make sure all source material is acknowledged. Please let me know if I have
missed something.

## Usage

### Installation 

The package can be installed via `pip`,

```shell
pip install tfimm
```

To load pretrained weights, `timm` needs to be installed. This is an optional 
dependency and can be installed via

```shell
pip install tfimm[timm]
```

### Creating models

To load pretrained models use

```python
import tfimm

model = tfimm.create_model("vit_tiny_patch16_224", pretrained="timm")
```

We can list available models with pretrained weights via

```python
import tfimm

print(tfimm.list_models(pretrained="timm"))
```

Most models are pretrained on ImageNet or ImageNet-21k. If we want to use them for other
tasks we need to change the number of classes in the classifier or remove the 
classifier altogether. We can do this by setting the `nb_classes` parameter in 
`create_model`. If `nb_classes=0`, the model will have no classification layer. If
`nb_classes` is set to a value different from the default model config, the 
classification layer will be randomly initialized, while all other weights will be
copied from the pretrained model.

The preprocessing function for each model can be created via
```python
import tensorflow as tf
import tfimm

preprocess = tfimm.create_preprocessing("vit_tiny_patch16_224", dtype="float32")
img = tf.ones((1, 224, 224, 3), dtype="uint8")
img_preprocessed = preprocess(img)
```

### Saving and loading models

All models are subclassed from `tf.keras.Model` (they are _not_ functional models).
They can still be saved and loaded using the `SavedModel` format.

```
>>> import tesnorflow as tf
>>> import tfimm
>>> model = tfimm.create_model("vit_tiny_patch16_224")
>>> type(model)
<class 'tfimm.architectures.vit.ViT'>
>>> model.save("/tmp/my_model")
>>> loaded_model = tf.keras.models.load_model("/tmp/my_model")
>>> type(loaded_model)
<class 'tfimm.architectures.vit.ViT'>
```

For this to work, the `tfimm` library needs to be imported before the model is loaded,
since during the import process, `tfimm` is registering custom models with Keras.
Otherwise, we obtain the following output

```
>>> import tensorflow as tf
>>> loaded_model = tf.keras.models.load_model("/tmp/my_model")
>>> type(loaded_model)
<class 'keras.saving.saved_model.load.Custom>ViT'>
```

## Models

The following architectures are currently available:

- CaiT (vision transformer) 
  [\[github\]](https://github.com/facebookresearch/deit/blob/main/README_cait.md)
  - Going deeper with Image Transformers 
    [\[arXiv:2103.17239\]](https://arxiv.org/abs/2103.17239)
- DeiT (vision transformer) 
  [\[github\]](https://github.com/facebookresearch/deit)
  - Training data-efficient image transformers & distillation through attention. 
    [\[arXiv:2012.12877\]](https://arxiv.org/abs/2012.12877) 
- ViT (vision transformer) 
  [\[github\]](https://github.com/google-research/vision_transformer)
  - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.
    [\[arXiv:2010.11929\]](https://arxiv.org/abs/2010.11929)
  - How to train your ViT? Data, Augmentation, and Regularization in Vision 
    Transformers. [\[arXiv:2106.10270\]](https://arxiv.org/abs/2106.10270)
  - Includes models trained with the SAM optimizer: Sharpness-Aware Minimization for 
    Efficiently Improving Generalization. 
    [\[arXiv:2010.01412\]](https://arxiv.org/abs/2010.01412)
  - Includes models from: ImageNet-21K Pretraining for the Masses
    [\[arXiv:2104.10972\]](https://arxiv.org/abs/2104.10972) 
    [\[github\]](https://github.com/Alibaba-MIIL/ImageNet21K)
- Swin Transformer 
  [\[github\]](https://github.com/microsoft/Swin-Transformer)
  - Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. 
    [\[arXiv:2103.14030\]](https://arxiv.org/abs/2103.14030)
  - Tensorflow code adapted from 
    [Swin-Transformer-TF](https://github.com/rishigami/Swin-Transformer-TF)
- MLP-Mixer and friends
  - MLP-Mixer: An all-MLP Architecture for Vision 
    [\[arXiv:2105.01601\]](https://arxiv.org/abs/2105.01601)
  - ResMLP: Feedforward networks for image classification... 
    [\[arXiv:2105.03404\]](https://arxiv.org/abs/2105.03404)
  - Pay Attention to MLPs (gMLP)
    [\[arXiv:2105.08050\]](https://arxiv.org/abs/2105.08050)
- ConvMixer 
  [\[github\]](https://github.com/tmp-iclr/convmixer)
  - Patches Are All You Need? 
    [\[ICLR 2022 submission\]](https://openreview.net/forum?id=TVHS5Y4dNvM)
- ResNet (work in progress, most available weights are from `timm`)
  - Deep Residual Learning for Image Recognition. 
    [\[arXiv:1512.03385\]](https://arxiv.org/abs/1512.03385)

## Profiling

To understand how big each of the models is, I have done some profiling to measure
- maximum batch size that fits in GPU memory and
- throughput in images/second
for both inference and backpropagation on K80 and V100 GPUs. For V100, measurements 
were done for both `float32` and mixed precision.

The results can be found in the `results/profiling_{k80, v100}.csv` files.

For backpropagation, we use as loss the mean of model outputs

```python
def backprop():
    with tf.GradientTape() as tape:
        output = model(x, training=True)
        loss = tf.reduce_mean(output)
        grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

## License

This repository is released under the Apache 2.0 license as found in the 
[LICENSE](LICENSE) file.