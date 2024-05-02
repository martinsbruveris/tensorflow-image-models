# TensorFlow Image Models

![Test Status](https://github.com/martinsbruveris/tensorflow-image-models/actions/workflows/tests.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/tfimm/badge/?version=latest)](https://tfimm.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Slack](https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=slack&logoColor=white)](https://join.slack.com/t/tfimm/shared_invite/zt-13dnaf3qo-5JJaCBFIQhugeBXBT3NK8A)

- [Introduction](#introduction)
- [Usage](#usage)
- [Models](#models)
- [Profiling](#profiling)
- [License](#license)
- [Contact](#contact)

## Introduction

TensorFlow Image Models (`tfimm`) is a collection of image models with pretrained
weights, obtained by porting architectures from 
[timm](https://github.com/rwightman/pytorch-image-models) to TensorFlow. The hope is
that the number of available architectures will grow over time. For now, it contains
vision transformers (ViT, DeiT, CaiT, PVT and Swin Transformers), MLP-Mixer models 
(MLP-Mixer, ResMLP, gMLP, PoolFormer and ConvMixer), various ResNet flavours (ResNet,
ResNeXt, ECA-ResNet, SE-ResNet), the EfficientNet family (including AdvProp, 
NoisyStudent, Edge-TPU, V2 and Lite versions), MobileNet-V2, VGG, as well as the recent 
ConvNeXt. `tfimm` has now expanded beyond classification and also includes Segment 
Anything.

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

To load pretrained weights, `timm` needs to be installed separately.

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

All models are subclassed from `tf_keras.Model` (they are _not_ functional models).
They can still be saved and loaded using the `SavedModel` format.

```
>>> import tesnorflow as tf
>>> import tfimm
>>> model = tfimm.create_model("vit_tiny_patch16_224")
>>> type(model)
<class 'tfimm.architectures.vit.ViT'>
>>> model.save("/tmp/my_model")
>>> loaded_model = tf_keras.models.load_model("/tmp/my_model")
>>> type(loaded_model)
<class 'tfimm.architectures.vit.ViT'>
```

For this to work, the `tfimm` library needs to be imported before the model is loaded,
since during the import process, `tfimm` is registering custom models with Keras.
Otherwise, we obtain the following output

```
>>> import tensorflow as tf
>>> loaded_model = tf_keras.models.load_model("/tmp/my_model")
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
- EfficientNet family
  - EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
    [\[arXiv:1905.11946\]](https://arxiv.org/abs/1905.11946)
  - Adversarial Examples Improve Image Recognition
    [\[arXiv:1911.09665\]](https://arxiv.org/abs/1911.09665)
  - Self-training with Noisy Student improves ImageNet classification
    [\[arXiv:1911.04252\]](https://arxiv.org/abs/1911.04252)
  - EfficientNet-EdgeTPU
    [\[Blog\]](https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html)
  - EfficientNet-Lite
    [\[Blog\]](https://blog.tensorflow.org/2020/03/higher-accuracy-on-vision-models-with-efficientnet-lite.html)
  - EfficientNetV2: Smaller Models and Faster Training
    [\[arXiv:2104.00298\]](https://arxiv.org/abs/2104.00298)
- MobileNet-V2
  - MobileNetV2: Inverted Residuals and Linear Bottlenecks
    [\[arXiv:1801.04381\]](https://arxiv.org/abs/1801.04381)
- Pyramid Vision Transformer 
  [\[github\]](https://github.com/whai362/PVT)
  - Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without
    Convolutions. [\[arXiv:2102.12122\]](https://arxiv.org/abs/2102.12122)
  - PVTv2: Improved Baselines with Pyramid Vision Transformer 
    [\[arXiv:2106.13797\]](https://arxiv.org/abs/2106.13797)
- ConvNeXt
  [\[github\]](https://github.com/facebookresearch/ConvNeXt)
  - A ConvNet for the 2020s. [\[arXiv:2201.03545\]](https://arxiv.org/abs/2201.03545)
- PoolFormer
  [\[github\]](https://github.com/sail-sg/poolformer)
  - PoolFormer: MetaFormer is Actually What You Need for Vision.
    [\[arXiv:2111.11418\]](https://arxiv.org/abs/2111.11418)
- Pooling-based Vision Transformers (PiT)
  - Rethinking Spatial Dimensions of Vision Transformers.
    [\[arXiv:2103.16302\]](https://arxiv.org/abs/2103.16302)
- ResNet, ResNeXt, ECA-ResNet, SE-ResNet and friends
  - Deep Residual Learning for Image Recognition. 
    [\[arXiv:1512.03385\]](https://arxiv.org/abs/1512.03385)
  - Exploring the Limits of Weakly Supervised Pretraining. 
    [\[arXiv:1805.00932\]](https://arxiv.org/abs/1805.00932)
  - Billion-scale Semi-Supervised Learning for Image Classification. 
    [\[arXiv:1905.00546\]](https://arxiv.org/abs/1905.00546)
  - ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks. 
    [\[arXiv1910.03151\]](https://arxiv.org/abs/1910.03151)
  - Revisiting ResNets. [\[arXiv:2103.07579\]](https://arxiv.org/abs/2103.07579)
  - Making Convolutional Networks Shift-Invariant Again. (anti-aliasing layer)
    [\[arXiv:1904.11486\]](https://arxiv.org/abs/1904.11486)
  - Squeeze-and-Excitation Networks. 
    [\[arXiv:1709.01507\]](https://arxiv.org/abs/1709.01507)
  - Big Transfer (BiT): General Visual Representation Learning
    [\[arXiv:1912.11370\]](https://arxiv.org/abs/1912.11370)
  - Knowledge distillation: A good teacher is patient and consistent
    [\[arXiv:2106:05237\]](https://arxiv.org/abs/2106.05237)
- Segment Anything Model (SAM) 
  [\[github\]](https://github.com/facebookresearch/segment-anything)
    - Segment Anything [\[arXiv:2304.02643\]](https://arxiv.org/abs/2304.02643)

### Loading pytorch models from HF hub

It is possible to load pre-trained model weights from the HF hub. See the 
[huggingface-model-weights](notebooks/huggingface-model-weights.ipynb) notebook for 
details. For this to work, it is important that the weight names and shapes on 
HF hub are compatible with one of the `tfimm` model configurations.

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

## Contact

All things related to `tfimm` can be discussed via 
[Slack](https://join.slack.com/t/tfimm/shared_invite/zt-13dnaf3qo-5JJaCBFIQhugeBXBT3NK8A).