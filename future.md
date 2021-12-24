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

Evaluation

- [ ] (must) Evaluate model on ImageNet
- [ ] (must) Run evaluation on GPU, e.g., Google Colab
- [ ] (optional) Evaluate model on ImageNet-ReaL
- [ ] (optional) Evaluate model on ImageNet-v2
- [ ] (optional) Evaluate model on ImageNet-A
- [ ] (optional) Evaluate model on ImageNet-R
- [ ] (optional) Evaluate model on ImageNet-C
- [ ] (must) Profiling single image inference time on CPU
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

To Do

- [ ] Add `interpolate_input` and `fixed_input_size` to all transformer models
- [ ] Add no weight_decay rule for certain layers
- [ ] Fix weight initialisation

Future

- [ ] (optional) Add pretrained [DINO models](https://github.com/facebookresearch/dino)
- [ ] (optional) Add [T2T-ViT models](https://github.com/yitu-opensource/T2T-ViT)
- [ ] (optional) Check compatibility with [tf-explain](https://github.com/sicara/tf-explain)

- [ ] PoolFormer github.com/sail-sg/poolformer
- [ ] Pretrained models: https://github.com/microsoft/simmim
- [ ] Pretrained models: https://github.com/bytedance/ibot
- [ ] Pretrained models: https://github.com/facebookresearch/moco-v3

- [ ] RepVGG https://github.com/DingXiaoH/RepVGG
- [ ] FaceX model zoo https://github.com/JDAI-CV/FaceX-Zoo

