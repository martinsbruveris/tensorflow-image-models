"""
We provide an implementation and pretrained weights for the PoolFormer models.

Paper: PoolFormer: MetaFormer is Actually What You Need for Vision.
`[arXiv:2111.11418] <https://arxiv.org/abs/2111.11418>`_.

Original pytorch code and weights from
`poolformer <https://github.com/sail-sg/poolformer>`_ repository.

The following models are available.

* ``poolformer_s12``
* ``poolformer_s24``
* ``poolformer_s36``
* ``poolformer_m36``
* ``poolformer_m48``
"""
# Copyright 2021 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright 2022 Marting Bruveris
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from tfimm.layers import ConvMLP, DropPath, PatchEmbeddings, norm_layer_factory
from tfimm.models import ModelConfig, keras_serializable, register_model
from tfimm.utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# Model registry will add each entrypoint fn to this
__all__ = ["PoolFormer", "PoolFormerConfig"]


@dataclass
class PoolFormerConfig(ModelConfig):
    """
    Configuration class for PoolFormer models.

    Parameters:
        name: Name of the model.
        url: URL for pretrained weights.
        nb_classes: Number of classes for classification head.
        in_channels: Number of input image channels.
        input_size: Input image size (height, width)

        embed_dim: Feature dimensions at each stage.
        nb_blocks: Number of blocks at each stage.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim

        drop_rate: Dropout rate.
        drop_path_rate: Dropout rate for stochastic depth.

        norm_layer: Normalization layer. See :func:`~norm_layer_factory` for possible
            values.
        act_layer: Activation function. See :func:`~act_layer_factory` for possible
            values.
        init_scale: Inital value for layer scale weights.

        crop_pct: Crop percentage for ImageNet evaluation.
        interpolation: Interpolation method for ImageNet evaluation.
        mean: Defines preprocessing function. If ``x`` is an image with pixel values
            in (0, 1), the preprocessing function is ``(x - mean) / std``.
        std: Defines preprpocessing function.

        first_conv: Name of first convolutional layer. Used by
            :func:`~tfimm.create_model` to adapt the number in input channels when
            loading pretrained weights.
        classifier: Name of classifier layer. Used by :func:`~tfimm.create_model` to
            adapt the classifier when loading pretrained weights.
    """

    nb_classes: int = 1000
    in_channels: int = 3
    input_size: Tuple[int, int] = (224, 224)
    embed_dim: Tuple = (64, 128, 320, 512)
    nb_blocks: Tuple = (2, 2, 6, 2)
    mlp_ratio: Tuple = (4.0, 4.0, 4.0, 4.0)
    # Regularization
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    # Other parameters
    norm_layer: str = "group_norm_1grp"
    act_layer: str = "gelu"
    init_scale: float = 1e-5
    # Parameters for inference
    crop_pct: float = 0.95
    interpolation: str = "bicubic"
    # Preprocessing
    mean: Tuple[float, float, float] = IMAGENET_DEFAULT_MEAN
    std: Tuple[float, float, float] = IMAGENET_DEFAULT_STD
    # Weight transfer
    first_conv: str = "patch_embed/proj"
    classifier: str = "head"


def _weight_initializers(seed=42):
    """Function returns initilializers to be used in the model."""
    kernel_initializer = tf.keras.initializers.TruncatedNormal(
        mean=0.0, stddev=0.02, seed=seed
    )
    bias_initializer = tf.keras.initializers.Zeros()
    return kernel_initializer, bias_initializer


class PoolFormerBlock(tf.keras.layers.Layer):
    """
    PoolFormer block.

    Args:
        embed_dim: Number of feature dimensions.
        mlp_ratio: Ratio of MLP hidden dim to embedding dim.
        drop_rate: Dropout rate.
        drop_path_rate: Dropout rate for stochastic depth/
        norm_layer: Normalization layer.
        act_layer: Activation function.
        init_scale: Initial value for layer scale weights.
    """

    def __init__(
        self,
        embed_dim: int,
        mlp_ratio: float,
        drop_rate: float,
        drop_path_rate: float,
        norm_layer: str,
        act_layer: str,
        init_scale: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        self.init_scale = init_scale

        norm_layer = norm_layer_factory(norm_layer)
        kernel_initializer, bias_initializer = _weight_initializers()

        self.norm1 = norm_layer(name="norm1")
        self.pool = tf.keras.layers.AveragePooling2D(
            pool_size=3, strides=1, padding="same"
        )
        self.norm2 = norm_layer(name="norm2")
        self.mlp = ConvMLP(
            hidden_dim=int(embed_dim * mlp_ratio),
            embed_dim=embed_dim,
            drop_rate=drop_rate,
            act_layer=act_layer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name="mlp",
        )
        self.layer_scale_1 = None
        self.layer_scale_2 = None
        self.drop_path = DropPath(drop_prob=drop_path_rate)

    def build(self, input_shape):
        self.layer_scale_1 = self.add_weight(
            shape=(self.embed_dim,),
            initializer=tf.keras.initializers.Constant(value=self.init_scale),
            trainable=True,
            name="layer_scale_1",
        )
        self.layer_scale_2 = self.add_weight(
            shape=(self.embed_dim,),
            initializer=tf.keras.initializers.Constant(value=self.init_scale),
            trainable=True,
            name="layer_scale_2",
        )

    def call(self, x, training=False):
        shortcut = x
        x = self.norm1(x, training=training)
        x = self.pool(x) - x  # Token mixer layer
        x = x * self.layer_scale_1
        x = self.drop_path(x, training=training)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x, training=training)
        x = self.mlp(x, training=training)
        x = x * self.layer_scale_2
        x = self.drop_path(x, training=training)
        x = x + shortcut
        return x


@keras_serializable
class PoolFormer(tf.keras.Model):
    """
    Class implementing a PoolFormer network.

    Paper: PoolFormer: MetaFormer is Actually What You Need for Vision.
    `[arXiv:2111.11418] <https://arxiv.org/abs/2111.11418>`_.

    Parameters:
        cfg: Configuration class for the model.
        **kwargs: Arguments are passed to ``tf.keras.Model``.
    """

    cfg_class = PoolFormerConfig

    def __init__(self, cfg: PoolFormerConfig, **kwargs):
        kwargs["name"] = kwargs.get("name", cfg.name)
        super().__init__(**kwargs)
        self.cfg = cfg
        norm_layer = norm_layer_factory(cfg.norm_layer)
        kernel_initializer, bias_initializer = _weight_initializers()

        self.patch_embed = PatchEmbeddings(
            patch_size=7,
            embed_dim=cfg.embed_dim[0],
            stride=4,
            padding=2,
            flatten=False,
            norm_layer="",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name="patch_embed",
        )

        # Stochastic depth
        dpr = np.linspace(0.0, cfg.drop_path_rate, sum(cfg.nb_blocks))
        dpr = np.split(dpr, np.cumsum(cfg.nb_blocks))

        self.blocks = OrderedDict()
        for j in range(len(cfg.nb_blocks)):
            for k in range(cfg.nb_blocks[j]):
                self.blocks[f"stage_{j}/block_{k}"] = PoolFormerBlock(
                    embed_dim=cfg.embed_dim[j],
                    mlp_ratio=cfg.mlp_ratio[j],
                    drop_rate=cfg.drop_rate,
                    drop_path_rate=dpr[j][k],
                    norm_layer=cfg.norm_layer,
                    act_layer=cfg.act_layer,
                    init_scale=cfg.init_scale,
                    name=f"network/{2*j}/{k}",
                )
            if j < len(cfg.nb_blocks) - 1:
                self.blocks[f"stage_{j}/downsample"] = PatchEmbeddings(
                    embed_dim=cfg.embed_dim[j + 1],
                    patch_size=3,
                    stride=2,
                    padding=1,
                    flatten=False,
                    norm_layer="",
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    name=f"network/{2*j+1}",
                )

        # Classifier head
        self.norm = norm_layer(name="norm")
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.head = (
            tf.keras.layers.Dense(units=cfg.nb_classes, name="head")
            if cfg.nb_classes > 0
            else tf.keras.layers.Activation("linear")  # Identity layer
        )

    @property
    def dummy_inputs(self) -> tf.Tensor:
        """Returns a tensor of the correct shape for inference."""
        return tf.zeros((1, *self.cfg.input_size, self.cfg.in_channels))

    @property
    def feature_names(self) -> List[str]:
        """
        Names of features, returned when calling ``call`` with ``return_features=True``.
        """
        _, features = self(self.dummy_inputs, return_features=True)
        return list(features.keys())

    def forward_features(
        self, x, training: bool = False, return_features: bool = False
    ):
        """
        Forward pass through model, excluding the classifier layer. This function is
        useful if the model is used as input for downstream tasks such as object
        detection.

        Arguments:
             x: Input to model
             training: Training or inference phase?
             return_features: If ``True``, we return not only the model output, but a
                dictionary with intermediate features.

        Returns:
            If ``return_features=True``, we return a tuple ``(y, features)``, where
            ``y`` is the model output and ``features`` is a dictionary with
            intermediate features.

            If ``return_features=False``, we return only ``y``.
        """
        features = OrderedDict()
        x = self.patch_embed(x, training=training)
        features["patch_embedding"] = x

        for key, block in self.blocks.items():
            x = block(x, training=training)
            features[key] = x

        x = self.norm(x, training=training)
        features["features_all"] = x
        x = self.pool(x)
        features["features"] = x

        return (x, features) if return_features else x

    def call(self, x, training: bool = False, return_features: bool = False):
        """
        Forward pass through the full model.

        Arguments:
             x: Input to model
             training: Training or inference phase?
             return_features: If ``True``, we return not only the model output, but a
                dictionary with intermediate features.

        Returns:
            If ``return_features=True``, we return a tuple ``(y, features)``, where
            ``y`` is the model output and ``features`` is a dictionary with
            intermediate features.

            If ``return_features=False``, we return only ``y``.
        """
        features = OrderedDict()
        x = self.forward_features(x, training, return_features)
        if return_features:
            x, features = x

        x = self.head(x)
        features["logits"] = x
        return (x, features) if return_features else x


@register_model
def poolformer_s12():
    """PoolFormer-S12 model, Params: 12M."""
    cfg = PoolFormerConfig(
        name="poolformer_s12",
        url="[pytorch]https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s12.pth.tar",  # noqa: E501
        embed_dim=(64, 128, 320, 512),
        nb_blocks=(2, 2, 6, 2),
        crop_pct=0.9,
    )
    return PoolFormer, cfg


@register_model
def poolformer_s24():
    """PoolFormer-S24 model, Params: 21M."""
    cfg = PoolFormerConfig(
        name="poolformer_s24",
        url="[pytorch]https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s24.pth.tar",  # noqa: E501
        embed_dim=(64, 128, 320, 512),
        nb_blocks=(4, 4, 12, 4),
        crop_pct=0.9,
    )
    return PoolFormer, cfg


@register_model
def poolformer_s36():
    """PoolFormer-S36 model, Params: 31M."""
    cfg = PoolFormerConfig(
        name="poolformer_s36",
        url="[pytorch]https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s36.pth.tar",  # noqa: E501
        embed_dim=(64, 128, 320, 512),
        nb_blocks=(6, 6, 18, 6),
        init_scale=1e-6,
        crop_pct=0.9,
    )
    return PoolFormer, cfg


@register_model
def poolformer_m36():
    """PoolFormer-M36 model, Params: 56M."""
    cfg = PoolFormerConfig(
        name="poolformer_m36",
        url="[pytorch]https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_m36.pth.tar",  # noqa: E501
        embed_dim=(96, 192, 384, 768),
        nb_blocks=(6, 6, 18, 6),
        init_scale=1e-6,
        crop_pct=0.95,
    )
    return PoolFormer, cfg


@register_model
def poolformer_m48():
    """PoolFormer-M48 model, Params: 73M."""
    cfg = PoolFormerConfig(
        name="poolformer_m48",
        url="[pytorch]https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_m48.pth.tar",  # noqa: E501
        embed_dim=(96, 192, 384, 768),
        nb_blocks=(8, 8, 24, 8),
        init_scale=1e-6,
        crop_pct=0.95,
    )
    return PoolFormer, cfg
