from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pytest
import tensorflow as tf
import tf_keras

import tfimm
from tfimm.architectures import ConvNeXt, ConvNeXtConfig, lora
from tfimm.models import create_model, register_model


# We test both rank 2 and rank 3 inputs, since they use different paths in call().
@pytest.mark.parametrize("input_shape", [(3, 4), (3, 4, 5)])
def test_lora_dense(input_shape):
    """Creating the layer and running inference in merged and non-merged modes."""
    layer = lora.LoRADense(units=2, use_bias=True, activation="swish")
    layer.build(input_shape=input_shape)
    assert not layer.merged

    # Set non-trivial weights, since bias and kernel_b are zero-initialised.
    layer.bias = tf.random.uniform(layer.bias.shape)
    layer.kernel_lora_a = tf.random.uniform(layer.kernel_lora_a.shape)
    layer.kernel_lora_b = tf.random.uniform(layer.kernel_lora_b.shape)

    x = tf.random.uniform(input_shape)
    res_1 = layer(x)

    assert not layer.merged
    layer.merge_weights()
    assert layer.merged
    res_2 = layer(x)

    tf.debugging.assert_near(res_1, res_2)


@pytest.mark.parametrize("use_bias", [True, False])
def test_lora_conv2d(use_bias):
    """Creating the layer and running inference in merged and non-merged modes."""
    layer = lora.LoRAConv2D(
        filters=5, kernel_size=(2, 3), use_bias=use_bias, activation="swish"
    )
    layer.build(input_shape=(1, 10, 10, 4))
    assert not layer.merged

    # Set non-trivial weights, since bias and kernel_b are zero-initialised.
    if use_bias:
        layer.bias = tf.random.uniform(layer.bias.shape)
    layer.kernel_lora_a = tf.random.uniform(layer.kernel_lora_a.shape)
    layer.kernel_lora_b = tf.random.uniform(layer.kernel_lora_b.shape)

    x = tf.random.uniform((1, 10, 10, 4))
    res_1 = layer(x)

    assert not layer.merged
    layer.merge_weights()
    assert layer.merged
    res_2 = layer(x)

    tf.debugging.assert_near(res_1, res_2)


def test_lora_registry():
    # Register class using subclassing
    class A:
        ...

    @lora.register_lora_architecture
    class B(A):
        cfg_class = None

    assert lora.lora_architecture(A) is B
    assert lora.lora_config(A) is None
    assert lora.lora_base_architecture(B) is A

    # Register class with explicit base class
    class C:
        ...

    @lora.register_lora_architecture(base_cls=C)
    class D:
        cfg_class = None

    assert lora.lora_architecture(C) is D
    assert lora.lora_config(C) is None
    assert lora.lora_base_architecture(D) is C


@register_model
def convnext_test_model():
    cfg = ConvNeXtConfig(
        name="convnext_test_model",
        nb_classes=12,
        input_size=(32, 32),
        embed_dim=(3, 4, 5, 6),
        nb_blocks=(1, 1, 1, 1),
    )
    return ConvNeXt, cfg


def test_convert_to_lora_model_and_back():
    model = create_model("convnext_test_model")
    assert type(model) is ConvNeXt
    lora_model = lora.convert_to_lora_model(model, lora_rank=1)
    assert type(lora_model) is lora.LoRAConvNeXt
    regular_model = lora.convert_to_regular_model(lora_model)
    assert type(regular_model) is ConvNeXt

    img = tf.random.uniform(model.dummy_inputs.shape)
    res_1 = model(img, training=False)
    res_2 = lora_model(img, training=False)
    res_3 = regular_model(img, training=False)

    tf.debugging.assert_near(res_1, res_2)
    tf.debugging.assert_near(res_1, res_3)


def test_create_model():
    # We test that non-LoRA parameters are passed correctly to the model
    model = lora.create_model("convnext_test_model", lora_rank=4, nb_classes=13)
    assert model.cfg.nb_classes == 13
    res = model(model.dummy_inputs)
    assert res.shape[-1] == 13

    # We test that we can also pass non-LoRA parameters to convert_to_lora_model
    base_model = create_model("convnext_test_model")
    model = lora.convert_to_lora_model(base_model, lora_rank=4, nb_classes=13)
    assert model.cfg.nb_classes == 13
    res = model(model.dummy_inputs)
    assert res.shape[-1] == 13


def _count(var_list: List[tf.Variable]) -> int:
    return np.sum([np.prod(v.get_shape()) for v in var_list]).item()


@pytest.mark.parametrize("use_bias", [True, False])
def test_lora_trainable_weights(use_bias):
    model = tf_keras.Sequential(
        [
            tf_keras.layers.Dense(units=2, use_bias=use_bias, name="fc1"),
            lora.LoRADense(units=3, use_bias=use_bias, lora_rank=3, name="fc2"),
        ],
    )
    model.build(input_shape=(4, 5))
    # Number of parameters
    # fc1: kernel 10=5*2, bias 2
    # fc2: kernel 6=2*3, lora_a 9=3*3, lora_b 6=2*3, bias 3
    nb_fc = 2 if use_bias else 0
    nb_lora = 3 if use_bias else 0

    # For the model everything is trainable
    assert _count(model.trainable_variables) == 10 + nb_fc + 6 + 9 + 6 + nb_lora
    assert _count(model.non_trainable_variables) == 0

    # Counting LoRA-only parameters
    assert _count(lora.lora_trainable_weights(model, "none")) == 9 + 6
    assert (
        _count(lora.lora_non_trainable_weights(model, "none"))
        == 10 + nb_fc + 6 + nb_lora
    )
    assert _count(lora.lora_trainable_weights(model, "lora_only")) == 9 + 6 + nb_lora
    assert _count(lora.lora_non_trainable_weights(model, "lora_only")) == 10 + nb_fc + 6
    assert _count(lora.lora_trainable_weights(model, "all")) == nb_fc + 9 + 6 + nb_lora
    assert _count(lora.lora_non_trainable_weights(model, "all")) == 10 + 6


def test_lora_trainable_weights_trainable_layers():
    # Setting `trainable_layers` does not play nicely with the layer naming in
    # sequential and functional models, so we use a subclassed model to test it.
    model = tfimm.create_model("convnext_test_model")
    trainable_weights = lora.lora_trainable_weights(
        model, train_bias="none", trainable_layers=[model.cfg.classifier]
    )
    trainable_weight_names = set(w.name for w in trainable_weights)
    res = {"convnext_test_model/head/fc/kernel:0", "convnext_test_model/head/fc/bias:0"}
    assert trainable_weight_names == res


def test_model_self_lora_version():
    # We want to enable a model to be already LoRA-aware, so we can use the same class
    # as both the regular and LoRA variants.
    @dataclass
    class Config:
        units: int
        lora_rank: Optional[int] = None

    class MyModel(tf_keras.Model):
        cfg_class = Config

        def __init__(self, cfg: Config, **kwargs):
            super().__init__(**kwargs)
            self.cfg = cfg

            if cfg.lora_rank is None:
                self.fc = tf_keras.layers.Dense(units=cfg.units, name="fc")
            else:
                self.fc = lora.LoRADense(
                    units=cfg.units, lora_rank=cfg.lora_rank, name="fc"
                )

        @property
        def dummy_inputs(self):
            return tf.zeros((1, 3))

        def call(self, x):
            return self.fc(x)

    lora.register_lora_architecture(MyModel, base_cls=MyModel)

    base_model = MyModel(cfg=Config(units=3))
    base_model(base_model.dummy_inputs)
    lora_model = lora.convert_to_lora_model(base_model, lora_rank=2)
    assert lora_model.cfg == Config(units=3, lora_rank=2)
    merged_model = lora.convert_to_regular_model(lora_model)
    assert merged_model.cfg == Config(units=3, lora_rank=None)
