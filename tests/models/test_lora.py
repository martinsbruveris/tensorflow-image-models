from typing import List

import numpy as np
import pytest
import tensorflow as tf

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


def _count(var_list: List[tf.Variable]) -> int:
    return np.sum([np.prod(v.get_shape()) for v in var_list]).item()


@pytest.mark.parametrize("use_bias", [True, False])
def test_lora_trainable_weights(use_bias):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(units=2, use_bias=use_bias, name="fc1"),
            lora.LoRADense(units=3, use_bias=use_bias, lora_rank=3, name="fc2"),
        ]
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
