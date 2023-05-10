import pytest
import numpy as np
import tensorflow as tf

from tfimm.architectures import lora, ConvNeXt, ConvNeXtConfig
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

    layer.merge_weights()
    assert layer.merged
    res_2 = layer(x)

    tf.debugging.assert_near(res_1, res_2)


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


def test_convert_to_lora_model():
    model = create_model("convnext_test_model")
    lora_model = lora.convert_to_lora_model(model, lora_rank=1)

    img = tf.random.uniform(model.dummy_inputs.shape)
    res_1 = model(img, training=False)
    res_2 = lora_model(img, training=False)

    tf.debugging.assert_near(res_1, res_2)


def _nb_parameters(model):
    trainable = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    non_trainable = np.sum(
        [np.prod(v.get_shape()) for v in model.non_trainable_weights]
    )
    return trainable, non_trainable


def test_set_only_lora_layers_trainable():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(units=2, use_bias=True, name="fc1"),
            lora.LoRADense(units=3, use_bias=True, lora_rank=3, name="fc2"),
        ]
    )
    model.build(input_shape=(4, 5))
    # Number of parameters
    # fc1: kernel 10=5*2, bias 2
    # fc2: kernel 6=2*3, lora_a 9=3*3, lora_b 6=2*3, bias 3

    # At the beginning everything is trainable
    assert _nb_parameters(model) == (10 + 2 + 6 + 9 + 6 + 3, 0)

    lora.set_only_lora_layers_trainable(model, train_bias="none")
    assert _nb_parameters(model) == (9 + 6, 10 + 2 + 6 + 3)
