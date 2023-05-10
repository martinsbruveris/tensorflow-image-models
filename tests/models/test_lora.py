import pytest
import tensorflow as tf

from tfimm.architectures import lora

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
