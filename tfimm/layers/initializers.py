import math

import tensorflow as tf


class FanoutInitializer(tf.keras.initializers.Initializer):
    """
    Fanout initializer as used by EfficientNet models.

    Args:
        nb_groups: Number of groups for groupwise convolutions. Fanout is performed
            for each group individually.
        depthwise: If True, we set the number of groups equal to the number of input
            channels. This overrides any value for nb_groups.
    """

    def __init__(self, nb_groups: int = 1, depthwise: bool = False):
        self.nb_groups = nb_groups
        self.depthwise = depthwise

    def __call__(self, shape, dtype=None, **kwargs):
        out_channels = shape[-1]
        fan_out = shape[0] * shape[1] * out_channels  # kernel_size * out_channels
        # Set nb_groups to in_channels for depthwise convolutions
        nb_groups = shape[-2] if self.depthwise else self.nb_groups
        fan_out /= nb_groups
        return tf.random.normal(
            shape, mean=0.0, stddev=math.sqrt(2.0 / fan_out), dtype=dtype
        )

    def get_config(self):
        return {"nb_groups": self.nb_groups, "depthwise": self.depthwise}
