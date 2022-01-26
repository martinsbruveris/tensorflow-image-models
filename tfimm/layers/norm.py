"""
Various normalisation layers
"""
import tensorflow as tf


class Affine(tf.keras.layers.Layer):
    """
    Affine normalisation as used in ResMLP networks.

    For NHWC ``x``, we return ``alpha * x + beta``, where ``alpha``, ``beta`` are C
    tensors.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.alpha = self.add_weight(
            shape=(input_shape[-1],),
            initializer="ones",
            trainable=True,
            name="alpha",
        )
        self.beta = self.add_weight(
            shape=(input_shape[-1],),
            initializer="zeros",
            trainable=True,
            name="beta",
        )

    def call(self, x):
        x = self.alpha * x + self.beta
        return x


def group_normalize(x, gamma, beta, nb_groups=None, group_size=None, eps=1e-5):
    """
    Applies group-normalization to NHWC ``x`` (see abs/1803.08494, go/dune-gn).

    This function just does the math, if you want a "layer" that creates the
    necessary variables etc., see ``group_norm`` below. You must either specify a fixed
    number of groups ``nb_groups``, which will automatically select a corresponding
    group size depending on the input's number of channels, or you must specify a
    ``group_size``, which leads to an automatic number of groups depending on the
    input's number of channels.

    Author: Lucas Beyer

    Args:
        x: N..C-tensor, the input to group-normalize. For images, this would be a
            NHWC-tensor, for time-series a NTC, for videos a NHWTC or NTHWC, all of
            them work, as normalization includes everything between N and C. Even just
            NC shape works, as C is grouped and normalized.
        gamma: tensor with C entries, learnable scale after normalization.
        beta: tensor with C entries, learnable bias after normalization.
        nb_groups: int, number of groups to normalize over (divides C).
        group_size: int, size of the groups to normalize over (divides C).
        eps: float, a small additive constant to avoid /sqrt(0).

    Returns:
        Group-normalized `x`, of the same shape and type as `x`.
    """
    assert (
        x.shape.ndims >= 2
    ), "Less than 2-dim Tensor passed to GroupNorm. Something's fishy."

    nb_channels = x.shape[-1]
    assert nb_channels is not None, "Cannot apply GroupNorm on dynamic channels."
    assert (nb_groups is None) != (
        group_size is None
    ), "You must specify exactly one of `num_groups`, `group_size`"

    if group_size is not None:
        nb_groups = nb_channels // group_size

    assert nb_channels % nb_groups == 0, "GroupNorm: {} not divisible by {}".format(
        nb_channels, nb_groups
    )

    orig_shape = tf.shape(x)

    # This shape is NHWGS where G is #groups and S is group-size.
    extra_shape = [nb_groups, nb_channels // nb_groups]
    group_shape = tf.concat([orig_shape[:-1], extra_shape], axis=-1)
    x = tf.reshape(x, group_shape)

    # The dimensions to normalize over: HWS for images, but more generally all
    # dimensions except N (batch, first) and G (cross-groups, next-to-last).
    # So more visually, normdims are the dots in N......G. (note the last one is
    # also a dot, not a full-stop, argh!)
    normdims = list(range(1, x.shape.ndims - 2)) + [x.shape.ndims - 1]
    mean, var = tf.nn.moments(x, normdims, keepdims=True)

    # Interestingly, we don't have a beta/gamma per group, but still one per
    # channel, at least according to the original paper. Reshape such that they
    # broadcast correctly.
    beta = tf.reshape(beta, extra_shape)
    gamma = tf.reshape(gamma, extra_shape)
    x = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
    return tf.reshape(x, orig_shape)


class GroupNormalization(tf.keras.layers.Layer):
    """
    A group-norm "layer" (see abs/1803.08494 go/dune-gn).

    This function creates beta/gamma variables in a name_scope, and uses them to apply
    ``group_normalize`` on the input ``x``. You can either specify a fixed number of
    groups ``nb_groups``, which will automatically select a corresponding group size
    depending on the input's number of channels, or you must specify a ``group_size``,
    which leads to an automatic number of groups depending on the input's number of
    channels. If you specify neither, the paper's recommended ``nb_groups=32`` is used.

    Authors: Lucas Beyer, Joan Puigcerver.

    Args:
        nb_groups: int, the number of channel-groups to normalize over.
        group_size: int, size of the groups to normalize over.
        eps: float, a small additive constant to avoid /sqrt(0).
        beta_init: initializer for bias, defaults to zeros.
        gamma_init: initializer for scale, defaults to ones.
        **kwargs: other tf.keras.layers.Layer arguments.
    """

    def __init__(
        self,
        nb_groups=None,
        group_size=None,
        eps=1e-5,
        beta_initializer=tf.zeros_initializer(),
        gamma_initializer=tf.ones_initializer(),
        **kwargs
    ):
        super(GroupNormalization, self).__init__(**kwargs)
        if nb_groups is None and group_size is None:
            nb_groups = 32

        self._nb_groups = nb_groups
        self._group_size = group_size
        self._eps = eps
        self._beta_init = beta_initializer
        self._gamma_init = gamma_initializer

    def build(self, input_size):
        channels = input_size[-1]
        assert channels is not None, "Cannot apply GN on dynamic channels."
        self._gamma = self.add_weight(
            name="gamma",
            shape=(channels,),
            initializer=self._gamma_init,
            dtype=self.dtype,
        )
        self._beta = self.add_weight(
            name="beta",
            shape=(channels,),
            initializer=self._beta_init,
            dtype=self.dtype,
        )
        super(GroupNormalization, self).build(input_size)

    def call(self, x):
        return group_normalize(
            x, self._gamma, self._beta, self._nb_groups, self._group_size, self._eps
        )
