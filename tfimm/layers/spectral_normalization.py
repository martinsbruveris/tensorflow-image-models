"""Spectral normalization layers.
## References:
[1] Yuichi Yoshida, Takeru Miyato. Spectral Norm Regularization for Improving
    the Generalizability of Deep Learning.
    _arXiv preprint arXiv:1705.10941_, 2017. https://arxiv.org/abs/1705.10941
[2] Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida.
    Spectral normalization for generative adversarial networks.
    In _International Conference on Learning Representations_, 2018.
[3] Henry Gouk, Eibe Frank, Bernhard Pfahringer, Michael Cree.
    Regularisation of neural networks by enforcing lipschitz continuity.
    _arXiv preprint arXiv:1804.04368_, 2018. https://arxiv.org/abs/1804.04368
"""
# `SpectralNormalizationConv2D` class copied from
# `https://github.com/tensorflow/models/blob/master/official/nlp/modeling/layers/spectral_normalization.py`
#
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

import tensorflow as tf


class SpectralNormalizationConv2D(tf.keras.layers.Wrapper):
    """Implements spectral normalization for Conv2D layer based on [3]."""

    def __init__(
        self,
        layer,
        iteration=1,
        norm_multiplier=0.95,
        training=True,
        aggregation=tf.VariableAggregation.MEAN,
        inhere_layer_name=False,
        legacy_mode=False,
        **kwargs,
    ):
        """Initializer.
        Args:
          layer: (tf.keras.layers.Layer) A TF Keras layer to apply normalization to.
          iteration: (int) The number of power iteration to perform to estimate
            weight matrix's singular value.
          norm_multiplier: (float) Multiplicative constant to threshold the
            normalization. Usually under normalization, the singular value will
            converge to this value.
          training: (bool) Whether to perform power iteration to update the singular
            value estimate.
          aggregation: (tf.VariableAggregation) Indicates how a distributed variable
            will be aggregated. Accepted values are constants defined in the class
            tf.VariableAggregation.
          inhere_layer_name: (bool) Whether to inhere the name of the input layer.
          legacy_mode: (bool) Whether to use the legacy implementation where the
            dimension of the u and v vectors are set to the batch size. It should
            not be enabled unless for backward compatibility reasons.
          **kwargs: (dict) Other keyword arguments for the layers.Wrapper class.
        """
        self.iteration = iteration
        self.do_power_iteration = training
        self.aggregation = aggregation
        self.norm_multiplier = norm_multiplier
        self.legacy_mode = legacy_mode

        # Set layer name.
        wrapper_name = kwargs.pop("name", None)
        if inhere_layer_name:
            wrapper_name = layer.name

        if not isinstance(layer, tf.keras.layers.Conv2D):
            raise ValueError(
                f"`layer` must be a `tf.keras.layers.Conv2D`. Observed `{layer}`"
            )
        # call __init__ of tf.keras.layers.Wrapper
        super().__init__(layer, name=wrapper_name, **kwargs)

    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)
        self.layer.kernel._aggregation = (
            self.aggregation
        )  # pylint: disable=protected-access
        self._dtype = self.layer.kernel.dtype

        # Shape (kernel_size_1, kernel_size_2, in_channel, out_channel).
        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()
        self.strides = self.layer.strides

        # Set the dimensions of u and v vectors.
        batch_size = input_shape[0]
        uv_dim = batch_size if self.legacy_mode else 1

        # Resolve shapes.
        in_height = input_shape[1]
        in_width = input_shape[2]
        in_channel = self.w_shape[2]

        out_height = in_height // self.strides[0]
        out_width = in_width // self.strides[1]
        out_channel = self.w_shape[3]

        self.in_shape = (uv_dim, in_height, in_width, in_channel)
        self.out_shape = (uv_dim, out_height, out_width, out_channel)

        self.v = self.add_weight(
            shape=self.in_shape,
            initializer=tf.initializers.random_normal(),
            trainable=False,
            name="v",
            dtype=self.dtype,
            aggregation=self.aggregation,
        )

        self.u = self.add_weight(
            shape=self.out_shape,
            initializer=tf.initializers.random_normal(),
            trainable=False,
            name="u",
            dtype=self.dtype,
            aggregation=self.aggregation,
        )

        super().build()

    def call(self, inputs):
        u_update_op, v_update_op, w_update_op = self.update_weights()
        output = self.layer(inputs)
        w_restore_op = self.restore_weights()

        # Register update ops.
        self.add_update(u_update_op)
        self.add_update(v_update_op)
        self.add_update(w_update_op)
        self.add_update(w_restore_op)

        return output

    def update_weights(self):
        """Computes power iteration for convolutional filters based on [3]."""
        # Initialize u, v vectors.
        u_hat = self.u
        v_hat = self.v

        if self.do_power_iteration:
            for _ in range(self.iteration):
                # Updates v.
                v_ = tf.nn.conv2d_transpose(
                    u_hat,
                    self.w,
                    output_shape=self.in_shape,
                    strides=self.strides,
                    padding="SAME",
                )
                v_hat = tf.nn.l2_normalize(tf.reshape(v_, [1, -1]))
                v_hat = tf.reshape(v_hat, v_.shape)

                # Updates u.
                u_ = tf.nn.conv2d(v_hat, self.w, strides=self.strides, padding="SAME")
                u_hat = tf.nn.l2_normalize(tf.reshape(u_, [1, -1]))
                u_hat = tf.reshape(u_hat, u_.shape)

        # Calculate denominator for normalization
        v_w_hat = tf.nn.conv2d(v_hat, self.w, strides=self.strides, padding="SAME")
        sigma = tf.matmul(tf.reshape(v_w_hat, [1, -1]), tf.reshape(u_hat, [-1, 1]))
        # Convert sigma from a 1x1 matrix to a scalar.
        sigma = tf.reshape(sigma, [])

        u_update_op = self.u.assign(u_hat)
        v_update_op = self.v.assign(v_hat)

        w_norm = tf.cond(
            (self.norm_multiplier / sigma) < 1,
            lambda: (self.norm_multiplier / sigma)  # pylint:disable=g-long-lambda
            * self.w,
            lambda: self.w,
        )

        w_update_op = self.layer.kernel.assign(w_norm)

        return u_update_op, v_update_op, w_update_op

    def restore_weights(self):
        """Restores layer weights to maintain gradient update (See Alg 1 of [1])."""
        return self.layer.kernel.assign(self.w)


class SpectralNormalizationDepthwiseConv2D(tf.keras.layers.Wrapper):
    """Implements spectral normalization for DepthwiseConv2D layer based on [3]."""

    def __init__(
        self,
        layer,
        iteration=1,
        norm_multiplier=0.95,
        training=True,
        aggregation=tf.VariableAggregation.MEAN,
        inhere_layer_name=False,
        legacy_mode=False,
        **kwargs,
    ):
        """Initializer.
        Args:
          layer: (tf.keras.layers.Layer) A TF Keras layer to apply normalization to.
          iteration: (int) The number of power iteration to perform to estimate
            weight matrix's singular value.
          norm_multiplier: (float) Multiplicative constant to threshold the
            normalization. Usually under normalization, the singular value will
            converge to this value.
          training: (bool) Whether to perform power iteration to update the singular
            value estimate.
          aggregation: (tf.VariableAggregation) Indicates how a distributed variable
            will be aggregated. Accepted values are constants defined in the class
            tf.VariableAggregation.
          inhere_layer_name: (bool) Whether to inhere the name of the input layer.
          legacy_mode: (bool) Whether to use the legacy implementation where the
            dimension of the u and v vectors are set to the batch size. It should
            not be enabled unless for backward compatibility reasons.
          **kwargs: (dict) Other keyword arguments for the layers.Wrapper class.
        """
        self.iteration = iteration
        self.do_power_iteration = training
        self.aggregation = aggregation
        self.norm_multiplier = norm_multiplier
        self.legacy_mode = legacy_mode

        # Set layer name.
        wrapper_name = kwargs.pop("name", None)
        if inhere_layer_name:
            wrapper_name = layer.name

        if not isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            raise ValueError(
                "`layer` must be a `tf.keras.layer.DepthwiseConv2D`. "
                + f"Observed `{layer}`"
            )
        # call __init__ of tf.keras.layers.Wrapper
        super().__init__(layer, name=wrapper_name, **kwargs)

    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)
        self.layer.depthwise_kernel._aggregation = (
            self.aggregation
        )  # pylint: disable=protected-access
        self._dtype = self.layer.depthwise_kernel.dtype

        # Shape (kernel_size_1, kernel_size_2, in_channel, out_channel).
        self.w = self.layer.depthwise_kernel
        self.w_shape = self.w.shape.as_list()
        self.strides = self.layer.strides

        # Set the dimensions of u and v vectors.
        batch_size = input_shape[0]
        uv_dim = batch_size if self.legacy_mode else 1

        # Resolve shapes.
        in_height = input_shape[1]
        in_width = input_shape[2]
        in_channel = self.w_shape[2]

        out_height = in_height // self.strides[0]
        out_width = in_width // self.strides[1]
        out_channel = self.w_shape[3]

        self.in_shape = (uv_dim, in_height, in_width, in_channel)
        self.out_shape = (uv_dim, out_height, out_width, out_channel)

        self.v = self.add_weight(
            shape=self.in_shape,
            initializer=tf.initializers.random_normal(),
            trainable=False,
            name="v",
            dtype=self.dtype,
            aggregation=self.aggregation,
        )

        self.u = self.add_weight(
            shape=self.out_shape,
            initializer=tf.initializers.random_normal(),
            trainable=False,
            name="u",
            dtype=self.dtype,
            aggregation=self.aggregation,
        )

        super().build()

    def call(self, inputs):
        u_update_op, v_update_op, w_update_op = self.update_weights()
        output = self.layer(inputs)
        w_restore_op = self.restore_weights()

        # Register update ops.
        self.add_update(u_update_op)
        self.add_update(v_update_op)
        self.add_update(w_update_op)
        self.add_update(w_restore_op)

        return output

    def update_weights(self):
        """Computes power iteration for convolutional filters based on [3]."""
        # Initialize u, v vectors.
        u_hat = self.u
        v_hat = self.v

        if self.do_power_iteration:
            for _ in range(self.iteration):
                # Updates v.
                v_ = tf.nn.conv2d_transpose(
                    u_hat,
                    self.w,
                    output_shape=self.in_shape,
                    strides=self.strides,
                    padding="SAME",
                )
                v_hat = tf.nn.l2_normalize(tf.reshape(v_, [1, -1]))
                v_hat = tf.reshape(v_hat, v_.shape)

                # Updates u.
                u_ = tf.nn.conv2d(v_hat, self.w, strides=self.strides, padding="SAME")
                u_hat = tf.nn.l2_normalize(tf.reshape(u_, [1, -1]))
                u_hat = tf.reshape(u_hat, u_.shape)

        # Calculate denominator for normalization
        v_w_hat = tf.nn.conv2d(v_hat, self.w, strides=self.strides, padding="SAME")
        sigma = tf.matmul(tf.reshape(v_w_hat, [1, -1]), tf.reshape(u_hat, [-1, 1]))
        # Convert sigma from a 1x1 matrix to a scalar.
        sigma = tf.reshape(sigma, [])

        u_update_op = self.u.assign(u_hat)
        v_update_op = self.v.assign(v_hat)

        w_norm = tf.cond(
            (self.norm_multiplier / sigma) < 1,
            lambda: (self.norm_multiplier / sigma)  # pylint:disable=g-long-lambda
            * self.w,
            lambda: self.w,
        )

        w_update_op = self.layer.depthwise_kernel.assign(w_norm)

        return u_update_op, v_update_op, w_update_op

    def restore_weights(self):
        """Restores layer weights to maintain gradient update (See Alg 1 of [1])."""
        return self.layer.depthwise_kernel.assign(self.w)
