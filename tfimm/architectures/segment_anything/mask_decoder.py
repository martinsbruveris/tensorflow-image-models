from typing import Tuple

import tensorflow as tf

from tfimm.layers import act_layer_factory, norm_layer_factory


class MaskDecoder(tf.keras.Model):
    def __init__(
        self,
        *,
        transformer: tf.keras.Model,
        embed_dim: int,
        nb_multimask_outputs: int,
        act_layer: str,
        iou_head_depth: int,
        iou_head_hidden_dim: int,
        **kwargs,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a tranformer
        architecture.

        Args:
            transformer: Transformer model used to predict masks.
            embed_dim: Dimension of image and prompt embeddings.
            nb_multimask_outputs: The number of masks to predict with disambiguating
                masks.
            act_layer: Activation layer.
            iou_head_depth: The depth of MLP used to predict mask quality.
            iou_head_hidden_dim: Hidden dimension of MLP used to predict mask quality.
        """
        super().__init__(**kwargs)
        self.transformer = transformer
        self.embed_dim = embed_dim
        self.nb_multimask_outputs = nb_multimask_outputs
        self.act_layer = act_layer
        self.iou_head_depth = iou_head_depth
        self.iou_head_hidden_dim = iou_head_hidden_dim

        self.iou_token = None
        self.mask_tokens = None
        self.nb_mask_tokens = self.nb_multimask_outputs + 1

        self.output_upscaling = OutputUpscaling(
            embed_dim=self.embed_dim, act_layer=self.act_layer, name="output_upscaling"
        )
        self.output_hypernetworks_mlps = [
            MLP(
                hidden_dim=self.embed_dim,
                output_dim=self.embed_dim // 8,
                nb_layers=3,
                name=f"output_hypernetworks_mlps/{j}",
            )
            for j in range(self.nb_mask_tokens)
        ]
        self.iou_prediction_head = MLP(
            hidden_dim=self.iou_head_hidden_dim,
            output_dim=self.nb_mask_tokens,
            nb_layers=self.iou_head_depth,
            name="iou_prediction_head",
        )

    def build(self, input_shape):
        self.iou_token = self.add_weight(
            shape=(1, self.embed_dim),
            initializer=tf.keras.initializers.RandomNormal(),
            trainable=True,
            name="iou_token/weight",
        )
        self.mask_tokens = self.add_weight(
            shape=(self.nb_mask_tokens, self.embed_dim),
            initializer=tf.keras.initializers.RandomNormal(),
            trainable=True,
            name="mask_tokens/weight",
        )

    def call(self, inputs, training=False, multimask_output: bool = False):
        """
        Predict masks given image and prompt embeddings.

        Args:
            inputs: Dictionary with the following entries.
                image_embeddings: Image embeddings of shape (N, H, W, C), where (H, W)
                    is the grid size (usually image size / 16) and C = embed_dim is the
                    dimension of image and prompt embeddings.
                image_pe: Positional encodings for the image, same shape as image
                    embeddings, i.e., (N, H, W, C).
                sparse_embeddings: Sparse prompt embeddings, shape (N, L, C), where
                    L can be 0 if no sparse embeddings are provided.
                dense_embeddings: Dense prompt embeddings, shape (N, H, W, C).
            training: Training of inference phase?
            multimask_output: If true, we return multiple predicted segmentation masks,
                the number determined by `nb_multimask_outputs`.

        Returns:
            An (N, K, H', W') tensor of predicted segmentation masks, where K is either
                1 or nb_multimask_outputs. (H', W') is the spatial resolution of the
                segmentation mask (usually image_size / 4).
            An (N, K) tensor of mask quality predictions.
        """
        image_embeddings = inputs["image_embeddings"]
        image_pe = inputs["image_pe"]
        sparse_embeddings = inputs["sparse_embeddings"]
        dense_embeddings = inputs["dense_embeddings"]

        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_embeddings=sparse_embeddings,
            dense_embeddings=dense_embeddings,
            training=training,
        )

        # Select the correct mask or masks for returning
        if multimask_output:
            return masks[:, 1:], iou_pred[:, 1:]
        else:
            return masks[:, 0:1], iou_pred[:, 0:1]

    def predict_masks(
        self,
        image_embeddings: tf.Tensor,
        image_pe: tf.Tensor,
        sparse_embeddings: tf.Tensor,
        dense_embeddings: tf.Tensor,
        training: bool,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Predicts masks. See `call` for more details."""
        n, h, w, c = tf.unstack(tf.shape(image_embeddings))

        # Concatenate output tokens
        output_tokens = tf.concat([self.iou_token, self.mask_tokens], axis=0)
        output_tokens = tf.expand_dims(output_tokens, axis=0)  # (1, M+1, C)
        output_tokens = tf.tile(output_tokens, (n, 1, 1))  # (N, M+1, C)
        tokens = tf.concat([output_tokens, sparse_embeddings], axis=1)  # (N, M+1+L, C)

        # Run the transformer
        tokens, image_embeddings = self.transformer(
            inputs={
                "point_embeddings": tokens,
                "image_embeddings": image_embeddings + dense_embeddings,
                "image_pe": image_pe,
            },
            training=training,
        )
        iou_token = tokens[:, 0, :]
        mask_tokens = tokens[:, 1 : (1 + self.nb_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        upscaled_embeddings = self.output_upscaling(image_embeddings, training=training)

        hyper_in_list = [
            self.output_hypernetworks_mlps[j](mask_tokens[:, j, :], training=training)
            for j in range(self.nb_mask_tokens)
        ]
        hyper_in = tf.stack(hyper_in_list, axis=1)

        n, h, w, c = tf.unstack(tf.shape(upscaled_embeddings))
        upscaled_embeddings = tf.reshape(upscaled_embeddings, (n, h * w, c))
        masks = tf.matmul(hyper_in, upscaled_embeddings, transpose_b=True)
        masks = tf.reshape(masks, (n, -1, h, w))

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token)

        return masks, iou_pred


class OutputUpscaling(tf.keras.Model):
    def __init__(
        self,
        embed_dim: int,
        act_layer: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.act_layer = act_layer

        norm_layer = norm_layer_factory("layer_norm_eps_1e-6")
        act_layer = act_layer_factory(self.act_layer)

        self.conv1 = tf.keras.layers.Conv2DTranspose(
            filters=self.embed_dim // 4,
            kernel_size=2,
            strides=2,
            use_bias=True,
            name="0",
        )
        self.norm1 = norm_layer(name="1")
        self.act1 = act_layer(name="2")
        self.conv2 = tf.keras.layers.Conv2DTranspose(
            filters=self.embed_dim // 8,
            kernel_size=2,
            strides=2,
            use_bias=True,
            name="3",
        )
        self.act2 = act_layer(name="4")

    def call(self, inputs, training=False):
        x = inputs
        x = self.conv1(x)
        x = self.norm1(x, training=training)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        return x


class MLP(tf.keras.Model):
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        nb_layers: int,
        sigmoid_output: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nb_layers = nb_layers
        self.sigmoid_output = sigmoid_output

        filters = [hidden_dim] * (nb_layers - 1) + [output_dim]
        self.blocks = [
            tf.keras.layers.Dense(units=f, use_bias=True, name=f"layers/{j}")
            for j, f in enumerate(filters)
        ]

    def call(self, inputs):
        x = inputs
        for j, block in enumerate(self.blocks):
            x = block(x)
            if j < self.nb_layers - 1:
                x = tf.nn.relu(x)
        if self.sigmoid_output:
            x = tf.nn.sigmoid(x)
        return x
