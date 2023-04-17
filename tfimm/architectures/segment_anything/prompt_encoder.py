from typing import Tuple

import numpy as np
import tensorflow as tf

from tfimm.layers import act_layer_factory, norm_layer_factory


class PromptEncoder(tf.keras.Model):
    def __init__(
        self,
        embed_dim: int,
        mask_hidden_dim: int,
        act_layer: str = "gelu",
        **kwargs,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
            embed_dim: The prompts' embedding dimension.
            mask_hidden_dim: The number of hidden channels for encoding input masks.
            act_layer: Activation layer.
        """
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mask_hidden_dim = mask_hidden_dim
        self.act_layer = act_layer

        self.pe_layer = PositionalEmbeddingRandom(
            embed_dim=self.embed_dim, name="pe_layer"
        )
        self.point_embeddings = None
        self.not_a_point_embed = None

        self.mask_downscaling = MaskDownscaling(
            embed_dim=self.embed_dim,
            mask_hidden_dim=self.mask_hidden_dim,
            act_layer=self.act_layer,
            name="mask_downscaling",
        )
        self.no_mask_embed = None

    @property
    def dummy_inputs(self):
        return {
            "points": tf.zeros((1, 1, 2), dtype=tf.float32),
            "labels": tf.zeros((1, 1), dtype=tf.int32),
            "boxes": tf.zeros((1, 1, 4), dtype=tf.float32),
            "masks": tf.zeros((1, 8, 8, 1), dtype=tf.float32),
        }

    def build(self, input_shape):
        nb_point_embeddings = 4  # foreground/background point + 2 box corners
        self.point_embeddings = [
            self.add_weight(
                shape=(1, self.embed_dim),
                initializer=tf.keras.initializers.RandomNormal(),
                trainable=True,
                name=f"point_embeddings/{j}/weight",
            )
            for j in range(nb_point_embeddings)
        ]
        self.not_a_point_embed = self.add_weight(
            shape=(1, self.embed_dim),
            initializer=tf.keras.initializers.RandomNormal(),
            trainable=True,
            name="not_a_point_embed/weight",
        )
        self.no_mask_embed = self.add_weight(
            shape=(1, self.embed_dim),
            initializer=tf.keras.initializers.RandomNormal(),
            trainable=True,
            name="no_mask_embed/weight",
        )

    def _embed_points(self, points: tf.Tensor, labels: tf.Tensor, input_size):
        """Embed point prompts."""
        points = points + 0.5  # Shift to center of pixel
        labels = tf.expand_dims(labels, axis=-1)  # (N, 1)
        embeddings = self.pe_layer.embed_points(points, input_size)
        embeddings += tf.where(
            labels == 0, self.point_embeddings[0], self.point_embeddings[1]
        )
        return embeddings

    def _embed_boxes(self, boxes: tf.Tensor, input_size):
        """Embed box promts."""
        # Shape of boxes is (N, M, 4)
        n, m, _ = tf.unstack(tf.shape(boxes))
        boxes = boxes + 0.5  # Shift to center of pixel
        corners = tf.reshape(boxes, (n * m, 2, 2))  # (N*M, 2, 2)
        embeddings = self.pe_layer.embed_points(
            corners, input_size  # (N*M, 2, embed_dim)
        )
        embeddings += tf.stack(
            [self.point_embeddings[2], self.point_embeddings[3]], axis=1
        )
        embeddings = tf.reshape(
            embeddings, (n, m, 2, self.embed_dim)
        )  # (N, M, 2, embed_dim)
        embeddings = tf.reshape(
            embeddings, (n, 2 * m, self.embed_dim)
        )  # (N, 2*M, embed_dim)
        return embeddings

    def _embed_masks(self, masks, training: bool = False):
        n, _, h, w = tf.unstack(tf.shape(masks))  # (N, M3, H', W')
        # If we have no masks, we return `no_mask_embed` broadcast across batch and
        # spatial dimensions.
        embeddings = tf.cond(
            tf.shape(masks)[1] == 0,
            lambda: tf.broadcast_to(
                input=tf.reshape(self.no_mask_embed, (1, 1, 1, -1)),
                shape=(n, h // 4, w // 4, self.embed_dim),
            ),
            lambda: self.mask_downscaling(masks, training=training),
        )
        return embeddings  # (N, H'', W'', D)

    def call(self, inputs, training=False):
        """
        Embeds different types of prompts, returning both sparse and dense embeddings.

        Arguments:
            inputs: Dictionary with the following entries.
                points: An (N, M1, 2) tensor of point coordinates.
                labels: An (N, M1) tensor of point labels. 1 indicates a foreground
                    point and 0 indicates a background point.
                boxes: An (N, M2, 4) tensor of bounding box coordinates.
                masks: An (N, M3, H', W') tensor for masks, where M3 can be 0 (no mask
                    to embed, tensor is empty). The spatial input size (H, W) should be
                    self.mask_size.
            training: Training or inference phase?

        Returns:
            Sparse embeddings of shape (N, M, D), where M is determined by the number
                of input points and boxes:
                 - M = 2*M2, if M1 = 0 (no points)
                 - M = M1 + 1, if M1 > 0 and M2 = 0 (points, no boxes) and
                 - M = M1 + 2*M2, if M1 > 0 and M2 > 0 (points and boxes)
            Dense mask embeddings of shape (N, H'', W'', D), where (H'', W'') is given
                by grid_size and D is embed_dim.
        """
        points = inputs["points"]
        labels = inputs["labels"]
        boxes = inputs["boxes"]
        masks = inputs["masks"]

        _, _, h, w = tf.unstack(tf.shape(masks))  # (N, M3, H', W')
        input_size = (4 * h, 4 * w)  # (H, W)

        point_embeddings = self._embed_points(points, labels, input_size)
        box_embeddings = self._embed_boxes(boxes, input_size)

        # If we have points, but no boxes, we append one `not_a_point_embed` embedding.
        # This is done in PyTorch via `pad=True` in `_embed_points()`.
        n = tf.shape(points)[0]
        pad_token = tf.cond(
            (tf.shape(points)[1] > 0) & (tf.shape(boxes)[1] == 0),
            lambda: tf.tile(  # We need this to get the right batch size
                input=tf.expand_dims(self.not_a_point_embed, axis=0),
                multiples=(n, 1, 1),
            ),
            lambda: tf.zeros((n, 0, self.embed_dim)),
        )

        sparse_embeddings = tf.concat(
            [point_embeddings, pad_token, box_embeddings], axis=1
        )
        dense_embeddings = self._embed_masks(masks, training=training)

        return sparse_embeddings, dense_embeddings

    def get_dense_pe(self, grid_size) -> tf.Tensor:
        """
        Returns the positional encoding used to encode point prompts, applied to a
        dense set of points the shape of the image encoding.

        Returns:
          Positional encoding with shape (*grid_size, embed_dim).
        """
        image_pe = self.pe_layer.embed_grid(grid_size)
        return image_pe


class MaskDownscaling(tf.keras.Model):
    def __init__(
        self,
        embed_dim: int,
        mask_hidden_dim: int,
        act_layer: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mask_hidden_dim = mask_hidden_dim
        self.act_layer = act_layer

        norm_layer = norm_layer_factory("layer_norm_eps_1e-6")
        act_layer = act_layer_factory(self.act_layer)

        self.conv1 = tf.keras.layers.Conv2D(
            filters=self.mask_hidden_dim // 4, kernel_size=2, strides=2, name="0"
        )
        self.norm1 = norm_layer(name="1")
        self.act1 = act_layer(name="2")
        self.conv2 = tf.keras.layers.Conv2D(
            filters=self.mask_hidden_dim, kernel_size=2, strides=2, name="3"
        )
        self.norm2 = norm_layer(name="4")
        self.act2 = act_layer(name="5")
        self.conv3 = tf.keras.layers.Conv2D(
            filters=self.embed_dim, kernel_size=1, name="6"
        )

    def call(self, inputs, training=False):
        x = inputs  # (N, M3, H', W')

        n, m, h, w = tf.unstack(tf.shape(x))
        x = tf.reshape(x, (n * m, h, w))  # (N*M3, H', W')
        x = tf.expand_dims(x, axis=-1)  # (N*M3, H', W', 1)

        x = self.conv1(x)
        x = self.norm1(x, training=training)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x, training=training)
        x = self.act2(x)
        x = self.conv3(x)  # (N*M3, H'', W'', D)

        _, h, w, d = tf.unstack(tf.shape(x))
        x = tf.reshape(x, (n, m, h, w, d))  # (N, M3, H'', W'', D)
        # If we are given multiple input masks, we sum the corresponding embeddings.
        x = tf.reduce_sum(x, axis=1)  # (N, H'', W'', D)

        return x


class PositionalEmbeddingRandom(tf.keras.layers.Layer):
    """
    Positional embedding using random spatial frequencies.
    """

    def __init__(
        self,
        embed_dim: int,
        scale: float = 1.0,
        **kwargs,
    ) -> None:
        """
        Args:
            embed_dim: Size of positional embeddings. Note that compared to the PyTorch
                code, embed_dim = 2 * num_pos_feats, but since we concatenate the
                x and y positional embeddings, the result has size embed_dim.
            scale: Norm of embeddings
        """
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.scale = scale

        self.positional_encoding_gaussian_matrix = None

    def build(self, input_shape):
        self.positional_encoding_gaussian_matrix = self.add_weight(
            shape=(2, self.embed_dim // 2),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.scale),
            trainable=False,
            name="positional_encoding_gaussian_matrix",
        )

    def call(self, x):
        """
        Positionally embed points that are normalised to [0, 1].

        Args:
            x: Tensor of shape (X, 2), X can be arbitrary.

        Returns:
            Tensor of shape (X, embed_dim).
        """
        x = 2 * x - 1  # (X, 2)
        x = tf.matmul(x, self.positional_encoding_gaussian_matrix)  # (X, D // 2)
        x = 2 * np.pi * x
        embed = tf.concat([tf.sin(x), tf.cos(x)], axis=-1)  # (X, D)
        return embed

    def embed_grid(self, size: Tuple[int, int]):
        """Generate positional embedding for a grid of the specified size."""
        h, w = size
        grid = tf.ones((h, w), dtype=tf.float32)
        x = (tf.cumsum(grid, axis=1) - 0.5) / tf.cast(w, tf.float32)
        y = (tf.cumsum(grid, axis=0) - 0.5) / tf.cast(h, tf.float32)
        points = tf.stack([x, y], axis=-1)
        return self(points)

    def embed_points(self, points, image_size):
        """Positionally embed points that are not normalized to [0,1]."""
        x = points[..., 0] / tf.cast(image_size[1], tf.float32)
        y = points[..., 1] / tf.cast(image_size[0], tf.float32)
        points = tf.stack([x, y], axis=-1)
        return self(points)
