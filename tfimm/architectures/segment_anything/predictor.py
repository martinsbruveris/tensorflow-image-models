# TODO: Add sample notebook
# TODO: Test notebook with Colab
# TODO: Add "Open in Colab" badge to notebook (see SAM)
# TODO: Test mixed precision behaviour

# TODO: Add module-level docstring to predictor.py
# TODO: Check docstring for predictor.__call__
# TODO: Add documentation for sam.py, compile documentation

# TODO: Convert PT models to TF and upload to GitHub
from typing import Callable, Optional, Tuple

import numpy as np
import tensorflow as tf

from tfimm.models.factory import create_preprocessing

from .sam import SegmentAnythingModel

class SAMPredictor:
    def __init__(
        self, model: SegmentAnythingModel, preprocessing: Optional[Callable] = None
    ):
        if preprocessing is None:
            preprocessing = create_preprocessing(
                model.cfg.name, in_channels=model.cfg.in_channels, dtype=tf.float32
            )

        self.model = model
        self.preprocessing = preprocessing

        # These attributes are set when calling `set_image()`.
        self.resizer : Optional[ResizeLongestSide] = None
        self.image_embedding = None
        self.image_set = False

    @property
    def input_size(self):
        return self.model.input_size

    @property
    def mask_size(self):
        return self.model.mask_size

    def set_image(self, image: np.ndarray):
        """
        Calculates the image embeddings for the provided image, allowing masks to be
        predicted much faster.

        Args:
            image: An array of shape (H, W, C) with pixel values in [0, 255].
        """
        self.resizer = ResizeLongestSide(
            src_size=image.shape[:2], dst_size=self.input_size
        )

        # We scale the image to the largest size possible that fits input_size.
        image = self.resizer.scale_image(image)
        image = self.resizer.pad_image(image)  # Pad to input_size
        image = image[np.newaxis, ...]  # Add batch dimension

        # Now we move to TF world and compute embeddings
        image = tf.convert_to_tensor(image)
        image = self.preprocessing(image)
        self.image_embedding = self.model.image_encoder(image, training=False)

        self.image_set = True

    def clear_image(self):
        self.resizer = None
        self.image_embedding = None
        self.image_set = False

    def preprocess_masks(self, mask: np.ndarray) -> np.ndarray:
        """
        Preprocesses a mask from the pixel space of the original image (H0, W0), to the
        correct input size to the model. Note that the mask should be a mask of
        logits and *not* the thresholded version.
        """
        # First we convert mask to full mask at input_size resolution
        mask = self.resizer.scale_image(mask, channels_last=False)
        mask = self.resizer.pad_image(mask, channels_last=False)

        # Then we rescale to mask_size
        mask = self.resizer.scale_to_size(
            mask, size=self.mask_size, channels_last=False
        )
        return mask

    def __call__(
        self,
        points: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        boxes: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ):
        """
        Predicts masks end-to-end for the given prompts. We assume that the image has
        already been set.

        The original image size is (H0, W0). After resizing and padding the image size
        becomes (H, W) as given by `input_size` (usually (1024, 1024). Mask input and
        logit output will have shape (H', W') given by `mask_size` (usually H'=H/4).

        One can use `preprocess_masks` to transform an input mask from (H0, W0) to
        (H', W').

        Args:
            points: An (M1, 2) array of point prompts with coordinates in pixel
                space of the original image (H0, W0).
            labels: An (M1,) array of labels for point prompts. 1 indicates a
                foreground point and 0 indicates a background point.
            boxes: An (M2, 4) tensor of box prompts of form (left, top, right,
                bottom) with coordinates in pixel space of the original image (H0, W0).
            masks: An (M3, H', W') tensor of mask inputs, where (H', W') is the mask
                size (usually H'=H/4).
            multimask_output: If True, we return multiple nested masks for each prompt.
            return_logits: If True, we don't threshold the upscaled mask.

        Returns:
            masks: An (K, H, W) bool tensor of binary masked predictions, where K is
                determined by the multimask_output parameter.
            quality: An (K,) array with the model's predictions of mask quality.
            logits: An (K, H', W') array with low resoulution logits, where usually
                H'=H/4 and W'=W/4. This can be passed as mask input to subsequent
                iterations of prediction.
        """
        if not self.image_set:
            raise ValueError("Need to set image before calling predict().")

        if points is None:
            points = np.zeros((0, 2), dtype=np.float32)
        if labels is None:
            labels = np.zeros((0,), dtype=np.int32)
        if boxes is None:
            boxes = np.zeros((0, 4), dtype=np.float32)
        if masks is None:
            masks = np.zeros((0, *self.mask_size), dtype=np.float32)

        points = self.resizer.scale_points(points)
        boxes = self.resizer.scale_boxes(boxes)

        points = tf.convert_to_tensor(points[np.newaxis])
        labels = tf.convert_to_tensor(labels[np.newaxis])
        boxes = tf.convert_to_tensor(boxes[np.newaxis])
        masks = tf.convert_to_tensor(masks[np.newaxis])

        masks, quality, logits = self._predict_tf(
            points, labels, boxes, masks, multimask_output
        )

        masks = masks[0].numpy()
        quality = quality[0].numpy()
        logits = logits[0].numpy()

        # Transform masks back to image size
        masks = self.resizer.postprocess_mask(masks)

        # We apply the threshold only at the very end
        if not return_logits:
            masks = masks > self.model.cfg.mask_threshold

        return masks, quality, logits

    def _predict_tf(
        self, points, labels, boxes, masks, multimask_output
    ):
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            inputs={"points": points, "labels": labels, "boxes": boxes, "masks": masks},
            training=False,
        )

        logits, quality = self.model.mask_decoder(
            inputs={
                "image_embeddings": self.image_embedding,
                "image_pe": self.model.get_image_pe(batch_size=1),
                "sparse_embeddings": sparse_embeddings,
                "dense_embeddings": dense_embeddings,
            },
            training=False,
            multimask_output=multimask_output,
        )

        masks = self.model._postprocess_logits(logits, return_logits=True)
        return masks, quality, logits


class ResizeLongestSide:
    """
    Utility class to resize images to the largest side fits in a given shape while
    preserving the aspect ratio. It also provides methods to resize coordinates and
    bounding boxes.
    """

    def __init__(self, src_size: Tuple[int, int], dst_size: Tuple[int, int]):
        self.src_size = src_size
        self.dst_size = dst_size

        self.scale, self.rescaled_size = self.get_scale()

    def get_scale(self):
        """Calculate rescaling parameters."""
        h_scale = self.dst_size[0] / self.src_size[0]
        w_scale = self.dst_size[1] / self.src_size[1]

        if h_scale >= w_scale:
            scale = w_scale
            rescaled_size = (int(scale * self.src_size[0]), self.dst_size[1])
        else:
            scale = h_scale
            rescaled_size = (self.dst_size[0], int(scale * self.src_size[1]))

        # This is just to avoid rounding issues. We make sure not to exceed dst_size.
        rescaled_size = (
            min(rescaled_size[0], self.dst_size[0]),
            min(rescaled_size[1], self.dst_size[1]),
        )
        return scale, rescaled_size

    @staticmethod
    def scale_to_size(
        image: np.ndarray,
        size: Tuple[int, int],
        channels_last: bool,
    ) -> np.ndarray:
        two_dim = image.ndim == 2  # This is the case for segmentation masks
        if two_dim:
            image = image[..., np.newaxis]  # TF resize needs at least 3D tensor
            # In a 2D tensor there is no channel dimension, so we can ignore whatever
            # the user provided.
            channels_last = True
        if not channels_last:  # TF resize expects HWC format.
            image = np.transpose(image, (1, 2, 0))

        # TF resize converts everything to floats, so we remember the dtype
        dtype = image.dtype

        # Why are we using TF here and not OpenCV? Because TFIMM does not have OpenCV
        # as a dependency and I don't want to introduce it just because of the one
        # resizing operation.
        image = tf.convert_to_tensor(image)
        image = tf.image.resize(
            image, size=size, method=tf.image.ResizeMethod.AREA
        )
        image = image.numpy().astype(dtype)

        if not channels_last:
            image = np.transpose(image, (2, 0, 1))
        if two_dim:
            image = image[..., 0]

        return image

    def scale_image(self, image: np.ndarray, channels_last: bool = True) -> np.ndarray:
        """Applies scaling to an image."""
        return self.scale_to_size(image, self.rescaled_size, channels_last)

    def unscale_image(self, image: np.ndarray, channels_last: bool = True) -> np.ndarray:
        """Reverses the scaling operation."""
        return self.scale_to_size(image, self.src_size, channels_last)

    def pad_image(self, image: np.ndarray, channels_last: bool = True) -> np.ndarray:
        two_dim = image.ndim == 2  # This is the case for segmentation masks
        if two_dim:
            image = image[..., np.newaxis]  # Add channel axis
            channels_last = True
        if not channels_last:  # TF resize expects HWC format.
            image = np.transpose(image, (1, 2, 0))

        # Pad shorter edge to model input size.
        pad_h = self.dst_size[0] - image.shape[0]
        pad_w = self.dst_size[1] - image.shape[1]
        if pad_h < 0 or pad_w < 0:
            raise ValueError("Cannot pad an image that is larger than dst_size.")

        image = np.pad(image, [[0, pad_h], [0, pad_w], [0, 0]])

        if not channels_last:
            image = np.transpose(image, (2, 0, 1))
        if two_dim:
            image = image[..., 0]

        return image

    def scale_points(self, points):
        return self.scale * points

    def scale_boxes(self, boxes):
        return self.scale * boxes

    def postprocess_mask(self, mask, threshold: Optional[float] = None):
        mask = mask[..., :self.rescaled_size[0], :self.rescaled_size[1]]
        mask = self.unscale_image(mask, channels_last=False)
        if threshold is not None:
            mask = mask > threshold
        return mask
