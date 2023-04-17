# TODO: Test mixed precision behaviour
# TODO: Convert PT models to TF and upload to GitHub
import math
from typing import Callable, Optional, Tuple

import numpy as np
import tensorflow as tf

from tfimm.models.factory import create_preprocessing

from .sam import SegmentAnythingModel


class SAMPredictor:
    """
    User-friendly interface to the Segment Anything model. Uses SAM to calculate the
    image embedding for an image, and then allows repeated, efficient mask prediction
    given prompts.

    While internally TF is used for inference, the inputs and return values in this
    class are numpy arrays for ease of use.

    Args:
        model: The model used for mask prediction.
        preprocessing: Preprocessing function for the model. If not provided we
            will query ``tfimm`` using the model name.
    """

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
        self.resizer: Optional[ImageResizer] = None
        self.image_embedding = None
        self.image_set = False

    def set_image(self, image: np.ndarray):
        """
        Calculates and stores the image embeddings for the provided image, allowing
        masks to be predicted much faster.

        Args:
            image: An array of shape (H, W, C) with pixel values in [0, 255]. The image
                can be any shape, and it will be resized and padded to the model input
                shape as necessary.

        Returns:
            Nothing. The image embedding and resizing information are stored in the
            class.
        """
        if self.model.cfg.fixed_input_size:
            self.resizer = ImageResizer(
                src_size=image.shape[:2], dst_size=self.model.cfg.input_size
            )
        else:
            # If the model allows flexible input sizes, we simply pad the image to
            # the nearest multiple of patch_size.
            patch_size = self.model.cfg.encoder_patch_size
            dst_size = (
                patch_size * math.ceil(image.shape[0] / patch_size),
                patch_size * math.ceil(image.shape[1] / patch_size),
            )
            self.resizer = ImageResizer(
                src_size=image.shape[:2], dst_size=dst_size, pad_only=True
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
        """Unsets the image and forgets the embedding."""
        self.resizer = None
        self.image_embedding = None
        self.image_set = False

    def input_size(self):
        """Returns the input size to the model."""
        if self.image_set:
            return self.resizer.dst_size
        elif self.model.cfg.fixed_input_size:
            return self.model.cfg.input_size
        else:
            raise ValueError(
                "To determine model input size need to set image or use a model with "
                "a fixed input size."
            )

    def mask_size(self):
        """Returns the mask prompt input size to the model."""
        return self.model.mask_size(self.input_size())

    def preprocess_masks(self, mask: np.ndarray) -> np.ndarray:
        """
        Preprocesses a mask from the pixel space of the original image (H0, W0), to the
        correct input size to the model. Note that the mask should be a mask of
        logits and *not* the thresholded version.

        Args:
            mask: An array of shape (M, H0, W0) or (N, M, H0, W0), where (H0, W0) is the
                original size of the input image.

        Returns:
            Preprocessed mask of shape (M, H', W') or (N, M, H', W') as given by
                ``mask_size``.
        """
        # First we convert mask to full mask at input_size resolution
        mask = self.resizer.scale_image(mask, channels_last=False)
        mask = self.resizer.pad_image(mask, channels_last=False)

        # Then we rescale to mask_size
        mask_size = self.mask_size()
        mask = self.resizer.scale_to_size(mask, size=mask_size, channels_last=False)
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
        becomes (H, W) given by ``input_size`` (usually (1024, 1024)). Mask input and
        logit output will have shape (H', W') given by ``mask_size`` (usually H'=H/4).

        One can use ``preprocess_masks`` to transform an input mask from (H0, W0) to
        (H', W').

        Prompts can also be batched, i.e., have the shape (N, M1, 2) for points;
        (N, M1) for point labels; (N, M2, 4) for boxes; and (N, M3, H', W') for mask
        prompts. Note that in this case we number and type of prompts is the same for
        each batch element. The return values will have the same batch dimension, i.e.,
        (N, K, H, W) for predicted masks, etc.

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
            * Masks, an (K, H, W) bool array of binary masked predictions, where K is
              determined by the multimask_output parameter. It is either 1, if
              ``multimask_output=False`` or given by the ``nb_multimask_outputs``
              parameter in the model configuration.
            * Scores, an (K,) array with the model's predictions of mask quality.
            * Logits, an (K, H', W') array with low resoulution logits, where usually
              H'=H/4 and W'=W/4. This can be passed as mask input to subsequent
              iterations of prediction.
        """
        if not self.image_set:
            raise ValueError("Need to set image before calling predict().")

        points = np.asarray(points) if points is not None else None
        labels = np.asarray(labels) if labels is not None else None
        boxes = np.asarray(boxes) if boxes is not None else None
        masks = np.asarray(masks) if masks is not None else None

        batch_shape = self._batch_shape(points, labels, boxes, masks)

        if points is None:
            points = np.zeros(batch_shape + (0, 2), dtype=np.float32)
        if labels is None:
            labels = np.zeros(batch_shape + (0,), dtype=np.int32)
        if boxes is None:
            boxes = np.zeros(batch_shape + (0, 4), dtype=np.float32)
        if masks is None:
            mask_size = self.mask_size()
            masks = np.zeros(batch_shape + (0, *mask_size), dtype=np.float32)

        # Check that batch shapes are compatible
        if (
            points.shape[:-2] != batch_shape
            or labels.shape[:-1] != batch_shape
            or boxes.shape[:-2] != batch_shape
            or masks.shape[:-3] != batch_shape
        ):
            raise ValueError("All prompts must have the same batch shape.")

        # Add batch dimension if needed
        if batch_shape == ():
            points = points[np.newaxis]
            labels = labels[np.newaxis]
            boxes = boxes[np.newaxis]
            masks = masks[np.newaxis]

        points = self.resizer.scale_points(points)
        boxes = self.resizer.scale_boxes(boxes)

        points = tf.convert_to_tensor(points)
        labels = tf.convert_to_tensor(labels)
        boxes = tf.convert_to_tensor(boxes)
        masks = tf.convert_to_tensor(masks)

        masks, scores, logits = self._predict_tf(
            points, labels, boxes, masks, multimask_output
        )

        masks = masks.numpy()
        scores = scores.numpy()
        logits = logits.numpy()

        # Transform masks back to image size.
        masks = self.resizer.postprocess_mask(masks)

        # Remove batch dimension, if input didn't have it.
        if batch_shape == ():
            masks = masks[0]
            scores = scores[0]
            logits = logits[0]

        # We apply the threshold only at the very end
        if not return_logits:
            masks = masks > self.model.mask_threshold

        return masks, scores, logits

    def _predict_tf(self, points, labels, boxes, masks, multimask_output):
        n = tf.shape(points)[0]
        image_embedding = tf.tile(self.image_embedding, (n, 1, 1, 1))

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            inputs={"points": points, "labels": labels, "boxes": boxes, "masks": masks},
            training=False,
        )

        logits, scores = self.model.mask_decoder(
            inputs={
                "image_embeddings": image_embedding,
                "image_pe": self.model.get_image_pe(image_embedding),
                "sparse_embeddings": sparse_embeddings,
                "dense_embeddings": dense_embeddings,
            },
            training=False,
            multimask_output=multimask_output,
        )

        masks = self.model.postprocess_logits(
            logits, input_size=self.input_size(), return_logits=True
        )
        return masks, scores, logits

    @staticmethod
    def _batch_shape(points, labels, boxes, masks):
        """Returns (), if there is no batch dimension and (n,) if there is."""
        if points is not None:
            return points.shape[:-2]
        elif labels is not None:
            return labels.shape[:-1]
        elif boxes is not None:
            return boxes.shape[:-2]
        elif masks is not None:
            return masks.shape[:-3]
        else:
            return ()


class ImageResizer:
    """
    Utility class to resize images to the largest side that fits in a given shape while
    preserving the aspect ratio. It also provides methods to resize coordinates and
    bounding boxes and pad images.

    Args:
        src_size: Size of image before resizing. The resize object is image
            specific, i.e., for each source image size it is recommended to create
            a new ``ImageResizer`` object.
        dst_size: The target size after resizing (and padding).
        pad_only: If True, we don't do any resizing and only pad the image to
            ``dst_size``.
    """

    def __init__(
        self,
        src_size: Tuple[int, int],
        dst_size: Tuple[int, int],
        pad_only: bool = False,
    ):
        self.src_size = src_size
        self.dst_size = dst_size
        self.pad_only = pad_only

        self.scale, self.rescaled_size = self._get_scale()

    def _get_scale(self) -> Tuple[float, Tuple[int, int]]:
        """Calculate rescaling parameters."""
        if self.pad_only:
            # If we only pad, then scale is 1 and the rescaled size equal input size.
            return 1.0, self.src_size

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
        """
        Scales an image to a given size. In this method we ignore ``dst_size`` and do
        not attempt to preserve the aspect ratio.

        Args:
            image: Image to be resized. Can be a 3D array (H, W, C) or a 4D array
                (N, H, W, C). We also accept channels first ordering (used for
                segmentation masks, i.e., (C, H, W) or (N, C, H, W).
            size: Target size.
            channels_last: If True, images are in HWC format and if False in CHW format.

        Returns:
            Resized image array.
        """
        no_batch_dim = image.ndim == 3  # Image without batch dimension
        if no_batch_dim:
            image = image[np.newaxis]  # Add batch dimension
        if not channels_last:  # TF resize expects HWC format.
            image = np.transpose(image, (0, 2, 3, 1))

        # TF resize converts everything to floats, so we remember the dtype
        dtype = image.dtype

        # Why are we using TF here and not OpenCV? Because TFIMM does not have OpenCV
        # as a dependency and I don't want to introduce it just because of the one
        # resizing operation.
        image = tf.convert_to_tensor(image)
        image = tf.image.resize(image, size=size, method=tf.image.ResizeMethod.AREA)
        image = image.numpy().astype(dtype)

        if not channels_last:
            image = np.transpose(image, (0, 3, 1, 2))
        if no_batch_dim:
            image = image[0]

        return image

    def scale_image(self, image: np.ndarray, channels_last: bool = True) -> np.ndarray:
        """
        Applies aspect-ratio preserving scaling to an image.

        Args:
            image: Image to be resized. Can be a 3D array (H, W, C) or a 4D array
                (N, H, W, C). We also accept channels first ordering (used for
                segmentation masks, i.e., (C, H, W) or (N, C, H, W).
            channels_last: If True, images are in HWC format and if False in CHW format.

        Returns:
            Resized image with spatial dimensions given by ``rescaled_size``. The
            longest edge of the image will be equal to ``dst_size``.
        """
        return self.scale_to_size(image, self.rescaled_size, channels_last)

    def unscale_image(
        self, image: np.ndarray, channels_last: bool = True
    ) -> np.ndarray:
        """
        Reverses the scaling operation.

        Args:
            image: Image to be rescaled back to original size given by ``src_size``. We
                assume that image has size ``rescaled_size``, otherwise aspect ratio
                will not be preserved. Image can be 3D or 4D with channels before or
                after spatial dimension.
            channels_last: If True, images are in HWC format and if False in CHW format.

        Returns:
            Resized image with size ``src_size``.
        """
        return self.scale_to_size(image, self.src_size, channels_last)

    def pad_image(self, image: np.ndarray, channels_last: bool = True) -> np.ndarray:
        """
        Apply zero padding to an image to size ``dst_size``.

        Args:
            image: Image to be padded. Can be 3D or 4D tensor with channel dimension
                before or after spatial dimensions.
            channels_last: If True, images are in HWC format and if False in CHW format.

        Returns:
            Zero padded image of size ``dst_size``.
        """
        no_batch_dim = image.ndim == 3  # Image without batch dimension
        if no_batch_dim:
            image = image[np.newaxis]  # Add batch dimension
        if not channels_last:  # TF resize expects HWC format.
            image = np.transpose(image, (0, 2, 3, 1))

        # Pad shorter edge to model input size.
        pad_h = self.dst_size[0] - image.shape[1]
        pad_w = self.dst_size[1] - image.shape[2]
        if pad_h < 0 or pad_w < 0:
            raise ValueError("Cannot pad an image that is larger than dst_size.")

        image = np.pad(image, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])

        if not channels_last:
            image = np.transpose(image, (0, 3, 1, 2))
        if no_batch_dim:
            image = image[0]

        return image

    def scale_points(self, points: np.ndarray) -> np.ndarray:
        """
        Scale points by the same factor as the image.

        Args:
            points: Points to be scaled.

        Returns:
            Scaled points.
        """
        return self.scale * points

    def scale_boxes(self, boxes: np.ndarray) -> np.ndarray:
        """
        Scale bounding boxes by the same factor as the image.

        Args:
            boxes: Boxes to be scaled.

        Returns:
            Scaled boxes.
        """
        return self.scale * boxes

    def postprocess_mask(
        self, mask: np.ndarray, threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Convert an upscaled segmentation mask from ``dst_size`` back to ``src_size``
        by removing padding and unscaling.

        Args:
            mask: Segmentation mask, i.e., image with channels_first ordering, of size
                ``dst_size``. Should be a logit-mask, i.e., before thresholding.
            threshold: Optionally, we can apply thresholding after resizing to obtain
                a boolean mask.

        Returns:
            Mask of size ``src_size``.
        """
        mask = mask[..., : self.rescaled_size[0], : self.rescaled_size[1]]
        mask = self.unscale_image(mask, channels_last=False)
        if threshold is not None:
            mask = mask > threshold
        return mask
