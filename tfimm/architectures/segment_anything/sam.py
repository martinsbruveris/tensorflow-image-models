"""
We provide an implementation and pretrained weights for the Segment Anything Models.

Paper: Segment Anything.
`[arXiv:2304.02643] <https://arxiv.org/abs/2304.02643>`_.

Original pytorch code and weights from
`Facebook Research <https://github.com/facebookresearch/segment-anything>`_.

The following models are available.

* ``sam_vit_b``
* ``sam_vit_l``
* ``sam_vit_h``

In the code we are trying to follow this convention in comments and docstrings.

* N is the batch dimension
* (H0, W0) is the dimension of the input image to ``SAMPRedictor``. There are no
  constraints to this size as the image will be resized and padded to the model input
  dimensions.
* (H, W) is the model input size. This is (1024, 1024) for the pretrained models.
* (H', W') is the input size for mask prompts and the output size for predicted
  mask logits. For the pretrained models this is (256, 256). To be precise, this is
  calculated as H'=4*H'' and same for W'.
* (H'', W'') is the spatial dimension of image embeddings. For pretrained models
  this is (64, 64). This is calculated as H''=H/patch_size with patch_size=16.
* C is the number of image input channels. Usually C=3.
* M1 is the number of point prompts given to the model.
* M2 is the number of box prompts given to the model. The PyTorch code only
  supports M2={0, 1}, so the accuracy with multiple box prompts might be
  limited.
* M3 is the number of mask prompts given to the model. The PyTorch code only
  supports M3={0, 1}, so the accuracy with multiple mask prompts might be limited.
* M is the number of tokens in the sparse embeddings returned by the prompt
  embedder. The number depends on M1 and M2.
* D is the embedding dimension, which is shared by both image, sparse and dense
  prompt embeddings. For the pretrained models this is 256.
* K is the number of masks returned by the model. This number is controlled by
  the model parameter ``nb_multimask_outputs`` (set to 3 in pretrained models). And
  also by the parameter ``multimask_output`` when calling ``SAMPredictor``.
"""
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple

import tensorflow as tf

from tfimm.models import ModelConfig, keras_serializable, register_model
from tfimm.utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from .image_encoder import ImageEncoder
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .transformer import TwoWayTransformer

# Model registry will add each entrypoint fn to this
__all__ = ["SegmentAnythingModel", "SegmentAnythingModelConfig"]


@dataclass
class SegmentAnythingModelConfig(ModelConfig):
    """
    Configuration class for SAM models.

    Parameters:
        name: Name of the model.
        url: URL for pretrained weights.
        in_channels: Number of input image channels.
        input_size: Input image size (height, width)
        fixed_input_size: If True, the model only accepts inputs of size ``input_size``.
            If False, the models accepts arbitrary input sizes by interpolating the
            positional encodings to account for the new input size.
        embed_dim: The shared embedding dimension of image and prompt embeddings.
        nb_multimask_outputs: Number of masks predicted by the model for each prompt.
        mask_threshold: Threshold for thresholding mask logits to a boolean mask.

        encoder_patch_size: Patchifying the image is implemented via a convolutional
            layer with kernel size and stride equal to ``patch_size``.
        encoder_embed_dim: Feature dimensions at each stage. These are hidden feature
            dimensions. The output dimension (which has to be compatible with the
            prompt embedding dimension) is given by ``embed_dim``.
        encoder_nb_blocks: Number of attention blocks in the image encoder.
        encoder_nb_heads: Number of self-attention heads in the image encoder.
        encoder_mlp_ratio: Ratio of mlp hidden dim to embedding dim

        encoder_drop_rate: Dropout rate
        encoder_attn_drop_rate: Attention dropout rate
        encoder_drop_path_rate: Dropout rate for stochastic depth

        encoder_norm_layer: Normalization layer. See :func:`~norm_layer_factory` for
            possible values.
        encoder_act_layer: Activation function. See :func:`~act_layer_factory` for
            possible values.
        encoder_qkv_bias: If True, add bias for qkv projection layers.
        encoder_global_attn_indices: Indexes for blocks using global attention. All
            other blocks use window attention.
        encoder_window_size: Window size for window attention blocks.

        prompt_mask_hidden_dim: Hidden dimension in the mask encoder network.

        decoder_nb_blocks: Number of attention blocks in the mask decoder.
        decoder_nb_heads: Number of self-attention heads in the mask decoder.
        decoder_mlp_channels: Number of channels in mlp layers.
        decoder_iou_head_depth: Number of layers in score predictor network.
        decoder_iou_hidden_dim: Number of hidden dimensions in score predictor network.

        mean: Defines preprocessing function. If ``x`` is an image with pixel values
            in (0, 1), the preprocessing function is ``(x - mean) / std``.
        std: Defines preprpocessing function.

        first_conv: Name of first convolutional layer. Used by
            :func:`~tfimm.create_model` to adapt the number in input channels when
            loading pretrained weights.
    """

    in_channels: int = 3
    input_size: Tuple[int, int] = (1024, 1024)
    fixed_input_size: bool = True
    embed_dim: int = 256
    nb_multimask_outputs: int = 3
    mask_threshold: float = 0.0

    # Image encoder
    encoder_patch_size: int = 16
    encoder_embed_dim: int = 768
    encoder_nb_blocks: int = 12
    encoder_nb_heads: int = 12
    encoder_mlp_ratio: float = 4.0

    encoder_drop_rate: float = 0.0
    encoder_attn_drop_rate: float = 0.0
    encoder_drop_path_rate: float = 0.0

    encoder_norm_layer: str = "layer_norm_eps_1e-6"
    encoder_act_layer: str = "gelu"
    encoder_qkv_bias: bool = True
    encoder_global_attn_indices: Tuple = (2, 5, 8, 11)
    encoder_window_size: int = 14

    # Prompt encoder
    prompt_mask_hidden_dim: int = 16

    # Mask decoder
    decoder_nb_blocks: int = 2
    decoder_nb_heads: int = 8
    decoder_mlp_channels: int = 2048
    decoder_iou_head_depth: int = 3
    decoder_iou_hidden_dim: int = 256

    # Preprocessing
    mean: Tuple[float, float, float] = IMAGENET_DEFAULT_MEAN
    std: Tuple[float, float, float] = IMAGENET_DEFAULT_STD

    # Weight transfer
    first_conv: str = "image_encoder/patch_embed/proj"

    @property
    def transform_weights(self):
        """
        Returns a dictionary of weights that need a custom transform function when
        loading to a model with a different input size. Dictionary contains weights
        and the corresponding transform function.
        """
        transforms = {"image_encoder/pos_embed": transform_pos_embed}
        # Only attention blocks that use global attention need to be transformed. For
        # the other blocks, rel_pos depends on the window size, which we assume is
        # constant
        for j in self.encoder_global_attn_indices:
            prefix = f"image_encoder/blocks/{j}/attn/rel_pos"
            transforms[prefix + "_h"] = partial(transform_rel_pos, axis=0)
            transforms[prefix + "_w"] = partial(transform_rel_pos, axis=1)
        return transforms


def transform_rel_pos(
    model, rel_pos, target_cfg: SegmentAnythingModelConfig, axis: int
):
    """
    Transform function to adapt the relative positional encodings from the attention
    blocks in the image encoder between image resolutions.
    """
    grid_dim = target_cfg.input_size[axis] // target_cfg.encoder_patch_size
    new_size = 2 * grid_dim - 1

    # rel_pos has shape (L, D), but tf.image.resize needs a 3D tensor.
    rel_pos = tf.expand_dims(rel_pos, axis=0)
    rel_pos = tf.image.resize(rel_pos, size=(1, new_size), method="bilinear")
    rel_pos = rel_pos[0]
    return rel_pos


def transform_pos_embed(model, pos_embed, target_cfg: SegmentAnythingModelConfig):
    """
    Transform function to adapt the absolute positional encodings in the image encoder
    between image resolutions.
    """
    grid_size = (
        target_cfg.input_size[0] // target_cfg.encoder_patch_size,
        target_cfg.input_size[1] // target_cfg.encoder_patch_size,
    )
    pos_embed = tf.image.resize(pos_embed, size=grid_size, method="bilinear")
    return pos_embed


@keras_serializable
class SegmentAnythingModel(tf.keras.Model):
    cfg_class = SegmentAnythingModelConfig

    def __init__(self, cfg: SegmentAnythingModelConfig, *args, **kwargs):
        kwargs["name"] = kwargs.get("name", cfg.name)
        super().__init__(*args, **kwargs)
        self.cfg = cfg

        self.image_encoder = ImageEncoder(
            input_size=cfg.input_size,
            fixed_input_size=cfg.fixed_input_size,
            patch_size=cfg.encoder_patch_size,
            in_channels=cfg.in_channels,
            embed_dim=cfg.encoder_embed_dim,
            nb_blocks=cfg.encoder_nb_blocks,
            nb_heads=cfg.encoder_nb_heads,
            mlp_ratio=cfg.encoder_mlp_ratio,
            out_channels=cfg.embed_dim,
            qkv_bias=cfg.encoder_qkv_bias,
            norm_layer=cfg.encoder_norm_layer,
            act_layer=cfg.encoder_act_layer,
            use_abs_pos=True,
            use_rel_pos=True,
            window_size=cfg.encoder_window_size,
            global_attn_indices=cfg.encoder_global_attn_indices,
            drop_rate=cfg.encoder_drop_rate,
            attn_drop_rate=cfg.encoder_attn_drop_rate,
            drop_path_rate=cfg.encoder_drop_path_rate,
            name="image_encoder",
        )
        self.prompt_encoder = PromptEncoder(
            embed_dim=cfg.embed_dim,
            mask_hidden_dim=cfg.prompt_mask_hidden_dim,
            act_layer="gelu",
            name="prompt_encoder",
        )
        self.mask_decoder = MaskDecoder(
            transformer=TwoWayTransformer(
                embed_dim=cfg.embed_dim,
                nb_blocks=cfg.decoder_nb_blocks,
                nb_heads=cfg.decoder_nb_heads,
                mlp_dim=cfg.decoder_mlp_channels,
                attention_downsample_rate=2,
                act_layer="relu",
                name="transformer",
            ),
            embed_dim=cfg.embed_dim,
            nb_multimask_outputs=cfg.nb_multimask_outputs,
            iou_head_depth=cfg.decoder_iou_head_depth,
            iou_head_hidden_dim=cfg.decoder_iou_hidden_dim,
            act_layer="gelu",
            name="mask_decoder",
        )

    def grid_size(
        self, input_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[int, int]:
        """
        Compute the grid size of image embeddings for the model given an image input
        size.

        Args:
            input_size: Image input size (H, W). If not provided use the input size
                from the model config.

        Returns:
            Spatial size of image embeddings (H'', W'').
        """
        input_size = input_size or self.cfg.input_size
        return (
            input_size[0] // self.cfg.encoder_patch_size,
            input_size[1] // self.cfg.encoder_patch_size,
        )

    def mask_size(
        self, input_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[int, int]:
        """
        Compute the size of (low res) masks for the model given an image input size.

        Args:
            input_size: Image input size (H, W). If not provided use the input size
                from the model config.

        Returns:
            Spatial size of low resolution masks (H', W').
        """
        grid_size = self.grid_size(input_size)
        return 4 * grid_size[0], 4 * grid_size[1]

    @property
    def mask_threshold(self):
        """Threshold for thresholding logit masks to boolean masks."""
        return self.cfg.mask_threshold

    @property
    def dummy_inputs(self):
        """Returns a (nested) tensor of the correct shape for inference."""
        inputs = {
            "images": tf.zeros((1, *self.cfg.input_size, self.cfg.in_channels)),
            "points": tf.zeros((1, 1, 2)),
            "labels": tf.zeros((1, 1)),
            "boxes": tf.zeros((1, 1, 4)),
            "masks": tf.zeros((1, 1, *self.mask_size(self.cfg.input_size))),
        }
        return inputs

    def get_image_pe(self, image_embeddings: tf.Tensor) -> tf.Tensor:
        """
        Returns image positional encodings compatible with the given image embeddings.

        Args:
            image_embeddings: Image embeddings returned by the image encoder.

        Returns:
            Image positional encoder to be passed to the mask decoder.
        """
        n, h, w, _ = tf.unstack(tf.shape(image_embeddings))
        image_pe = self.prompt_encoder.get_dense_pe((h, w))  # (H'', W'', D)
        image_pe = tf.expand_dims(image_pe, axis=0)  # (1, H'', W'', D)
        image_pe = tf.tile(image_pe, (n, 1, 1, 1))  # (N, H'', W'', D)
        return image_pe

    def postprocess_logits(self, logits, input_size, return_logits: bool):
        """
        Upscales and optionally thresholds logits returned from the mask decoder to
        segmentation masks.

        Args:
            logits: Low-resolution logits returned by the mask decoder.
            input_size: Image input size (H, W).
            return_logits: If True, we don't apply a threshold.

        Returns:
            Segmentation mask (thresholded or not) of size (H, W).
        """
        _, _, h, w = tf.unstack(tf.shape(logits))  # (N, K, H', W')
        masks = tf.transpose(logits, (0, 2, 3, 1))  # (N, H', W', K)
        masks = tf.image.resize(
            masks, size=input_size, method=tf.image.ResizeMethod.BILINEAR
        )  # (N, H, W, K)
        masks = tf.transpose(masks, (0, 3, 1, 2))  # (N, K, H, W)
        if not return_logits:
            masks = masks > self.mask_threshold
        return masks

    # TODO: Add return_features
    def call(self, inputs, training=False, multimask_output=False, return_logits=False):
        """
        Predicts masks end-to-end from provided images and prompts. If prompts are not
        known in advance, using SamPredictor is recommended over calling the model
        directly.

        Args:
            inputs: A dictionary with the following entries

             * images: An (N, H, W, C) tensor of preprocessed input images.
             * points: An (N, M1, 2) tensor of point prompts with coordinates in pixel
               space, i.e., values between 0 and H or W.
             * labels: An (N, M1) tensor of labels for point prompts. 1 indicates a
               foreground point and 0 indicates a background point.
             * boxes: An (N, M2, 4) tensor of box prompts of form (left, top, right,
               bottom) with coordinates in pixel space.
             * masks: An (N, M3, H', W') tensor of mask inputs, where M3 is
               either 1 or 0 (no mask provided).

            training: Training or inference phase?
            multimask_output: If True, we return multiple nested masks for each prompt.
            return_logits: If True, we don't threshold the upscaled mask. This is useful
                if we want to resize the mask back to original image size first and
                then apply the threshold.

        Returns:

            * Masks, an (N, K, H, W) bool tensor of binary masked predictions, where K
              is determined by the multimask_output parameter.
            * Scores, an (N, K) tensor with the model's predictions of mask quality.
            * Logits, an (N, K, H', W') tensor with low resoulution logits, where
              usually H'=H/4 and W'=W/4. This can be passed as mask input to subsequent
              iterations of prediction.
        """
        # Shape (N, H'', W'', D), where H'' = H / 16 (grid size).
        image_embeddings = self.image_encoder(inputs["images"], training=training)

        # Sparse shape: (N, M, D); Dense shape: (N, H'', W'', D).
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            inputs={
                "points": inputs["points"],
                "labels": inputs["labels"],
                "boxes": inputs["boxes"],
                "masks": inputs["masks"],
            },
            training=training,
        )

        # Logits shape: (N, K, H', W'); Scores shape: (N, K).
        logits, scores = self.mask_decoder(
            inputs={
                "image_embeddings": image_embeddings,
                "image_pe": self.get_image_pe(image_embeddings),
                "sparse_embeddings": sparse_embeddings,
                "dense_embeddings": dense_embeddings,
            },
            training=training,
            multimask_output=multimask_output,
        )

        masks = self.postprocess_logits(
            logits,
            input_size=tf.shape(inputs["images"])[1:3],
            return_logits=return_logits,
        )
        return masks, scores, logits


@register_model
def sam_vit_b():
    """SAM ViT-Base"""
    cfg = SegmentAnythingModelConfig(
        name="sam_vit_b",
        url=(
            "[pytorch]https://dl.fbaipublicfiles.com/segment_anything/"
            "sam_vit_b_01ec64.pth"
        ),
        encoder_embed_dim=768,
        encoder_nb_blocks=12,
        encoder_nb_heads=12,
        encoder_global_attn_indices=(2, 5, 8, 11),
    )
    return SegmentAnythingModel, cfg


@register_model
def sam_vit_l():
    """SAM ViT-Large"""
    cfg = SegmentAnythingModelConfig(
        name="sam_vit_l",
        url=(
            "[pytorch]https://dl.fbaipublicfiles.com/segment_anything/"
            "sam_vit_l_0b3195.pth"
        ),
        encoder_embed_dim=1024,
        encoder_nb_blocks=24,
        encoder_nb_heads=16,
        encoder_global_attn_indices=(5, 11, 17, 23),
    )
    return SegmentAnythingModel, cfg


@register_model
def sam_vit_h():
    """SAM ViT-Huge"""
    cfg = SegmentAnythingModelConfig(
        name="sam_vit_h",
        url=(
            "[pytorch]https://dl.fbaipublicfiles.com/segment_anything/"
            "sam_vit_h_4b8939.pth"
        ),
        encoder_embed_dim=1280,
        encoder_nb_blocks=32,
        encoder_nb_heads=16,
        encoder_global_attn_indices=(7, 15, 23, 31),
    )
    return SegmentAnythingModel, cfg
