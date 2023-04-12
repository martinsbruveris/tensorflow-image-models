from dataclasses import dataclass
from typing import List, Tuple

import tensorflow as tf

from tfimm.models import ModelConfig, keras_serializable, register_model
from tfimm.utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from .image_encoder import ImageEncoder

# model_registry will add each entrypoint fn to this
__all__ = ["SegmentAnythingModel", "SegmentAnythingModelConfig"]


# TODO: Add dropout parameters to config
# TODO: Fix documentation of parameters
@dataclass
class SegmentAnythingModelConfig(ModelConfig):
    """
    Configuration class for SAM models.

    Parameters:
        name: Name of the model.
        url: URL for pretrained weights.
        in_channels: Number of input image channels.
        input_size: Input image size (height, width)

        patch_size: Patchifying the image is implemented via a convolutional layer with
            kernel size and stride equal to ``patch_size``.
        embed_dim: Feature dimensions at each stage.
        nb_blocks: Number of blocks at each stage.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim
        conv_mlp_block: There are two equivalent implementations of the ConvNeXt block,
            using either (1) 1x1 convolutions or (2) fully connected layers. In PyTorch
            option (2) also requires permuting channels, which is not needed in
            TensorFlow. We offer both implementations here, because some ``timm`` models
            use (1) while others use (2).

        drop_rate: Dropout rate.
        drop_path_rate: Dropout rate for stochastic depth.

        norm_layer: Normalization layer. See :func:`~norm_layer_factory` for possible
            values.
        act_layer: Activation function. See :func:`~act_layer_factory` for possible
            values.
        init_scale: Inital value for layer scale weights.

        crop_pct: Crop percentage for ImageNet evaluation.
        interpolation: Interpolation method for ImageNet evaluation.
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
    # Image encoder
    encoder_patch_size: int = 16
    encoder_embed_dim: int = 768
    encoder_nb_blocks: int = 12
    encoder_nb_heads: int = 12
    encoder_mlp_ratio: float = 4.0
    encoder_norm_layer: str = "layer_norm_eps_1e-6"
    encoder_act_layer: str = "gelu"
    encoder_qkv_bias: bool = True
    encoder_global_attn_indices: Tuple = (2, 5, 8, 11)
    encoder_window_size: int = 14
    # Prompt encoder
    prompt_embed_dim: int = 256
    prompt_mask_in_channels: int = 16
    # Mask decoder
    decoder_nb_multimask_outputs: int = 3
    decoder_nb_blocks: int = 2
    decoder_nb_heads: int = 8
    decoder_mlp_channels: int = 2048
    decoder_iou_head_depth: int = 3
    decoder_iou_head_channels: int = 256
    # Preprocessing
    mean: Tuple[float, float, float] = IMAGENET_DEFAULT_MEAN
    std: Tuple[float, float, float] = IMAGENET_DEFAULT_STD
    # Weight transfer
    first_conv: str = "stem/0"
    classifier: str = "head/fc"


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
            out_channels=cfg.prompt_embed_dim,
            qkv_bias=cfg.encoder_qkv_bias,
            norm_layer=cfg.encoder_norm_layer,
            act_layer=cfg.encoder_act_layer,
            use_abs_pos=True,
            use_rel_pos=True,
            window_size=cfg.encoder_window_size,
            global_attn_indices=cfg.encoder_global_attn_indices,
        )

    @property
    def dummy_inputs(self):
        inputs = {
            "images": tf.zeros((1, *self.cfg.input_size, self.cfg.in_channels)),
            "original_size": tf.convert_to_tensor([self.cfg.input_size]),
            "point_coords": tf.zeros((1, 0, 2)),
            "point_labels": tf.zeros((1, 0)),
            "boxes": tf.zeros((1, 0, 4)),
            "mask_inputs": tf.zeros((1, 0, *self.cfg.input_size)),
        }
        return inputs

    # TODO: Add all other input bits to preprocessing
    def preprocess(self, img, dtype=None):
        """Normalize pixel values and pad to a square input."""
        if dtype is not None:
            img = tf.cast(img, dtype)

        img = img / 255.0
        # TODO: Adapt mean and std to number of channels
        img = (img - self.cfg.mean) / self.cfg.std

        h, w = tf.unstack(tf.shape(img)[1:3])
        pad_h = self.cfg.input_size[0] - h
        pad_w = self.cfg.input_size[1] - w
        img = tf.pad(img, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])

        return img

    # TODO: Add return_features
    def call(self, inputs, training=False, multimask_output=False):
        """
        Predicts masks end-to-end from provided images and prompts. If prompts are not
        known in advance, using SamPredictor is recommended over calling the model
        directly.

        Args:
            inputs: A dictionary with the following entries
                images: An (N, H, W, C) tensor of preprocessed input images.
                original_size: An (N, 2) tensor of original image sizes before padding.
                point_coords: An (N, B1, 2) tensor of point prompts.
                point_labels: An (N, B1) tensor of labels for point prompts
                boxes: An (N, B2, 4) tensor of box prompts, transformed to the input
                    frame of the model.
                mask_inputs: An (N, B3, H, W) tensor of mask inputs.
            training: Training or inference phase?
            multimask_output: If True, we return multiple nested masks for each prompt.

        Returns:
            masks: An (N, B, C, H, W) tensor of binary masked predictions, where
                B=B1+B2+B3 is the total number of input prompts, and C is determined
                by the multimask_output parameter.
            iou_predictions: An (N, B, C) tensor with the model's predictions of mask
                quality.
            low_res_logits: An (N, B, C, H', W') tensor with low resoulution logits,
                with H'=W'=256 by default. This can be passed as mask input to
                subsequent iterations of prediction.
        """
        # TODO: Customize resolution of low_res_logits
        img = inputs["images"]
        image_embeddings = self.image_encoder(img, training=training)

        return image_embeddings, None, None


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
