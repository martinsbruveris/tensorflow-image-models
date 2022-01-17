from dataclasses import dataclass

import tensorflow as tf

from tfimm.models import (
    EmbeddingModel,
    create_model,
    create_preprocessing,
    model_config,
)

from .registry import cfg_serializable


@dataclass
class ModelConfig:
    model_name: str
    pretrained: str = ""
    model_path: str = ""
    input_size: tuple = ()
    in_channels: int = -1
    nb_classes: int = -1
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    attn_drop_rate: float = 0.0


@cfg_serializable
class ModelFactory:
    """Wrapper around `tfimm` models to be used by the training framework."""

    cfg_class = ModelConfig

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    def __call__(self):
        # fmt: off
        kwargs = {}
        # We only pass arguments to `create_model`, if they take non-default values.
        for arg, default in [
            ("input_size", ()), ("in_channels", -1), ("nb_classes", -1),
            ("drop_rate", 0.0), ("drop_path_rate", 0.0), ("attn_drop_rate", 0.0)
        ]:
            if getattr(self.cfg, arg) != default:
                kwargs[arg] = getattr(self.cfg, arg)
        # fmt: on

        model = create_model(
            self.cfg.model_name,
            pretrained=self.cfg.pretrained,
            model_path=self.cfg.model_path,
            **kwargs,
        )
        preprocessing = create_preprocessing(self.cfg.model_name)
        return model, preprocessing

    @property
    def tf_input_shape(self):
        """
        Returns the input shape to be used for keras.Input layers. Some models, e.g.,
        ResNets, can operate on arbitrary sized inputs, others, e.g., transformers,
        need fixed-sized inputs.
        """
        cfg = model_config(self.cfg.model_name)
        if not getattr(cfg, "interpolate_input", True):
            input_shape = *self.cfg.input_size, self.cfg.in_channels
        else:
            input_shape = (None, None, self.cfg.in_channels)
        return input_shape


@dataclass
class SavedModelConfig:
    # Path to where the model is saved.
    path: str
    # Preprocessing function will return a tensor of this dtype. If not provided we
    # default to using `tf.keras.backend.floatx()`.
    dtype: str = ""
    # Preprocessing will subtract the mean and divide by the standard deviation, i.e.,
    # `img = (img - mean) / std`. With the default values preprocessing remains the
    # identity.
    mean: tuple = (0.0, 0.0, 0.0)
    std: tuple = (1.0, 1.0, 1.0)


@cfg_serializable
class SavedModel:
    """
    Class provides a wrapper to a model saved on disk, either as an h5 file or as a
    saved model.
    """

    cfg_class = SavedModelConfig

    def __init__(self, cfg: SavedModelConfig):
        self.cfg = cfg

    def __call__(self):
        model_path = str(self.cfg.path)
        # First we try loading the model as a `keras.Model`
        try:
            model = tf.keras.models.load_model(model_path)
        except ValueError:
            # If that doesn't work we try the saved model format.
            model = tf.saved_model.load(model_path)
        dtype = self.cfg.dtype or tf.keras.backend.floatx()

        def _preprocess(img: tf.Tensor) -> tf.Tensor:
            img = tf.cast(img, dtype=dtype)
            img = (img - self.cfg.mean) / self.cfg.std
            return img

        return model, _preprocess


@dataclass
class EmbeddingModelConfig:
    backbone_name: str
    backbone_pretrained: str = ""
    backbone_model_path: str = ""
    input_size: tuple = -1
    in_channels: int = -1
    embed_dim: int = 512
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    attn_drop_rate: float = 0.0


@cfg_serializable
class EmbeddingModelFactory:
    """
    Creates a model to be used for embedding learning problems such as face recognition.
    """

    cfg_class = EmbeddingModelConfig

    def __init__(self, cfg: EmbeddingModelConfig):
        self.cfg = cfg

    def __call__(self):
        # fmt: off
        kwargs = {}
        # We only pass arguments to `create_model`, if they take non-default values.
        for arg, default in [
            ("input_size", ()), ("in_channels", -1), ("drop_rate", 0.0),
            ("drop_path_rate", 0.0), ("attn_drop_rate", 0.0)
        ]:
            if getattr(self.cfg, arg) != default:
                kwargs[arg] = getattr(self.cfg, arg)
        # fmt: on

        backbone = create_model(
            self.cfg.backbone_name,
            pretrained=self.cfg.backbone_pretrained,
            model_path=self.cfg.backbone_model_path,
            nb_classes=0,
            **kwargs,
        )
        model = EmbeddingModel(backbone=backbone, embed_dim=self.cfg.embed_dim)
        preprocessing = create_preprocessing(self.cfg.backbone_name)
        return model, preprocessing

    @property
    def tf_input_shape(self):
        """
        Returns the input shape to be used for keras.Input layers. Some models, e.g.,
        ResNets, can operate on arbitrary sized inputs, others, e.g., transformers,
        need fixed-sized inputs.
        """
        cfg = model_config(self.cfg.backbone_name)
        if not getattr(cfg, "interpolate_input", True):
            input_shape = *self.cfg.input_size, self.cfg.in_channels
        else:
            input_shape = (None, None, self.cfg.in_channels)
        return input_shape
