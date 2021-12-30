from dataclasses import dataclass

from tfimm.models import create_model, create_preprocessing, model_config
from tfimm.train.registry import cfg_serializable


@dataclass
class ModelConfig:
    model_name: str
    pretrained: str
    input_size: tuple
    nb_channels: int
    nb_classes: int
    drop_rate: float = 0.0


@cfg_serializable
class ModelFactory:
    cfg_class = ModelConfig

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    def __call__(self):
        model = create_model(
            self.cfg.model_name,
            pretrained=self.cfg.pretrained,
            input_size=self.cfg.input_size,
            in_channels=self.cfg.nb_channels,
            nb_classes=self.cfg.nb_classes,
            drop_rate=self.cfg.drop_rate,
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
        if getattr(cfg, "fixed_input_size", False):
            input_shape = *self.cfg.input_size, self.cfg.nb_channels
        else:
            input_shape = (None, None, self.cfg.nb_channels)
        return input_shape
