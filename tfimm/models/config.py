from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Shared base class for model configurations."""

    name: str = ""
    url: str = ""
