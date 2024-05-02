import tempfile

import numpy as np
import tf_keras

from tfimm.models import EmbeddingModel
from tfimm.models.factory import create_model

from . import architectures  # noqa: F401


def test_save_load_model():
    """Tests ability to use keras save() and load() functions."""
    backbone = create_model("resnet_test_model_1")
    model = EmbeddingModel(backbone=backbone, embed_dim=32)
    model(model.dummy_inputs)

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save(tmpdir)
        loaded_model = tf_keras.models.load_model(tmpdir)

    assert type(model) is type(loaded_model)

    img = np.random.rand(1, *backbone.cfg.input_size, backbone.cfg.in_channels)
    y_1 = model(img).numpy()
    y_2 = loaded_model(img).numpy()

    assert (np.max(np.abs(y_1 - y_2))) < 1e-6
