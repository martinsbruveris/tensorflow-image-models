"""
Script to convert the Deepware scanner model from PyTorch to TensorFlow.

First, download the models from the official
[repository](https://github.com/deepware/deepfake-scanner) into `models/`.

Then run this script.
"""
import numpy as np
import tensorflow as tf
import timm
import torch

import tfimm


class EffNet(torch.nn.Module):
    def __init__(self, arch="b3"):
        super(EffNet, self).__init__()
        fc_size = {
            "b1": 1280,
            "b2": 1408,
            "b3": 1536,
            "b4": 1792,
            "b5": 2048,
            "b6": 2304,
            "b7": 2560,
        }
        assert arch in fc_size.keys()
        effnet_model = getattr(timm.models.efficientnet, "tf_efficientnet_%s_ns" % arch)
        self.encoder = effnet_model()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = torch.nn.Dropout(0.2)
        self.fc = torch.nn.Linear(fc_size[arch], 1)

    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


@tf.keras.utils.register_keras_serializable
class DeepwareEffnet(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = None
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.drop = tf.keras.layers.Dropout(rate=0.2)
        self.fc = tf.keras.layers.Dense(1, name="fc")

    def build(self, input_shape):
        self.encoder = tfimm.create_model("efficientnet_b7_ns", name="encoder")

    @property
    def dummy_inputs(self) -> tf.Tensor:
        return tf.zeros((1, 224, 224, 3))

    def call(self, x, training=False):
        x = self.encoder.forward_features(x, training=training)
        x = self.pool(x)
        x = self.drop(x, training=training)
        x = self.fc(x)
        return x


def convert_model(model_name: str):
    print(f"Converting model {model_name}.")
    pt_model_path = f"models/{model_name}.pt"
    print("Loading PyTorch model.")
    pt_state_dict = torch.load(pt_model_path, map_location="cpu")
    print("Creating TF model.")
    model = DeepwareEffnet()
    print("Transferring weights.")
    tfimm.utils.timm.load_pytorch_weights_in_tf2_model(model, pt_state_dict)
    print("Saving model.")
    model.save(f"models/{model_name}")
    print("Done.")


def test_conversion(model_name: str):
    print(f"Testing conversion of model {model_name}.")
    pt_model = EffNet(arch="b7")
    pt_model_path = f"models/{model_name}.pt"
    pt_state_dict = torch.load(pt_model_path, map_location="cpu")
    pt_model.load_state_dict(pt_state_dict)
    pt_model.eval()

    tf_model = tf.keras.models.load_model(f"models/{model_name}")

    img = np.random.rand(5, 224, 224, 3).astype("float32")

    pt_img = torch.Tensor(img.transpose([0, 3, 1, 2]))
    pt_res = pt_model.forward(pt_img)
    pt_res = pt_res.detach().numpy()

    tf_img = tf.constant(img)
    tf_res = tf_model(tf_img, training=False)
    tf_res = tf_res.numpy()

    print(np.max(np.abs(tf_res - pt_res)))
    print(np.max(np.abs(pt_res)))


def main():
    convert_model("deepware")
    test_conversion("deepware")


if __name__ == "__main__":
    main()
