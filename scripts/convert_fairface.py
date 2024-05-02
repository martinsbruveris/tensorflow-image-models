"""
Script to convert the FairFace PyTorch models to TensorFlow.

First, download the models from the official
[repository](https://github.com/dchen236/FairFace) into `models/`.

Then run this script.

The conversion is simple, because torchvision ResNet-34 models are compatible with
the models contained in tfimm.
"""
import numpy as np
import tensorflow as tf
import torch
import torchvision
import tf_keras

import tfimm


def convert_model(model_name: str, nb_classes: int):
    print(f"Converting model {model_name}.")
    pt_model_path = f"models/{model_name}.pt"
    print("Loading PyTorch model.")
    pt_state_dict = torch.load(pt_model_path, map_location="cpu")
    print("Creating TF model.")
    model = tfimm.create_model("resnet34", nb_classes=nb_classes)
    print("Transferring weights.")
    tfimm.utils.timm.load_pytorch_weights_in_tf2_model(model, pt_state_dict)
    print("Saving model.")
    model.save(f"models/{model_name}")
    print("Done.")


def test_conversion(model_name: str):
    print(f"Testing conversion of model {model_name}.")
    pt_model = torchvision.models.resnet34(pretrained=False)
    pt_model.fc = torch.nn.Linear(pt_model.fc.in_features, 18)
    pt_model_path = f"models/{model_name}.pt"
    pt_state_dict = torch.load(pt_model_path, map_location="cpu")
    pt_model.load_state_dict(pt_state_dict)
    pt_model.eval()

    tf_model = tf_keras.models.load_model(f"models/{model_name}")

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
    # Outputs
    #  - 0:7 race (white, black, latino hispanic, east asian, southeast asian,
    #       indian, middle eastern)
    #  - 7:9 gender (male, female)
    #  - 9:18 age (0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+)
    convert_model("res34_fair_align_multi_4_20190809", nb_classes=18)
    # Outputs
    #  - 0:4 race (white, black, asian, indian)
    # But model returns 18 classes for some reason...
    convert_model("res34_fair_align_multi_7_20190809", nb_classes=18)

    test_conversion("res34_fair_align_multi_4_20190809")
    test_conversion("res34_fair_align_multi_7_20190809")


if __name__ == "__main__":
    main()
