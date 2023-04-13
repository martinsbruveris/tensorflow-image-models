"""
Script to convert Meta's Segment Anything Models from PyTorch to TensorFlow
"""
import numpy as np
import tensorflow as tf
import torch
from torch.hub import load_state_dict_from_url

import tfimm
import tfimm.architectures.segment_anything.torch as pt_sam

# # Look at names of weights
# state_dict = load_state_dict_from_url(
#     url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"  # noqa: E501
# )
# for k in state_dict.keys():
#     print(k)


image = 255 * np.random.rand(1, 1024, 1024, 3).astype(np.float32)
points = np.random.rand(1, 3, 2).astype(np.float32)
labels = np.asarray([[1, 1, 0]]).astype(np.int32)
boxes = np.random.rand(1, 1, 4).astype(np.float32)
masks = np.random.rand(1, 1, 256, 256).astype(np.float32)
pt_inputs = {
    "image": torch.Tensor(image)[0].permute((2, 0, 1)),
    "point_coords": torch.Tensor(points),
    "point_labels": torch.Tensor(labels),
    "boxes": torch.Tensor(boxes),
    "mask_inputs": torch.Tensor(masks),
    "original_size": (1024, 1024),
}
tf_inputs = {
    "images": tf.constant(image),
    "points": tf.constant(points),
    "labels": tf.constant(labels),
    "boxes": tf.constant(boxes),
    "masks": tf.constant(masks),
}

pt_model = pt_sam.build_sam_vit_b(checkpoint="sam_vit_b_01ec64.pth")

# # Testing the user-friendly predictor functionality...
# predictor = pt_sam.predictor.SamPredictor(sam_model=pt_model)
# predictor.set_image(img.astype(np.uint8))
# predictor.predict(
#     point_coords=np.random.rand(3, 2),
#     point_labels=np.asarray([1, 1, 0]),
#     box=np.random.rand(4),
#     mask_input=np.random.rand(1, 256, 256),
#     multimask_output=True,
#     return_logits=True,
# )

pt_res = pt_model(batched_input=[pt_inputs], multimask_output=True)
# pt_res = pt_res.permute((0, 2, 3, 1))
# print(pt_res.shape)
# pt_res = pt_res.detach().numpy()

pt_masks = pt_res[0]["masks"]
pt_quality = pt_res[0]["iou_predictions"]
pt_logits = pt_res[0]["low_res_logits"]

pt_masks = pt_masks.detach().numpy()
pt_quality = pt_quality.detach().numpy()
pt_logits = pt_logits.detach().numpy()

print(pt_masks.shape)
print(pt_quality.shape)
print(pt_logits.shape)

tf_model = tfimm.create_model("sam_vit_b", pretrained=True)
# PyTorch model does preprocessing in the forward function.
tf_inputs["images"] = tf_model.preprocess(tf_inputs["images"])
tf_res = tf_model(tf_inputs, training=False, multimask_output=True)
# print(tf_res.shape)
# tf_res = tf_res.numpy()
tf_masks, tf_quality, tf_logits = tf_res

tf_masks = tf_masks.numpy()
tf_quality = tf_quality.numpy()
tf_logits = tf_logits.numpy()

print(tf_masks.shape)
print(tf_quality.shape)
print(tf_logits.shape)

# print(np.max(np.abs(tf_res - pt_res)))
# print(np.max(np.abs(pt_res)))

print(np.sum(tf_masks != pt_masks))
print(np.max(np.abs(tf_quality - pt_quality)))
print(np.max(np.abs(pt_quality)))
print(np.max(np.abs(tf_logits - pt_logits)))
print(np.max(np.abs(pt_logits)))
