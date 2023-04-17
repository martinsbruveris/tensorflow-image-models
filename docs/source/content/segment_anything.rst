Segment Anything
================

.. py:module:: tfimm.architectures.segment_anything.sam

.. automodule:: tfimm.architectures.segment_anything.sam

.. autoclass:: SegmentAnythingModelConfig
.. autoclass:: SegmentAnythingModel
   :members: grid_size, mask_size, mask_threshold, dummy_inputs, call

.. py:module:: tfimm.architectures.segment_anything.predictor

.. autoclass:: SAMPredictor
   :members: set_image, clear_image, preprocess_masks, __call__
.. autoclass:: ImageResizer
   :members: scale_to_size, scale_image, unscale_image, pad_image, scale_points,
             scale_boxes, postprocess_mask
