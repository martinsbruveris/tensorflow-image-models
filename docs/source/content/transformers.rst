Transformer Models
==================

Here we describe some features that are common to most transformer models.

Changing input shape
--------------------

Most parts of a transformer architecture are independent of input resolution. Changing
the input resolution results in a different number of patches. The projection,
self-attention and MLP layers work on arbitrary length inputs. The only part that needs
to be adapted are the position embeddings.

Position embeddings can adjusted via 2D interpolation to the new input resolution.
However, since position embeddings are learnt, after interpolation they may no longer
be meaningful. Thus, by default, transformer models can only run inference at the
resolution specified by ``input_size``.

If we want to fine-tune a model at a different resolution, we can specify the new
resolution when creating the model. In that case, the position embeddings will be
interpolated for the new resolution.

.. code-block:: python

  # Default model with `input_size=(224, 224)`
  model_224 = create_model("vit_tiny_patch16_224")

  # Model with interpolated position embeddings
  model_384 = create_model("vit_tiny_patch16_224", input_size=(384, 384))


Transforming weights
--------------------

Internally adjusting model input size is done via the ``transform_weights`` field in
the config. The field ``transform_weights`` is a dictionary of the form

.. code-block:: python

  cfg.transform_weights = {"pos_embed": ViT.transform}

The function ``ViT.transform`` is called as ``src_model.transform(tgt_cfg)`` and returns
the corresponding weight transformed for ``tgt_cfg``.

Inference at arbitrary resolution
---------------------------------

We can enable inference at arbitrary resolution, by setting the parameter
``interpolate_input=True`` when constructing the model.

.. code-block:: python

   model = create_model("vit_tiny_patch16_224", interpolate_input=True)
   logits = model(np.zeros((1, 256, 256, 3), dtype="float32"))

To avoid accidental inference at the wrong resolution, the default is
``interpolate_input=False``.
