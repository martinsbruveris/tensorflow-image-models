from tfimm.models.factory import create_model, transfer_weights


def test_transform_pos_embed():
    """
    We test if we can transfer weights between ViT models with different input sizes,
    which requires interpolation of position embeddings during weight transfer. This
    should be done via the transfer_weight hook in the config.
    """
    model_name = "vit_tiny_patch16_224"
    src_model = create_model(model_name)
    dst_model = create_model(model_name, input_size=(112, 112))
    transfer_weights(src_model, dst_model)

    img = dst_model.dummy_inputs
    dst_model(img)
