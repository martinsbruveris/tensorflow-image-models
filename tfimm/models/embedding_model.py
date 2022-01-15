import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class EmbeddingModel(tf.keras.Model):
    """
    This model can be used for embedding learning tasks. It adds to any TFIMM backbone
    a fully connected layer without bias followed by batch norm. Possible applications
    include face recognition.
    """

    def __init__(self, backbone: tf.keras.Model, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.embed_dim = embed_dim

        self.fc = tf.keras.layers.Dense(embed_dim, name="emb/fc")
        self.bn = tf.keras.layers.BatchNormalization(
            axis=-1, scale=False, name="emb/bn"
        )

    @property
    def dummy_inputs(self) -> tf.Tensor:
        return self.backbone.dummy_inputs

    def call(self, x, training=False, return_features=False):
        features = {}
        x = self.backbone.forward_features(x, training, return_features)
        if return_features:
            x, features = x
        x = self.fc(x)
        x = self.bn(x, training=training)
        features["embeddings"] = x
        return (x, features) if return_features else x

    def get_config(self):
        return {
            "backbone": tf.keras.utils.serialize_keras_object(self.backbone),
            "embed_dim": self.embed_dim,
        }

    @classmethod
    def from_config(cls, config):
        return EmbeddingModel(
            backbone=tf.keras.models.model_from_config(config["backbone"]),
            embed_dim=config["embed_dim"],
        )
