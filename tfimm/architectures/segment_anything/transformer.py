import tensorflow as tf

from tfimm.layers import norm_layer_factory

from .common import MLPBlock


class TwoWayTransformer(tf.keras.Model):
    def __init__(
        self,
        embed_dim: int,
        nb_blocks: int,
        nb_heads: int,
        mlp_dim: int,
        attention_downsample_rate: int,
        act_layer: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.nb_blocks = nb_blocks
        self.nb_heads = nb_heads
        self.mlp_dim = mlp_dim
        self.attention_downsample_rate = attention_downsample_rate
        self.act_layer = act_layer

        norm_layer = norm_layer_factory("layer_norm")

        self.blocks = [
            TwoWayAttentionBlock(
                embed_dim=self.embed_dim,
                nb_heads=self.nb_heads,
                mlp_dim=self.mlp_dim,
                attention_downsample_rate=self.attention_downsample_rate,
                skip_first_layer_pe=j == 0,
                act_layer=self.act_layer,
                name=f"layers/{j}",
            )
            for j in range(self.nb_blocks)
        ]
        self.final_attn_token_to_image = DownsampleAttention(
            embed_dim=self.embed_dim,
            nb_heads=self.nb_heads,
            downsample_rate=self.attention_downsample_rate,
            name="final_attn_token_to_image",
        )
        self.norm_final_attn = norm_layer(name="norm_final_attn")

    def call(self, inputs, training=False):
        """
        Args:
            inputs: Dictionary with the following entries.
                point_embeddings: Point embedding, should have shape (B, N, embed_dim)
                        for any N.
                image_embeddings: Image to attend to. Should have the shape
                    (B, H, W, embed_dim) for any (H, W).
                image_pe: Positional encoding to add to the image. Must have the same
                    shape as image_embedding.
            training: Training or inference mode?

        Returns:
            The processed point_embedding, same shape as input.
            The processed image_embedding, same shape as input. Note that this differs
                from PyTorch, where the output has shape (B, H*W, embed_dim).
        """
        image_embeddings = inputs["image_embeddings"]
        image_pe = inputs["image_pe"]
        point_embeddings = inputs["point_embeddings"]
        b, h, w, c = tf.unstack(tf.shape(image_embeddings))

        image_embedding = tf.reshape(image_embeddings, (b, -1, c))  # (B, H*W, C)
        image_pe = tf.reshape(image_pe, (b, -1, c))  # (B, H*W, C)

        # Prepare queries
        queries = point_embeddings
        keys = image_embedding

        # Apply transformer blocks
        for block in self.blocks:
            queries, keys = block(
                {"q": queries, "k": keys, "q_pe": point_embeddings, "k_pe": image_pe},
                training=training,
            )

        # Apply final attention layer from the points to the image
        attn = self.final_attn_token_to_image(
            {"q": queries + point_embeddings, "k": keys + image_pe, "v": keys},
            training=training,
        )
        queries = queries + attn
        queries = self.norm_final_attn(queries, training=training)

        # Reshape back to (B, H, W, C)
        keys = tf.reshape(keys, (b, h, w, c))

        return queries, keys


class TwoWayAttentionBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        embed_dim: int,
        nb_heads: int,
        mlp_dim: int,
        attention_downsample_rate: int,
        skip_first_layer_pe: bool,
        act_layer: str,
        **kwargs,
    ):
        """
        A transformer block with four layers:
            (1) self-attention of sparse inputs,
            (2) cross attention of sparse inputs to dense inputs,
            (3) mlp block on sparse inputs, and
            (4) cross attention of dense inputs to sparse inputs.

        Args:
          embed_dim: the channel dimension of the embeddings
          num_heads: the number of heads in the attention layers
          mlp_dim: the hidden dimension of the mlp block
          skip_first_layer_pe: skip the PE on the first layer
          act_layer: the activation of the mlp block
        """
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.nb_heads = nb_heads
        self.mlp_dim = mlp_dim
        self.attention_downsample_rate = attention_downsample_rate
        self.skip_first_layer_pe = skip_first_layer_pe
        self.act_layer = act_layer

        norm_layer = norm_layer_factory("layer_norm")

        self.self_attn = DownsampleAttention(
            embed_dim=embed_dim, nb_heads=nb_heads, downsample_rate=1, name="self_attn"
        )
        self.norm1 = norm_layer(name="norm1")

        self.cross_attn_token_to_image = DownsampleAttention(
            embed_dim=embed_dim,
            nb_heads=nb_heads,
            downsample_rate=attention_downsample_rate,
            name="cross_attn_token_to_image",
        )
        self.norm2 = norm_layer(name="norm2")

        self.mlp = MLPBlock(
            hidden_dim=self.mlp_dim,
            embed_dim=self.embed_dim,
            act_layer=self.act_layer,
            drop_rate=0.0,
            name="mlp",
        )
        self.norm3 = norm_layer(name="norm3")

        self.cross_attn_image_to_token = DownsampleAttention(
            embed_dim=embed_dim,
            nb_heads=nb_heads,
            downsample_rate=attention_downsample_rate,
            name="cross_attn_image_to_token",
        )
        self.norm4 = norm_layer(name="norm4")

    def call(self, inputs, training=False):
        q, k, q_pe, k_pe = inputs["q"], inputs["k"], inputs["q_pe"], inputs["k_pe"]

        # Self-attention block
        if self.skip_first_layer_pe:
            q = self.self_attn({"q": q, "k": q, "v": q}, training=training)
        else:
            attn = self.self_attn({"q": q + q_pe, "k": q + q_pe, "v": q})
            q = q + attn
        q = self.norm1(q, training=training)

        # Cross-attention block, tokens attending to image embedding
        attn = self.cross_attn_token_to_image(
            {"q": q + q_pe, "k": k + k_pe, "v": k}, training=training
        )
        q = q + attn
        q = self.norm2(q, training=training)

        # MLP block
        mlp = self.mlp(q, training=training)
        q = q + mlp
        q = self.norm3(q, training=training)

        # Cross-attention block, image embeddings attending to tokens
        attn = self.cross_attn_image_to_token(
            {"q": k + k_pe, "k": q + q_pe, "v": q}, training=training
        )
        k = k + attn
        k = self.norm4(k, training=training)

        return q, k


class DownsampleAttention(tf.keras.layers.Layer):
    """
    An attention layer that allows for downscaling the size of the embedding after
    projection to queries, keys, and values.
    """

    def __init__(
        self,
        embed_dim: int,
        nb_heads: int,
        downsample_rate: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.nb_heads = nb_heads
        self.downsample_rate = downsample_rate

        internal_dim = self.embed_dim // self.downsample_rate
        self.q_proj = tf.keras.layers.Dense(internal_dim, use_bias=True, name="q_proj")
        self.k_proj = tf.keras.layers.Dense(internal_dim, use_bias=True, name="k_proj")
        self.v_proj = tf.keras.layers.Dense(internal_dim, use_bias=True, name="v_proj")
        self.out_proj = tf.keras.layers.Dense(
            units=self.embed_dim, use_bias=True, name="out_proj"
        )

    def _separate_heads(self, x: tf.Tensor):
        b, m, c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]  # (B, M, C)
        x = tf.reshape(x, (b, m, self.nb_heads, c // self.nb_heads))  # (B, M, Hd, C/Hd)
        x = tf.transpose(x, (0, 2, 1, 3))  # (B, Hd, M, C/Hd)
        return x

    def _recombine_heads(self, x: tf.Tensor):
        # Shape of x is (B, Hd, M, C/Hd)
        batch_size, _, seq_length, _ = tf.unstack(tf.shape(x))
        x = tf.transpose(x, (0, 2, 1, 3))  # (B, M, Hd, C/Hd)
        x = tf.reshape(x, (batch_size, seq_length, -1))  # (B, M, C)
        return x

    def call(self, inputs):
        q, k, v = inputs["q"], inputs["k"], inputs["v"]

        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q)
        k = self._separate_heads(k)
        v = self._separate_heads(v)

        # Attention
        d = tf.shape(q)[-1]  # D=C/Hd
        attn = tf.matmul(q, k, transpose_b=True)  # (B, Hd, M, M)
        attn = attn / tf.sqrt(tf.cast(d, tf.float32))
        attn = tf.nn.softmax(attn, axis=-1)  # (B, Hd, M, M)

        # Get output
        x = tf.matmul(attn, v)  # (B, Hd, M, C/Hd)
        x = self._recombine_heads(x)  # (B, M, C)
        x = self.out_proj(x)

        return x
