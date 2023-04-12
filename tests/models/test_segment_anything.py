from typing import Tuple

import numpy as np
import pytest
import tensorflow as tf
import torch

from tfimm.architectures.segment_anything.image_encoder import (
    add_decomposed_rel_pos as tf_add_decomposed_rel_pos,
)
from tfimm.architectures.segment_anything.image_encoder import (
    get_rel_pos as tf_get_rel_pos,
)
from tfimm.architectures.segment_anything.image_encoder import (
    window_partition as tf_window_partition,
)
from tfimm.architectures.segment_anything.image_encoder import (
    window_unpartition as tf_window_unpartition,
)
from tfimm.architectures.segment_anything.prompt_encoder import (
    PositionalEmbeddingRandom as TFPositionalEmbeddingRandom,
)
from tfimm.architectures.segment_anything.prompt_encoder import (
    PromptEncoder as TFPromptEncoder,
)
from tfimm.architectures.segment_anything.torch.modeling.image_encoder import (
    add_decomposed_rel_pos as pt_add_decomposed_rel_pos,
)
from tfimm.architectures.segment_anything.torch.modeling.image_encoder import (
    get_rel_pos as pt_get_rel_pos,
)
from tfimm.architectures.segment_anything.torch.modeling.image_encoder import (
    window_partition as pt_window_partiton,
)
from tfimm.architectures.segment_anything.torch.modeling.image_encoder import (
    window_unpartition as pt_window_unpartition,
)
from tfimm.architectures.segment_anything.torch.modeling.prompt_encoder import (
    PositionEmbeddingRandom as PTPositionalEmbeddingRandom,
)
from tfimm.architectures.segment_anything.torch.modeling.prompt_encoder import (
    PromptEncoder as PTPromptEncoder,
)
from tfimm.architectures.segment_anything.torch.modeling.transformer import (
    Attention as PTAttention,
)
from tfimm.architectures.segment_anything.torch.modeling.transformer import (
    TwoWayAttentionBlock as PTTwoWayAttentionBlock,
)
from tfimm.architectures.segment_anything.torch.modeling.transformer import (
    TwoWayTransformer as PTTwoWayTransformer,
)
from tfimm.architectures.segment_anything.transformer import Attention as TFAttention
from tfimm.architectures.segment_anything.transformer import (
    TwoWayAttentionBlock as TFTwoWayAttentionBlock,
)
from tfimm.architectures.segment_anything.transformer import (
    TwoWayTransformer as TFTwoWayTransformer,
)
from tfimm.utils.timm import load_pytorch_weights_in_tf2_model


@pytest.mark.parametrize("shape", [(2, 8, 7, 3), (2, 3, 6, 4)])
def test_window_partition(shape: Tuple[int, ...]):
    x = np.random.rand(*shape)
    h, w = x.shape[1:3]
    tf_x = tf.convert_to_tensor(x)
    pt_x = torch.Tensor(x)

    tf_y, (tf_hp, tf_wp) = tf_window_partition(tf_x, 3)
    pt_y, (pt_hp, pt_wp) = pt_window_partiton(pt_x, 3)

    tf_res = tf_y.numpy()
    pt_res = pt_y.detach().numpy()

    np.testing.assert_almost_equal(tf_res, pt_res)
    assert tf_hp == pt_hp
    assert tf_wp == pt_wp

    tf_z = tf_window_unpartition(tf_y, (tf_hp, tf_wp), (h, w))
    pt_z = pt_window_unpartition(pt_y, 3, (pt_hp, pt_wp), (h, w))

    tf_res = tf_z.numpy()
    pt_res = pt_z.detach().numpy()

    np.testing.assert_almost_equal(tf_res, pt_res)
    np.testing.assert_almost_equal(tf_res, x)


@pytest.mark.parametrize(
    "q_size, k_size, pos_size, interpolate_pos",
    [
        (3, 3, 5, False),  # Simple case, q=k, no interpolation needed
        (3, 4, 7, False),  # Now, q != k, but interpolation still not needed
        (3, 3, 10, True),  # q=k, but need interpolation
        (3, 4, 9, True),  # General case
    ],
)
def test_get_rel_pos(q_size, k_size, pos_size, interpolate_pos):
    rel_pos = np.random.uniform(size=(pos_size, 4))
    tf_rel_pos = tf.convert_to_tensor(rel_pos)
    pt_rel_pos = torch.Tensor(rel_pos)

    tf_res = tf_get_rel_pos(q_size, k_size, tf_rel_pos, interpolate_pos)
    pt_res = pt_get_rel_pos(q_size, k_size, pt_rel_pos)

    np.testing.assert_almost_equal(tf_res.numpy(), pt_res.detach().numpy())


@pytest.mark.parametrize("qh, qw, kh, kw", [(5, 4, 3, 2)])
def test_add_decomposed_rel_pos(qh, qw, kh, kw):
    n = 3
    c = 7

    attn = np.random.uniform(size=(n, qh * qw, kh * kw))
    q = np.random.uniform(size=(n, qh * qw, c))
    rel_pos_h = np.random.uniform(size=(2 * max(qh, kh) - 1, c))
    rel_pos_w = np.random.uniform(size=(2 * max(qw, kw) - 1, c))

    tf_res = tf_add_decomposed_rel_pos(
        attn=tf.convert_to_tensor(attn),
        q=tf.convert_to_tensor(q),
        rel_pos_h=tf.convert_to_tensor(rel_pos_h),
        rel_pos_w=tf.convert_to_tensor(rel_pos_w),
        q_size=(qh, qw),
        k_size=(kh, kw),
        interpolate_pos=False,
    )
    pt_res = pt_add_decomposed_rel_pos(
        attn=torch.Tensor(attn),
        q=torch.Tensor(q),
        rel_pos_h=torch.Tensor(rel_pos_h),
        rel_pos_w=torch.Tensor(rel_pos_w),
        q_size=(qh, qw),
        k_size=(kh, kw),
    )
    np.testing.assert_almost_equal(tf_res.numpy(), pt_res.detach().numpy(), decimal=6)


def test_prompt_encoder_empty():
    """We test that the prompt encoder behaves correctly for empty prompts."""
    prompt_encoder = TFPromptEncoder(
        input_size=(32, 64),
        grid_size=(8, 16),
        embed_dim=6,
        mask_channels=9,
    )
    prompt_encoder(prompt_encoder.dummy_inputs)

    embeddings = prompt_encoder._embed_points(
        points=tf.ones((3, 0, 2), dtype=tf.float32),
        labels=tf.ones((3, 0), dtype=tf.int32),
    )
    tf.ensure_shape(embeddings, (3, 0, 6))

    embeddings = prompt_encoder._embed_boxes(boxes=tf.ones((3, 0, 4), dtype=tf.float32))
    tf.ensure_shape(embeddings, (3, 1, 6))

    embeddings = prompt_encoder._embed_masks(
        masks=tf.ones((3, *prompt_encoder.mask_size, 0), dtype=tf.float32)
    )
    tf.ensure_shape(embeddings, (3, 8, 16, 6))


def test_prompt_encoder():
    points = np.random.uniform(size=(1, 3, 2)).astype(np.float32)
    labels = np.asarray([[1, 1, 0]], dtype=np.int32)
    boxes = np.random.uniform(size=(1, 1, 4)).astype(np.float32)
    masks = np.random.uniform(size=(1, 32, 64, 1)).astype(np.float32)

    pt_prompt_encoder = PTPromptEncoder(
        embed_dim=6,
        image_embedding_size=(8, 16),
        input_image_size=(32, 64),
        mask_in_chans=9,
    )
    tf_prompt_encoder = TFPromptEncoder(
        input_size=(32, 64),
        grid_size=(8, 16),
        embed_dim=6,
        mask_channels=9,
    )
    load_pytorch_weights_in_tf2_model(
        tf_prompt_encoder,
        pt_prompt_encoder.state_dict(),
    )

    pt_sparse, pt_dense = pt_prompt_encoder.forward(
        points=(torch.Tensor(points), torch.Tensor(labels)),
        boxes=torch.Tensor(boxes),
        masks=torch.Tensor(masks).permute((0, 3, 1, 2)),
    )
    pt_dense = pt_dense.permute((0, 2, 3, 1))
    tf_sparse, tf_dense = tf_prompt_encoder(
        {"points": points, "labels": labels, "boxes": boxes, "masks": masks}
    )
    np.testing.assert_almost_equal(tf_sparse.numpy(), pt_sparse.detach().numpy())
    np.testing.assert_almost_equal(
        tf_dense.numpy(), pt_dense.detach().numpy(), decimal=5
    )


def test_positional_embedding_random():
    pt_embedder = PTPositionalEmbeddingRandom(num_pos_feats=7, scale=1.0)
    tf_embedder = TFPositionalEmbeddingRandom(embed_dim=14, scale=1.0)
    load_pytorch_weights_in_tf2_model(
        tf_embedder, pt_embedder.state_dict(), tf_inputs=tf.zeros((3, 2))
    )

    points = np.random.rand(3, 2)
    pt_res = pt_embedder._pe_encoding(torch.Tensor(points))
    tf_res = tf_embedder(tf.convert_to_tensor(points))
    np.testing.assert_almost_equal(tf_res.numpy(), pt_res.detach().numpy())

    points = np.random.uniform(0, 12, (1, 4, 2)).astype(np.float32)
    pt_res = pt_embedder.forward_with_coords(torch.Tensor(points), (12, 16))
    tf_res = tf_embedder.embed_points(tf.convert_to_tensor(points), (12, 16))
    np.testing.assert_almost_equal(tf_res.numpy(), pt_res.detach().numpy())

    pt_res = pt_embedder.forward((5, 7))
    tf_res = tf_embedder.embed_grid((5, 7))
    pt_res = pt_res.permute((1, 2, 0))
    np.testing.assert_almost_equal(tf_res.numpy(), pt_res.detach().numpy())


def test_mask_attention():
    """Test attention module used for the mask decoder."""
    pt_attention = PTAttention(embedding_dim=12, num_heads=2, downsample_rate=3)
    tf_attention = TFAttention(embed_dim=12, nb_heads=2, downsample_rate=3)
    load_pytorch_weights_in_tf2_model(
        tf_attention,
        pt_attention.state_dict(),
        tf_inputs={key: tf.zeros((1, 1, 12)) for key in {"q", "k", "v"}},
    )

    q = np.random.rand(2, 3, 12).astype(np.float32)
    k = np.random.rand(2, 3, 12).astype(np.float32)
    v = np.random.rand(2, 3, 12).astype(np.float32)
    pt_res = pt_attention(q=torch.Tensor(q), k=torch.Tensor(k), v=torch.Tensor(v))
    tf_res = tf_attention(
        inputs={"q": tf.constant(q), "k": tf.constant(k), "v": tf.constant(v)}
    )
    np.testing.assert_almost_equal(tf_res.numpy(), pt_res.detach().numpy())


@pytest.mark.parametrize("skip_first_layer_pe", [True, False])
def test_two_way_attention(skip_first_layer_pe):
    pt_attention = PTTwoWayAttentionBlock(
        embedding_dim=12,
        num_heads=2,
        mlp_dim=3,
        skip_first_layer_pe=skip_first_layer_pe,
    )
    tf_attention = TFTwoWayAttentionBlock(
        embed_dim=12,
        nb_heads=2,
        mlp_dim=3,
        attention_downsample_rate=2,
        skip_first_layer_pe=skip_first_layer_pe,
        act_layer="relu",
    )
    load_pytorch_weights_in_tf2_model(
        tf_attention,
        pt_attention.state_dict(),
        tf_inputs={key: tf.zeros((1, 1, 12)) for key in {"q", "k", "q_pe", "k_pe"}},
    )

    q = np.random.rand(2, 3, 12).astype(np.float32)
    k = np.random.rand(2, 3, 12).astype(np.float32)
    q_pe = np.random.rand(2, 3, 12).astype(np.float32)
    k_pe = np.random.rand(2, 3, 12).astype(np.float32)
    inputs = {"q": q, "k": k, "q_pe": q_pe, "k_pe": k_pe}
    pt_q, pt_k = pt_attention(
        queries=torch.Tensor(q),
        keys=torch.Tensor(k),
        query_pe=torch.Tensor(q_pe),
        key_pe=torch.Tensor(k_pe),
    )
    tf_q, tf_k = tf_attention(
        inputs={key: tf.constant(value) for key, value in inputs.items()}
    )
    np.testing.assert_almost_equal(tf_q.numpy(), pt_q.detach().numpy(), decimal=5)
    np.testing.assert_almost_equal(tf_k.numpy(), pt_k.detach().numpy(), decimal=5)


def test_two_way_transformer():
    pt_transformer = PTTwoWayTransformer(
        depth=2,
        embedding_dim=12,
        num_heads=2,
        mlp_dim=3,
    )
    tf_transformer = TFTwoWayTransformer(
        embed_dim=12,
        nb_blocks=2,
        nb_heads=2,
        mlp_dim=3,
        attention_downsample_rate=2,
        act_layer="relu",
    )
    load_pytorch_weights_in_tf2_model(
        tf_transformer,
        pt_transformer.state_dict(),
        tf_inputs={
            "point_embedding": tf.zeros((1, 1, 12)),
            "image_embedding": tf.zeros((1, 1, 1, 12)),
            "image_pe": tf.zeros((1, 1, 1, 12)),
        },
    )

    pts_emb = np.random.rand(2, 3, 12).astype(np.float32)
    img_emb = np.random.rand(2, 4, 5, 12).astype(np.float32)
    img_pe = np.random.rand(2, 4, 5, 12).astype(np.float32)
    pt_q, pt_k = pt_transformer(
        point_embedding=torch.Tensor(pts_emb),
        image_embedding=torch.Tensor(img_emb).permute((0, 3, 1, 2)),
        image_pe=torch.Tensor(img_pe).permute((0, 3, 1, 2)),
    )
    pt_k = pt_k.reshape(img_emb.shape)
    tf_q, tf_k = tf_transformer(
        {
            "point_embedding": tf.constant(pts_emb),
            "image_embedding": tf.constant(img_emb),
            "image_pe": tf.constant(img_pe),
        }
    )
    np.testing.assert_almost_equal(tf_q.numpy(), pt_q.detach().numpy(), decimal=5)
    np.testing.assert_almost_equal(tf_k.numpy(), pt_k.detach().numpy(), decimal=5)


def test_mask_decoder():
    ...
