# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import operations as P
from mindspore import Parameter, Tensor

from .token_learner import TokenLearnerModule


class VITMultiheadAttention(nn.Cell):
    def __init__(self, d_model, n_head):
        """

        :param d_model: width of tensor/embedding dim
        :param n_head: output of mutlithead attention/num_heads
        """
        super(VITMultiheadAttention, self).__init__()
        self.embed_dim = d_model
        self.num_heads = n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.in_proj = nn.Dense(self.embed_dim, 3 * self.embed_dim)
        self.out_proj = nn.Dense(self.embed_dim, self.embed_dim)
        self.split = ops.Split(-1, 3)
        self.expand_dims = P.ExpandDims()
        self.softmax = nn.Softmax(-1)
        self.transpose = ops.Transpose()
        self.scaling = self.head_dim ** -0.5

    def construct(self, query, key, value):
        tgt_len, bsz, embed_dim = query.shape
        qkv = self.in_proj(query).view(tgt_len, bsz, 3, embed_dim).transpose((2, 0, 1, 3))
        q = qkv[0:1]
        k = qkv[1:2]
        v = qkv[2:3]
        q = ops.Squeeze(0)(q)
        k = ops.Squeeze(0)(k)
        v = ops.Squeeze(0)(v)
        q = q * self.scaling
        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose((1, 0, 2))  # (bs) x (HW + 1) x h
        k = k.view(-1, bsz * self.num_heads, self.head_dim).transpose((1, 0, 2))  # (bs) x (HW + 1) x h
        v = v.view(-1, bsz * self.num_heads, self.head_dim).transpose((1, 0, 2))  # (bs) x (HW + 1) x h
        attn_output_weights = ops.matmul(q, k.transpose((0, 2, 1)))    # bs x (HW + 1) x (HW + 1)
        attn_output_weights = self.softmax(attn_output_weights)   # bs x (HW + 1) x (HW + 1)
        attn_output = ops.matmul(attn_output_weights, v)  # bs x (HW + 1) x h
        attn_output = self.transpose(attn_output, (1, 0, 2))
        attn_output = attn_output.view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output


class QuickGELU(nn.Cell):
    def __init__(self):
        super(QuickGELU, self).__init__()
        self.ratio = 1.702
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        return x * self.sigmoid(self.ratio * x)


class VITAttentionWithMask(nn.Cell):
    def __init__(self, d_model, n_head):
        super(VITAttentionWithMask, self).__init__()
        self.attn = VITMultiheadAttention(d_model, n_head)

    def construct(self, x):
        return self.attn(x, x, x)


class VITResidualAttentionBlock(nn.Cell):
    def __init__(self, d_model, n_head):
        super(VITResidualAttentionBlock, self).__init__()
        self.attn = VITAttentionWithMask(d_model, n_head)
        self.ln_1 = nn.LayerNorm([d_model])
        self.c_fc = nn.Dense(d_model, d_model * 4)
        self.gelu = QuickGELU()
        self.c_proj = nn.Dense(d_model * 4, d_model)
        self.mlp = nn.SequentialCell([
            self.c_fc,
            self.gelu,
            self.c_proj
        ])
        self.ln_2 = nn.LayerNorm([d_model])

    def construct(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class VITTransformer(nn.Cell):
    def __init__(self, width, layers, heads):
        super(VITTransformer, self).__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.SequentialCell(
            *[VITResidualAttentionBlock(width, heads) for _ in range(layers)]
        )

    def construct(self, x):
        return self.resblocks(x)


class VisualTransformer(nn.Cell):
    def __init__(self,
                 input_resolution,
                 patch_size,
                 width,
                 layers,
                 output_dim,
                 token_learner=None,
                 heads=None):
        super(VisualTransformer, self).__init__()
        if heads is None:
            heads = width // 64
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(3, width, patch_size, patch_size)
        self.num_tokens = (input_resolution // patch_size) ** 2 + 1
        self.num_tokens_side = input_resolution // patch_size
        scale = width ** -0.5
        self.class_embedding = Parameter(scale * Tensor(np.random.normal(0, 1, size=(width)).astype(np.float32)))
        self.positional_embedding = Parameter(
            scale * Tensor(np.random.normal(
                size=(self.num_tokens, width)).astype(np.float32)))
        self.ln_pre = nn.LayerNorm([width])
        self.transformer = VITTransformer(width, layers, heads)
        self.ln_post = nn.LayerNorm([width])
        self.proj = Parameter(scale * Tensor(np.random.normal(0, 1, size=(width, output_dim)).astype(np.float32)))
        if token_learner is not None:
            self.token_learner = TokenLearnerModule(in_channels=output_dim, **token_learner)
        self.width = width
        self.cat = ops.Concat(1)
        self.tile = ops.Tile()
        self.expand_dim = ops.ExpandDims()

    def construct(self, x):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.transpose(0, 2, 1)
        class_embedding = self.tile(self.class_embedding, (x.shape[0], 1, 1))
        x = self.cat((class_embedding, x))
        x = x + self.positional_embedding
        x = self.ln_pre(x)
        x = x.transpose(1, 0, 2)
        x = self.transformer(x)
        x = x.transpose(1, 0, 2)

        x = self.ln_post(x)
        x = ops.matmul(x, self.proj)
        if hasattr(self, "token_learner"):
            cls_token, x = x[:, 0, :], x[:, 1:, :]
            x = x.view(-1, self.num_tokens_side, self.num_tokens_side, self.output_dim)
            x, _ = self.token_learner(x)
            cls_token = self.expand_dim(cls_token, 1).astype(x.dtype)
            x = self.cat((cls_token, x))
        return x


class ClipVisualTransformer(VisualTransformer):
    def __init__(self, *args, **kwargs):
        super(ClipVisualTransformer, self).__init__(*args, **kwargs)

    def construct(self, x):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.transpose(0, 2, 1)
        class_embedding = self.tile(self.class_embedding, (x.shape[0], 1, 1))
        x = self.cat((class_embedding, x))
        x = x + self.positional_embedding
        x = self.ln_pre(x)
        x = x.transpose(1, 0, 2)
        x = self.transformer(x)
        x = x.transpose(1, 0, 2)

        x = x[:, 0, :]
        x = self.ln_post(x)
        x = ops.matmul(x, self.proj)
        return x
