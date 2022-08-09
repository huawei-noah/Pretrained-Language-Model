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
from mindspore import dtype as mstype
from mindspore.ops import operations as P
from mindspore import Parameter, Tensor
from mindspore.common.initializer import TruncatedNormal, initializer


class BERTMultiheadAttention(nn.Cell):
    def __init__(self, d_model, n_head):
        """

        :param d_model: width of tensor/embedding dim
        :param n_head: output of mutlithead attention/num_heads
        """
        super(BERTMultiheadAttention, self).__init__()
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

    def construct(self, query, key, value, attn_mask):
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
        attn_output_weights += self.expand_dims(attn_mask, 0)
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


class BERTAttentionWithMask(nn.Cell):
    def __init__(self, d_model, n_head, attn_mask):
        super(BERTAttentionWithMask, self).__init__()
        self.attn = BERTMultiheadAttention(d_model, n_head)
        self.attn_mask = attn_mask

    def construct(self, x):
        return self.attn(x, x, x, self.attn_mask)


class BERTResidualAttentionBlock(nn.Cell):
    def __init__(self, d_model, n_head, attn_mask):
        super(BERTResidualAttentionBlock, self).__init__()
        self.attn = BERTAttentionWithMask(d_model, n_head, attn_mask)
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


class BERTTransformer(nn.Cell):
    def __init__(self, width, layers, heads, attn_mask):
        super(BERTTransformer, self).__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.SequentialCell(
            *[BERTResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def construct(self, x):
        return self.resblocks(x)


class BERT_Wukong(nn.Cell):
    def __init__(self,
                 context_length,
                 vocab_size,
                 output_dim,
                 width,
                 layers,
                 heads,
                 return_full_embed=True):
        super(BERT_Wukong, self).__init__()
        self.width = width
        self.layers = layers
        self.vocab_size = vocab_size
        self.return_full_embed = return_full_embed
        self.embedding_table = Parameter(initializer(TruncatedNormal(0.02), [vocab_size, width]))
        self.gather = ops.Gather()
        self.reshape = ops.Reshape()
        self.cast = ops.Cast()

        self.positional_embedding = Parameter(initializer(TruncatedNormal(0.01), [context_length, width]))
        self.ln_final = nn.LayerNorm([self.width])
        self.text_projection = Parameter(
            Tensor(np.random.normal(0, self.width ** -0.5, size=(self.width, output_dim)).astype(np.float32)))
        self.transformer_layer = BERTTransformer(width, layers, heads, self.build_attntion_mask(context_length))

    @staticmethod
    def build_attntion_mask(context_length):
        mask = np.triu(np.full((context_length, context_length), -np.inf).astype(np.float32), 1)
        mask = Tensor(mask)
        return mask

    def construct(self, text):
        bsz, ctx_len = text.shape
        flatten_id = text.flatten()
        gather_result = self.gather(self.embedding_table, flatten_id, 0)

        x = self.reshape(gather_result, (bsz, ctx_len, -1))
        x = x + self.positional_embedding
        x = x.transpose(1, 0, 2)
        x = self.transformer_layer(x)
        x = x.transpose(1, 0, 2)
        x = self.ln_final(x)
        if not self.return_full_embed:
            x = x[nn.Range(x.shape[0])(),
                  self.cast(self.cast(text != 0, mstype.float32).sum(axis=-1), mstype.int32) - 1]
        x = ops.matmul(x, self.text_projection)
        return x
