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
import logging
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import dtype as mstype


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FilipTemplateEncoder(nn.Cell):
    def __init__(self, text_encoder):
        super(FilipTemplateEncoder, self).__init__()
        self.text_encoder = text_encoder
        self.text_norm = nn.Norm(axis=-1, keep_dims=True)
        self.concat = ops.Concat()
        self.expand_dims = ops.ExpandDims()

    def construct(self, text_tokens):
        n_class, n_templates, token_len = text_tokens.shape
        text_tokens = text_tokens.reshape((n_class * n_templates, token_len))
        res = []
        batch_num = n_class * n_templates // 100
        for i in range(100):
            text_tokens_part = text_tokens[batch_num * i: batch_num * (i + 1), :]
            text_tokens_features_part = self.text_encoder(text_tokens_part)
            text_tokens_features_part = text_tokens_features_part / self.text_norm(text_tokens_features_part)
            text_pad_mask = text_tokens_part > 0
            text_pad_mask = self.expand_dims(text_pad_mask, -1)
            text_tokens_features_part = text_tokens_features_part * text_pad_mask
            res.append(text_tokens_features_part)
        text_features = self.concat(res)
        if n_templates > 1:
            text_features = text_features.reshape(n_class, n_templates, token_len, -1)
        return text_features, n_templates


class ClipTemplateEncoder(nn.Cell):
    def __init__(self, text_encoder):
        super(ClipTemplateEncoder, self).__init__()
        self.text_encoder = text_encoder
        self.text_norm = nn.Norm(axis=-1, keep_dims=True)
        self.concat = ops.Concat()
        self.expand_dims = ops.ExpandDims()

    def construct(self, text_tokens):
        n_class, n_templates, token_len = text_tokens.shape
        text_tokens = text_tokens.reshape((n_class * n_templates, token_len))
        res = []
        batch_num = n_class * n_templates // 100
        for i in range(100):
            text_tokens_part = text_tokens[batch_num * i: batch_num * (i + 1), :]
            text_tokens_features_part = self.text_encoder(text_tokens_part)
            text_tokens_features_part = text_tokens_features_part / self.text_norm(text_tokens_features_part)
            res.append(text_tokens_features_part)
        text_features = self.concat(res)
        if n_templates > 1:
            text_features = text_features.reshape(n_class, n_templates, -1)
            text_features = text_features.mean(1)
            text_features = text_features / self.text_norm(text_features)
        return text_features, n_templates


class LateSimilarity(nn.Cell):
    def __init__(self):
        super(LateSimilarity, self).__init__()
        self.matmul = ops.MatMul(transpose_b=True)
        self.concat = ops.Concat(1)

    def construct(self, rep1, rep2):
        batch_size1, n_token1, feat_dim = rep1.shape
        _, n_token2, _ = rep2.shape
        out = self.matmul(rep1.reshape(-1, feat_dim), rep2.reshape(-1, feat_dim))
        out = out.reshape(batch_size1, n_token1, -1, n_token2).max(3)
        out = out.mean(1)
        return out


class FilipEval(nn.Cell):
    def __init__(self, text_features, n_template, image_encoder, text_encoder):
        super(FilipEval, self).__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.text_features = text_features
        self.n_template = n_template
        self.image_norm = nn.Norm(axis=-1, keep_dims=True)
        self.text_norm = nn.Norm(axis=-1, keep_dims=True)
        self.equal = ops.Equal()
        self.cast = ops.Cast()
        self.sim_func = LateSimilarity()
        self.softmax = ops.Softmax()
        self.softmax.add_prim_attr('primitive_target', 'CPU')
        self.expand_dims = ops.ExpandDims()
        self.expand_dims.add_prim_attr('primitive_target', 'CPU')
        self.topk = ops.TopK(sorted=True)
        self.topk.add_prim_attr('primitive_target', 'CPU')
        self.concat = ops.Concat()
        self.concat.add_prim_attr('primitive_target', 'CPU')
        self.mean = ops.ReduceMean()
        self.mean.add_prim_attr('primitive_target', 'CPU')

    def construct(self, images, targets):
        # text_tokens: #class x #templates x token_length
        image_features = self.image_encoder(images)
        total = image_features.shape[0]
        image_features = image_features[:, 1:, :]
        image_features = image_features / self.image_norm(image_features)
        if self.n_template > 1:
            all_sim = []
            for i in range(self.text_features.shape[1]):
                text_feat = self.text_features[:, i, :, :]
                sim_one = self.softmax(self.sim_func(image_features, text_feat))
                sim_one = self.expand_dims(sim_one, 0)
                all_sim.append(sim_one)
            similarity = self.concat(all_sim)
            similarity = self.mean(similarity, 0)
        else:
            similarity = self.sim_func(image_features, self.text_features)
        pred = self.topk(similarity, 5)[1].transpose()
        correct = self.equal(pred, targets.view(1, -1).expand_as(pred))
        correct = self.cast(correct, mstype.float32)
        return correct[:1].sum(0), correct[:5].sum(0), self.cast(total, mstype.float32)


class ClipEval(nn.Cell):
    def __init__(self, text_features, n_template, image_encoder, text_encoder):
        super(ClipEval, self).__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.image_norm = nn.Norm(axis=-1, keep_dims=True)
        self.text_features = text_features
        self.n_template = n_template
        self.softmax = ops.Softmax()
        self.softmax.add_prim_attr('primitive_target', 'CPU')
        self.matmul = ops.MatMul(transpose_b=True)
        self.cast = ops.Cast()
        self.topk = ops.TopK(sorted=True)
        self.topk.add_prim_attr('primitive_target', 'CPU')
        self.equal = ops.Equal()

    def construct(self, images, targets):
        image_features = self.image_encoder(images)
        total = image_features.shape[0]
        image_features = image_features / self.image_norm(image_features)
        image_features = self.cast(image_features, mstype.float16)

        similarity = self.softmax(self.matmul(image_features, self.text_features))
        pred = self.topk(similarity, 5)[1].transpose()
        correct = self.equal(pred, targets.view(1, -1).expand_as(pred))
        correct = self.cast(correct, mstype.float32)
        return correct[:1].sum(0), correct[:5].sum(0), self.cast(total, mstype.float32)
