# coding=utf-8
# 2021.12.30-Changed for SuperPLM modeling
#      Huawei Technologies Co., Ltd. <yinyichun@huawei.com>
# Copyright 2021 Huawei Technologies Co., Ltd.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from .file_utils import cached_path, WEIGHTS_NAME, CONFIG_NAME

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}
BERT_CONFIG_NAME = 'bert_config.json'


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertConfig(object):
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12):
        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                                                               and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        return_unused_kwargs = kwargs.pop('return_unused_kwargs', False)

        if os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        else:
            config_file = pretrained_model_name_or_path

        # Load config
        config = cls.from_json_file(config_file)

        if hasattr(config, 'pruned_heads'):
            config.pruned_heads = dict((int(key), value) for key, value in config.pruned_heads.items())

        # Update config with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info("Model config %s", str(config))
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())


class SuperBertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(SuperBertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

        self.sample_weight = None
        self.sample_bias = None

    def set_sample_config(self, sample_hidden_dim):
        self.sample_weight = self.weight[:sample_hidden_dim]
        self.sample_bias = self.bias[:sample_hidden_dim]

    def calc_sampled_param_num(self):
        weight_numel = self.sample_weight.numel()
        bias_numel = self.sample_bias.numel()

        assert weight_numel != 0
        assert bias_numel != 0

        return weight_numel + bias_numel

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.sample_weight * x + self.sample_bias


class SuperLinear(nn.Linear):
    def __init__(self, super_in_dim, super_out_dim, bias=True,
                 uniform_=None, non_linear='linear'):

        super().__init__(super_in_dim, super_out_dim, bias=bias)

        # super_in_dim and super_out_dim indicate the largest network!
        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim

        # input_dim and output_dim indicate the current sampled size
        self.sample_in_dim = None
        self.sample_out_dim = None

        self.samples = {}

        self._reset_parameters(bias, uniform_, non_linear)
        self.profiling = False

    def profile(self, mode=True):
        self.profiling = mode

    def sample_parameters(self, resample=False):
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def _reset_parameters(self, bias, uniform_, non_linear):
        nn.init.xavier_uniform_(self.weight) if uniform_ is None else uniform_(
            self.weight, non_linear=non_linear)
        if bias:
            nn.init.constant_(self.bias, 0.)

    def set_sample_config(self, sample_in_dim, sample_out_dim, in_index=None, out_index=None):
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim

        self._sample_parameters(in_index=in_index, out_index=out_index)

    def _sample_parameters(self, in_index=None, out_index=None):
        self.samples['weight'] = sample_weight(self.weight, self.sample_in_dim, self.sample_out_dim,
                                               in_index=in_index, out_index=out_index)
        self.samples['bias'] = self.bias
        if self.bias is not None:
            self.samples['bias'] = sample_bias(self.bias, self.sample_out_dim, out_index=out_index)
        return self.samples

    def forward(self, x):
        if self.bias is not None:
            return F.linear(x, self.samples['weight'].to(x.device), self.samples['bias'].to(x.device))
        else:
            return F.linear(x, self.samples['weight'].to(x.device))

    def calc_sampled_param_num(self):
        assert 'weight' in self.samples.keys()
        weight_numel = self.samples['weight'].numel()

        if self.samples['bias'] is not None:
            bias_numel = self.samples['bias'].numel()
        else:
            bias_numel = 0

        return weight_numel + bias_numel


# two stages
def sample_weight(weight, sample_in_dim, sample_out_dim, in_index=None, out_index=None):

    if in_index is None:
        sample_weight = weight[:, :sample_in_dim]
    else:
        sample_weight = weight.index_select(1, in_index.to(weight.device))

    if out_index is None:
        sample_weight = sample_weight[:sample_out_dim, :]
    else:
        sample_weight = sample_weight.index_select(0, out_index.to(sample_weight.device))

    return sample_weight


def sample_bias(bias, sample_out_dim, out_index=None):

    if out_index is None:
        sample_bias = bias[:sample_out_dim]
    else:
        sample_bias = bias.index_select(0, out_index.to(bias.device))

    return sample_bias


class SuperEmbedding(nn.Module):
    def __init__(self, dict_size, embd_size, padding_idx=None):
        super(SuperEmbedding, self).__init__()
        self.embedding = nn.Embedding(dict_size, embd_size, padding_idx=padding_idx)
        self.sample_embedding_weight = None

    def set_sample_config(self, sample_embed_dim):
        self.sample_embedding_weight = self.embedding.weight[..., :sample_embed_dim]

    def calc_sampled_param_num(self):
        weight_numel = self.sample_embedding_weight.numel()
        assert weight_numel != 0
        return weight_numel

    def forward(self, input_ids):
        return F.embedding(input_ids, self.sample_embedding_weight.to(input_ids.device), self.embedding.padding_idx,
                           self.embedding.max_norm, self.embedding.norm_type,
                           self.embedding.scale_grad_by_freq, self.embedding.sparse)


class SuperBertEmbeddings(nn.Module):
    def __init__(self, config):
        super(SuperBertEmbeddings, self).__init__()
        self.word_embeddings = SuperEmbedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = SuperEmbedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = SuperEmbedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = SuperBertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sample_embed_dim = None

    def set_sample_config(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim
        self.word_embeddings.set_sample_config(sample_embed_dim)
        self.position_embeddings.set_sample_config(sample_embed_dim)
        self.token_type_embeddings.set_sample_config(sample_embed_dim)
        self.LayerNorm.set_sample_config(sample_embed_dim)

    def calc_sampled_param_num(self):
        w_emb_numel = self.word_embeddings.calc_sampled_param_num()
        p_emb_numel = self.position_embeddings.calc_sampled_param_num()
        t_emb_numel = self.token_type_embeddings.calc_sampled_param_num()
        ln_numel = self.LayerNorm.calc_sampled_param_num()

        logger.info('w_emb: {}\n'.format(w_emb_numel))
        logger.info('p_emb: {}\n'.format(p_emb_numel))
        logger.info('t_emb: {}\n'.format(t_emb_numel))
        logger.info('ln_emb: {}\n'.format(ln_numel))

        return w_emb_numel + p_emb_numel + t_emb_numel + ln_numel

    def forward(self, input_ids, sample_embed_dim=-1, token_type_ids=None):
        self.set_sample_config(sample_embed_dim)

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings

        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings += token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SuperBertSelfAttention(nn.Module):
    def __init__(self, config):
        super(SuperBertSelfAttention, self).__init__()
        try:
            qkv_size = config.qkv_size
        except:
            qkv_size = config.hidden_size

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(qkv_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = SuperLinear(config.hidden_size, self.all_head_size)
        self.key = SuperLinear(config.hidden_size, self.all_head_size)
        self.value = SuperLinear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.sample_num_attention_head = None
        self.sample_attention_head_size = None
        self.sample_qkv_size = None

    def set_sample_config(self, sample_embed_dim, num_attention_head, qkv_size,
                          in_index=None, out_index=None):
        assert qkv_size % num_attention_head == 0
        self.sample_qkv_size = qkv_size
        self.sample_attention_head_size = qkv_size // num_attention_head
        self.sample_num_attention_head = num_attention_head

        self.query.set_sample_config(sample_embed_dim, qkv_size, in_index=in_index, out_index=out_index)
        self.key.set_sample_config(sample_embed_dim, qkv_size, in_index=in_index, out_index=out_index)
        self.value.set_sample_config(sample_embed_dim, qkv_size, in_index=in_index, out_index=out_index)

    def calc_sampled_param_num(self):
        query_numel = self.query.calc_sampled_param_num()
        key_numel = self.key.calc_sampled_param_num()
        value_numel = self.value.calc_sampled_param_num()

        logger.info('query_numel: {}\n'.format(query_numel))
        logger.info('key_numel: {}\n'.format(key_numel))
        logger.info('value_numel: {}\n'.format(value_numel))

        return query_numel + key_numel + value_numel

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.sample_num_attention_head, self.sample_attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, sample_embed_dim=-1, num_attention_head=-1, qkv_size=-1,
                in_index=None, out_index=None):
        self.set_sample_config(sample_embed_dim, num_attention_head, qkv_size,
                               in_index=in_index, out_index=out_index)

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.sample_attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        context_layer = torch.matmul(self.dropout(attention_probs), value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.sample_qkv_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_scores


class SuperBertSelfOutput(nn.Module):
    def __init__(self, config):
        super(SuperBertSelfOutput, self).__init__()
        try:
            qkv_size = config.qkv_size
        except:
            qkv_size = config.hidden_size

        self.dense = SuperLinear(qkv_size, config.hidden_size)
        self.LayerNorm = SuperBertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def set_sample_config(self, qkv_size, sample_embed_dim, in_index=None):
        self.dense.set_sample_config(qkv_size, sample_embed_dim, in_index=in_index)
        self.LayerNorm.set_sample_config(sample_embed_dim)

    def calc_sampled_param_num(self):
        dense_numel = self.dense.calc_sampled_param_num()
        ln_numel = self.LayerNorm.calc_sampled_param_num()

        logger.info('dense_numel: {}\n'.format(dense_numel))
        logger.info('ln_numel: {}\n'.format(ln_numel))

        return dense_numel + ln_numel

    def forward(self, hidden_states, input_tensor, qkv_size=-1, sample_embed_dim=-1, in_index=None):
        self.set_sample_config(qkv_size, sample_embed_dim, in_index=in_index)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SuperBertAttention(nn.Module):
    def __init__(self, config):
        super(SuperBertAttention, self).__init__()
        self.self = SuperBertSelfAttention(config)
        self.output = SuperBertSelfOutput(config)

    def set_sample_config(self, sample_embed_dim, num_attention_head, qkv_size):
        self.self.set_sample_config(sample_embed_dim, num_attention_head, qkv_size)
        self.output.set_sample_config(qkv_size, sample_embed_dim)

    def calc_sampled_param_num(self):
        self_numel = self.self.calc_sampled_param_num()
        output_numel = self.output.calc_sampled_param_num()

        logger.info('self_numel: {}\n'.format(self_numel))
        logger.info('output_numel: {}\n'.format(output_numel))

        return self_numel + output_numel

    def forward(self, input_tensor, attention_mask, sample_embed_dim=-1,
                num_attention_head=-1, qkv_size=-1,
                in_index=None, out_index=None):

        self_output = self.self(input_tensor, attention_mask, sample_embed_dim,
                                num_attention_head, qkv_size, in_index=in_index, out_index=out_index)
        self_output, layer_att = self_output
        attention_output = self.output(self_output, input_tensor, qkv_size,
                                       sample_embed_dim, in_index=out_index)
        return attention_output, layer_att


class SuperBertIntermediate(nn.Module):
    def __init__(self, config):
        super(SuperBertIntermediate, self).__init__()
        self.dense = SuperLinear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def set_sample_config(self, sample_embed_dim, intermediate_size):
        self.dense.set_sample_config(sample_embed_dim, intermediate_size)

    def calc_sampled_param_num(self):
        dense_numel = self.dense.calc_sampled_param_num()

        logger.info('dense_numel: {}\n'.format(dense_numel))
        return dense_numel

    def forward(self, hidden_states, sample_embed_dim=-1, intermediate_size=-1):
        self.set_sample_config(sample_embed_dim, intermediate_size)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class SuperBertOutput(nn.Module):
    def __init__(self, config):
        super(SuperBertOutput, self).__init__()
        self.dense = SuperLinear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = SuperBertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def set_sample_config(self, intermediate_size, sample_embed_dim):
        self.dense.set_sample_config(intermediate_size, sample_embed_dim)
        self.LayerNorm.set_sample_config(sample_embed_dim)

    def calc_sampled_param_num(self):
        dense_numel = self.dense.calc_sampled_param_num()
        ln_numel = self.LayerNorm.calc_sampled_param_num()

        logger.info('dense_numel: {}\n'.format(dense_numel))
        logger.info('ln_numel: {}\n'.format(ln_numel))
        return dense_numel + ln_numel

    def forward(self, hidden_states, input_tensor, intermediate_size=-1, sample_embed_dim=-1):
        self.set_sample_config(intermediate_size, sample_embed_dim)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SuperBertLayer(nn.Module):
    def __init__(self, config):
        super(SuperBertLayer, self).__init__()
        self.attention = SuperBertAttention(config)
        self.intermediate = SuperBertIntermediate(config)
        self.output = SuperBertOutput(config)

    def set_sample_config(self, sample_embed_dim, intermediate_size, num_attention_head, qkv_size):
        self.attention.set_sample_config(sample_embed_dim, num_attention_head, qkv_size)
        self.intermediate.set_sample_config(sample_embed_dim, intermediate_size)
        self.output.set_sample_config(intermediate_size, sample_embed_dim)

    def calc_sampled_param_num(self):
        attention_numel = self.attention.calc_sampled_param_num()
        intermediate_numel = self.intermediate.calc_sampled_param_num()
        output_numel = self.output.calc_sampled_param_num()

        logger.info('attention_numel: {}\n'.format(attention_numel))
        logger.info('intermediate_numel: {}\n'.format(intermediate_numel))
        logger.info('output_numel: {}\n'.format(output_numel))

        return attention_numel + intermediate_numel + output_numel

    def forward(self, hidden_states, attention_mask, sample_embed_dim=-1,
                intermediate_size=-1, num_attention_head=-1, qkv_size=-1,
                in_index=None, out_index=None):

        attention_output = self.attention(hidden_states, attention_mask,
                                          sample_embed_dim, num_attention_head, qkv_size,
                                          in_index=in_index, out_index=out_index)
        attention_output, layer_att = attention_output

        intermediate_output = self.intermediate(attention_output, sample_embed_dim,
                                                intermediate_size)
        layer_output = self.output(intermediate_output, attention_output, intermediate_size, sample_embed_dim)
        return layer_output, layer_att


class SuperBertEncoder(nn.Module):
    def __init__(self, config):
        super(SuperBertEncoder, self).__init__()
        layer = SuperBertLayer(config)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])
        self.sample_layer_num = None

    def set_sample_config(self, subbert_config):
        self.sample_layer_num = subbert_config['sample_layer_num']
        sample_embed_dim = subbert_config['sample_hidden_size']
        num_attention_heads = subbert_config['sample_num_attention_heads']
        itermediate_sizes = subbert_config['sample_intermediate_sizes']
        qkv_sizes = subbert_config['sample_qkv_sizes']
        for layer, num_attention_head, intermediate_size, qkv_size in zip(self.layers[:self.sample_layer_num],
                                                                          num_attention_heads,
                                                                          itermediate_sizes,
                                                                          qkv_sizes):
            layer.set_sample_config(sample_embed_dim, intermediate_size, num_attention_head, qkv_size)

    def calc_sampled_param_num(self):
        layers_numel = 0

        for layer in self.layers[:self.sample_layer_num]:
            layers_numel += layer.calc_sampled_param_num()

        logger.info('layer_numel: {}'.format(layers_numel))

        return layers_numel

    def forward(self, hidden_states, attention_mask, subbert_config=None, kd=False,
                in_index=None, out_index=None):
        all_encoder_layers = []
        all_encoder_att = []

        sample_embed_dim = subbert_config['sample_hidden_size']
        num_attention_heads = subbert_config['sample_num_attention_heads']
        itermediate_sizes = subbert_config['sample_intermediate_sizes']
        qkv_sizes = subbert_config['sample_qkv_sizes']
        sample_layer_num = subbert_config['sample_layer_num']

        for i, layer_module in enumerate(self.layers[:sample_layer_num]):
            all_encoder_layers.append(hidden_states)
            hidden_states = layer_module(all_encoder_layers[i], attention_mask,
                                         sample_embed_dim, itermediate_sizes[i],
                                         num_attention_heads[i],
                                         qkv_sizes[i], in_index=in_index, out_index=out_index)
            hidden_states, layer_att = hidden_states
            all_encoder_att.append(layer_att)

        all_encoder_layers.append(hidden_states)

        if not kd:
            return all_encoder_layers, all_encoder_att
        else:
            return all_encoder_layers[-1], all_encoder_att[-1]


class SuperBertPooler(nn.Module):
    def __init__(self, config):
        super(SuperBertPooler, self).__init__()
        self.dense = SuperLinear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def set_sample_config(self, sample_hidden_dim):
        self.dense.set_sample_config(sample_hidden_dim, sample_hidden_dim)

    def calc_sampled_param_num(self):
        dense_numel = self.dense.calc_sampled_param_num()
        logger.info('dense_numel: {}'.format(dense_numel))

        return dense_numel

    def forward(self, hidden_states, sample_hidden_dim):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        self.set_sample_config(sample_hidden_dim)
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPreTrainedModel(nn.Module):
    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, SuperBertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_scratch(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        resolved_config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        if not os.path.exists(resolved_config_file):
            resolved_config_file = os.path.join(pretrained_model_name_or_path, BERT_CONFIG_NAME)
        config = BertConfig.from_json_file(resolved_config_file)
        logger.info("Model config {}".format(config))
        model = cls(*inputs, **kwargs)
        return model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        state_dict = kwargs.get('state_dict', None)
        kwargs.pop('state_dict', None)
        cache_dir = kwargs.get('cache_dir', None)
        kwargs.pop('cache_dir', None)
        from_tf = kwargs.get('from_tf', False)
        kwargs.pop('from_tf', None)

        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            archive_file = pretrained_model_name_or_path
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name_or_path,
                    ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                    archive_file))
            return None
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(
                archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file) or from_tf:
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        # Load config
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        if not os.path.exists(config_file):
            # Backward compatibility with old naming format
            config_file = os.path.join(serialization_dir, BERT_CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(*inputs, **kwargs)
        if state_dict is None and not from_tf:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path, map_location='cpu')
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None

            if 'bert' not in key:
                new_key = 'bert.' + key

            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)

        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                model.__class__.__name__, "\n\t".join(error_msgs)))
        return model


class SuperBertModel(BertPreTrainedModel):
    def __init__(self, config, fit_size=768):
        super(SuperBertModel, self).__init__(config)
        self.embeddings = SuperBertEmbeddings(config)
        self.encoder = SuperBertEncoder(config)
        self.pooler = SuperBertPooler(config)
        self.dense_fit = SuperLinear(config.hidden_size, fit_size)

        self.hidden_size = config.hidden_size
        self.qkv_size = self.hidden_size

        try:
            self.qkv_size = config.qkv_size
        except:
            self.qkv_size = config.hidden_size

        self.fit_size = fit_size
        self.head_number = config.num_attention_heads
        self.apply(self.init_bert_weights)

    def set_sample_config(self, subbert_config):
        self.embeddings.set_sample_config(subbert_config['sample_hidden_size'])
        self.encoder.set_sample_config(subbert_config)
        self.pooler.set_sample_config(subbert_config['sample_hidden_size'])

    def calc_sampled_param_num(self):
        emb_numel = self.embeddings.calc_sampled_param_num()
        encoder_numel = self.encoder.calc_sampled_param_num()
        pooler_numel = self.pooler.calc_sampled_param_num()

        logger.info('===========================')
        logger.info('emb_numel: {}\n'.format(emb_numel))
        logger.info('encoder_numel: {}\n'.format(encoder_numel))
        logger.info('pooler_numel: {}\n'.format(pooler_numel))
        logger.info('all parameters: {}\n'.format(emb_numel + encoder_numel + pooler_numel))
        logger.info('===========================')
        return emb_numel + encoder_numel + pooler_numel

    def forward(self, input_ids, subbert_config,
                attention_mask=None, token_type_ids=None,
                kd=False, kd_infer=False):

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        head_number = self.head_number
        qkv_size = self.qkv_size
        sample_qkv_size = subbert_config['sample_qkv_sizes'][0]

        in_out_index = None
        if kd:
            in_dim_per_head = int(qkv_size / head_number)
            in_sample_per_head = int(sample_qkv_size / head_number)

            in_out_index = []
            for i in range(head_number):
                start_ind = in_dim_per_head * i
                in_out_index.extend(range(start_ind, start_ind + in_sample_per_head))

            in_out_index = torch.tensor(in_out_index)
            in_out_index.to(input_ids.device)

        embedding_output = self.embeddings(input_ids, subbert_config['sample_hidden_size'],
                                           token_type_ids=token_type_ids)

        if kd:
            last_rep, last_att = self.encoder(embedding_output, extended_attention_mask, subbert_config,
                                              kd=True, out_index=in_out_index)
            self.dense_fit.set_sample_config(subbert_config['sample_hidden_size'], self.fit_size)
            last_rep = self.dense_fit(last_rep)

            if not kd_infer:
                return last_rep, last_att
            else:
                pooled_output = self.pooler(last_rep, subbert_config['sample_hidden_size'])
                return last_rep, pooled_output
        else:
            all_encoder_layers, all_encoder_att = self.encoder(embedding_output, extended_attention_mask,
                                                               subbert_config, kd=False, out_index=in_out_index)
            sequence_output = all_encoder_layers[-1]
            pooled_output = self.pooler(sequence_output, subbert_config['sample_hidden_size'])
            return sequence_output, pooled_output


class SuperBertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(SuperBertPredictionHeadTransform, self).__init__()
        # Need to unty it when we separate the dimensions of hidden and emb
        self.dense = SuperLinear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or \
                (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = SuperBertLayerNorm(config.hidden_size, eps=1e-12)

    def set_sample_config(self, sample_hidden_dim):
        self.dense.set_sample_config(sample_hidden_dim, sample_hidden_dim)
        self.LayerNorm.set_sample_config(sample_hidden_dim)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class SuperBertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(SuperBertLMPredictionHead, self).__init__()
        self.transform = SuperBertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = SuperLinear(bert_model_embedding_weights.size(1),
                                   bert_model_embedding_weights.size(0),
                                   bias=False)

        self.dict_size = bert_model_embedding_weights.size(0)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def set_sample_config(self, sample_hidden_dim):
        self.decoder.set_sample_config(sample_hidden_dim, self.dict_size)
        self.transform.set_sample_config(sample_hidden_dim)

    def forward(self, hidden_states, sample_hidden_dim):
        self.set_sample_config(sample_hidden_dim)
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class SuperTinyBertForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super(SuperTinyBertForPreTraining, self).__init__(config)
        self.bert = SuperBertModel(config)
        self.apply(self.init_bert_weights)
        self.config = config

    def forward(self, input_ids, subbert_config, token_type_ids=None, attention_mask=None, kd=True):
        last_rep, last_att = self.bert(input_ids, subbert_config, token_type_ids,
                                       attention_mask, kd=kd)
        return last_rep, last_att


class SuperBertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(SuperBertPreTrainingHeads, self).__init__()
        self.predictions = SuperBertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = SuperLinear(config.hidden_size, 2)

    def set_sample_config(self, sample_hidden_dim):
        self.predictions.set_sample_config(sample_hidden_dim)
        self.seq_relationship.set_sample_config(sample_hidden_dim, 2)

    def forward(self, sequence_output, pooled_output, sample_hidden_dim):
        self.set_sample_config(sample_hidden_dim)
        prediction_scores = self.predictions(sequence_output, sample_hidden_dim)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class SuperBertForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super(SuperBertForPreTraining, self).__init__(config)
        self.bert = SuperBertModel(config)
        self.cls = SuperBertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.embedding.weight)
        self.apply(self.init_bert_weights)
        self.config = config

    def set_sample_config(self, subbert_config):
        self.bert.set_sample_config(subbert_config)
        self.cls.set_sample_config(subbert_config['sample_hidden_size'])

    def forward(self, input_ids, subbert_config, token_type_ids=None, attention_mask=None,
                masked_lm_labels=None, next_sentence_label=None):
        self.set_sample_config(subbert_config)

        encoder_layers, pooled_output = self.bert(input_ids, subbert_config, token_type_ids, attention_mask)
        sequence_output = encoder_layers
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output,
                                                             subbert_config['sample_hidden_size'])

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss
        elif masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            total_loss = masked_lm_loss
            return total_loss
        else:
            return prediction_scores, seq_relationship_score


class SuperBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(SuperBertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = SuperBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = SuperLinear(config.hidden_size, self.config.num_labels)

        self.apply(self.init_bert_weights)

    def set_sample_config(self, subbert_config):
        self.bert.set_sample_config(subbert_config)
        self.classifier.set_sample_config(subbert_config['sample_hidden_size'], self.num_labels)

    def calc_sampled_param_num(self):
        return self.bert.calc_sampled_param_num()

    def save_pretrained(self, save_directory):

        assert os.path.isdir(save_directory), "Saving path should be a directory where " \
                                              "the model and configuration can be saved"

        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self

        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("Model weights saved in {}".format(output_model_file))

    def forward(self, input_ids, subbert_config, attention_mask=None, token_type_ids=None,
                labels=None, kd_infer=False):

        if not kd_infer:
            encoded_layers, pooled_output = self.bert(input_ids, subbert_config,
                                                      attention_mask=attention_mask,
                                                      token_type_ids=token_type_ids,
                                                      kd=False, kd_infer=False)
        else:
            last_rep, pooled_output = self.bert(input_ids, subbert_config, attention_mask=attention_mask,
                                                token_type_ids=token_type_ids, kd=True, kd_infer=True)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class SuperBertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super(SuperBertForQuestionAnswering, self).__init__(config)
        self.bert = SuperBertModel(config)
        self.qa_outputs = SuperLinear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)

    def set_sample_config(self, subbert_config):
        self.bert.set_sample_config(subbert_config)
        self.qa_outputs.set_sample_config(subbert_config['sample_hidden_size'], 2)

    def calc_sampled_param_num(self):
        return self.bert.calc_sampled_param_num()

    def save_pretrained(self, save_directory):

        assert os.path.isdir(save_directory), "Saving path should be a directory where " \
                                              "the model and configuration can be saved"

        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self

        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("Model weights saved in {}".format(output_model_file))

    def forward(self, input_ids, subbert_config, attention_mask=None, token_type_ids=None,
                start_positions=None, end_positions=None, kd_infer=False):

        if not kd_infer:
            encoded_layers, pooled_output = self.bert(input_ids, subbert_config,
                                                      attention_mask=attention_mask,
                                                      token_type_ids=token_type_ids,
                                                      kd=False, kd_infer=False)
            last_sequence_output = encoded_layers[-1]
        else:
            last_sequence_output, pooled_output = self.bert(input_ids, subbert_config, attention_mask=attention_mask,
                                                            token_type_ids=token_type_ids, kd=True, kd_infer=True)

        logits = self.qa_outputs(last_sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        logits = (start_logits, end_logits)
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss

        return logits


