# coding=utf-8
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

import numpy as np

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

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
    """Configuration class to store the configuration of a `BertModel`.
    """

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
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            layer_norm_eps: The epsilon used by LayerNorm.
        """
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


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")


    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
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


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, logits_fit=False):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        context_layer = torch.matmul(self.dropout(attention_probs), value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer, attention_scores


class GlobalLocalStandardAttention(nn.Module):
    def __init__(self, config):
        super(GlobalLocalStandardAttention, self).__init__()

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # b, h, l, d
        key_layer = self.transpose_for_scores(mixed_key_layer)  # b, h, l, d
        value_layer = self.transpose_for_scores(mixed_value_layer)  # b, h, l, d

        # First stage to get new_global_vec
        global_query_vec = query_layer[:, :, 0:1, :]  # b, h, 1, d
        global_scores = torch.matmul(global_query_vec, key_layer.transpose(-2, -1))  # b, h, 1, l
        global_scores = global_scores / math.sqrt(self.attention_head_size) + attention_mask

        global_attentions = nn.Softmax(dim=-1)(global_scores)
        new_global_vec = torch.matmul(global_attentions, value_layer)  # b, h, 1, d
        # Second stage to get new_local_vecs
        local_query_vecs = query_layer[:, :, 1:, :].unsqueeze(-2)  # b, h, l-1, 1, d
        local_key_layer = key_layer[:, :, 1:, :].unsqueeze(-1)  # b, h, l-1, d, 1
        local_key_global_vec = key_layer[:, :, 0:1, :].unsqueeze(-1).expand_as(local_key_layer)  # b, h, l-1, d, 1
        local_key_layer = torch.cat([local_key_layer, local_key_global_vec], dim=-1)

        local_value_layer = value_layer[:, :, 1:, :].unsqueeze(-1)
        local_value_global_vec = value_layer[:, :, 0:1, :].unsqueeze(-1).expand_as(local_value_layer)
        local_value_layer = torch.cat([local_value_layer, local_value_global_vec], dim=-1)
        local_value_layer = local_value_layer.transpose(-2, -1)  # b, h, l-1, 2, d

        local_scores = torch.matmul(local_query_vecs, local_key_layer) / math.sqrt(
            self.attention_head_size)  # b, h, l-1, 1, 2
        local_attentions = nn.Softmax(dim=-1)(local_scores)
        new_local_vecs = torch.matmul(local_attentions, local_value_layer).squeeze(-2)  # b, h, l-1, d

        context_layer = torch.cat([new_global_vec, new_local_vecs], dim=-2)  # b, h, l, d
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # b, l, h, d
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  # b, l, d'
        return context_layer, local_attentions


class GlobalLocalAttention(nn.Module):
    def __init__(self, config):
        super(GlobalLocalAttention, self).__init__()

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key_value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key_value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # b, h, l, d
        key_layer = self.transpose_for_scores(mixed_key_layer)  # b, h, l, d
        value_layer = self.transpose_for_scores(mixed_key_layer)  # b, h, l, d

        # First stage to get new_global_vec
        global_query_vec = query_layer[:, :, 0:1, :]  # b, h, 1, d
        global_scores = torch.matmul(global_query_vec, key_layer.transpose(-2, -1))  # b, h, 1, l
        global_scores = global_scores / math.sqrt(self.attention_head_size) + attention_mask

        global_attentions = nn.Softmax(dim=-1)(global_scores)
        new_global_vec = torch.matmul(global_attentions, value_layer)  # b, h, 1, d
        # Second stage to get new_local_vecs
        local_query_vecs = query_layer[:, :, 1:, :].unsqueeze(-2)  # b, h, l-1, 1, d
        local_key_layer = key_layer[:, :, 1:, :].unsqueeze(-1)  # b, h, l-1, d, 1
        local_key_global_vec = new_global_vec.unsqueeze(-1).expand_as(local_key_layer)  # b, h, l-1, d, 1

        local_key_layer = torch.cat([local_key_layer, local_key_global_vec], dim=-1)  # b, h, l-1, d, 2
        local_value_layer = local_key_layer.transpose(-2, -1)  # b, h, l-1, 2, d
        local_scores = torch.matmul(local_query_vecs, local_key_layer) / math.sqrt(self.attention_head_size)
        # b, h, l-1, 1, 2
        local_attentions = nn.Softmax(dim=-1)(local_scores)

        new_local_vecs = torch.matmul(local_attentions, local_value_layer).squeeze(-2)  # b, h, l-1, d
        context_layer = torch.cat([new_global_vec, new_local_vecs], dim=-2)  # b, h, l, d
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # b, l, h, d
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  # b, l, d'
        return context_layer, local_attentions


class GlobalLocalAttentionV2(nn.Module):
    def __init__(self, config):
        super(GlobalLocalAttentionV2, self).__init__()

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key_value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key_value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # b, h, l, d
        key_layer = self.transpose_for_scores(mixed_key_layer)  # b, h, l, d
        value_layer = self.transpose_for_scores(mixed_key_layer)  # b, h, l, d

        # First stage to get new_global_vec
        global_query_vec = query_layer[:, :, 0:1, :]  # b, h, 1, d
        global_scores = torch.matmul(global_query_vec, key_layer.transpose(-2, -1))  # b, h, 1, l
        global_scores = global_scores / math.sqrt(self.attention_head_size) + attention_mask

        global_attentions = nn.Softmax(dim=-1)(global_scores)
        new_global_vec = torch.matmul(global_attentions, value_layer)  # b, h, 1, d
        # Second stage to get new_local_vecs
        local_query_vecs = query_layer[:, :, 1:, :].unsqueeze(-2)  # b, h, l-1, 1, d
        local_key_layer = key_layer[:, :, 1:, :].unsqueeze(-1)  # b, h, l-1, d, 1
        local_last_key_layer = key_layer[:, :, :-1, :].unsqueeze(-1)
        local_next_key_layer = torch.cat([key_layer[:, :, 2:, :], key_layer[:, :, 0:1, :]], dim=-2).unsqueeze(-1)
        local_key_global_vec = new_global_vec.unsqueeze(-1).expand_as(local_key_layer)  # b, h, l-1, d, 1

        local_key_layer = torch.cat([local_key_layer, local_key_global_vec, local_last_key_layer, local_next_key_layer],
                                    dim=-1)  # b, h, l-1, d, 4
        local_value_layer = local_key_layer.transpose(-2, -1)  # b, h, l-1, 4, d
        local_scores = torch.matmul(local_query_vecs, local_key_layer) / math.sqrt(
            self.attention_head_size)  # b, h, l-1, 1, 4
        local_attentions = nn.Softmax(dim=-1)(local_scores)

        new_local_vecs = torch.matmul(local_attentions, local_value_layer).squeeze(-2)  # b, h, l-1, d
        context_layer = torch.cat([new_global_vec, new_local_vecs], dim=-2)  # b, h, l, d
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # b, l, h, d
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  # b, l, d'
        return context_layer, local_attentions


def _generate_relative_positions_matrix(length, max_relative_position,
                                        cache=False):
    """Generates matrix of relative positions between inputs."""
    if not cache:
        range_vec = torch.arange(length)
        range_mat = range_vec.repeat(length).view(length, length)
        distance_mat = range_mat - torch.t(range_mat)
    else:
        distance_mat = torch.arange(-length + 1, 1, 1).unsqueeze(0)

    distance_mat_clipped = torch.clamp(distance_mat, -max_relative_position, max_relative_position)
    final_mat = distance_mat_clipped + max_relative_position
    return final_mat


def _generate_relative_positions_embeddings(length, depth, max_relative_position=64):
    vocab_size = max_relative_position * 2 + 1
    range_vec = torch.arange(length)
    range_mat = range_vec.repeat(length).view(length, length)
    distance_mat = range_mat - torch.t(range_mat)
    distance_mat_clipped = torch.clamp(distance_mat, -max_relative_position, max_relative_position)
    final_mat = distance_mat_clipped + max_relative_position
    embeddings_table = np.zeros([vocab_size, depth])
    for pos in range(vocab_size):
        for i in range(depth // 2):
            embeddings_table[pos, 2 * i] = np.sin(pos / np.power(10000, 2 * i / depth))
            embeddings_table[pos, 2 * i + 1] = np.cos(pos / np.power(10000, 2 * i / depth))

    embeddings_table_tensor = torch.tensor(embeddings_table).float()
    flat_relative_positions_matrix = final_mat.view(-1)
    one_hot_relative_positions_matrix = torch.nn.functional.one_hot(flat_relative_positions_matrix,
                                                                    num_classes=vocab_size).float()
    embeddings = torch.matmul(one_hot_relative_positions_matrix, embeddings_table_tensor)
    my_shape = list(final_mat.size())
    my_shape.append(depth)
    embeddings = embeddings.view(my_shape)
    return embeddings


### Test: print(_generate_relative_positions_embeddings(6, 32, 4)[0, 0, :])

class NeZhaSelfAttention(nn.Module):
    def __init__(self, config):
        super(NeZhaSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.relative_positions_embeddings = _generate_relative_positions_embeddings(
            length=256, depth=self.attention_head_size, max_relative_position=64)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        try:
            self.use_minilm_trick = config.use_minilm_trick
        except:
            self.use_minilm_trick = False

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        device = 'cpu'
        if hidden_states.is_cuda:
            device = hidden_states.get_device()
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        batch_size, num_attention_heads, from_seq_length, to_seq_length = attention_scores.size()

        relations_keys = self.relative_positions_embeddings.clone()[:to_seq_length, :to_seq_length, :].detach().to(
            device)
        # relations_keys = embeddings.clone().detach().to(device)
        query_layer_t = query_layer.permute(2, 0, 1, 3)
        query_layer_r = query_layer_t.contiguous().view(from_seq_length, batch_size * num_attention_heads,
                                                        self.attention_head_size)
        key_position_scores = torch.matmul(query_layer_r, relations_keys.permute(0, 2, 1))
        key_position_scores_r = key_position_scores.view(from_seq_length, batch_size,
                                                         num_attention_heads, from_seq_length)
        key_position_scores_r_t = key_position_scores_r.permute(1, 2, 0, 3)
        attention_scores = attention_scores + key_position_scores_r_t
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        if self.use_minilm_trick:
            attention_vv_scores = torch.matmul(value_layer, value_layer.transpose(-1, -2))
            attention_vv_scores = attention_vv_scores / math.sqrt(self.attention_head_size)
            attention_vv_scores = attention_vv_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        relations_values = self.relative_positions_embeddings.clone()[:to_seq_length, :to_seq_length, :].detach().to(
            device)
        attention_probs_t = attention_probs.permute(2, 0, 1, 3)
        attentions_probs_r = attention_probs_t.contiguous().view(from_seq_length, batch_size * num_attention_heads,
                                                                 to_seq_length)
        value_position_scores = torch.matmul(attentions_probs_r, relations_values)
        value_position_scores_r = value_position_scores.view(from_seq_length, batch_size,
                                                             num_attention_heads,
                                                             self.attention_head_size)
        value_position_scores_r_t = value_position_scores_r.permute(1, 2, 0, 3)
        context_layer = context_layer + value_position_scores_r_t

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        if self.use_minilm_trick:
            return context_layer, (attention_scores, attention_vv_scores)
        else:
            return context_layer, attention_scores


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        try:
            self.use_relative_position = config.use_relative_position
        except:
            self.use_relative_position = False

        try:
            self.use_global_local_attn = config.use_global_local_attn
        except:
            self.use_global_local_attn = False

        if self.use_relative_position:
            self.self = NeZhaSelfAttention(config)
        elif self.use_global_local_attn:
            self.self = GlobalLocalAttention(config)
        else:
            self.self = BertSelfAttention(config)

        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, logits_fit=False):
        self_output = self.self(input_tensor, attention_mask, logits_fit=logits_fit)
        self_output, layer_att = self_output
        attention_output = self.output(self_output, input_tensor)
        return attention_output, layer_att


class BertIntermediate(nn.Module):
    def __init__(self, config, factor=1):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, int(config.intermediate_size * factor))
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config, factor=1):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(int(config.intermediate_size * factor), int(config.hidden_size * factor))
        self.LayerNorm = BertLayerNorm(int(config.hidden_size * factor), eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.factor = factor

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if self.factor == 1:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        else:
            hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config, last_layer=False):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

        try:
            self.use_diff_last_ffn = config.use_diff_last_ffn
        except:
            self.use_diff_last_ffn = -1

        try:
            self.remove_ffn = config.remove_ffn
        except:
            self.remove_ffn = False

        if last_layer:
            self.intermediate = BertIntermediate(config, factor=self.use_diff_last_ffn)
            self.output = BertOutput(config, factor=self.use_diff_last_ffn)

    def forward(self, hidden_states, attention_mask, logits_fit=False):
        attention_output = self.attention(hidden_states, attention_mask, logits_fit=logits_fit)
        attention_output, layer_att = attention_output

        if not self.remove_ffn:
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)
        else:
            layer_output = attention_output
        return layer_output, layer_att


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        try:
            self.use_diff_last_ffn = config.use_diff_last_ffn
        except:
            self.use_diff_last_ffn = -1

        try:
            self.qa_bert = config.qa_bert
        except:
            self.qa_bert = False

        if self.use_diff_last_ffn < 0:
            self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])
        else:
            last_layer = BertLayer(config, last_layer=True)
            self.layer = nn.ModuleList(
                [copy.deepcopy(layer) for _ in range(config.num_hidden_layers - 1)] + [last_layer])

    def forward(self, hidden_states, attention_mask, original_attention_mask=None, logits_fit=False):
        all_encoder_layers = []
        all_encoder_att = []
        layer_num = len(self.layer)

        for i, layer_module in enumerate(self.layer):
            if self.qa_bert:
                factor = pow(2, layer_num - 1 - i)
                previous_shape = hidden_states.size()

                if i != 0:
                    hidden_states = hidden_states.view(previous_shape[0] // (factor * 2), factor * 2,
                                                       previous_shape[1], previous_shape[2])
                    hidden_states = hidden_states.view(hidden_states.shape[0],
                                                       hidden_states.shape[1] * hidden_states.shape[2],
                                                       hidden_states.shape[3])

                    # logger.info('i: {}; previous_shape: {}; hidden_state#1: {}'.format(i, previous_shape,
                    #                                                                    hidden_states.size()))

                hidden_states = hidden_states.view(hidden_states.shape[0], factor,
                                                   hidden_states.shape[1] // factor,
                                                   hidden_states.shape[2])

                hidden_states = hidden_states.view(hidden_states.shape[0] * hidden_states.shape[1],
                                                   hidden_states.shape[2],
                                                   hidden_states.shape[3])

                # logger.info('hidden_state#2: {}'.format(hidden_states.size()))

                previous_shape = original_attention_mask.size()
                extended_attention_mask = original_attention_mask.view(previous_shape[0], factor,
                                                                       previous_shape[1] // factor)
                extended_attention_mask = extended_attention_mask.view(previous_shape[0] * factor,
                                                                       previous_shape[1] // factor)
                extended_attention_mask = extended_attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = extended_attention_mask.to(
                    dtype=next(self.parameters()).dtype)  # fp16 compatibility
                extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
                attention_mask = extended_attention_mask
                # logger.info('previous_shape: {}, extended_attention_mask: {},'
                #             ' attention_mask: {}'.format(previous_shape, extended_attention_mask.size(),
                #                                          attention_mask.size()))

            all_encoder_layers.append(hidden_states)
            hidden_states = layer_module(all_encoder_layers[i], attention_mask, logits_fit=logits_fit)
            hidden_states, layer_att = hidden_states
            all_encoder_att.append(layer_att)

        all_encoder_layers.append(hidden_states)
        return all_encoder_layers, all_encoder_att


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        try:
            self.use_diff_last_ffn = config.use_diff_last_ffn
        except:
            self.use_diff_last_ffn = -1

        if self.use_diff_last_ffn < 0:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        else:
            self.dense = nn.Linear(int(config.hidden_size * self.use_diff_last_ffn),
                                   config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

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
        elif isinstance(module, BertLayerNorm):
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
        model = cls(config, *inputs, **kwargs)
        return model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
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
        model = cls(config, *inputs, **kwargs)
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
            if 'tcpnet' in key:
                new_key = key.replace('tcpnet', 'bert')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            ####
            if 'electra' in key:
                new_key = key.replace('electra.', 'bert.')

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


class BertModel(BertPreTrainedModel):
    def __init__(self, config, fit_size=768):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_attention_mask=False,
                model_distillation=True, last_strategy=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output, extended_attention_mask, attention_mask)
        encoded_layers, attention_layers = encoded_layers
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if output_attention_mask:
            return encoded_layers, attention_layers, pooled_output, extended_attention_mask
        if model_distillation:
            if last_strategy:
                return encoded_layers[-1], attention_layers[-1]
            return encoded_layers, attention_layers
        return encoded_layers, attention_layers, pooled_output


class KDBertModel(BertPreTrainedModel):
    def __init__(self, config, fit_size=768):
        super(KDBertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.dense_fit = nn.Linear(config.hidden_size, fit_size)
        self.fit_size = fit_size

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, is_student=False, logits_fit=False):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output, extended_attention_mask, attention_mask, logits_fit=logits_fit)
        encoded_layers, attention_layers = encoded_layers

        last_rep = encoded_layers[-1]
        if is_student:
            last_rep = self.dense_fit(last_rep)
        last_att = attention_layers[-1]

        return last_rep, last_att


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        # Need to unty it when we separate the dimensions of hidden and emb
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or \
                (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)
        self.bert = BertModel(config)

        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                masked_lm_labels=None, next_sentence_label=None):
        encoder_layers, _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = encoder_layers[-1]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

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


class TinyBertDistill(BertPreTrainedModel):
    def __init__(self, config, fit_size=768):
        super(TinyBertDistill, self).__init__(config)
        self.config = config
        self.fit_size = fit_size
        self.bert = BertModel(config)
        if config.hidden_size != fit_size:
            self.fit_denses = nn.ModuleList([nn.Linear(config.hidden_size, fit_size)
                                             for _ in range(config.num_hidden_layers + 1)])
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        encoded_layers, attention_layers = self.bert(input_ids, token_type_ids,
                                                     attention_mask, model_distillation=True)
        if self.config.hidden_size != self.fit_size:
            encoded_layers = [self.fit_denses[i](encoded_layers[i]) for i in range(len(self.fit_denses))]
        return encoded_layers, attention_layers


class TinyBertForPreTraining(BertPreTrainedModel):
    def __init__(self, config, fit_size=768):
        super(TinyBertForPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight)

        self.apply(self.init_bert_weights)

        try:
            self.use_diff_last_ffn = config.use_diff_last_ffn
            self.fit_dense_2 = nn.Linear(int(config.hidden_size * self.use_diff_last_ffn), fit_size)
        except:
            self.use_diff_last_ffn = -1

        try:
            self.multiple_fit = config.multple_fit
        except:
            self.multiple_fit = False

        if not self.multiple_fit:
            self.fit_dense = nn.Linear(config.hidden_size, fit_size)
        else:
            self.fit_dense = nn.ModuleList([nn.Linear(config.hidden_size, fit_size)
                                            for _ in range(config.num_hidden_layers + 1)])

    def forward(self, input_ids, token_type_ids=None,
                attention_mask=None, masked_lm_labels=None,
                next_sentence_label=None, labels=None):
        sequence_output, att_output, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask)
        tmp = []

        if not self.multiple_fit:
            for s_id, sequence_layer in enumerate(sequence_output[:-1]):
                tmp.append(self.fit_dense(sequence_layer))
        else:
            for s_id, sequence_layer in enumerate(sequence_output[:-1]):
                tmp.append(self.fit_dense[s_id](sequence_layer))

        if self.use_diff_last_ffn < 0:
            tmp.append(self.fit_dense(sequence_output[-1]))
        else:
            tmp.append(self.fit_dense_2(sequence_output[-1]))

        sequence_output = tmp

        return att_output, sequence_output


class BertForJoint(BertPreTrainedModel):
    def __init__(self, config, num_intent_labels, num_slot_labels):
        super(BertForJoint, self).__init__(config)
        self.num_intent_labels = num_intent_labels
        self.num_slot_labels = num_slot_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.intent_classifier = nn.Linear(config.hidden_size, num_intent_labels)
        self.slot_classifier = nn.Linear(config.hidden_size, num_slot_labels)
        self.apply(self.init_bert_weights)

    def soft_cross_entropy(self, predicts, targets):
        student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        return - (targets_prob * student_likelihood).mean()

    def forward(self, input_ids, token_type_ids=None,
                attention_mask=None, intent_labels=None, slot_labels=None, soft_cross_entropy=False):
        encoded_layers, attention_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        intent_logits = self.intent_classifier(self.dropout(pooled_output))

        last_encoded_layer = encoded_layers[-1]
        slot_logits = self.slot_classifier(self.dropout(last_encoded_layer))
        tmp = []
        if intent_labels is not None and slot_labels is not None:
            if soft_cross_entropy:
                intent_loss = self.soft_cross_entropy(intent_logits, intent_labels)
                slot_loss = self.soft_cross_entropy(slot_logits, slot_labels)
            else:
                loss_fct = CrossEntropyLoss()
                intent_loss = loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_labels.view(-1))
                if attention_mask is not None:
                    active_slot_loss = attention_mask.view(-1) == 1
                    active_slot_logits = slot_logits.view(-1, self.num_slot_labels)[active_slot_loss]
                    active_slot_labels = slot_labels.view(-1)[active_slot_loss]
                    slot_loss = loss_fct(active_slot_logits, active_slot_labels)
                else:
                    slot_loss = loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels.view(-1))

            return intent_loss, slot_loss
        else:
            return intent_logits, slot_logits


class BertForMaskedLM(BertPreTrainedModel):
    """BERT model with the masked language modeling head.
    This module comprises the BERT model followed by the masked language modeling head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]

    Outputs:
        if `masked_lm_labels` is  not `None`:
            Outputs the masked language modeling loss.
        if `masked_lm_labels` is `None`:
            Outputs the masked language modeling logits of shape [batch_size, sequence_length, vocab_size].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForMaskedLM(config)
    masked_lm_logits_scores = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                output_att=False, infer=False):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask,
                                       output_all_encoded_layers=True, output_att=output_att)

        if output_att:
            sequence_output, att_output = sequence_output
        prediction_scores = self.cls(sequence_output[-1])

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            if not output_att:
                return masked_lm_loss
            else:
                return masked_lm_loss, att_output
        else:
            if not output_att:
                return prediction_scores
            else:
                return prediction_scores, att_output


class BertForJointLSTM(BertPreTrainedModel):
    def __init__(self, config, num_intent_labels, num_slot_labels):
        super(BertForJointLSTM, self).__init__(config)
        self.num_intent_labels = num_intent_labels
        self.num_slot_labels = num_slot_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.intent_classifier = nn.Linear(config.hidden_size, num_intent_labels)
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=300,
            batch_first=True,
            bidirectional=True

        )
        self.slot_classifier = nn.Linear(300 * 2, num_slot_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None,
                attention_mask=None, intent_labels=None, slot_labels=None):
        encoded_layers, attention_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        intent_logits = self.intent_classifier(self.dropout(pooled_output))

        last_encoded_layer = encoded_layers[-1]
        slot_logits, _ = self.lstm(last_encoded_layer)
        slot_logits = self.slot_classifier(slot_logits)
        tmp = []
        if intent_labels is not None and slot_labels is not None:
            loss_fct = CrossEntropyLoss()
            intent_loss = loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_labels.view(-1))
            if attention_mask is not None:
                active_slot_loss = attention_mask.view(-1) == 1
                active_slot_logits = slot_logits.view(-1, self.num_slot_labels)[active_slot_loss]
                active_slot_labels = slot_labels.view(-1)[active_slot_loss]
                slot_loss = loss_fct(active_slot_logits, active_slot_labels)
            else:
                slot_loss = loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels.view(-1))

            return intent_loss, slot_loss
        else:
            return intent_logits, slot_logits


class TinyBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels, fit_size=768):
        super(TinyBertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.fit_dense = nn.Linear(config.hidden_size, fit_size)
        self.apply(self.init_bert_weights)

        try:
            self.use_diff_last_ffn = config.use_diff_last_ffn
            self.fit_dense_2 = nn.Linear(int(config.hidden_size * self.use_diff_last_ffn), fit_size)
        except:
            self.use_diff_last_ffn = -1

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, is_student=False):

        sequence_output, att_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)

        logits = self.classifier(torch.relu(pooled_output))

        tmp = []
        for s_id, sequence_layer in enumerate(sequence_output[:-1]):
            tmp.append(self.fit_dense(sequence_layer))

        if self.use_diff_last_ffn < 0:
            tmp.append(self.fit_dense(sequence_output[-1]))
        else:
            tmp.append(self.fit_dense_2(sequence_output[-1]))

        sequence_output = tmp
        return logits, att_output, sequence_output
