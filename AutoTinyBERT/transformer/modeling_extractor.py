# coding=utf-8
# 2021.12.30-Changed for modeling extractor
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


class SuperBertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(SuperBertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

        self.hidden_size = hidden_size

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

    def prune_layer_norm_layer(self, layer, sample_hidden_dim, dim=0):
        """Interpolates between linear layers.
            If the second layer is not provided, interpolate with random"""
        dim = (dim + 100) % 2
        W = layer.weight[:sample_hidden_dim].clone().detach()
        if layer.bias is not None:
            if dim == 1:
                b = layer.bias.clone().detach()
            else:
                b = layer.bias[:sample_hidden_dim].clone().detach()

        new_layer = SuperBertLayerNorm(sample_hidden_dim)
        new_layer.weight.requires_grad = False
        new_layer.weight.copy_(W.contiguous())
        new_layer.weight.requires_grad = True
        if layer.bias is not None:
            new_layer.bias.requires_grad = False
            new_layer.bias.copy_(b.contiguous())
            new_layer.bias.requires_grad = True
        return new_layer


class SuperLinear(nn.Linear):
    def __init__(self, super_in_dim, super_out_dim, bias=True, uniform_=None, non_linear='linear'):
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

        return self._sample_parameters(in_index=in_index, out_index=out_index)

    def _sample_parameters(self, in_index=None, out_index=None):

        new_layer = nn.Linear(self.sample_in_dim, self.sample_out_dim)
        new_weight_value = sample_weight(self.weight, self.sample_in_dim, self.sample_out_dim,
                                         in_index=in_index, out_index=out_index)

        new_layer.weight[:] = new_weight_value
        self.samples['bias'] = self.bias
        if self.bias is not None:
            new_bias_value = sample_bias(self.bias, self.sample_out_dim, out_index=out_index)

        new_layer.bias[:] = new_bias_value
        return new_layer

    def calc_sampled_param_num(self):
        assert 'weight' in self.samples.keys()
        weight_numel = self.samples['weight'].numel()

        if self.samples['bias'] is not None:
            bias_numel = self.samples['bias'].numel()
        else:
            bias_numel = 0

        return weight_numel + bias_numel


# why use two stages ?
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
                 num_labels=-1,
                 pre_trained='',
                 training=''):
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
            self.pre_trained = pre_trained
            self.training = training
            self.num_labels = num_labels
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


class SuperEmbedding(nn.Module):
    def __init__(self, dict_size, embd_size, padding_idx=None):
        super(SuperEmbedding, self).__init__()
        self.embedding = nn.Embedding(dict_size, embd_size, padding_idx=padding_idx)
        self.sample_embedding_weight = None
        self.dict_size = dict_size
        self.padding_idx = padding_idx

    def set_sample_config(self, sample_embed_dim):
        new_embedding = nn.Embedding(self.dict_size, sample_embed_dim, padding_idx=self.padding_idx)
        new_embedding_value = self.embedding.weight[..., :sample_embed_dim].clone()
        new_embedding.weight[:] = new_embedding_value

        self.embedding = new_embedding

    def calc_sampled_param_num(self):
        weight_numel = self.sample_embedding_weight.numel()
        assert weight_numel !=0
        return weight_numel


class SuperBertEmbeddings(nn.Module):
    def __init__(self, config):
        super(SuperBertEmbeddings, self).__init__()
        self.word_embeddings = SuperEmbedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = SuperEmbedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = SuperEmbedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = SuperBertLayerNorm(config.hidden_size, eps=1e-12) #config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sample_embed_dim = None

    def set_sample_config(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim
        self.word_embeddings.set_sample_config(sample_embed_dim)
        self.position_embeddings.set_sample_config(sample_embed_dim)
        self.token_type_embeddings.set_sample_config(sample_embed_dim)
        self.LayerNorm = self.LayerNorm.prune_layer_norm_layer(self.LayerNorm,
                                                               sample_embed_dim)

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


class SuperBertSelfAttention(nn.Module):
    def __init__(self, config):
        super(SuperBertSelfAttention, self).__init__()

        ### Main chages ###
        ### BEGIN
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
        #### END

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
        self.query = self.query.set_sample_config(sample_embed_dim, qkv_size, in_index=in_index, out_index=out_index)
        self.key = self.key.set_sample_config(sample_embed_dim, qkv_size, in_index=in_index, out_index=out_index)
        self.value = self.value.set_sample_config(sample_embed_dim, qkv_size, in_index=in_index, out_index=out_index)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.sample_num_attention_head, self.sample_attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def calc_sampled_param_num(self):
        query_numel = self.query.calc_sampled_param_num()
        key_numel = self.key.calc_sampled_param_num()
        value_numel = self.value.calc_sampled_param_num()

        logger.info('query_numel: {}\n'.format(query_numel))
        logger.info('key_numel: {}\n'.format(key_numel))
        logger.info('value_numel: {}\n'.format(value_numel))
        return query_numel + key_numel + value_numel


class SuperBertSelfOutput(nn.Module):
    def __init__(self, config):
        super(SuperBertSelfOutput, self).__init__()
        ### Main Changes
        ### BEGIN
        try:
            qkv_size = config.qkv_size
        except:
            qkv_size = config.hidden_size

        self.dense = SuperLinear(qkv_size, config.hidden_size)
        ### END
        # self.dense = SuperLinear(config.hidden_size, config.hidden_size)

        self.LayerNorm = SuperBertLayerNorm(config.hidden_size, eps=1e-12) #config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def set_sample_config(self, qkv_size, sample_embed_dim, in_index=None):
        self.dense = self.dense.set_sample_config(qkv_size, sample_embed_dim, in_index=in_index)
        self.LayerNorm = self.LayerNorm.prune_layer_norm_layer(self.LayerNorm,
                               sample_embed_dim)

    def calc_sampled_param_num(self):
        dense_numel = self.dense.calc_sampled_param_num()
        ln_numel = self.LayerNorm.calc_sampled_param_num()

        logger.info('dense_numel: {}\n'.format(dense_numel))
        logger.info('ln_numel: {}\n'.format(ln_numel))

        return dense_numel + ln_numel

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SuperBertAttention(nn.Module):
    def __init__(self, config):
        super(SuperBertAttention, self).__init__()
        self.self = SuperBertSelfAttention(config)
        self.output = SuperBertSelfOutput(config)

    def set_sample_config(self, sample_embed_dim, num_attention_head, qkv_size, in_index=None, out_index=None):
        self.self.set_sample_config(sample_embed_dim, num_attention_head, qkv_size, in_index=in_index,
                                    out_index=out_index)
        self.output.set_sample_config(qkv_size, sample_embed_dim, in_index=out_index)

    def calc_sampled_param_num(self):
        self_numel = self.self.calc_sampled_param_num()
        output_numel = self.output.calc_sampled_param_num()

        logger.info('self_numel: {}\n'.format(self_numel))
        logger.info('output_numel: {}\n'.format(output_numel))

        return self_numel + output_numel

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        self_output, layer_att = self_output
        attention_output = self.output(self_output, input_tensor)
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
        self.dense = self.dense.set_sample_config(sample_embed_dim, intermediate_size)

    def calc_sampled_param_num(self):
        dense_numel = self.dense.calc_sampled_param_num()

        logger.info('dense_numel: {}\n'.format(dense_numel))
        return dense_numel


class SuperBertOutput(nn.Module):
    def __init__(self, config):
        super(SuperBertOutput, self).__init__()
        self.dense = SuperLinear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = SuperBertLayerNorm(config.hidden_size, eps=1e-12) #config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def set_sample_config(self, intermediate_size, sample_embed_dim):
        self.dense = self.dense.set_sample_config(intermediate_size, sample_embed_dim)
        self.LayerNorm = self.LayerNorm.prune_layer_norm_layer(self.LayerNorm, sample_embed_dim)

    def calc_sampled_param_num(self):
        dense_numel = self.dense.calc_sampled_param_num()
        ln_numel = self.LayerNorm.calc_sampled_param_num()

        logger.info('dense_numel: {}\n'.format(dense_numel))
        logger.info('ln_numel: {}\n'.format(ln_numel))
        return dense_numel + ln_numel


class SuperBertLayer(nn.Module):
    def __init__(self, config):
        super(SuperBertLayer, self).__init__()
        self.attention = SuperBertAttention(config)
        self.intermediate = SuperBertIntermediate(config)
        self.output = SuperBertOutput(config)

    def set_sample_config(self, sample_embed_dim, intermediate_size, num_attention_head, qkv_size,
                          in_index=None, out_index=None):
        self.attention.set_sample_config(sample_embed_dim, num_attention_head, qkv_size,
                                         in_index=in_index, out_index=out_index)
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


class SuperBertEncoder(nn.Module):
    def __init__(self, config):
        super(SuperBertEncoder, self).__init__()
        layer = SuperBertLayer(config)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])
        self.sample_layer_num = None

    def set_sample_config(self, subbert_config,
                          in_index=None, out_index=None):
        self.sample_layer_num = subbert_config['sample_layer_num']

        self.layers = self.layers[:self.sample_layer_num]
        sample_embed_dim = subbert_config['sample_hidden_size']
        num_attention_heads = subbert_config['sample_num_attention_heads']
        itermediate_sizes = subbert_config['sample_intermediate_sizes']
        qkv_sizes = subbert_config['sample_qkv_sizes']
        for layer, num_attention_head, intermediate_size, qkv_size in zip(self.layers[:self.sample_layer_num],
                                                                          num_attention_heads,
                                                                          itermediate_sizes,
                                                                          qkv_sizes):
            layer.set_sample_config(sample_embed_dim, intermediate_size, num_attention_head, qkv_size,
                                    in_index=in_index, out_index=out_index)

    def calc_sampled_param_num(self):
        layers_numel = 0

        for layer in self.layers[:self.sample_layer_num]:
            layers_numel += layer.calc_sampled_param_num()

        logger.info('layer_numel: {}'.format(layers_numel))

        return layers_numel


class SuperBertPooler(nn.Module):
    def __init__(self, config):
        super(SuperBertPooler, self).__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = SuperLinear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def set_sample_config(self, sample_hidden_dim):
        self.dense = self.dense.set_sample_config(sample_hidden_dim, sample_hidden_dim)

    def calc_sampled_param_num(self):
        dense_numel = self.dense.calc_sampled_param_num()
        logger.info('dense_numel: {}'.format(dense_numel))

        return dense_numel


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
        model = cls(config, *inputs, **kwargs)
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
            ####
            if 'embeddings.weight' in key:
                new_key = key.replace('embeddings.weight', 'embeddings.embedding.weight')
            if 'layer.' in key:
                new_key = key.replace('layer.', 'layers.')

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

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'electra' in key:
                new_key = key.replace
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

        logger.info('load finished!')

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
    def __init__(self, config):
        super(SuperBertModel, self).__init__(config)
        self.embeddings = SuperBertEmbeddings(config)
        self.encoder = SuperBertEncoder(config)
        self.pooler = SuperBertPooler(config)
        self.hidden_size = config.hidden_size
        self.head_number = config.num_attention_heads
        self.apply(self.init_bert_weights)

    def set_sample_config(self, subbert_config, kd=True):
        head_number = self.head_number
        hidden_size = self.hidden_size
        sample_qkv_size = subbert_config['sample_qkv_sizes'][0]

        in_dim_per_head = int(hidden_size / head_number)
        in_sample_per_head = int(sample_qkv_size / head_number)

        in_out_index = []
        for i in range(head_number):
            start_ind = in_dim_per_head * i
            in_out_index.extend(range(start_ind, start_ind + in_sample_per_head))

        if not kd:
            in_out_index = []
            in_out_index.extend(range(0, sample_qkv_size))

        in_out_index = torch.tensor(in_out_index)
        # in_out_index.to(self.embeddings.device)

        self.embeddings.set_sample_config(subbert_config['sample_hidden_size'])
        self.encoder.set_sample_config(subbert_config, out_index=in_out_index)
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
