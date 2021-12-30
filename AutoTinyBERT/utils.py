# coding=utf-8
# Copyright 2021 Huawei Technologies Co., Ltd.
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

import random


def sample_arch_4_kd(layer_numbers, hidden_sizes, ffn_sizes, qkv_sizes,
                     reset_rand_seed=False, rand_seed=0):

    if reset_rand_seed:
        random.seed(rand_seed)

    config = dict()

    layer_num = random.choice(layer_numbers)

    config['sample_layer_num'] = layer_num
    config['sample_hidden_size'] = random.choice(hidden_sizes)
    config['sample_intermediate_sizes'] = [random.choice(ffn_sizes)] * layer_num
    config['sample_num_attention_heads'] = [12] * layer_num
    config['sample_qkv_sizes'] = [random.choice(qkv_sizes)] * layer_num
    return config


def sample_arch_4_mlm(layer_numbers, hidden_sizes, ffn_sizes,
                      head_numbers, reset_rand_seed=False, rand_seed=0):

    if reset_rand_seed:
        random.seed(rand_seed)

    config = dict()

    layer_num = random.choice(layer_numbers)
    head_num = random.choice(head_numbers)

    config['sample_layer_num'] = layer_num
    config['sample_hidden_size'] = random.choice(hidden_sizes)
    config['sample_intermediate_sizes'] = [random.choice(ffn_sizes)] * layer_num
    config['sample_num_attention_heads'] = [head_num] * layer_num
    config['sample_qkv_sizes'] = [head_num * 64] * layer_num
    return config

