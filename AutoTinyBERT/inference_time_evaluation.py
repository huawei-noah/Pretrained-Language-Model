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
"""Inference time evaluation."""

import argparse
import time
import os
import random

import torch
import numpy as np

from transformer.modeling_super_kd import SuperTinyBertForPreTraining, BertConfig
from transformer.tokenization import BertTokenizer


def text_padding(max_seq_length, device, batch_size):
    input_ids = [9333] * max_seq_length
    input_masks = max_seq_length * [1]
    input_segments = max_seq_length * [0]

    input_ids = torch.tensor([input_ids]*batch_size, dtype=torch.long).to(device)
    input_masks = torch.tensor([input_masks]*batch_size, dtype=torch.long).to(device)
    input_segments = torch.tensor([input_segments]*batch_size, dtype=torch.long).to(device)
    return input_ids, input_masks, input_segments


def arch_cpu_time(model, arch, args):
    aver_time = 0.
    infer_cnt = args.infer_cnt
    for i in range(infer_cnt):
        input_ids, input_masks, input_segments = text_padding(args.max_seq_length,
                                                              device,
                                                              args.batch_size)

        start = time.time()
        with torch.no_grad():
            model(input_ids, arch, input_masks, kd=not args.mlm)

        end = time.time()
        sep = 1000 * (end - start)

        if i == 0:
            continue
        else:
            aver_time += sep / (args.infer_cnt - 1)

    print('{}\t{}'.format(arch, aver_time))
    with open('./latency_dataset/lat.tmp', 'a') as f:
        f.write(f'{arch}\t{aver_time}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model", default='tinybert_model/4l/', type=str)
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--seed", default=0, type=int)

    # Search space for sub_bart architecture
    parser.add_argument('--layer_num_space', nargs='+', type=int, default=[1, 8])
    parser.add_argument('--hidden_size_space', nargs='+', type=int, default=[128, 768])
    parser.add_argument('--qkv_size_space', nargs='+', type=int, default=[180, 768])
    parser.add_argument('--head_num_space', nargs='+', type=int, default=[1, 12])
    parser.add_argument('--intermediate_size_space', nargs='+', type=int, default=[128, 3072])
    parser.add_argument('--mlm', action='store_true')

    parser.add_argument('--infer_cnt', type=int, default=10)

    args = parser.parse_args()
    
    config = BertConfig.from_pretrained(os.path.join(args.bert_model, 'config.json'))
    model = SuperTinyBertForPreTraining.from_scratch(args.bert_model, config)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

    device = 'cpu'
    model.to(device)
    model.eval()

    torch.set_num_threads(1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # build arch space
    min_hidden_size, max_hidden_size = args.hidden_size_space
    min_ffn_size, max_ffn_size = args.intermediate_size_space
    min_qkv_size, max_qkv_size = args.qkv_size_space
    min_head_size, max_head_size = args.head_num_space

    hidden_step = 16
    ffn_step = 12
    qkv_step = 12
    head_step = 1

    number_hidden_step = int((max_hidden_size - min_hidden_size) / hidden_step)
    number_ffn_step = int((max_ffn_size - min_ffn_size) / ffn_step)
    number_qkv_step = int((max_qkv_size - min_qkv_size) / qkv_step)
    number_head_step = int((max_head_size - min_head_size) / head_step)

    layer_numbers = args.layer_num_space
    hidden_sizes = [i * hidden_step + min_hidden_size for i in range(number_hidden_step + 1)]
    ffn_sizes = [i * ffn_step + min_ffn_size for i in range(number_ffn_step + 1)]
    qkv_sizes = [i * qkv_step + min_qkv_size for i in range(number_qkv_step + 1)]
    head_sizes = [i * head_step + min_head_size for i in range(number_head_step + 1)]

    # Test BERT-base time
    config = dict()
    config['sample_layer_num'] = 12
    config['sample_num_attention_heads'] = [12] * 12
    config['sample_hidden_size'] = 768
    config['sample_intermediate_sizes'] = [3072] * 12
    config['sample_qkv_sizes'] = [768] * 12
    arch_cpu_time(model, config, args)

    config['sample_layer_num'] = 4
    config['sample_hidden_size'] = 320
    config['sample_intermediate_sizes'] = [512] * 4
    config['sample_num_attention_heads'] = [5]*4
    config['sample_qkv_sizes'] = [320] * 4


    arch_cpu_time(model, config, args)

    input('')

    for layer_num in layer_numbers:
        config = dict()
        config['sample_layer_num'] = layer_num

        if not args.mlm:
            config['sample_num_attention_heads'] = [12] * layer_num

            for hidden_size in hidden_sizes:
                config['sample_hidden_size'] = hidden_size

                for ffn_size in ffn_sizes:
                    config['sample_intermediate_sizes'] = [ffn_size] * layer_num

                    for qkv_size in qkv_sizes:
                        config['sample_qkv_sizes'] = [qkv_size] * layer_num

                        arch_cpu_time(model, config, args)
        else:
            for head_size in head_sizes:
                config['sample_num_attention_heads'] = [head_size] * layer_num
                config['sample_qkv_sizes'] = [head_size * 64] * layer_num

                for hidden_size in hidden_sizes:
                    config['sample_hidden_size'] = hidden_size

                    for ffn_size in ffn_sizes:
                        config['sample_intermediate_sizes'] = [ffn_size] * layer_num

                        arch_cpu_time(model, config, args)
