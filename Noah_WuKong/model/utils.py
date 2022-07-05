#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2022, Huawei Technologies Co., Ltd. All rights reserved.
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
import math
import os

import torch
import torch.distributed as dist
from mmcv.runner import load_checkpoint as load_checkpoint_
from torch.utils.checkpoint import checkpoint_sequential


def token_wise_similarity(rep1, rep2, mask=None, chunk_size=1024):
    batch_size1, n_token1, feat_dim = rep1.shape
    batch_size2, n_token2, _ = rep2.shape
    num_folds = math.ceil(batch_size2 / chunk_size)
    output = []
    for i in range(num_folds):
        rep2_seg = rep2[i * chunk_size:(i + 1) * chunk_size]
        out_i = rep1.reshape(-1, feat_dim) @ rep2_seg.reshape(-1, feat_dim).T
        out_i = out_i.reshape(batch_size1, n_token1, -1, n_token2).max(3)[0]
        if mask is None:
            out_i = out_i.mean(1)
        else:
            out_i = out_i.sum(1)
        output.append(out_i)
    output = torch.cat(output, dim=1)
    if mask is not None:
        output = output / mask.sum(1, keepdim=True).clamp_(min=1)
    return output


def auto_grad_checkpoint(layer, x, chunks=3):
    use_grad_checkpoint = getattr(auto_grad_checkpoint, 'use_grad_checkpoint', False)
    chunks = getattr(auto_grad_checkpoint, 'chunks', chunks)  # for globally set chunks
    chunks = min(len(layer), chunks)
    need_grad = next(layer.parameters()).requires_grad
    if use_grad_checkpoint and need_grad and chunks > 0:
        return checkpoint_sequential(layer, chunks, x)
    return layer(x)


def load_checkpoint(model, pretrained):
    load_checkpoint_(model, pretrained, map_location='cpu')


def is_distributed():
    return get_world_size() > 1


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_local_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    return local_rank


def is_master():
    return get_rank() == 0


def is_local_master():
    return get_local_rank() == 0


def get_local_proc_group(group_size=8):
    world_size = get_world_size()
    if world_size <= group_size or group_size == 1:
        return None
    assert world_size % group_size == 0, f'world size ({world_size}) should be evenly divided by group size ({group_size}).'
    process_groups = getattr(get_local_proc_group, 'process_groups', dict())
    if group_size not in process_groups:
        num_groups = dist.get_world_size() // group_size
        groups = [list(range(i * group_size, (i + 1) * group_size)) for i in range(num_groups)]
        process_groups.update({group_size: [torch.distributed.new_group(group) for group in groups]})
        get_local_proc_group.process_groups = process_groups

    group_idx = get_rank() // group_size
    process_groups = get_local_proc_group.process_groups.get(group_size)[group_idx]
    return process_groups
