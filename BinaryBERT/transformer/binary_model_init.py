# coding=utf-8
# Copyright 2020 Huawei Technologies Co., Ltd.
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
# intialize the weights of a binary model from
# (1) a trained full-precision model
# (2) a trained ternary model
# (3) the full-precision model accompanying a ternary model
# have not taken into account the learnable clipping

import torch
from collections import OrderedDict

def binarize(input, layerwise, double=True):
    # binarize double precision
    if double:
        input = input.double()
    if layerwise:
        s = input.size()
        m = input.norm(p=1).div(input.nelement())
        result = input.sign().mul(m.expand(s))
    else:
        n = input[0].nelement()  # W of size axb, return a vector of  ax1
        s = input.size()
        m = input.norm(1, 1, keepdim=True).div(n)
        result = input.sign().mul(m.expand(s))
    return result


def ternarize(input, layerwise, double=True):
    # ternarize double precision
    if double:
        input = input.double()
    if layerwise:
        m = input.norm(p=1).div(input.nelement())
        thres = 0.7 * m
        pos = (input > thres).double() if double else (input > thres).float()
        neg = (input < -thres).double() if double else (input < -thres).float()
        mask = (input.abs() > thres).double() if double else (input.abs() > thres).float()
        alpha = (mask * input).abs().sum() / mask.sum()
        result = alpha * pos - alpha * neg

    else:  # row-wise only for embed / weight
        n = input[0].nelement()
        m = input.data.norm(p=1, dim=1).div(n)
        thres = (0.7 * m).view(-1, 1).expand_as(input)
        pos = (input > thres).double() if double else (input > thres).float()
        neg = (input < -thres).double() if double else (input < -thres).float()
        mask = (input.abs() > thres).double() if double else (input.abs() > thres).float()
        alpha = ((mask * input).abs().sum(dim=1) / mask.sum(dim=1)).view(-1, 1)
        result = alpha * pos - alpha * neg
    return result


def tws_split(source_fp_model_dir, target_model_dir):

    state_dict = torch.load(source_fp_model_dir, map_location='cpu')
    new_state_dict = OrderedDict()
    processed_keys = []
    use_double = True  # NOTE: use double gives high precision for the conversion
    for k, v in state_dict.items():
        if 'LayerNorm' in k or 'position_embeddings.weight' in k or 'token_type_embeddings.weight' in k \
            or not k.startswith('bert'):
            new_state_dict[k] = v
            processed_keys.append(k)
        else:
            if use_double:
                v = v.double()
            if 'word_embeddings.weight' in k:
                processed_keys.append(k)
                v_ternary = ternarize(v, False, use_double) # quantize on the fly
                v_zero = torch.zeros_like(v).double() if use_double else torch.zeros_like(v)
                # v_small = torch.Tensor(v.shape[0], v.shape[1])
                x = (v_ternary == 0).sum(dim=1).double() if use_double else (v_ternary == 0).sum(dim=1).float()
                v_small = ((torch.max(v_ternary, dim=1)[0] * v[0].nelement() - v.abs().sum(dim=1)) / (
                        2.0 * x)).view(-1, 1).expand_as(v)
                a11 = torch.where(v_ternary == 0 , v_zero, v)
                a12 = torch.where(v_ternary == 0 , torch.where(v>0, v+v_small, v_small), v_zero)
                a22 = torch.where(v_ternary == 0, torch.where(v<0, v-v_small, -v_small), v_zero)
                m = ((a11.abs().sum(dim=1) + a22.abs().sum(dim=1) - a12.abs().sum(dim=1)) / (
                        2.0*a11.abs().sum(dim=1))).view(-1, 1).expand_as(v)
                # v_small.data.fill_(1e-8)
                k1 = ''.join([k[:-7], '_1', k[-7:]])
                k2 = ''.join([k[:-7], '_2', k[-7:]])
                v1 = torch.where(v_ternary == 0 , torch.where(v>0, v+v_small, v_small), m*v)
                v2 = torch.where(v_ternary == 0, torch.where(v<0, v-v_small, -v_small), (1-m)*v)
                new_state_dict[k1] = v1.clone().detach().float()
                new_state_dict[k2] = v2.clone().detach().float()

            elif 'weight' in k:
                processed_keys.append(k)
                v_ternary = ternarize(v, True, use_double) # quantize on the fly
                v_zero = torch.zeros_like(v)
                v_small = torch.Tensor(v.shape[0], v.shape[1])
                if use_double:
                    v_zero = v_zero.double()
                    v_small = v_small.double()

                x = (v_ternary == 0).sum().double() if use_double else (v_ternary == 0).sum().float()
                tmp = (torch.max(v_ternary).item() *v.nelement() - v.abs().sum())/(
                        2.0*x)
                v_small.data.fill_(tmp)
                a11 = torch.where(v_ternary == 0 , v_zero, v)
                a12 = torch.where(v_ternary == 0 , torch.where(v>0, v+v_small, v_small), v_zero)
                a22 = torch.where(v_ternary == 0, torch.where(v<0, v-v_small, -v_small), v_zero)
                m = (a11.abs().sum() + a22.abs().sum() - a12.abs().sum()) / (2*a11.abs().sum())

                k1 = ''.join([k[:-7], '_1', k[-7:]])
                k2 = ''.join([k[:-7], '_2', k[-7:]])
                v1 = torch.where(v_ternary == 0 , torch.where(v>0, v+v_small, v_small), m*v)
                v2 = torch.where(v_ternary == 0, torch.where(v<0, v-v_small, -v_small), (1-m)*v)
                # scaling = torch.max(v_ternary)[0]*v.nelement()/(v1.abs()+v2.abs()).sum()
                new_state_dict[k1] = v1.clone().detach().float()
                new_state_dict[k2] = v2.clone().detach().float()
            elif 'bias' in k:
                processed_keys.append(k)
                k1 = ''.join([k[:-5], '_1', k[-5:]])
                k2 = ''.join([k[:-5], '_2', k[-5:]])
                v1 = v / 2
                v2 = v / 2
                new_state_dict[k1] = v1
                new_state_dict[k2] = v2
            elif 'clip_query' in k or 'clip_key' in k or 'clip_value' in k or 'clip_attn' in k:
                processed_keys.append(k)
                new_state_dict[k] = v
            elif 'input_clip_val' in k:
                processed_keys.append(k)
                ind = k.find('.input_clip_val')
                k1 = ''.join([k[:ind], '_1', k[ind:]])
                k2 = ''.join([k[:ind], '_2', k[ind:]])
                new_state_dict[k1] = v
                new_state_dict[k2] = v

    # check if all keys are processed
    ckpt_keys = list(state_dict.keys())
    for key in ckpt_keys:
        if key not in processed_keys:
            print('Key: %s not processed' % key)

    torch.save(new_state_dict, target_model_dir)
    return target_model_dir