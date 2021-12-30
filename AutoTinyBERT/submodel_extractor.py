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

import os
import json
import torch
import argparse

from transformer.modeling_extractor import SuperBertModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument('--arch',
                        type=str,
                        required=True)
    parser.add_argument('--output',
                        type=str,
                        required=True)
    parser.add_argument('--kd', action='store_true')

    args = parser.parse_args()

    model = SuperBertModel.from_pretrained(args.model)
    size = 0
    for n, p in model.named_parameters():
        size += p.nelement()
        print('n: {}#@#p: {}'.format(n, p.nelement()))

    print('the model size is : {}'.format(size))

    arch = json.loads(json.dumps(eval(args.arch)))

    print('kd: {}'.format(args.kd))

    kd = True if args.kd else False
    model.module.set_sample_config(arch, kd) if hasattr(model, 'module') \
        else model.set_sample_config(arch, kd)

    size = 0
    for n, p in model.named_parameters():
        size += p.nelement()
        print('n: {}#@#p: {}'.format(n, p.nelement()))

    print('the extracted model size is : {}'.format(size))

    model_to_save = model.module if hasattr(model, 'module') else model

    model_output = os.path.join(args.output, 'pytorch_model.bin')
    torch.save(model_to_save.state_dict(), model_output)


if __name__ == "__main__":
    main()

