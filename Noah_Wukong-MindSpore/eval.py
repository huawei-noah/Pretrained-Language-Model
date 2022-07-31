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
import os
import json
import logging
import argparse

import yaml
from tqdm import tqdm
import numpy as np
from mindspore import Tensor
from mindspore import context
from mindspore import dtype as mstype

from src.model import FilipTemplateEncoder, ClipTemplateEncoder, FilipEval, ClipEval
from src.tools import generate_zh_template, load_visual_model, load_text_model
from src.dataset import get_dataset


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


def main():
    parser = argparse.ArgumentParser(description='evaluation for wukong dataset')
    parser.add_argument('--config_path', help='model configuration file path', required=True)
    parser.add_argument('--ckpt_path', help="checkpoint file path for torch model", required=True)
    parser.add_argument('--dataset_path', help="ILSVRC dataset path root", required=True)
    parser.add_argument('--batch_size', help="evaluate dataset batch size", type=int, default=4)
    args = parser.parse_args()
    config_path = args.config_path
    ckpt_path = args.ckpt_path
    dataset_path = args.dataset_path

    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    text_encoder = load_text_model(config['model']['text'], ckpt_path)
    text_encoder = text_encoder.to_float(mstype.float16)
    val_dataset = get_dataset(dataset_path, args.batch_size)
    dataset_size = val_dataset.get_dataset_size()

    logger.info("start generating template feature")
    class_name_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'imagenet_class_name_zh.json'
    )
    mapping = json.load(open(class_name_file, 'r'))
    sort_keys = sorted(list(mapping.keys()))
    dataset_labels = [mapping[key] for key in sort_keys]

    template_tokens = generate_zh_template(dataset_labels)
    template_tokens = Tensor(template_tokens)
    if config['eval'] == 'filip':
        template_encoder = FilipTemplateEncoder(text_encoder)
    elif config['eval'] == 'clip':
        template_encoder = ClipTemplateEncoder(text_encoder)
    else:
        raise NotImplementedError
    template_encoder.set_train(False)
    template_feature, n_template = template_encoder(template_tokens)
    logger.info("template feature generated successfully")
    logger.info("==========================")

    visual_encoder = load_visual_model(config['model']['visual'], ckpt_path)
    visual_encoder = visual_encoder.to_float(mstype.float16)

    if config['eval'] == 'filip':
        eval_matric = FilipEval(template_feature, n_template, visual_encoder, text_encoder)
    elif config['eval'] == 'clip':
        eval_matric = ClipEval(template_feature, n_template, visual_encoder, text_encoder)
    else:
        raise NotImplementedError
    eval_matric.set_train(False)
    eval_matric = eval_matric.to_float(mstype.float16)

    correct_1 = []
    correct_5 = []
    logger.info('total iter: %d', dataset_size)
    for data in tqdm(val_dataset, total=dataset_size):
        output = eval_matric(*data)
        acc1, acc5 = output[0].asnumpy(), output[1].asnumpy()
        correct_1.append(acc1)
        correct_5.append(acc5)
    correct_1 = np.hstack(correct_1)
    correct_1 = correct_1.mean()
    correct_5 = np.hstack(correct_5)
    correct_5 = correct_5.mean()
    logger.info("model %s result:", config_path)
    logger.info("correct @1: {:.2f}; correct @5: {:.2f}".format(correct_1 * 100, correct_5 * 100))


if __name__ == '__main__':
    main()
