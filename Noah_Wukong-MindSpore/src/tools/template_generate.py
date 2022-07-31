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
from .simple_tokenizer import set_tokenizer_lang, tokenize


def generate_zh_template(label_list):
    set_tokenizer_lang('zh', 32)
    template_list = []
    template_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'zh_templates.txt'
    )

    templates = []
    for line in open(template_path, 'r'):
        templates.append(line.strip())
    num_prompts = len(templates)
    num_labels = len(label_list)
    for label in label_list:
        for template in templates:
            template_list.append(template.replace('{}', label))
    token = tokenize(template_list).reshape((num_labels, num_prompts, -1))
    return token
