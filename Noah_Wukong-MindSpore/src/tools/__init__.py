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
from .template_generate import generate_zh_template
from .model_utils import load_visual_model, load_text_model
from .simple_tokenizer import set_tokenizer_lang, tokenize


__all__ = ['generate_zh_template', 'load_visual_model',
           'load_text_model', 'set_tokenizer_lang', 'tokenize']
