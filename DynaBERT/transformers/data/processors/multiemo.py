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
""" GLUE processors and helpers """

import logging
import os
from typing import List

import numpy as np
from .utils import DataProcessor, InputExample, InputFeatures
from ...file_utils import is_tf_available

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def multiemo_convert_examples_to_features(
        examples, tokenizer,
        max_length=512,
        task=None,
        label_list=None,
        output_mode=None,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """

    if task is not None:
        _, lang, domain, kind = task.split('_')
        processor = MultiemoProcessor(lang, domain, kind)
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = multiemo_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))

        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                            max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids),
                                                                                            max_length)
        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label=label))

    return features


class MultiemoProcessor(DataProcessor):
    """Processor for the Multiemo data2 set"""

    def __init__(self, lang: str, domain: str, kind: str):
        super(MultiemoProcessor, self).__init__()
        self.lang = lang.lower()
        self.domain = domain.lower()
        self.kind = kind.lower()

    def get_train_examples(self, data_dir: str) -> List[InputExample]:
        """See base class."""
        file_path = self.get_set_type_path(data_dir, 'train')
        logger.info(f"LOOKING AT {file_path}")
        return self._create_examples(self._read_txt(file_path), "train")

    def get_dev_examples(self, data_dir: str) -> List[InputExample]:
        """See base class."""
        file_path = self.get_set_type_path(data_dir, 'dev')
        return self._create_examples(self._read_txt(file_path), "dev")

    def get_test_examples(self, data_dir: str) -> List[InputExample]:
        """See base class."""
        file_path = self.get_set_type_path(data_dir, 'test')
        return self._create_examples(self._read_txt(file_path), "test")

    def get_set_type_path(self, data_dir: str, set_type: str) -> str:
        return os.path.join(data_dir, self.domain + '.' + self.kind + '.' + set_type + '.' + self.lang + '.txt')

    def get_labels(self) -> List[str]:
        """See base class."""
        if self.kind == 'text':
            return ["meta_amb", "meta_minus_m", "meta_plus_m", "meta_zero"]
        else:
            return ["z_amb", "z_minus_m", "z_plus_m", "z_zero"]

    @staticmethod
    def _create_examples(lines: List[str], set_type: str) -> List[InputExample]:
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            split_line = line.split('__label__')
            text_a = split_line[0]
            label = split_line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


multiemo_tasks_num_labels = {
    "multiemo": 4,
}

multiemo_output_modes = {
    "multiemo": "classification"
}
