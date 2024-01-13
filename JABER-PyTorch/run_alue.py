# Modified version of run_glue script
# Source: https://github.com/huggingface/transformers/blob/v2.7.0/examples/run_glue.py

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
""" Finetuning the library models for sequence classification on ALUE (T5)."""


import logging

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import trange

from transformers import (
    AdamW,
    AutoConfig,
    T5ForConditionalGeneration,
    get_constant_schedule,
    Adafactor,
    BertPreTrainedModel,
    BertModel,
    T5EncoderModel,
    T5PreTrainedModel
)

from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from torch.cuda.amp import autocast, GradScaler

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from generate_data import *
# from modeling import *

logger = logging.getLogger(__name__)


###################
### BERT Models ###
###################


class BertForSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier_old.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, num_labels):
        super().__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids=None, inputs_embeds=None, token_type_ids=None, attention_mask=None, labels=None):
        """
                output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        """
        outputs = self.bert(input_ids=input_ids,
                            inputs_embeds=inputs_embeds,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            output_attentions=True,
                            output_hidden_states=True)
        pooled_output = outputs[1]
        task_output = self.dropout(pooled_output)
        logits = self.classifier(task_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            if self.num_labels == 1:
                # We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs # (loss), logits, (hidden_states), (attentions)


class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    """
    Bert Model transformer with a multi-label sequence classification head on top
    (a linear layer with sigmoid activation on top of the pooled output).
    """
    def __init__(self, config, num_labels):
        super().__init__(config)
        self.num_labels = num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, self.num_labels)
        self.classifier = nn.Sigmoid()

        #self.apply(self.init_bert_weights)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.bert(input_ids=input_ids,
                            inputs_embeds=inputs_embeds,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            output_attentions=True, output_hidden_states=True)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        pooled_output = self.linear(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            # Apply loss before the `Sigmoid` layer, as `BCEWithLogitsLoss`
            # internally applies `Sigmoid` in a more numerically stable fashion.
            loss = loss_fct(pooled_output, labels.type_as(pooled_output))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels=2):
        super().__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, num_labels)

    # [hel, #lo, wor, #ld]
    # [0, 2]
    # [hidden(hel), hidden(#lo), hidden(wor), hidden(#ld)]
    # [hidden(hel), hidden(wor), 0, 0]
    def gather_indexes(self, sequence_tensor, positions):
        """Gathers the vectors at the specific positions over a minibatch."""
        batch_size = sequence_tensor.size()[0]
        seq_length = sequence_tensor.size()[1]
        width = sequence_tensor.size()[2]
        device = positions.device

        flat_offsets = torch.reshape(torch.range(0, batch_size - 1, dtype=torch.int32) * seq_length, [-1, 1]).to(device)
        flat_positions = torch.reshape(positions + flat_offsets, [-1])
        flat_sequence_tensor = torch.reshape(sequence_tensor, [batch_size * seq_length, width])
        output_tensor = torch.index_select(flat_sequence_tensor, 0, flat_positions)
        output_tensor = torch.reshape(output_tensor, [batch_size, seq_length, width])

        return output_tensor

    def sequence_mask(self, lengths, max_len=None):
        """
        Creates a boolean mask from sequence lengths.
        """
        batch_size = lengths.numel()
        max_len = max_len or lengths.max()
        return (torch.arange(0, max_len)
                .type_as(lengths)
                .repeat(batch_size, 1)
                .lt(lengths.unsqueeze(1)))

    def forward(self, input_ids=None, inputs_embeds=None, token_type_ids=None, attention_mask=None, positions=None, seq_len=None, labels=None):
        r'''
        Return:
        loss
        pred (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, self.num_labels)`):

        '''
        outputs = self.bert(input_ids=input_ids, inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask, token_type_ids=token_type_ids,
                            output_attentions=True, output_hidden_states=True
        )
        sequence_output = outputs[0]
        output_seq = self.gather_indexes(sequence_output, positions)
        logits = self.dense(output_seq)  ## : shape (batch_size, sequence_length, num_labels)
        # pred = F.softmax(logits,dim=-1)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if seq_len is None:
            seq_len = torch.sum(attention_mask, dim=-1)

        if labels is not None:
            ## caclulate loss
            flat_logits = torch.reshape(logits, [-1, self.num_labels])
            labels = labels.contiguous().view(-1)  ## : shape (batch_size*sequence_length, )
            # cls_weights = torch.tensor([self.tagger.cls_weights]).to(device)
            tok_weights = torch.reshape(self.sequence_mask(seq_len, input_ids.shape[1]), [-1])
            # temp_loss = F.cross_entropy(flat_logits, labels, reduction='none')
            # loss_fct = CrossEntropyLoss()
            loss_fct = CrossEntropyLoss(reduction='none')
            numerator = torch.sum(loss_fct(flat_logits, labels) * tok_weights)
            denominator = torch.sum(tok_weights) + 1e-5
            loss = numerator / denominator

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            output_attentions=True, output_hidden_states=True)
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        stack_logits = torch.vstack([start_logits, end_logits])
        # print(start_logits.shape, end_logits.shape, stack_logits.shape)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss, stack_logits
        else:
            return stack_logits

############################
### T5 Encoder-only Model ##
############################

class Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class T5EncForSequenceClassification(T5PreTrainedModel):

    def __init__(self, config, model_path, num_labels):
        super().__init__(config)
        self.num_labels = num_labels
        self.enc = T5EncoderModel.from_pretrained(model_path, config=config)
        self.pooler = Pooler(config)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.enc(input_ids=input_ids, attention_mask=attention_mask)

        sequence_output = outputs.last_hidden_state
        # pooled_output = torch.mean(sequence_output, dim=1)
        pooled_output = sequence_output[:, 0, :]  # Take the first token
        # pooled_output = self.pooler(sequence_output)
        task_output = self.dropout(pooled_output)

        logits = self.classifier(task_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                # We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)


class T5EncForMultiLabelSequenceClassification(T5PreTrainedModel):
    def __init__(self, config, model_path, num_labels):
        super().__init__(config)
        self.num_labels = num_labels
        self.enc = T5EncoderModel.from_pretrained(model_path, config=config)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.linear = nn.Linear(config.hidden_size, self.num_labels)
        self.classifier = nn.Sigmoid()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
    ):
        outputs = self.enc(input_ids=input_ids, attention_mask=attention_mask)

        sequence_output = outputs.last_hidden_state
        # pooled_output = torch.mean(sequence_output, dim=1)
        pooled_output = sequence_output[:, 0, :]  # Take the first token
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.linear(pooled_output)

        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            # Apply loss before the `Sigmoid` layer, as `BCEWithLogitsLoss`
            # internally applies `Sigmoid` in a more numerically stable fashion.
            loss = loss_fct(pooled_output, labels.type_as(pooled_output))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class T5EncForTokenClassification(T5PreTrainedModel):
    def __init__(self, config, model_path, num_labels):
        super().__init__(config)
        self.num_labels = num_labels
        self.enc = T5EncoderModel.from_pretrained(model_path, config=config)
        self.dense = nn.Linear(config.hidden_size, num_labels)

    def gather_indexes(self, sequence_tensor, positions):
        """Gathers the vectors at the specific positions over a minibatch."""
        batch_size = sequence_tensor.size()[0]
        seq_length = sequence_tensor.size()[1]
        width = sequence_tensor.size()[2]
        device = positions.device

        flat_offsets = torch.reshape(torch.range(0, batch_size - 1, dtype=torch.int32) * seq_length, [-1, 1]).to(device)
        flat_positions = torch.reshape(positions + flat_offsets, [-1])
        flat_sequence_tensor = torch.reshape(sequence_tensor, [batch_size * seq_length, width])
        output_tensor = torch.index_select(flat_sequence_tensor, 0, flat_positions)
        output_tensor = torch.reshape(output_tensor, [batch_size, seq_length, width])

        return output_tensor

    def sequence_mask(self, lengths, max_len=None):
        """
        Creates a boolean mask from sequence lengths.
        """
        batch_size = lengths.numel()
        max_len = max_len or lengths.max()
        return (torch.arange(0, max_len)
                .type_as(lengths)
                .repeat(batch_size, 1)
                .lt(lengths.unsqueeze(1)))

    def forward(self, input_ids, attention_mask=None, positions=None, seq_len=None, labels=None):
        r'''
        Return:
        loss
        pred (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, self.num_labels)`):

        '''
        outputs = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        output_seq = self.gather_indexes(sequence_output, positions)
        logits = self.dense(output_seq)  ## : shape (batch_size, sequence_length, num_labels)
        # pred = F.softmax(logits,dim=-1)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if seq_len is None:
            seq_len = torch.sum(attention_mask, dim=-1)

        if labels is not None:
            ## caclulate loss
            flat_logits = torch.reshape(logits, [-1, self.num_labels])
            labels = labels.contiguous().view(-1)  ## : shape (batch_size*sequence_length, )
            # cls_weights = torch.tensor([self.tagger.cls_weights]).to(device)
            tok_weights = torch.reshape(self.sequence_mask(seq_len, input_ids.shape[1]), [-1])
            # temp_loss = F.cross_entropy(flat_logits, labels, reduction='none')
            # loss_fct = CrossEntropyLoss()
            loss_fct = CrossEntropyLoss(reduction='none')
            numerator = torch.sum(loss_fct(flat_logits, labels) * tok_weights)
            denominator = torch.sum(tok_weights) + 1e-5
            loss = numerator / denominator

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)



class T5EncForQuestionAnswering(T5PreTrainedModel):
    def __init__(self, config, model_path, num_labels=2):
        super().__init__(config)
        self.enc = T5EncoderModel.from_pretrained(model_path, config=config)
        self.qa_outputs = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, start_positions=None, end_positions=None):
        outputs = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        stack_logits = torch.vstack([start_logits, end_logits])
        # print(start_logits.shape, end_logits.shape, stack_logits.shape)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss, stack_logits
        else:
            return stack_logits

############################
##### model utils methods ##
############################
def torch_init_model(model, init_checkpoint, delete_module=False):
    state_dict = torch.load(init_checkpoint, map_location='cpu')
    # state_dict = dict(state_dict["state_dict"])
    # print(state_dict["state_dict"])
    # for key in state_dict.keys():
    #     print(key)
    # return
    state_dict_new = {}
    # delete module.
    if delete_module:
        for key in state_dict.keys():
            v = state_dict[key]
            new_key = key.replace('gamma', 'weight').replace('beta', 'bias').replace('module.', '')
            state_dict_new[new_key] = v

        state_dict = state_dict_new
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

    # for n, p in model.named_parameters():
    #     print(n)
    prefix = ''
    if not hasattr(model, 'bert') and not hasattr(model, 'enc'): prefix  = 'bert.'
    # if not hasattr(model, 'enc'): prefix = 'enc.'
    load(model, prefix=prefix)

    print("missing keys:{}".format(missing_keys))
    print('unexpected keys:{}'.format(unexpected_keys))
    print('error msgs:{}'.format(error_msgs))


class MyDataCollator:
    def __init__(self, args, data_processor: DataProcessor, is_train=True):
        self.task_type = data_processor.task_type
        self.is_gen = data_processor.is_gen
        self.arch = args.arch
        self.is_pretrain_jaber = MODEL_CONF_MAP[args.model_name] == "pretrain_jaber" and self.arch != "t5"
        self.pad_idx = data_processor.pad_idx
        self.bos_idx, self.eos_idx = data_processor.bos_idx, data_processor.eos_idx

        self.is_train = is_train
        self.max_seq_len = args.max_seq_len
        self.max_max_seq_len = 512

    def __call__(self, features):
        max_seq_len = self.max_seq_len if self.is_train else self.max_max_seq_len
        batch = self._pad([f["input_ids"][:max_seq_len] for f in features])

        if self.task_type == "ner" and not self.is_gen:
            batch["seq_len"] = torch.tensor([len(f["positions"]) for f in features], dtype=torch.long)
            msl = batch["input_ids"].shape[-1]
            for k in ["positions", "labels"]:
                batch[k] = torch.tensor([f[k] + [0] * (msl-len(f[k])) for f in features], dtype=torch.long)
            return batch

        if self.task_type == "qa" and not self.is_gen:
            start_positions, end_positions =  zip(*[f["labels"] for f in features])
            batch["start_positions"] = torch.tensor(start_positions, dtype=torch.long)
            batch["end_positions"] = torch.tensor(end_positions, dtype=torch.long)
            return batch

        if self.task_type in ["mlc", "gen"] or self.is_gen:
            labels = [f["labels"] for f in features]
        else:
            labels = [f["labels"][0] for f in features]

        if self.is_gen:
            batch.update(self._pad_labels(labels))
        else:
            dtype = torch.float if self.task_type == "reg" and not self.is_gen else torch.long
            batch["labels"] = torch.tensor(labels, dtype=dtype)

        return batch

    def _pad(self, input_ids_lst):
        msq = max(len(lst) for lst in input_ids_lst)
        input_ids, attention_mask = [], []
        for ii in input_ids_lst:
            val = msq - len(ii)
            input_ids.append(ii + [self.pad_idx] * val)
            attention_mask.append([1] * len(ii) + [0] * val)
        input_ids = torch.as_tensor(input_ids, dtype=torch.long)
        attention_mask = torch.as_tensor(attention_mask, dtype=torch.long)
        batch = {"input_ids": input_ids, "attention_mask": attention_mask}

        if self.is_pretrain_jaber:
            batch["token_type_ids"] = self.get_segment_ids(input_ids)
            # print(batch["token_type_ids"])
        return batch

    def _pad_labels(self, labels_lst):

        msq = max(len(lst) for lst in labels_lst)
        labels, decoder_attention_mask, decoder_input_ids = [], [], []
        for lbl in labels_lst:
            seq_len = len(lbl)
            labels.append(lbl + [self.eos_idx] + [-100] * (msq - seq_len))
            decoder_input_ids.append([self.pad_idx]+lbl + [0] * (msq - seq_len)) #self.bos_idx
            decoder_attention_mask.append([1] * (seq_len+1) + [0] * (msq - seq_len))

        labels = torch.as_tensor(labels, dtype=torch.long)
        decoder_attention_mask = torch.as_tensor(decoder_attention_mask, dtype=torch.long)
        decoder_input_ids = torch.as_tensor(decoder_input_ids, dtype=torch.long)
        return {"labels": labels,
                "decoder_input_ids": decoder_input_ids, "decoder_attention_mask": decoder_attention_mask}

    def get_segment_ids(self, input_ids):
        batch_size, max_seq_len = input_ids.shape
        rng = torch.tile(torch.unsqueeze(torch.arange(max_seq_len), 0),
                         (batch_size, 1))  # , device=torch.device('cuda')
        zero_arr, one_arr = torch.zeros_like(input_ids), torch.ones_like(input_ids)
        cond = input_ids == (self.eos_idx * one_arr)
        segment_ids, _ = torch.min(torch.where(cond, rng, max_seq_len * one_arr), dim=-1)
        segment_ids = torch.tile(torch.unsqueeze(segment_ids, 1), (1, max_seq_len))
        cond = torch.logical_and(torch.greater(rng, segment_ids), torch.not_equal(input_ids, zero_arr))
        segment_ids = torch.where(cond, one_arr, zero_arr)

        return segment_ids


def set_seed(args):
    if args.seed == -1:
        return
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

BEST_ACCURACY = -1
BEST_PRED_DICT = {}

def train(args, data_processor: DataProcessor, model, data_collator: MyDataCollator):
    global BEST_ACCURACY
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    dataset_dict, sampler_dict, dataloader_dict = load_dataloaders(args, data_processor, data_collator)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(dataloader_dict["train"]) // args.gradient_accumulation_steps) + 1
    else:
        args.logging_steps = len(dataloader_dict["train"]) // args.gradient_accumulation_steps
        t_total = len(dataloader_dict["train"]) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "layer_norm.weight"]#
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    if args.arch == "bert":
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    else:
        optimizer = Adafactor(model.parameters(),  # optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              relative_step=False,
                              warmup_init=False)

    scheduler = get_constant_schedule(optimizer)
    scaler = GradScaler()
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_path, "scheduler.pt")))

    # if args.fp16:
    #     try:
    #         from apex import amp
    #     except ImportError:
    #         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(dataset_dict["train"]))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_path):
        # set global_step to global_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(dataloader_dict["train"]) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(dataloader_dict["train"]) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility
    for epoch_index in train_iterator:
        epoch_iterator = tqdm(dataloader_dict["train"], desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            inputs = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                # last step in epoch but step is always smaller than gradient_accumulation_steps
                len(epoch_iterator) <= args.gradient_accumulation_steps
                and (step + 1) == len(epoch_iterator)
            ):
                if args.fp16:
                    # otherwise, optimizer.step() is skipped.
                    scaler.step(optimizer)
                    # scaler.step(scheduler)
                    # Updates the scale for next iteration.
                    scaler.update()
                else:
                    optimizer.step()
                # scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step == 1:
                # if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                if args.local_rank in [-1, 0] and args.save_epochs and (step + 1) == len(epoch_iterator):#  or True
                    logs = {}
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well

                        portion = "dev"
                        eval_sample_num = len(dataset_dict[portion])

                        eval_loss = run_eval(args, data_processor, model,
                                             dataloader_dict[portion], eval_sample_num, portion="dev")
                        results = data_processor.compute_score(portion)
                        metrics = data_processor.final_metric(results)
                        data_processor.reset_pred()

                        logger.info("***** Eval results {} *****".format(portion))
                        logger.info("  %s = %s", "eval_loss", str(eval_loss))
                        for key in sorted(results.keys()):
                            logger.info("  %s = %s", key, str(results[key]))

                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))

                    ## save best model
                    output_dir = args.output_dir
                    print("Intermediate evaluate: ", metrics)
                    if BEST_ACCURACY < metrics:
                        BEST_ACCURACY = metrics
                        global BEST_PRED_DICT

                        if args.save_model:
                            checkpoint_name = "pytorch_model.bin"

                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = (
                                model.module if hasattr(model, "module") else model
                            )  # Take care of distributed/parallel training

                            output_model_file = os.path.join(args.output_dir, checkpoint_name)
                            logger.info("Saving model checkpoint to %s", output_model_file)
                            torch.save(model_to_save.state_dict(), output_model_file)
                            print("Best Model Saved!")
                            output_config_file = os.path.join(args.output_dir, "config.json")
                            with open(output_config_file, 'w') as f:
                                f.write(model_to_save.config.to_json_string())

                            # logger.info("Saving optimizer states to %s", output_dir)
                            # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))

                        # save best result
                        output_eval_file = os.path.join(output_dir, "best_eval_results.txt")
                        with open(output_eval_file, "w") as writer:
                            logger.info("***** Best Eval results *****")
                            for key in sorted(results.keys()):
                                logger.info("  %s = %s", key, str(results[key]))
                                writer.write("%s = %s\n" % (key, str(results[key])))

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def run_eval(args, data_processor: DataProcessor, model, eval_dataloader, eval_sample_num, portion):
    eval_output_dir = args.output_dir

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(portion))
    logger.info("  Num examples = %d", eval_sample_num)
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        inputs = {k: v.to(args.device) for k, v in batch.items()}
        with torch.no_grad():
            if data_processor.is_gen:
                decoder_seq_len = inputs["labels"].shape[-1]
                if decoder_seq_len == 1:
                    decoder_seq_len = inputs["input_ids"].shape[-1]
                inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
                # if portion == "test":
                generation_output = model.generate(**inputs,
                                                   num_beams=5,
                                                   max_length=decoder_seq_len,
                                                   early_stopping=True,
                                                   return_dict_in_generate=True,
                                                   output_scores=True,
                                                   )
                decoder_output_ids = generation_output.sequences.detach().cpu().numpy().tolist()
                # print(generation_output.sequences)
                logits = generation_output.scores#.detach().cpu().numpy().tolist()
                data_processor.y_logits[portion].append(logits)
                data_processor.y_pred[portion].append(decoder_output_ids)

            else:
                model_output = model(**inputs)
                tmp_eval_loss, logits = model_output[:2]
                eval_loss += tmp_eval_loss.mean().item()
                logits = logits.detach().cpu().numpy().tolist()
                data_processor.y_logits[portion].append(logits)

        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    data_processor.process_logits(portion)
    return eval_loss



def load_model(args, data_processor: DataProcessor, is_infer=False):
    # model_path = os.path.join(config["data_dir"], "models", args.model_name)

    ckpt_name = "pytorch_model-%s.bin" % args.step if args.step != "-1" else "pytorch_model.bin"

    def get_disc_lbl_num():
        if data_processor.task_type == "mlc": return len(MLC_LBL_DICT[data_processor.task_name])
        elif data_processor.task_type == "ner": return len(data_processor.id2label)
        elif data_processor.task_type == "qa": return 2
        return len(data_processor.id2label) if data_processor.id2label else 1

    is_pretrain_jaber = MODEL_CONF_MAP[args.model_name] == "pretrain_jaber"
    if args.arch == "bert":
        num_labels = get_disc_lbl_num()
        if is_pretrain_jaber:# and args.task_name in SEQ_PAIR_TASK:
            model = _load_jaber(args, data_processor.task_type, num_labels)
        else:
            bert_config = AutoConfig.from_pretrained(
                    args.model_path,
                    hidden_dropout_prob=args.dropout_rate,
                )

            if data_processor.task_type == "mlc":
                model = BertForMultiLabelSequenceClassification(config=bert_config, num_labels=num_labels)
            elif data_processor.task_type == "ner":
                model = BertForTokenClassification(config=bert_config, num_labels=num_labels)
            elif data_processor.task_type == "qa":
                model = BertForQuestionAnswering(config=bert_config)
            else:
                model = BertForSequenceClassification(config=bert_config, num_labels=num_labels)
            torch_init_model(model, os.path.join(args.model_path, ckpt_name))

    elif args.arch == "t5":
        t5_config = AutoConfig.from_pretrained(args.model_path, dropout_rate=args.dropout_rate)
        if data_processor.is_gen:
            model = T5ForConditionalGeneration.from_pretrained(
                os.path.join(args.model_path, ckpt_name),
                config=t5_config)
        else:
            num_labels = get_disc_lbl_num()
            cls = {"mlc": T5EncForMultiLabelSequenceClassification,
                   "ner": T5EncForTokenClassification,
                   "qa": T5EncForQuestionAnswering,
                   "cls": T5EncForSequenceClassification,
                   "reg": T5EncForSequenceClassification}[data_processor.task_type]

            model = cls(config=t5_config,
                        model_path=os.path.join(args.model_path, ckpt_name),
                        num_labels=num_labels)
            if is_infer:
                torch_init_model(model, os.path.join(args.model_path, ckpt_name), delete_module=False)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total Model Parameters=%s" % pytorch_total_params)
    return model


def _load_jaber(args, task_type, num_labels):
    from NEZHA_PyTorch.modeling_nezha import BertForSequenceClassification
    from NEZHA_PyTorch.modeling_nezha import BertForMultiLabelSequenceClassification
    from NEZHA_PyTorch.modeling_nezha import NeZhaForTokenClassification
    from NEZHA_PyTorch.modeling_nezha import NeZhaForQuestionAnswering

    if task_type == "mlc":
        model = BertForMultiLabelSequenceClassification.from_pretrained(args.model_path,
                                                                        hidden_dropout_prob=args.dropout_rate,
                                                                        num_labels=num_labels)
    elif task_type == "ner":
        model = NeZhaForTokenClassification.from_pretrained(args.model_path,
                                                            hidden_dropout_prob=args.dropout_rate,
                                                            num_labels=num_labels)
    elif task_type == "qa":
        model = NeZhaForQuestionAnswering.from_pretrained(args.model_path,
                                                          hidden_dropout_prob=args.dropout_rate)
    else:
        model = BertForSequenceClassification.from_pretrained(args.model_path,
                                                              hidden_dropout_prob=args.dropout_rate,
                                                              num_labels=num_labels)

    return model


def load_initials(args):
    # load the dp class
    key = (args.task_name, args.is_gen, args.model_name)
    filename = os.path.join("./raw_datasets", "dp.%s.%s.%s.pkl" % key)
    with open(filename, 'rb') as fp:
        data_processor = pickle.load(fp)

    # load model
    model = load_model(args, data_processor)
    model.to(args.device)

    # init data collator
    data_collator = MyDataCollator(args, data_processor)

    return data_processor, model, data_collator


def load_dataloaders(args, data_processor, data_collator, overwrite_train=False):
    dataset_dict, sampler_dict, dataloader_dict = {}, {}, {}

    # load train dataset and dataloader
    portion = "train"
    key = ("dataset", data_processor.task_name, data_processor.model_name, data_processor.is_gen, portion)
    cache_url = os.path.join("./raw_datasets", "%s_%s_%s_%s_%s" % key)
    dataset_dict[portion] = datasets.load_from_disk(cache_url, keep_in_memory=False)
    if args.local_rank == -1:
        sampler_dict[portion] = RandomSampler(dataset_dict[portion])
    else:
        sampler_dict[portion] = DistributedSampler(dataset_dict[portion])
    dataloader_dict[portion] = DataLoader(
        dataset_dict[portion], sampler=sampler_dict[portion],
        collate_fn=data_collator, batch_size=args.per_gpu_train_batch_size
    )

    # load dataset and dataloader for no train portion
    for portion in data_processor.data_dict.keys():
        if not overwrite_train and portion == "train": continue
        key = ("dataset", data_processor.task_name, data_processor.model_name, data_processor.is_gen, portion)
        cache_url = os.path.join("./raw_datasets", "%s_%s_%s_%s_%s" % key)
        dataset_dict[portion] = datasets.load_from_disk(cache_url, keep_in_memory=False)
        sampler_dict[portion] = SequentialSampler(dataset_dict[portion])
        dataloader_dict[portion] = DataLoader(
            dataset_dict[portion], sampler=sampler_dict[portion], collate_fn=data_collator,
            batch_size=args.per_gpu_eval_batch_size
        )

    return dataset_dict, sampler_dict, dataloader_dict


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dropout_rate",
        default=0.1,
        type=float,
        help="The hidden_dropout_prob in bert config file",
    )

    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--per_gpu_train_batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=16, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=10.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_portion", default=0, type=float, help="Linear warmup over total_steps*warmup_portion.")

    parser.add_argument("--logging_steps", type=int, default=-1, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=-1, help="Save checkpoint every X updates steps.")
    parser.add_argument("--save_epochs", type=bool, default=True, help="Save checkpoint at the end of each epoch.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        required=True,
        help="name of the model (e.g. pytorch_model-835820.bin)",
    )
    parser.add_argument(
        "--task_name",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--log_file",
        default=None,
        type=str,
        help="The name of the log file",
    )
    parser.add_argument(
        "--step",
        default=None,
        type=str,
        help="The name of the log file",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name",
    )
    parser.add_argument(
        "--save_model",
        default=0,
        type=int,
        help="save bets ckpt or not",
    )
    parser.add_argument(
        "--is_gen",
        default=0,
        type=int,
        help="If seq2seq task formulation or not",
    )
    parser.add_argument(
        "--max_seq_len",
        default=512,
        type=int,
        help="index of fold for few shot setting",
    )
    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    for arch, model_name_set in MODEL_ARCH_MAP.items():
        if args.model_name in model_name_set:
            args.arch = arch
            break

    # if args.arch == "t5":
    #     args.per_gpu_eval_batch_size = 8
    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    data_processor, model, data_collator = load_initials(args)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        global_step, tr_loss = train(args, data_processor, model, data_collator)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

if __name__ == "__main__":
    main()
