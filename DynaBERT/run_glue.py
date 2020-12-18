# coding=utf-8
# 2020.08.28 - Changed regular fine-tuning to fine-tuning with adaptive width and depth
#              Huawei Technologies Co., Ltd <houlu3@huawei.com>
# Copyright (c) 2020, Huawei Technologies Co., Ltd.  All rights reserved.
# Copyright 2018 The Google AI Language Team Authors,  the HuggingFace Inc.
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
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import math
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm, trange
from torch.nn import  MSELoss

from transformers import (BertConfig, BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer)

from transformers import AdamW, WarmupLinearSchedule

from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features


logger = logging.getLogger(__name__)
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"

def soft_cross_entropy(predicts, targets):
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    return -torch.sum(targets_prob * student_likelihood, dim=-1).mean()


loss_mse = MSELoss()
ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig)), ())
MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, teacher_model=None):
    """ Train the model """

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    if args.model_type == 'roberta':
        args.warmup_steps = int(t_total*0.06)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    global_step = 0
    tr_loss = 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)

    current_best = 0
    output_eval_file = os.path.join(args.output_dir, 'eval_results.txt')

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[3],
                      'token_type_ids': batch[2] if args.model_type in ['bert'] else None}

            # prepare the hidden states and logits of the teacher model
            if args.training_phase == 'dynabertw' and teacher_model:
                with torch.no_grad():
                    _, teacher_logit, teacher_reps, _, _ = teacher_model(**inputs)
            elif args.training_phase == 'dynabert' and teacher_model:
                hidden_max_all, logits_max_all = [], []
                for width_mult in sorted(args.width_mult_list, reverse=True):
                    with torch.no_grad():
                        _, teacher_logit, teacher_reps, _, _ = teacher_model(**inputs)
                        hidden_max_all.append(teacher_reps)
                        logits_max_all.append(teacher_logit)

            # accumulate grads for all sub-networks
            for depth_mult in sorted(args.depth_mult_list, reverse=True):
                model.apply(lambda m: setattr(m, 'depth_mult', depth_mult))
                # select teacher model layers for matching
                if args.training_phase == 'dynabert' or 'final_finetuning':
                    model = model.module if hasattr(model, 'module') else model
                    base_model = getattr(model, model.base_model_prefix, model)
                    n_layers = base_model.config.num_hidden_layers
                    depth = round(depth_mult * n_layers)
                    kept_layers_index = []
                    for i in range(depth):
                        kept_layers_index.append(math.floor(i / depth_mult))
                    kept_layers_index.append(n_layers)

                # adjust width
                width_idx = 0
                for width_mult in sorted(args.width_mult_list, reverse=True):
                    model.apply(lambda m: setattr(m, 'width_mult', width_mult))
                    # stage 1: width-adaptive
                    if args.training_phase == 'dynabertw':
                        if getattr(args, 'data_aug'):
                            loss, student_logit, student_reps, _, _ = model(**inputs)

                            # distillation loss of logits
                            if args.output_mode == "classification":
                                logit_loss = soft_cross_entropy(student_logit, teacher_logit.detach())
                            elif args.output_mode == "regression":
                                logit_loss = 0

                            # distillation loss of hidden states
                            rep_loss = 0
                            for student_rep, teacher_rep in zip(student_reps, teacher_reps):
                                tmp_loss = loss_mse(student_rep, teacher_rep.detach())
                                rep_loss += tmp_loss

                            loss = args.width_lambda1 * logit_loss + args.width_lambda2 * rep_loss
                        else:
                            loss = model(**inputs)[0]

                    # stage 2: width- and depth- adaptive
                    elif args.training_phase == 'dynabert':
                        loss, student_logit, student_reps, _, _ = model(**inputs)

                        # distillation loss of logits
                        if args.output_mode == "classification":
                            logit_loss = soft_cross_entropy(student_logit, logits_max_all[width_idx].detach())
                        elif args.output_mode == "regression":
                            logit_loss = 0

                        # distillation loss of hidden states
                        rep_loss = 0
                        for student_rep, teacher_rep in zip(
                                student_reps, list(hidden_max_all[width_idx][i] for i in kept_layers_index)):
                            tmp_loss = loss_mse(student_rep, teacher_rep.detach())
                            rep_loss += tmp_loss

                        loss = args.depth_lambda1 * logit_loss + args.depth_lambda2 * rep_loss  # ground+truth and distillation
                        width_idx += 1  # move to the next width

                    # stage 3: final finetuning
                    else:
                        loss = model(**inputs)[0]

                    print(loss)
                    if args.n_gpu > 1:
                        loss = loss.mean()
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    loss.backward()

            # clip the accumulated grad from all widths
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # evaluate
                if global_step > 0 and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.evaluate_during_training:
                        acc = []
                        if args.task_name == "mnli":   # for both MNLI-m and MNLI-mm
                            acc_both = []

                        # collect performance of all sub-networks
                        for depth_mult in sorted(args.depth_mult_list, reverse=True):
                            model.apply(lambda m: setattr(m, 'depth_mult', depth_mult))
                            for width_mult in sorted(args.width_mult_list, reverse=True):
                                model.apply(lambda m: setattr(m, 'width_mult', width_mult))
                                results = evaluate(args, model, tokenizer)

                                logger.info("********** start evaluate results *********")
                                logger.info("depth_mult: %s ", depth_mult)
                                logger.info("width_mult: %s ", width_mult)
                                logger.info("results: %s ", results)
                                logger.info("********** end evaluate results *********")

                                acc.append(list(results.values())[0])
                                if args.task_name == "mnli":
                                    acc_both.append(list(results.values())[0:2])

                        # save model
                        if sum(acc) > current_best:
                            current_best = sum(acc)
                            if args.task_name == "mnli":
                                print("***best***{}\n".format(acc_both))
                                with open(output_eval_file, "a") as writer:
                                    writer.write("{}\n".format(acc_both))
                            else:
                                print("***best***{}\n".format(acc))
                                with open(output_eval_file, "a") as writer:
                                    writer.write("{}\n" .format(acc))

                            logger.info("Saving model checkpoint to %s", args.output_dir)
                            model_to_save = model.module if hasattr(model, 'module') else model
                            model_to_save.save_pretrained(args.output_dir)
                            torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
                            model_to_save.config.to_json_file(os.path.join(args.output_dir, CONFIG_NAME))
                            tokenizer.save_vocabulary(args.output_dir)

            if 0 < t_total < global_step:
                epoch_iterator.close()
                break

        if 0 < t_total < global_step:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    """ Evaluate the model """
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + 'MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                     batch_size=args.eval_batch_size)

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[3]}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert'] else None
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        if eval_task == 'mnli-mm':
            results.update({'acc_mm':result['acc']})
        else:
            results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt") # wirte all the results to the same file
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
            writer.write("\n")
    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False):

    processor = processors[task]()
    output_mode = output_modes[task]
    logger.info("Creating features from dataset file at %s", args.data_dir)
    label_list = processor.get_labels()
    if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
        label_list[1], label_list[2] = label_list[2], label_list[1]
    examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
    if not evaluate and args.data_aug:
        examples_aug = processor.get_train_examples_aug(args.data_dir)
        examples = examples + examples_aug
    features = convert_examples_to_features(examples,
                                            tokenizer,
                                            label_list=label_list,
                                            max_length=args.max_seq_length,
                                            output_mode=output_mode,
                                            pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
    )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def compute_neuron_head_importance(args, model, tokenizer):
    """ This method shows how to compute:
        - neuron importance scores based on loss according to http://arxiv.org/abs/1905.10650
    """
    # prepare things for heads
    model = model.module if hasattr(model, 'module') else model
    base_model = getattr(model, model.base_model_prefix, model)
    n_layers, n_heads = base_model.config.num_hidden_layers, base_model.config.num_attention_heads
    head_importance = torch.zeros(n_layers, n_heads).to(args.device)
    head_mask = torch.ones(n_layers, n_heads).to(args.device)
    head_mask.requires_grad_(requires_grad=True)

    # collect weights
    intermediate_weight = []
    intermediate_bias = []
    output_weight = []
    for name, w in model.named_parameters():
        if 'intermediate' in name:
            if w.dim() > 1:
                intermediate_weight.append(w)
            else:
                intermediate_bias.append(w)

        if 'output' in name and 'attention' not in name:
            if w.dim() > 1:
                output_weight.append(w)

    neuron_importance = []
    for w in intermediate_weight:
        neuron_importance.append(torch.zeros(w.shape[0]).to(args.device))

    model.to(args.device)

    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + 'MM') if args.task_name == "mnli" else (args.output_dir,)

    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, input_mask, _, label_ids = batch
            segment_ids = batch[2] if args.model_type=='bert' else None  # RoBERTa does't use segment_ids

            # calculate head importance
            outputs = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids,
                            head_mask=head_mask)
            loss = outputs[0]
            loss.backward()
            head_importance += head_mask.grad.abs().detach()

            # calculate  neuron importance
            for w1, b1, w2, current_importance in zip(intermediate_weight, intermediate_bias, output_weight, neuron_importance):
                current_importance += ((w1 * w1.grad).sum(dim=1) + b1 * b1.grad).abs().detach()
                current_importance += ((w2 * w2.grad).sum(dim=0)).abs().detach()

    return head_importance, neuron_importance


def reorder_neuron_head(model, head_importance, neuron_importance):
    """ reorder neurons based on their importance.

        Arguments:
            model: bert model
            head_importance: 12*12 matrix for head importance in 12 layers
            neuron_importance: list for neuron importance in 12 layers.
    """
    model = model.module if hasattr(model, 'module') else model
    base_model = getattr(model, model.base_model_prefix, model)

    # reorder heads and ffn neurons
    for layer, current_importance in enumerate(neuron_importance):
        # reorder heads
        idx = torch.sort(head_importance[layer], descending=True)[-1]
        base_model.encoder.layer[layer].attention.reorder_heads(idx)
        # reorder neurons
        idx = torch.sort(current_importance, descending=True)[-1]
        base_model.encoder.layer[layer].intermediate.reorder_neurons(idx)
        base_model.encoder.layer[layer].output.reorder_neurons(idx)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_dir", default=None, type=str, required=True,
                        help="The student (and teacher) model dir.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where trained model is saved.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--evaluate_during_training", default=True,
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", default=True,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float,
                        help="dropout rate on hidden states.")
    parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float,
                        help="dropout rate on attention probs.")

    parser.add_argument('--data_aug', action='store_true', help="whether using data augmentation")
    # for depth direction
    parser.add_argument('--depth_mult_list', type=str, default='1.',
                        help="the possible depths used for training, e.g., '1.' is for default")
    parser.add_argument("--depth_lambda1", default=1.0, type=float,
                        help="logit matching coef.")
    parser.add_argument("--depth_lambda2", default=1.0, type=float,
                        help="hidden states matching coef.")
    # for width direction
    parser.add_argument('--width_mult_list', type=str, default='1.',
                        help="the possible widths used for training, e.g., '1.' is for separate training "
                             "while '0.25,0.5,0.75,1.0' is for vanilla slimmable training")
    parser.add_argument("--width_lambda1", default=1.0, type=float,
                        help="logit matching coef.")
    parser.add_argument("--width_lambda2", default=0.1, type=float,
                        help="hidden states matching coef.")

    parser.add_argument("--training_phase", default="dynabertw", type=str,
                        help="can be finetuning, dynabertw, dynabert, final_finetuning")

    args = parser.parse_args()

    args.width_mult_list = [float(width) for width in args.width_mult_list.split(',')]
    args.depth_mult_list = [float(depth) for depth in args.depth_mult_list.split(',')]

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Set seed
    set_seed(args)

    # Prepare GLUE task: provide num_labels here
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # prepare model, tokernizer and config
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_dir, num_labels=num_labels, finetuning_task=args.task_name)
    config.output_attentions, config.output_hidden_states, config.output_intermediate = True,True,True
    tokenizer = tokenizer_class.from_pretrained(args.model_dir, do_lower_case=args.do_lower_case)

    # load teacher model if necessary
    if args.training_phase == 'dynabertw' or args.training_phase == 'dynabert':
        teacher_model = model_class.from_pretrained(args.model_dir, config=config)
        teacher_model.to(args.device)
    else:
        teacher_model = None

    # load student model if necessary
    model = model_class.from_pretrained(args.model_dir, config=config)

    if args.training_phase == 'dynabertw':
        # rewire the network according to the importance of attention heads and neurons
        head_importance, neuron_importance = compute_neuron_head_importance(args, model, tokenizer)
        reorder_neuron_head(model, head_importance, neuron_importance)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        if teacher_model:
            global_step, tr_loss = train(args, train_dataset, model, tokenizer, teacher_model)
        else:
            global_step, tr_loss = train(args, train_dataset, model, tokenizer)

        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


if __name__ == "__main__":
    main()
