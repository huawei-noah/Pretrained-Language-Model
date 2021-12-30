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
from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import sys
import pickle
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler,TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch.nn import MSELoss

from transformer import BertForQuestionAnswering,WEIGHTS_NAME, CONFIG_NAME
from transformer.modeling_quant import BertForQuestionAnswering as QuantBertForQuestionAnswering
from transformer import BertTokenizer
from transformer import BertAdam
from transformer import BertConfig

from utils_squad import *

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logger = logging.getLogger()

def soft_cross_entropy(predicts, targets):
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    return (- targets_prob * student_likelihood).mean()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default='data/',
                        type=str,
                        help="The data directory.")
    parser.add_argument("--model_dir",
                        default='models/',
                        type=str,
                        help="The models directory.")
    parser.add_argument("--teacher_model",
                        default=None,
                        type=str,
                        help="The models directory.")
    parser.add_argument("--student_model",
                        default=None,
                        type=str,
                        help="The models directory.")
    parser.add_argument("--output_dir",
                        default='output',
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--version_2_with_negative',
                        action='store_true', 
                        help="Squadv2.0 if true else Squadv1.1 ")

    # default
    parser.add_argument("--max_seq_length",
                        default=384,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--doc_stride", 
                        default=128, 
                        type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", 
                        default=64, 
                        type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--n_best_size", 
                        default=20, 
                        type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", 
                        default=30, 
                        type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", 
                        default=0, 
                        type=int)
    parser.add_argument('--null_score_diff_threshold',
                        type=float, 
                        default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--do_lower_case',
                        #action='store_true',
                        default=True,
                        help="do lower case")
    

    parser.add_argument("--per_gpu_batch_size",
                        default=16,
                        type=int,
                        help="Per GPU batch size for training.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--eval_step', 
                        type=int, 
                        default=200,
                        help="Evaluate every X training steps")
    
    parser.add_argument('--pred_distill',
                        action='store_true',
                        help="Whether to distil with task layer")
    parser.add_argument('--intermediate_distill',
                        action='store_true',
                        help="Whether to distil with intermediate layers")
    parser.add_argument('--save_fp_model',
                        action='store_true',
                        help="Whether to save fp32 model")
    parser.add_argument('--save_quantized_model',
                        action='store_true',
                        help="Whether to save quantized model")
    
    
    parser.add_argument("--weight_bits",
                        default=2,
                        type=int,
                        choices=[2,8],
                        help="Quantization bits for weight.")
    parser.add_argument("--input_bits",
                        default=8,
                        type=int,
                        help="Quantization bits for activation.")
    parser.add_argument("--clip_val",
                        default=2.5,
                        type=float,
                        help="Initial clip value.")

    

    args = parser.parse_args()
    summaryWriter = SummaryWriter(args.output_dir)

    if args.teacher_model is None:
        args.teacher_model = args.model_dir
    if args.student_model is None:
        args.student_model = args.model_dir

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    args.batch_size = args.n_gpu*args.per_gpu_batch_size

    logger.info(f'The args: {args}')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained(args.teacher_model, do_lower_case=True)

    # preparing training data
    input_file = 'train-v2.0' if args.version_2_with_negative else 'train-v1.1'
    input_file = os.path.join(args.data_dir,input_file)
    if os.path.exists(input_file):
        train_features = pickle.load(open(input_file,'rb'))
    else:
        input_file = 'train-v2.0.json' if args.version_2_with_negative else 'train-v1.1.json'
        input_file = os.path.join(args.data_dir,input_file)
        _, train_examples = read_squad_examples(
                        input_file=input_file, is_training=True,
                        version_2_with_negative=args.version_2_with_negative)
        train_features = convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=True)
        
    num_train_optimization_steps = int(
        len(train_features) / args.batch_size) * args.num_train_epochs
    logger.info("***** Running training *****")
    logger.info("  Num split examples = %d", len(train_features))
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_start_positions, all_end_positions)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    input_file = 'dev-v2.0.json' if args.version_2_with_negative else 'dev-v1.1.json'
    args.dev_file = os.path.join(args.data_dir,input_file)
    dev_dataset, eval_examples = read_squad_examples(
                        input_file=args.dev_file, is_training=False,
                        version_2_with_negative=args.version_2_with_negative)
    eval_features = convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=False)
    logger.info("***** Running predictions *****")
    logger.info("  Num orig examples = %d", len(eval_examples))
    logger.info("  Num split examples = %d", len(eval_features))
    logger.info("  Batch size = %d", args.batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

    teacher_model = BertForQuestionAnswering.from_pretrained(args.teacher_model)
    teacher_model.to(args.device)
    teacher_model.eval()
    if args.n_gpu > 1:
        teacher_model = torch.nn.DataParallel(teacher_model)
    result = do_eval(args,teacher_model, eval_dataloader,eval_features,eval_examples,args.device, dev_dataset)
    em,f1 = result['exact_match'],result['f1']
    logger.info(f"Full precision teacher exact_match={em},f1={f1}")
    
    student_config = BertConfig.from_pretrained(args.student_model, 
                                                quantize_act=True,
                                                weight_bits = args.weight_bits,
                                                input_bits = args.input_bits,
                                                clip_val = args.clip_val)
    student_model = QuantBertForQuestionAnswering.from_pretrained(args.student_model,config = student_config)
    student_model.to(args.device)

    if args.n_gpu > 1:
        student_model = torch.nn.DataParallel(student_model)
    # Prepare optimizer
    param_optimizer = list(student_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    schedule = 'warmup_linear'
    optimizer = BertAdam(optimizer_grouped_parameters,
                            schedule=schedule,
                            lr=args.learning_rate,
                            warmup=0.1,
                            t_total=num_train_optimization_steps)

    loss_mse = MSELoss()
    # Train and evaluate
    global_step = 0
    best_dev_f1 = 0.0
    flag_loss = float('inf')
    previous_best = None
    tr_loss = 0.
    tr_att_loss = 0.
    tr_rep_loss = 0.
    tr_cls_loss = 0.
    for epoch_ in range(int(args.num_train_epochs)):
        for step, batch in enumerate(train_dataloader):
            student_model.train()
            batch = tuple(t.to(args.device) for t in batch)

            input_ids, input_mask, segment_ids, start_positions, end_positions = batch
            att_loss = 0.
            rep_loss = 0.
            cls_loss = 0.
            loss = 0

            student_logits, student_atts, student_reps = student_model(input_ids,segment_ids,input_mask)
            with torch.no_grad():
                teacher_logits, teacher_atts, teacher_reps = teacher_model(input_ids, segment_ids, input_mask)

            if args.pred_distill:
                soft_start_ce_loss = soft_cross_entropy(student_logits[0], teacher_logits[0])
                soft_end_ce_loss = soft_cross_entropy(student_logits[1], teacher_logits[1])
                cls_loss = soft_start_ce_loss + soft_end_ce_loss
                loss += cls_loss
                tr_cls_loss += cls_loss.item()
            
            if args.intermediate_distill:
                for student_att, teacher_att in zip(student_atts, teacher_atts):
                    student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(args.device),
                                                student_att)
                    teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(args.device),
                                                teacher_att)
                    tmp_loss = loss_mse(student_att, teacher_att)
                    att_loss += tmp_loss

                for student_rep, teacher_rep in zip(student_reps, teacher_reps):
                    tmp_loss = loss_mse(student_rep, teacher_rep)
                    rep_loss += tmp_loss

                loss += rep_loss + att_loss
                tr_att_loss += att_loss.item()
                tr_rep_loss += rep_loss.item()
                

            if args.n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            tr_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            save_model = False
            if global_step % args.eval_step == 0 or global_step == num_train_optimization_steps-1:
                logger.info("***** Running evaluation *****")
                logger.info(f"  Epoch = {epoch_} iter {global_step} step")
                if previous_best is not None:
                    logger.info(f"Previous best = {previous_best}")

                student_model.eval()
                result = do_eval(args,student_model, eval_dataloader,eval_features,eval_examples,args.device, dev_dataset)
                em,f1 = result['exact_match'],result['f1']
                logger.info(f'{em}/{f1}')
                if f1 > best_dev_f1:
                    previous_best = f"exact_match={em},f1={f1}"
                    best_dev_f1 = f1
                    save_model = True

                summaryWriter.add_scalars('performance',{'exact_match':em,
                                            'f1':f1},global_step)
                loss = tr_loss / global_step
                cls_loss = tr_cls_loss / global_step
                att_loss = tr_att_loss / global_step
                rep_loss = tr_rep_loss / global_step

                summaryWriter.add_scalar('total_loss',loss,global_step)
                summaryWriter.add_scalars('distill_loss',{'att_loss':att_loss,
                                            'rep_loss':rep_loss,
                                            'cls_loss':cls_loss},global_step)

                
            #save quantiozed model
            if save_model:
                logger.info(previous_best)
                if args.save_fp_model:
                    logger.info("******************** Save full precision model ********************")
                    model_to_save = student_model.module if hasattr(student_model, 'module') else student_model
                    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

                    torch.save(model_to_save.state_dict(), output_model_file)
                    model_to_save.config.to_json_file(output_config_file)
                    tokenizer.save_vocabulary(args.output_dir)
                if args.save_quantized_model:
                    logger.info("******************** Save quantized model ********************")
                    output_quant_dir = os.path.join(args.output_dir, 'quant')
                    if not os.path.exists(output_quant_dir):
                        os.makedirs(output_quant_dir)
                    model_to_save = student_model.module if hasattr(student_model, 'module') else student_model
                    quant_model = copy.deepcopy(model_to_save)
                    for name, module in quant_model.named_modules():
                        if hasattr(module,'weight_quantizer'):
                            module.weight.data = module.weight_quantizer.apply(module.weight,module.weight_clip_val,
                                                                         module.weight_bits,True)

                    output_model_file = os.path.join(output_quant_dir, WEIGHTS_NAME)
                    output_config_file = os.path.join(output_quant_dir, CONFIG_NAME)

                    torch.save(quant_model.state_dict(), output_model_file)
                    model_to_save.config.to_json_file(output_config_file)
                    tokenizer.save_vocabulary(output_quant_dir)


if __name__ == "__main__":
    main()
