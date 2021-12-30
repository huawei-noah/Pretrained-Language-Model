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
from torch.nn import CrossEntropyLoss, MSELoss

from transformer import BertForSequenceClassification,WEIGHTS_NAME, CONFIG_NAME
from transformer.modeling_quant import BertForSequenceClassification as QuantBertForSequenceClassification
from transformer import BertTokenizer
from transformer import BertAdam
from transformer import BertConfig
from utils_glue import *

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logger = logging.getLogger()

def get_tensor_data(output_mode, features):
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    all_seq_lengths = torch.tensor([f.seq_length for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,all_label_ids, all_seq_lengths)
    return tensor_data, all_label_ids

def do_eval(model, task_name, eval_dataloader,
            device, output_mode, eval_labels, num_labels):
    eval_loss = 0
    nb_eval_steps = 0
    preds = []

    for _,batch_ in enumerate(eval_dataloader):
        batch_ = tuple(t.to(device) for t in batch_)
        with torch.no_grad():
            input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch_
            logits, _, _ = model(input_ids, segment_ids, input_mask)

        # create eval loss and other metric required by the task
        if output_mode == "classification":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif output_mode == "regression":
            loss_fct = MSELoss()
            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    preds = preds[0]
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(task_name, preds, eval_labels.numpy())
    result['eval_loss'] = eval_loss
    return result

def soft_cross_entropy(predicts, targets):
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    return (- targets_prob * student_likelihood).mean()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default='data',
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_dir",
                        default='models/tinybert',
                        type=str,
                        help="The model dir.")
    parser.add_argument("--teacher_model",
                        default=None,
                        type=str,
                        help="The models directory.")
    parser.add_argument("--student_model",
                        default=None,
                        type=str,
                        help="The models directory.")
    parser.add_argument("--task_name",
                        default='sst-2',
                        type=str,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default='output',
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    
    parser.add_argument('--aug_train',
                        action='store_false',
                        help="Whether to use augmented data or not")
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
    assert args.pred_distill or args.intermediate_distill, "'pred_distill' and 'intermediate_distill', at least one must be True"
    summaryWriter = SummaryWriter(args.output_dir)
    logger.info('The args: {}'.format(args))
    task_name = args.task_name.lower()
    data_dir = os.path.join(args.data_dir,task_name)
    output_dir = os.path.join(args.output_dir,task_name)
    # processed_data_dir = os.path.join(args.data_dir,'preprocessed',task_name)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    if args.student_model is None:
        args.student_model = os.path.join(args.model_dir,task_name)
    if args.teacher_model is None:
        args.teacher_model = os.path.join(args.model_dir,task_name)

    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mnli-mm": MnliMismatchedProcessor,
        "mrpc": MrpcProcessor,
        "sst-2": Sst2Processor,
        "sts-b": StsbProcessor,
        "qqp": QqpProcessor,
        "qnli": QnliProcessor,
        "rte": RteProcessor
    }

    output_modes = {
        "cola": "classification",
        "mnli": "classification",
        "mrpc": "classification",
        "sst-2": "classification",
        "sts-b": "regression",
        "qqp": "classification",
        "qnli": "classification",
        "rte": "classification"
    }

    default_params = {
        "cola": {"max_seq_length": 64,"batch_size":16,"eval_step":50},
        "mnli": {"max_seq_length": 128,"batch_size":32,"eval_step":1000},
        "mrpc": {"max_seq_length": 128,"batch_size":32,"eval_step":200},
        "sst-2": {"max_seq_length": 64,"batch_size":32,"eval_step":200},
        "sts-b": {"max_seq_length": 128,"batch_size":32,"eval_step":50},
        "qqp": {"max_seq_length": 128,"batch_size":32,"eval_step":1000},
        "qnli": {"max_seq_length": 128,"batch_size":32,"eval_step":1000},
        "rte": {"max_seq_length": 128,"batch_size":32,"eval_step":100}
    }

    acc_tasks = ["mnli", "mrpc", "sst-2", "qqp", "qnli", "rte"]
    corr_tasks = ["sts-b"]
    mcc_tasks = ["cola"]

    # Prepare devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    # Prepare seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
 
    if task_name in default_params:
        args.batch_size = default_params[task_name]["batch_size"]
        if n_gpu > 0:
            args.batch_size = int(args.batch_size*n_gpu)
        args.max_seq_length = default_params[task_name]["max_seq_length"]
        args.eval_step = default_params[task_name]["eval_step"]

    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.student_model, do_lower_case=True)
    
    if args.aug_train:
        try:
            train_file = os.path.join(processed_data_dir,'aug_data')
            train_features = pickle.load(open(train_file,'rb'))
        except:
            train_examples = processor.get_aug_examples(data_dir)
            train_features = convert_examples_to_features(train_examples, label_list,
                                            args.max_seq_length, tokenizer, output_mode)
    else:
        try:
            train_file = os.path.join(processed_data_dir,'train_data')
            train_features = pickle.load(open(train_file,'rb'))
        except:
            train_examples = processor.get_train_examples(data_dir)
            train_features = convert_examples_to_features(train_examples, label_list,
                                            args.max_seq_length, tokenizer, output_mode)

    num_train_optimization_steps = int(len(train_features) / args.batch_size) * args.num_train_epochs
    train_data, _ = get_tensor_data(output_mode, train_features)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
    
    try:
        dev_file = train_file = os.path.join(processed_data_dir,'dev_data')
        eval_features = pickle.load(open(dev_file,'rb'))
    except:
        eval_examples = processor.get_dev_examples(data_dir)
        eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)

    eval_data, eval_labels = get_tensor_data(output_mode, eval_features)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)
    if task_name == "mnli":
        processor = processors["mnli-mm"]()
        try:
            dev_mm_file = train_file = os.path.join(processed_data_dir,'dev-mm_data')
            mm_eval_features = pickle.load(open(dev_mm_file,'rb'))
        except:
            mm_eval_examples = processor.get_dev_examples(data_dir)
            mm_eval_features = convert_examples_to_features(
                mm_eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        
        mm_eval_data, mm_eval_labels = get_tensor_data(output_mode, mm_eval_features)
        logger.info("  Num examples = %d", len(mm_eval_features))

        mm_eval_sampler = SequentialSampler(mm_eval_data)
        mm_eval_dataloader = DataLoader(mm_eval_data, sampler=mm_eval_sampler,
                                        batch_size=args.batch_size)

    teacher_model = BertForSequenceClassification.from_pretrained(args.teacher_model)
    teacher_model.to(device)
    teacher_model.eval()
    if n_gpu > 1:
        teacher_model = torch.nn.DataParallel(teacher_model)

    result = do_eval(teacher_model, task_name, eval_dataloader,
                    device, output_mode, eval_labels, num_labels)
    if task_name in acc_tasks:
        if task_name in ['sst-2','mnli','qnli','rte']:
            fp32_performance = f"acc:{result['acc']}"
        elif task_name in ['mrpc','qqp']:
            fp32_performance = f"f1/acc:{result['f1']}/{result['acc']}"
    if task_name in corr_tasks:
        fp32_performance = f"pearson/spearmanr:{result['pearson']}/{result['spearmanr']}"

    if task_name in mcc_tasks:
        fp32_performance = f"mcc:{result['mcc']}"

    if task_name == "mnli":
        result = do_eval(teacher_model, 'mnli-mm', mm_eval_dataloader,
                            device, output_mode, mm_eval_labels, num_labels)
        fp32_performance += f"  mm-acc:{result['acc']}"
    fp32_performance = task_name +' fp32   ' + fp32_performance
    student_config = BertConfig.from_pretrained(args.teacher_model, 
                                                quantize_act=True,
                                                weight_bits = args.weight_bits,
                                                input_bits = args.input_bits,
                                                clip_val = args.clip_val)
    student_model = QuantBertForSequenceClassification.from_pretrained(args.student_model, config = student_config, num_labels=num_labels)
    student_model.to(device)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_features))
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    if n_gpu > 1:
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
    global_step = 0
    best_dev_acc = 0.0
    previous_best = None
    
    tr_loss = 0.
    tr_att_loss = 0.
    tr_rep_loss = 0.
    tr_cls_loss = 0.
    for epoch_ in range(int(args.num_train_epochs)):
        nb_tr_examples, nb_tr_steps = 0, 0

        for step, batch in enumerate(train_dataloader):
            student_model.train()
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch
            att_loss = 0.
            rep_loss = 0.
            cls_loss = 0.
            loss = 0.

            student_logits, student_atts, student_reps = student_model(input_ids, segment_ids, input_mask)

            with torch.no_grad():
                teacher_logits, teacher_atts, teacher_reps = teacher_model(input_ids, segment_ids, input_mask)
            
            if args.pred_distill:
                if output_mode == "classification":
                    cls_loss = soft_cross_entropy(student_logits,teacher_logits)
                elif output_mode == "regression":
                    cls_loss = loss_mse(student_logits, teacher_logits)

                loss = cls_loss
                tr_cls_loss += cls_loss.item()

            if args.intermediate_distill:
                for student_att, teacher_att in zip(student_atts, teacher_atts):
                    student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                                student_att)
                    teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                                teacher_att)
                    tmp_loss = loss_mse(student_att, teacher_att)
                    att_loss += tmp_loss

                for student_rep, teacher_rep in zip(student_reps, teacher_reps):
                    tmp_loss = loss_mse(student_rep, teacher_rep)
                    rep_loss += tmp_loss

                loss += rep_loss + att_loss
                tr_att_loss += att_loss.item()
                tr_rep_loss += rep_loss.item()

            if n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            tr_loss += loss.item()
            nb_tr_examples += label_ids.size(0)
            nb_tr_steps += 1
            if global_step % args.eval_step == 0 or global_step == num_train_optimization_steps-1:
                logger.info("***** Running evaluation *****")
                logger.info("  {} step of {} steps".format(global_step, num_train_optimization_steps))
                if previous_best is not None:
                    logger.info(f"{fp32_performance}\nPrevious best = {previous_best}")

                student_model.eval()

                loss = tr_loss / (step + 1)
                cls_loss = tr_cls_loss / (step + 1)
                att_loss = tr_att_loss / (step + 1)
                rep_loss = tr_rep_loss / (step + 1)

                result = do_eval(student_model, task_name, eval_dataloader,
                                    device, output_mode, eval_labels, num_labels)
                result['global_step'] = global_step
                result['cls_loss'] = cls_loss
                result['att_loss'] = att_loss
                result['rep_loss'] = rep_loss
                result['loss'] = loss
                summaryWriter.add_scalar('total_loss',loss,global_step)
                summaryWriter.add_scalars('distill_loss',{'att_loss':att_loss,
                                            'rep_loss':rep_loss,
                                            'cls_loss':cls_loss},global_step)

                if task_name=='cola':
                    summaryWriter.add_scalar('mcc',result['mcc'],global_step)
                elif task_name in ['sst-2','mnli','mnli-mm','qnli','rte','wnli']:
                    summaryWriter.add_scalar('acc',result['acc'],global_step)
                elif task_name in ['mrpc','qqp']:
                    summaryWriter.add_scalars('performance',{'acc':result['acc'],
                                                'f1':result['f1'],
                                                'acc_and_f1':result['acc_and_f1']},global_step)
                else:
                    summaryWriter.add_scalar('corr',result['corr'],global_step)

                save_model = False

                if task_name in acc_tasks and result['acc'] > best_dev_acc:
                    if task_name in ['sst-2','mnli','qnli','rte']:
                        previous_best = f"acc:{result['acc']}"
                    elif task_name in ['mrpc','qqp']:
                        previous_best = f"f1/acc:{result['f1']}/{result['acc']}"
                    best_dev_acc = result['acc']
                    save_model = True

                if task_name in corr_tasks and result['corr'] > best_dev_acc:
                    previous_best = f"pearson/spearmanr:{result['pearson']}/{result['spearmanr']}"
                    best_dev_acc = result['corr']
                    save_model = True

                if task_name in mcc_tasks and result['mcc'] > best_dev_acc:
                    previous_best = f"mcc:{result['mcc']}"
                    best_dev_acc = result['mcc']
                    save_model = True

                if save_model:
                    # Test mnli-mm
                    if task_name == "mnli":
                        result = do_eval(student_model, 'mnli-mm', mm_eval_dataloader,
                                            device, output_mode, mm_eval_labels, num_labels)
                        previous_best+= f"mm-acc:{result['acc']}"
                    logger.info(fp32_performance)
                    logger.info(previous_best)
                    if args.save_fp_model:
                        logger.info("******************** Save full precision model ********************")
                        model_to_save = student_model.module if hasattr(student_model, 'module') else student_model
                        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
                        output_config_file = os.path.join(output_dir, CONFIG_NAME)

                        torch.save(model_to_save.state_dict(), output_model_file)
                        model_to_save.config.to_json_file(output_config_file)
                        tokenizer.save_vocabulary(output_dir)
                    if args.save_quantized_model:
                        logger.info("******************** Save quantized model ********************")
                        output_quant_dir = os.path.join(output_dir, 'quant')
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
