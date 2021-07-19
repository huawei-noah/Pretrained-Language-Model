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
import os
import random
import numpy as np
import torch
import copy
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.nn import MSELoss
from transformer.configuration_bert import BertConfig
from transformer.tokenization import BertTokenizer
from transformer.optimization import BertAdam, WarmupLinearSchedule
from transformer.file_utils import WEIGHTS_NAME, CONFIG_NAME
from transformer.binary_model_init import tws_split

from helper import *
import pickle
from kd_learner_squad import KDLearner
from utils_squad import write_predictions,read_squad_examples,InputFeatures,convert_examples_to_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", default='tmp', type=str, help='Jobid to save training logs')
    parser.add_argument("--data_dir",default=None,type=str,help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--teacher_model",default=None,type=str,help="The teacher model dir.")
    parser.add_argument("--student_model",default=None,type=str,help="The student model dir.")
    parser.add_argument("--output_dir",default='output',type=str,help="The output directory where the model predictions and checkpoints will be written.")

    # default params for SQuAD
    parser.add_argument('--version_2_with_negative', 
                        action='store_true')
    parser.add_argument("--max_seq_length",
                        default=384,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument('--null_score_diff_threshold',
                        type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")
    
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--weight_decay', '--wd',
                        default=1e-4,
                        type=float,
                        metavar='W',
                        help='weight decay')
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--eval_step', type=int, default=200)
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    
    parser.add_argument('--do_eval',default = 0,type=int)
     # distillation params
    parser.add_argument('--aug_train', action='store_true',
                        help="Whether using data augmentation or not")
    parser.add_argument('--kd_type', default='no_kd', choices=['no_kd', 'two_stage', 'logit_kd', 'joint_kd'],
                        help="choose one of the kd type")
    parser.add_argument('--distill_logit', action='store_true',
                        help="Whether using distillation over logits or not")
    parser.add_argument('--distill_rep_attn', action='store_true',
                        help="Whether using distillation over reps and attns or not")
    parser.add_argument('--temperature', type=float, default=1.)
    # quantization params
    parser.add_argument("--weight_bits", default=32, type=int, help="number of bits for weight")
    parser.add_argument("--weight_quant_method", default='twn', type=str,
                        choices=['twn', 'bwn', 'uniform', 'laq'],
                        help="weight quantization methods, can be bwn, twn, laq")
    parser.add_argument("--input_bits",  default=32, type=int,
                        help="number of bits for activation")
    parser.add_argument("--input_quant_method", default='uniform', type=str, choices=['uniform', 'lsq'],
                        help="weight quantization methods, can be bwn, twn, or symmetric quantization for default")

    parser.add_argument('--learnable_scaling', action='store_true', default=True)
    parser.add_argument("--ACT2FN", default='gelu', type=str,
                        help='activation fn for ffn-mid. A8 uses uq + gelu; A4 uses lsq + relu.')
    # training config
    parser.add_argument('--sym_quant_ffn_attn', action='store_true',
                        help='whether use sym quant for attn score and ffn after act') # default asym
    parser.add_argument('--sym_quant_qkvo', action='store_true',  default=True,
                        help='whether use asym quant for Q/K/V and others.') # default sym
    # layerwise quantization config
    parser.add_argument('--clip_init_file', default='threshold_std.pkl', help='files to restore init clip values.')
    parser.add_argument('--clip_init_val', default=2.5, type=float, help='init value of clip_vals, default to (-2.5, +2.5).')
    parser.add_argument('--clip_lr', default=1e-4, type=float, help='Use a seperate lr for clip_vals / stepsize')
    parser.add_argument('--clip_wd', default=0.0, type=float, help='weight decay for clip_vals / stepsize')

    # layerwise quantization config
    parser.add_argument('--embed_layerwise', default=False, type=lambda x: bool(int(x)))
    parser.add_argument('--weight_layerwise', default=True, type=lambda x: bool(int(x)))
    parser.add_argument('--input_layerwise', default=True, type=lambda x: bool(int(x)))

    ### spliting
    parser.add_argument('--split', action='store_true',
                        help='whether to conduct tws spliting. NOTE this is only for training binarybert')
    parser.add_argument('--is_binarybert', action='store_true',
                        help='whether to use binarybert modelling.')

    args = parser.parse_args()

    log_dir = os.path.join(args.output_dir, 'record_%s.log' % args.job_id)
    init_logging(log_dir)
    print_args(vars(args))

    # Prepare devices
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logging.info("device: {} n_gpu: {}".format(device, n_gpu))

    # Prepare seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    tokenizer = BertTokenizer.from_pretrained(args.teacher_model, do_lower_case=True)
    config = BertConfig.from_pretrained(args.teacher_model)
    config.num_labels = 2

    student_config = copy.deepcopy(config)
    student_config.weight_bits = args.weight_bits
    student_config.input_bits = args.input_bits
    student_config.weight_quant_method = args.weight_quant_method
    student_config.input_quant_method = args.input_quant_method
    student_config.clip_init_val = args.clip_init_val
    student_config.learnable_scaling = args.learnable_scaling
    student_config.sym_quant_qkvo = args.sym_quant_qkvo
    student_config.sym_quant_ffn_attn = args.sym_quant_ffn_attn
    student_config.embed_layerwise = args.embed_layerwise
    student_config.weight_layerwise = args.weight_layerwise
    student_config.input_layerwise = args.input_layerwise
    student_config.hidden_act = args.ACT2FN
    logging.info("***** Training data *****")
    input_file = 'train-v2.0.json' if args.version_2_with_negative else 'train-v1.1.json'
    input_file = os.path.join(args.data_dir,input_file)

    if os.path.exists(input_file+'.features.pkl'):
        logging.info("  loading from cache %s", input_file+'.features.pkl')
        train_features = pickle.load(open(input_file+'.features.pkl', 'rb'))
    else:
        _, train_examples = read_squad_examples(input_file=input_file, is_training=True, 
                                                version_2_with_negative=args.version_2_with_negative)
        train_features = convert_examples_to_features(
                examples=train_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=True)
        pickle.dump(train_features, open(input_file+'.features.pkl','wb'))
        
    args.batch_size = args.batch_size // args.gradient_accumulation_steps
    num_train_optimization_steps = int(
        len(train_features) / args.batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

    logging.info("  Num examples = %d", len(train_features))
    logging.info("  Num total steps = %d", num_train_optimization_steps)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_start_positions, all_end_positions)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    logging.info("***** Evaluation data *****")
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

    logging.info("  Num examples = %d", len(eval_features))

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

    if not args.do_eval:
        from transformer.modeling_dynabert import BertForQuestionAnswering
        teacher_model = BertForQuestionAnswering.from_pretrained(args.teacher_model, config = config)
        teacher_model.to(device)
        if n_gpu > 1:
            teacher_model = torch.nn.DataParallel(teacher_model)

    if args.split:
        # rename the checkpoint to restore
        split_model_dir = os.path.join(args.output_dir,'binary_model_init')
        if not os.path.exists(split_model_dir):
            os.mkdir(split_model_dir)
        # copy the json file, avoid over-writing
        source_model_dir = os.path.join(args.student_model, CONFIG_NAME)
        target_model_dir = os.path.join(split_model_dir, CONFIG_NAME)
        os.system('cp -v %s %s' % (source_model_dir, target_model_dir))

        # create the split model ckpt
        source_model_dir = os.path.join(args.student_model, WEIGHTS_NAME)
        target_model_dir = os.path.join(split_model_dir, WEIGHTS_NAME)
        target_model_dir = tws_split(source_model_dir, target_model_dir)
        args.student_model = split_model_dir  # over-write student_model dir
        print("transformed binary model stored at: {}".format(target_model_dir))

    if args.is_binarybert:
        from transformer.modeling_dynabert_binary import BertForQuestionAnswering
        student_model = BertForQuestionAnswering.from_pretrained(args.student_model, config=student_config)
    else:
        from transformer.modeling_dynabert_quant import BertForQuestionAnswering
        student_model = BertForQuestionAnswering.from_pretrained(args.student_model, config=student_config)
    student_model.to(device)
    if n_gpu > 1:
        student_model = torch.nn.DataParallel(student_model)

    learner = KDLearner(args, device, student_model, teacher_model,num_train_optimization_steps)

    if args.do_eval:
        """ evaluation """
        learner.eval(student_model, eval_dataloader, eval_features, eval_examples, dev_dataset)
        return 0

    """ perform training """
    if args.kd_type == 'joint_kd':
        learner.args.distill_logit = True
        learner.args.distill_rep_attn = True
        learner.build()
        learner.train(train_dataloader, eval_dataloader, eval_features, eval_examples, dev_dataset)

    elif args.kd_type == 'logit_kd':
        # only perform the logits kd
        learner.args.distill_logit = True
        learner.args.distill_rep_attn = False
        learner.build(lr=args.learning_rate)
        learner.train(train_dataloader, eval_dataloader, eval_features, eval_examples, dev_dataset)

    elif args.kd_type == 'two_stage':
        # stage 1: intermediate layer distillation
        learner.args.distill_logit = False
        learner.args.distill_rep_attn = True
        learner.build(lr=2.5*args.learning_rate)
        learner.train(train_dataloader, eval_dataloader, eval_features, eval_examples, dev_dataset)

        # stage 2: prediction layer distillation
        learner.student_model.load_state_dict(torch.load(os.path.join(learner.output_dir,WEIGHTS_NAME)))
        learner.args.distill_logit = True
        learner.args.distill_rep_attn = False

        learner.build(lr=args.learning_rate)  # prepare the optimizer again.
        learner.train(train_dataloader, eval_dataloader, eval_features, eval_examples, dev_dataset)

    else:
        assert args.kd_type == 'no_kd'
        # NO kd training, vanilla cross entropy with hard label
        learner.build(lr=args.learning_rate)  # prepare the optimizer again.
        learner.train(train_dataloader, eval_dataloader, eval_features, eval_examples, dev_dataset)

    del learner
    return 0



if __name__ == "__main__":
    main()
