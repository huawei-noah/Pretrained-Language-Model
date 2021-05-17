#-*- coding:utf-8 -*-
#The MIT License (MIT)
#Copyright (c) 2021 Huawei Technologies Co., Ltd.

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import time
import os
import torch
import numpy as np
import random
from train import train
from importlib import import_module
import argparse
from models.Config import Config
from utils import build_dataset, get_time_dif, build_dataloader

parser = argparse.ArgumentParser(description='HyperText Text Classification')
parser.add_argument('--model', type=str, default='HyperText',
                    help='HyperText')
parser.add_argument('--embedding', default='random', type=str, help='using random init word embedding or using pretrained')
parser.add_argument('--use_word_segment', default=True, type=bool, help='True for word, False for char')
parser.add_argument('--datasetdir', default='./data/tnews_public', type=str, help='dataset dir')
parser.add_argument('--outputdir', default='./output', type=str, help='output dir')
parser.add_argument('--dropout', default=0.0, type=float, help='dropout')
parser.add_argument('--require_improvement', default=6000, type=int, help='require_improvement steps')
parser.add_argument('--num_epochs', default=50, type=int, help='num_epochs')
parser.add_argument('--batch_size', default=32, type=int, help='batch_size')
parser.add_argument('--max_length', default=40, type=int, help='max_length')
parser.add_argument('--learning_rate', default=1.1e-2, type=float, help='learning_rate')
parser.add_argument('--embed_dim', default=200, type=int, help='embed_dim')
parser.add_argument('--bucket', default=1500000, type=int, help='total ngram bucket size')
parser.add_argument('--wordNgrams', default=2, type=int, help='use max n-grams, eg: 2 or 3 ...')
parser.add_argument('--eval_per_batchs', default=100, type=int, help='eval_per_batchs')
parser.add_argument('--min_freq', default=1, type=int, help='min word frequents of construct vocab')
parser.add_argument('--lr_decay_rate', default=0.96, type=float, help='lr_decay_rate')#0.96,0.87

args = parser.parse_args()

def print_config(config):
    print("hyperparamper config:")
    for k, v in config.__dict__.items():
        print("%s=%s" % (str(k), str(v)))

if __name__ == '__main__':
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    dataset = args.datasetdir
    outputdir =  args.outputdir

    embedding = ''
    if args.embedding == 'random':
        embedding = 'random'
    else:
        embedding=args.embedding

    model_name = args.model
    print(model_name)

    x = import_module('models.' + model_name)
    config = Config(dataset, outputdir, embedding)

    # reset config
    config.model_name = args.model
    config.save_path = os.path.join(outputdir, args.model + '.ckpt')
    config.log_path = os.path.join(outputdir, args.model + '.log')
    config.dropout = float(args.dropout)
    config.require_improvement = int(args.require_improvement)
    config.num_epochs = int(args.num_epochs)
    config.batch_size = int(args.batch_size)
    config.max_length = int(args.max_length)
    config.learning_rate = float(args.learning_rate)
    config.embed = int(args.embed_dim)
    config.bucket = int(args.bucket)
    config.wordNgrams = int(args.wordNgrams)
    config.lr_decay_rate=float(args.lr_decay_rate)

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.use_word_segment,
                                                           min_freq=int(args.min_freq))
    time_dif = get_time_dif(start_time)
    print("Finished Data loaded...")
    print("Time usage:", time_dif)

    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)

    print_config(config)
    print(model.parameters)
    train(config, model, train_data, dev_data, test_data, int(args.eval_per_batchs))
