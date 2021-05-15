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

import os
import torch
import numpy as np


class Config(object):
    """hyperparameter configuration"""

    def __init__(self, datasetdir, outputdir, embedding):
        self.model_name = 'HyperText'
        self.train_path = os.path.join(datasetdir, 'train.txt')
        self.dev_path = os.path.join(datasetdir, 'dev.txt')
        self.test_path = os.path.join(datasetdir, 'test.txt')
        self.class_list = []
        self.vocab_path = os.path.join(datasetdir, 'vocab.txt')
        self.labels_path = os.path.join(datasetdir, 'labels.txt')
        self.save_path = os.path.join(outputdir, self.model_name + '.ckpt')
        self.log_path = os.path.join(outputdir, self.model_name + '.log')
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        self.embedding_pretrained = torch.tensor(
            np.load(os.path.join(datasetdir, embedding))["embeddings"].astype('float32')) \
            if embedding != 'random' else None  # pretrained word embedding
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # cuda or cpu
        self.dropout = 0.5
        self.require_improvement = 2500  # max patient steps when early stoping
        self.num_classes = len(self.class_list)  # label number
        self.n_vocab = 0
        self.num_epochs = 30
        self.wordNgrams = 2
        self.batch_size = 32
        self.max_length = 1000
        self.learning_rate = 1e-2
        self.embed = self.embedding_pretrained.size(1) \
            if self.embedding_pretrained is not None else 100
        self.bucket = 20000  # word and ngram vocab size
        self.lr_decay_rate = 0.96
