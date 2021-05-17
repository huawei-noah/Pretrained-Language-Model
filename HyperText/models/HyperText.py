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


import torch.nn as nn
from hyperbolic.poincare import *
from hyperbolic.mobius_linear import *


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.c_seed=1.0
        self.manifold = PoincareBall()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            emb = torch.Tensor(config.n_vocab, config.embed)
            nn.init.xavier_normal_(emb)
            with torch.no_grad():
                emb[0].fill_(0)
            self.embedding = ManifoldParameter(emb, requires_grad=True,
                                               manifold=self.manifold, c=self.c_seed)
        emb_wordngram = torch.Tensor(config.bucket, config.embed)

        #nn.init.uniform_(emb_wordngram, -0.01, 0.01)
        nn.init.xavier_normal_(emb_wordngram)
        with torch.no_grad():
            emb_wordngram[0].fill_(0)
        self.embedding_wordngram = ManifoldParameter(emb_wordngram, requires_grad=True,
                                                     manifold=self.manifold, c=self.c_seed)
        self.dropout = nn.Dropout(config.dropout)
        self.hyperLinear =MobiusLinear(self.manifold, config.embed,
                                       config.num_classes, c=self.c_seed)

    def forward(self, x):
        out_word = self.embedding[x[0]]
        out_wordngram = self.embedding_wordngram[x[1]]
        out = torch.cat((out_word, out_wordngram), 1)
        out = self.dropout(out)
        out = self.manifold.einstein_midpoint(out, c=self.c_seed)
        out = self.hyperLinear(out)
        out = self.manifold.logmap0(out, self.c_seed)

        return out
