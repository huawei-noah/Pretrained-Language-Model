# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE

from ..mapper_coef_grid_search import grid_search

if __name__ == '__main__':
    score_keepers, golds = grid_search('pipeline', '../gold', ['friends_s02_e03.json', 'friends_s02_e09.json'])

    for i, score_keeper in enumerate(score_keepers):
        print(score_keeper.name, score_keeper.best_map)
        if i == 0:
            print('gold', golds)
        scores = [x[:2] for x in score_keeper.scores]
        scores = sorted(scores, key=lambda x: x[1])
        for i, (coefs, mae), in enumerate(scores[:20]):
            print(i, mae, coefs)
        print('-' * 80)
