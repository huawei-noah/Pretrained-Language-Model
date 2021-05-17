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

import json
import numpy as np
from pathlib import Path

import config
from mapper import SentenceEncoderWrapper, preprocess, dtw_restricted, compute_mae


def get_film(film_path, gold_path, film):
    with open(film_path / film) as f:
        return json.load(f), np.load(gold_path / film.replace('.json', '_solved.npy'))


class ScoreKeeper:
    def __init__(self, name):
        self.name = name

        self.scores = []
        self.best_mae = 1000
        self.additional_mae = []

        self.best_coefs = None
        self.best_map = None

    def update(self, coefs, mae, additional_mae, syn2scenes):
        self.scores.append((coefs, mae, additional_mae))
        if mae < self.best_mae:
            self.best_mae = mae
            self.additional_mae = additional_mae
            self.best_coefs = coefs
            self.best_map = syn2scenes

    def print_best_so_far(self, max_sym_jac, max_one_sided_jac):
        fixed_coefs = {p: round(v, 2) for (p, v) in self.best_coefs.items()}
        print(self.name,
              f'cur_max [{round(max_sym_jac, 2)}, {round(max_one_sided_jac, 2)}], '
              f'best_mae {round(self.best_mae, 3)} additional_mae',
              f'{[round(x, 3) for x in self.additional_mae]} best_coefs {fixed_coefs}')


def compare_with_gold(score, gold):
    def evaluate_ext(mapping, ref_mapping, name):
        print(name, 'mae =', compute_mae(mapping, ref_mapping), '#clusters =', len(set(mapping)))

    print('-' * 80)
    print(score.name)
    evaluate_ext(gold, gold, 'ref')
    evaluate_ext(score.best_map, gold, 'coef')


def grid_search(pipeline_path, gold_path, films):
    pipeline_path, gold_path = Path(pipeline_path), Path(gold_path)
    sent_encoder = SentenceEncoderWrapper()

    films_path = pipeline_path / 'scenes_and_plot'
    film_pairs = []
    for film in films:
        j, gold = get_film(films_path, gold_path, film)
        film_pairs.append((j, gold))

    score_keepers = [ScoreKeeper('total')]
    for j, _ in film_pairs:
        score_keepers.append(ScoreKeeper(j['film_name']))

    scorers = []
    for j, _ in film_pairs:
        _, _, _, _, scorer, _ = preprocess(j.copy(), sent_encoder, add_empty_strings=True)
        scorers.append(scorer)

    try:
        for sym_jac in np.linspace(0, 1, 11):
            for one_sided_jac in np.linspace(0, 1, 11):
                for sim_mean in np.linspace(0, 1, 11):
                    for sim_max in np.linspace(0, 1, 11):
                        coefs = {'sim_max': sim_max, 'sim_mean': sim_mean, 'sym_jac': sym_jac,
                                 'one_sided_jac': one_sided_jac}

                        syn2scenes = [dtw_restricted(scorer, coefs)[1::2] for scorer in scorers]

                        pred_concat = sum(syn2scenes, [])
                        gold_concat = sum([list(g) for _, g in film_pairs], [])
                        mae = [compute_mae(pred_concat, gold_concat)]
                        for syn2scene, (_, gold) in zip(syn2scenes, film_pairs):
                            mae.append(compute_mae(syn2scene, gold))

                        score_keepers[0].update(coefs, mae[0], mae[1:], syn2scenes)
                        for i, score_keeper in enumerate(score_keepers):
                            if i == 0:
                                continue
                            score_keeper.update(coefs, mae[i], mae[1:i] + mae[i + 1:], syn2scenes[i - 1])

                for score in score_keepers:
                    score.print_best_so_far(sym_jac, one_sided_jac)
                print('-' * 80)
    finally:
        print('RESULT:')
        for score_keeper, (_, gold) in zip(score_keepers[1:], film_pairs):
            compare_with_gold(score_keeper, gold)
        return score_keepers, [g for j, g in film_pairs]


if __name__ == '__main__':
    pipeline_path = 'combined'
    gold_path = 'gold'
    films = ['avengersthe2012.json', '12monkeys.json', 'friends_s02_e03.json', 'friends_s02_e09.json']
    grid_search(pipeline_path, gold_path, films)
