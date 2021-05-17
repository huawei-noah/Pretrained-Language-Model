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

import os
import string
import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from movie_metadata import search_movie

PUNCT_TRANS = str.maketrans('', '', string.punctuation)


def reduce(s):
    return s.translate(PUNCT_TRANS).lower().replace(' ', '')


def check_year(cand):
    if 'year' in cand.data:
        return cand.data['year'] < 2016
    return True


def get_possible_names(filename):
    name = filename[:-4]  # removing .txt
    if not name.endswith('the'):
        return [name]
    return [name[:-3], 'the' + name[:-3], name]


def choose_by_roles(candidates, roles):
    candidates_with_roles, movies, scores = [], [], []
    for cand in candidates:
        if cand.cast:
            candidates_with_roles.append(cand)
            scores.append(len(roles & cand.cast))

    sorted_scores = sorted(scores, key=lambda x: -x) + [0, 0]
    if sorted_scores[0] - sorted_scores[1] > 2:
        best_score_pos = np.argmax(scores)
        return candidates_with_roles[best_score_pos], sorted_scores[:-2]
    return None, sorted_scores[:-2]


def not_found(final_candidate, sorted_scores):
    return final_candidate is None and (len(sorted_scores) == 0 or sorted_scores[0] <= 2)


def filter_equal(name, cands, equal=True):
    equality_func = (lambda x, y: x == y) if equal else (lambda x, y: x != y)
    return [cand for cand in cands if equality_func(name, cand.get('title'))]


def retrieve(film_df, save_path):
    def save():
        film_df.to_csv(save_path, index=False)
        print('SAVED')

    for i, row in tqdm(film_df.iterrows()):
        if row.status != 'not_processed':
            continue
        try:
            cands = search_movie(row.film_name)
            get_film = lambda equal: choose_by_roles(filter_equal(row.film_name, cands, equal=equal), row.roles)
            final_candidate, sorted_scores = get_film(equal=True)
            if not_found(final_candidate, sorted_scores):
                print(f'{sorted_scores}, USING UNEQUAL FILMS')
                final_candidate, sorted_scores = get_film(equal=False)

            if not_found(final_candidate, sorted_scores):
                film_df.loc[i, 'status'] = 'not_found'
                print(row.filename, 'NOT FOUND', sorted_scores)
                continue

            if final_candidate is None:
                film_df.loc[i, 'status'] = 'ambiguous'
                print(row.filename, 'AMBIGUOUS', sorted_scores)
                continue

            film_df.loc[i, 'status'] = 'correct'
            film_df.loc[i, 'imdb_id'] = final_candidate.imdb_id
            film_df.loc[i, 'retrieved_name'] = final_candidate.title
            film_df.loc[i, 'ref_roles'] = final_candidate.cast_clean
            film_df.loc[i, 'plot'] = final_candidate.get_plot()

            print(row.filename, '->', final_candidate.title, sorted_scores)
        except KeyboardInterrupt as e:
            film_df.loc[i, 'status'] = 'not_processed'
            raise e

        if (i + 1) % 10 == 0:
            save()

    save()


if __name__ == '__main__':
    load_path = config.PIPELINE_PATH / 'names_and_roles.csv'
    save_path = config.PIPELINE_PATH / 'names_and_plot.csv'
    if os.path.exists(save_path):
        load_path = save_path

    film_df = pd.read_csv(load_path, na_filter=False)
    film_df['roles'] = film_df.roles.apply(eval)
    if load_path != save_path:
        film_df['ref_roles'] = film_df.roles.apply(lambda x: list())

    retrieve(film_df, save_path)

    print('OVERALL:')
    print(film_df.status.value_counts())
    print()
    print('WITH PLOT:', np.sum(film_df.plot != ''))
