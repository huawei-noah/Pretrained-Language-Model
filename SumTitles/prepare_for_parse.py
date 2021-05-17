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
import shutil
import pandas as pd
from tqdm import tqdm

import config


def create_or_clean(path, clean=True):
    if not os.path.exists(path):
        os.makedirs(path)
    elif clean:
        for file in os.listdir(path):
            os.remove(path / file)


if __name__ == '__main__':
    input_path = config.PIPELINE_PATH / 'scraped_scripts'
    output_path = config.PIPELINE_PATH / 'flattened_scripts'
    create_or_clean(output_path)

    film_df = pd.DataFrame(columns=('filename', 'imdb_id', 'retrieved_name', 'status', 'synopsis'))

    films = set()
    for genre in tqdm(os.listdir(input_path)):
        if genre.startswith('.'):
            continue

        for filename in os.listdir(input_path / genre):
            if filename not in films:
                films.add(filename)
                shutil.copy(input_path / genre / filename, output_path)
                row = {'filename': filename, 'imdb_id': '', 'retrieved_name': '', 'status': '', 'synopsis': ''}
                film_df = film_df.append(row, ignore_index=True)

    film_df['status'] = 'not_processed'

    print('Found', len(film_df), 'films')
    print(film_df.head())
    film_df.to_csv(config.PIPELINE_PATH / 'only_names.csv', index=False)
