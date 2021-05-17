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
from argparse import ArgumentParser

import pandas as pd
import logging
from imdb import IMDb
from pathlib import Path
from tqdm import tqdm

from ..kvtoraman_scraper.proxy_kon import PROXY
from movie_metadata import get_movie_by_id

logger = logging.getLogger('imdbpy')
logger.disabled = True


def load_df(save_path):
    film_df = pd.DataFrame(columns=('imdb_id', 'title', 'year', 'sub_count', 'names', 'roles', 'plot'))
    if not os.path.exists(save_path):
        print('No saved file available.')
        return film_df
    temp_df = pd.read_csv(save_path, na_filter=False)
    temp_df = temp_df.astype(str)
    temp_df['sub_count'] = temp_df.sub_count.astype(int)
    temp_df['roles'] = temp_df.roles.apply(eval)
    temp_df['summary'] = temp_df.summary.apply(eval)

    film_df = temp_df
    return film_df


def save(save_path):
    film_df.to_csv(save_path, index=False)
    print('SAVED')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--subtitles-path", default='../en/OpenSubtitles/xml/en')
    parser.add_argument("--save-path", default='statistics.csv')
    args = parser.parse_args()

    subs_path = Path(args.subtitles_path)

    for year in os.listdir(subs_path):
        year_dir = os.listdir(subs_path / year)
        with tqdm(enumerate(year_dir), total=len(year_dir)) as t:
            for i, imdb_id in t:
                t.set_description(year)
                if imdb_id in film_df.imdb_id.values:
                    continue

                xmls = [f for f in os.listdir(subs_path / year / imdb_id) if f.endswith('.xml')]

                movie = get_movie_by_id(imdb_id)
                names = ''
                for subs in xmls:
                    with open(subs_path / year / imdb_id / subs, encoding='utf-8') as f:
                        if '<w alternative=' in f.read():
                            names = subs

                film_df = film_df.append({'imdb_id': imdb_id, 'title': movie.title, 'year': year,
                                          'sub_count': len(xmls), 'names': names,
                                          'roles': movie.cast_clean, 'plot': movie.get_plot()}, ignore_index=True)
                if i % 50 == 0:
                    save(args.save_path)

    save(args.save_path)

