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
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from prepare_for_parse import create_or_clean
from mapper import Sentencizer, get_plot_sents


def get_longest_plot(plot, char_threshold=10):
    parts = []

    cur_symb = None
    cur_count = 0
    st = 0
    in_sep = False
    for pos, symb in enumerate(plot):
        if symb != cur_symb:
            cur_symb = symb
            cur_count = 0
            if in_sep:
                in_sep = False
                st = pos
        cur_count += 1
        if cur_count == char_threshold:
            parts.append(plot[st:pos - char_threshold + 1].strip())
            in_sep = True

    parts.append(plot[st:].strip())

    longest_part_pos = np.argmax([len(part) for part in parts])
    return parts[longest_part_pos]


if __name__ == '__main__':
    df_path = config.PIPELINE_PATH / 'names_and_plot.csv'
    scripts_path = config.PIPELINE_PATH / 'parsed_scenes'
    output_path = config.PIPELINE_PATH / 'scenes_and_plot'
    create_or_clean(output_path)

    film_df = pd.read_csv(df_path, na_filter=False).set_index('filename')
    film_df['ref_roles'] = film_df.ref_roles.apply(eval)

    sentencizer = Sentencizer()

    for filename in tqdm(os.listdir(scripts_path)):
        if not filename.endswith('json'):
            continue
        with open(scripts_path / filename) as f:
            j = json.load(f)

        film_row = film_df.loc[filename[:-4] + 'txt']
        if film_row.plot != "":
            del j['full_plot']

            j['imdb_id'] = film_row.imdb_id
            j['plot_sents'] = get_plot_sents(get_longest_plot(film_row.plot), sentencizer)
            j['roles'] = film_row.ref_roles

            scenes = j['scenes']
            del j['scenes']
            for scene in scenes:
                del scene['plot']

                new_turns = []
                for turn in scene['turns']:
                    new_turns.append([turn['speaker'], turn['utterance']])
                scene['turns'] = new_turns

            j['clusters'] = [
                {
                    'syn_ids': list(),
                    'scene_ids': [i],
                    'plot': list(),
                    'scenes': [scene],
                }
                for i, scene in enumerate(scenes)
            ]

            with open(output_path / filename, 'w') as f:
                json.dump(j, f)
