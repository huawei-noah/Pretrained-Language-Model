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
import requests
from pathlib import Path
from bs4 import BeautifulSoup

from ..mapper import Sentencizer, get_plot_sents

FRIENDS_PLOTS_URL = "http://www.friends-tv.org/epshort.html"


def convert_episode(episode, summary_map, sentencizer):
    episode_id = episode['episode_id']
    title, summary = summary_map[episode_id]
    film_name = f"Friends {episode_id.replace('_', ' ')}: {title}"
    filename = f'friends_{episode_id}.json'
    out_j = {
        'filename': filename,
        'film_name': film_name,
        'roles': set(),
        'imdb_id': '',
        'plot_sents': get_plot_sents(summary, sentencizer),
        'clusters': []
    }

    for i, scene in enumerate(episode['scenes']):
        converted_scene = {'description': '', 'turns': []}
        converted_cluster = {'syn_ids': [], 'scene_ids': [i], 'synopsis': [], 'scenes': [converted_scene]}
        for turn in scene['utterances']:
            if len(turn['speakers']) == 0 or turn['transcript'] == '':
                continue
            speaker = turn['speakers'][0]
            converted_scene['turns'].append([speaker.upper(),  turn['transcript']])
            out_j['roles'].add(speaker)

        if converted_scene['turns']:
            out_j['clusters'].append(converted_cluster)

    out_j['roles'] = list(out_j['roles'])
    return out_j


def convert_season(season, summary_map, sentencizer):
    return [convert_episode(episode, summary_map, sentencizer) for episode in season['episodes']]


def get_summary_map(summary_html):
    summary_text = [line.strip() for line in summary_html.get_text().split('\n\n')]
    summary_text = summary_text[summary_text.index('First Season Plots'):]
    summary_text = [line for line in summary_text if line and line[0].isnumeric()]

    summary_map = dict()
    for line in summary_text:
        name, summary = line.split('\n', 1)
        number, title = name.split(' ', 1)
        s, e = [int(x) for x in number.split('.')]
        number = f's{s:02d}_e{e:02d}'

        summary = summary.replace('\n', ' ').strip()
        summary_map[number] = (title, summary)

    return summary_map


if __name__ == '__main__':
    res = requests.get(FRIENDS_PLOTS_URL)
    if not res.ok:
        raise Exception("Cannot download plots from " + FRIENDS_PLOTS_URL)
    summary_html = BeautifulSoup(str(res.content, encoding='ISO-8859-1'), features='lxml')
    summary_map = get_summary_map(summary_html)
    sentencizer = Sentencizer()

    raw_path = Path('repo/json')
    out_path = Path('pipeline/scenes_and_synopsis')
    if not out_path.exists():
        os.makedirs(out_path)
    for file in os.listdir(out_path):
        os.remove(out_path / file)

    for file in os.listdir(raw_path):
        with open(raw_path / file) as f:
            raw_j = json.load(f)

        out_j_list = convert_season(raw_j, summary_map, sentencizer)

        for j in out_j_list:
            with open(out_path / j['filename'], 'w') as f:
                json.dump(j, f)

        for episode in out_j_list:
            print(f"Processed {episode['film_name']}, got {len(episode['clusters'])} scenes")
