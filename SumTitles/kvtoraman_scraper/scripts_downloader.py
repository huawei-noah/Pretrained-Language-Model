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
import urllib.request
from bs4 import BeautifulSoup
from pathlib import Path

punct_trans = str.maketrans('', '', string.punctuation)


def reduce(s):
    return s.translate(punct_trans).lower().replace(' ', '')


def starts_index(prefix, li):
    for index, s in enumerate(li):
        if s.startswith(prefix):
            return index


def create_or_clean(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        for file in os.listdir(path):
            os.remove(path / file)


def get_lines(url):
    with urllib.request.urlopen(url) as opened:
        html = opened.read()
    soup = BeautifulSoup(html)

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()  # rip it out

    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]

    all_scripts_line = starts_index('ALL SCRIPTS', lines)
    writers_line = starts_index('Writers :', lines)
    try:
        return lines[all_scripts_line + 1:writers_line - 1]
    except Exception as e:
        return []


# PRETTY WOMAN ISSUE!!!


if __name__ == '__main__':
    save_path = Path('scraped_films')
    create_or_clean(save_path)
    error_count = 0

    with open('all_name_script.txt') as urls:
        urls = urls.readlines()
        for i, line in enumerate(urls):
            name, url = line.split('\t')
            name = name.strip()
            lines = get_lines(url)

            key = name
            if key.endswith('Script'):
                key = key[:-len('Script')]
            file = reduce(key) + '.txt'

            if name.endswith(', The'):
                name = 'The ' + name[:-len(', The')]

            if lines:
                with open(save_path / file, 'w', encoding="utf-8") as f:
                    print(name, file=f)
                    print('\n'.join(lines), file=f)
            else:
                error_count += 1
            print(f'{i + 1}/{len(urls)} {bool(lines)}, {error_count} errors {name}')
