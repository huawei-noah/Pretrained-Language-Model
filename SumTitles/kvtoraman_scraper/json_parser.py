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
from bs4 import BeautifulSoup
from urllib import request
import sys

with open(sys.argv[1]) as data_file:
    data = json.load(data_file)

with open('all_name_script.txt', 'w') as out:
    for i, movie in enumerate(data):
        url = movie['link'].replace(' ', '%20')
        name = movie['name'].replace('\t', ' ')
        name = name.replace('Script', '')

        # print(url)
        html = request.urlopen(url).read().decode('utf8', errors='ignore')
        soup = BeautifulSoup(html, 'html.parser')
        links_in_page = soup.find_all('a')
        found = False
        for link in links_in_page:
            text = link.get_text()
            if 'Read' in text and 'Script' in text:
                found = True
                script_link = 'http://www.imsdb.com' + link.get('href')
                print(f'{i + 1}/{len(data)}', name + '\t' + script_link)
                out.write(name + '\t' + script_link + '\n')
                break
        if not found:
            print('NOTFOUND', name + '\t' + script_link)
