# coding=utf-8
# Copyright 2020 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import sys
import six
import unicodedata
import collections
import base64

count = 0
output = open(sys.argv[1], "w", encoding = "utf-8")
b16 = {}
byteVocab = {}

Bvocab = open("../byteVocab.txt", 'r', encoding = 'utf-8')

for line in Bvocab:
    tokens = line.strip().split('\t')
    print(tokens[0])
    print(tokens[1])
    byteVocab[tokens[1]] = tokens[0]


def getChinese(context):
#    context = context.decode("utf-8") # convert context from str to unicode
    filtrate = re.compile(u'[^\u4E00-\u9FA5]') # non-Chinese unicode range
    context = filtrate.sub(r'', context) # remove all non-Chinese characters
#    context = context.encode("utf-8") # convert unicode back to str
    return context

for i in range(10):
    b16[i] = str(i) 

b16[10] = 'A'
b16[11] = 'B'
b16[12] = 'C'
b16[13] = 'D'
b16[14] = 'E'
b16[15] = 'F'

b256tob16 = {}
def base16decode(s):
    result = 0
    for c in s:
        result = result * 16 + b16[c]
    return result

def base16encode(n):
    result = ''
    n = int(n)
    while n > 0:
        n = int(n)
        result = b16[n%16] + result
        n /= 16
        n = int(n)
    return result

def base256encode(n):
    return chr(n)
    result = ''
    while n > 0:
        n = int(n)
        result = chr(n%256) + result
        n /= 256
    return result
for i in range(256):
    b256tob16[str(base256encode(i))] = i
for line in sys.stdin:
    print(line)
    line = line.split('\t') #bytes(line.strip(), encoding="utf-8")
#    output.writelines("{}\t".format(line[0]))
    print(line)
    if len(getChinese(line[0])) == 0: output.writelines("##")
    for token in line[0]:
        if token == ' ': continue
        token16 = token
        if len(getChinese(token[0])) > 0:
            token16 =str(chr(int(byteVocab[token[0]]))) + token[1:]
        tk = (str(base16encode((b256tob16[token16]))))
        output.writelines("{}".format(tk))
        print(tk)
    output.writelines("\t{}".format(str(line[1])))
    print(line[1])
    

for i in range(16):
    for j in range(16):
#        if i < 4: 
#            output.writelines("{}\t{}{}\t{}\n".format("C"+str(10*i+j), b16[i], b16[j], "1"))
#            continue
#        output.writelines("{}\t{}{}\t{}\n".format(chr(10*i+j), b16[i], b16[j], "1"))
        output.writelines("{}{}\t{}\n".format( b16[i], b16[j], "1"))
        output.writelines("##{}{}\t{}\n".format( b16[i], b16[j], "1"))
