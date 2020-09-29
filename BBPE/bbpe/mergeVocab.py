# coding=utf-8
# Copyright 2019 Huawei Noah's Ark Lab and Cloud BU.
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
b16 = {}

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
    while n > 0:
        n = int(n)
        result = b16[n%16] + result
        n /= 16
        n = int(n)
    return result

def base256encode(n):
    return chr(n)

for i in range(256):
    b256tob16[str(base256encode(i))] = i
vocab = {}

byteVocab = open(sys.argv[1], 'r')
Vocab = open(sys.argv[2], 'r')

for line in byteVocab:
    line = line.strip()
#    print(line)
    tokens = line.split('\t')
#    print("tokens: " + tokens[0] + " " + tokens[1] + "\n")
    tk = tokens[0]#(str(base16encode((b256tob16[tokens[0]]))))
    #vocab[tk+'\t'+tk] = int(tokens[1])
    vocab[tk] = int(tokens[1])

for line in Vocab:
    tokens = line.strip().split('\t')
    print(line)
#    if tokens[0] not in vocab:
        #vocab[tokens[0]+'\t'+tokens[1]] = int(tokens[2])
    vocab[tokens[0]] = int(tokens[1])


mergedVocab = sorted(vocab.items(), key=lambda item:item[1], reverse=True)

output = open(sys.argv[3], 'w')
for item in mergedVocab:
    output.writelines("{}\t{}\n".format(item[0], item[1]))
