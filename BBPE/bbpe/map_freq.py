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

output = open(sys.argv[1], "w", encoding = "utf-8")
vocab = open(sys.argv[2], "w", encoding = "utf-8")

vocabulary = {}

def print_3byte(start, token):
    count = 0
    if len(token) < start + 6: return -1
    tk = token[start:start+6]
    try:
        output.write(base64.b16decode(tk).decode('utf-8'))
        return 1
    except:
        count+=1
    finally:
        count+=1
    return 0

def print_2byte(start, token):
    count = 0
    if len(token) < start + 4: return -1
    tk = token[start:start+4]
    try:
        output.write(base64.b16decode(tk).decode('utf-8'))
        return 1
    except:
        count+=1
    finally:
        count+=1
    return 0

def print_1byte(start, token):
    count = 0
    if len(token) < start + 2: return -1
    tk = token[start:start+2]
    try:
        output.write(base64.b16decode(tk).decode('utf-8'))
        return 1
    except:
        count+=1
    finally:
        count+=1
    return 0

for line in sys.stdin:
#    if i%2==0:
#        question = getChinese(line.strip())
#    else:
#    line = line.decode("utf-8")
    line = line.strip() #bytes(line.strip(), encoding="utf-8")
#    for token in line:
    tokens = line.split("\t")
    start = 0 
    end = 0
    output.write(tokens[0] + " | ")
    token = tokens[0]
#    print(token)
#        print(str(base64.b16encode(token.encode("utf-8"))))
#        output.write(str(base64.b16encode(token.encode("utf-8")))[2:-1])
#        output.write((base64.b16decode(str(base64.b16encode(token.encode("utf-8")))[2:-1]).decode('utf-8')))
    if len(token) > 0:
        if token not in vocabulary:    
            vocab.write(token)
            vocab.write("\n")
            vocabulary[token] = 1
        while True:
  #          print(tk)
            if start >= len(token): break
                #print(base64.b16decode(tk.encode("utf-8")).decode('utf-8'))
                #output.write(base64.b16decode(tk).decode('utf-8'))
            if print_3byte(start, token) > 0: 
                start += 6
                continue
            elif print_2byte(start, token) > 0: 
                start += 4
                continue
            elif print_1byte(start, token) > 0:
                start +=2
            else:
                if token[start:start+2] == '##':
                    output.write('##')
                else:
                    output.write(" 0x"+str(token[start:start+2])+" ")
                start +=2
  #          if start == 0 and "##" + tokens[0] not in vocabulary: 
  #              vocab.write("##")
  #              vocab.write(tokens[0])
  #              vocab.write("\n")
  #              vocabulary["##"+tokens[0]] = 1
                
    output.write(" | "+tokens[1])

    output.write("\n")
