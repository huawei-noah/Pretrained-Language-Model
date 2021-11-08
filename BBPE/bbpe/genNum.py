# -*- coding: UTF-8 -*-
import re
import sys
import unicodedata
import collections
import base64

def base256encode(n):
    return chr(n)
    result = ''
    while n > 0:
        n = int(n)
        result = chr(n%256) + result
        n /= 256
    return result

charvocab = open("charVocab.txt", "r")

vocab = {}
for line in charvocab:
    line = line.strip()
#    print(line)
    tokens = line.split('\t')
#    print("tokens: " + tokens[0] + " " + tokens[1] + "\n")
    tk = tokens[0]#(str(base16encode((b256tob16[tokens[0]]))))
    #vocab[tk+'\t'+tk] = int(tokens[1])
    vocab[tk] = int(tokens[1])

for i in range(1000):
#    print(str(i).encode("utf-8"))
    token = (str(base64.b16encode(str(i).encode("utf-8")))[2:-1])
    if token not in vocab: vocab[token] = 1

mergedVocab = sorted(vocab.items(), key=lambda item:item[1], reverse=True)

output = open('charNumVocab.txt', 'w')
for item in mergedVocab:
    output.writelines("{}\t{}\n".format(item[0], item[1]))
    output.writelines("##{}\t{}\n".format(item[0], item[1]))

for i in range(10):
    token = (str(base64.b16encode(('00'+str(i)).encode("utf-8")))[2:-1])
    output.writelines("##{}\t{}\n".format(token, 1))

for i in range(100):
    if i < 10: continue
    token = (str(base64.b16encode(('0'+str(i)).encode("utf-8")))[2:-1])
    output.writelines("##{}\t{}\n".format(token, 1))
