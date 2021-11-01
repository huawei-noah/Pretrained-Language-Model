# -*- coding: UTF-8 -*-
import re
import sys
import unicodedata
import collections
import base64

output = open("byteVocab.txt", "w")

corpus = open("cn_wiki_sample.txt", "r")

vocab = {}

def getChinese(context):
#    context = context.decode("utf-8") # convert context from str to unicode
    filtrate = re.compile(u'[^\u4E00-\u9FA5]') # non-Chinese unicode range
    context = filtrate.sub(r'', context) # remove all non-Chinese characters
#    context = context.encode("utf-8") # convert unicode back to str
    return context

i = 0

for line in corpus:
    line = line.strip()
#    print(line)
    tokens = line #.split()
    print(tokens)

    for token in tokens: #range(len(tokens)):
       # token = tokens[i]    
        print(token)
        if len(getChinese(token)) > 0 and token not in vocab:
            vocab[token] = i #int(tokens[1])
            i += 1
        if i>= 512: break
    if i >= 512: break

mergedVocab = sorted(vocab.items(), key=lambda item:item[1], reverse=False)

for item in mergedVocab:
    output.writelines("{}\t{}\n".format(item[1], item[0]))
