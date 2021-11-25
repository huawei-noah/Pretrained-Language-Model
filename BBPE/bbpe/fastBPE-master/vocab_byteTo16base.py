# -*- coding: UTF-8 -*-
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
#    print(tokens[0])
#    print(tokens[1])
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
#    print(line)
    line = line.split('\t') #bytes(line.strip(), encoding="utf-8")
#    output.writelines("{}\t".format(line[0]))
    print(line)
    #if len(getChinese(line[0])) == 0: output.writelines("##")
    if line[0][0] == '_': line[0] = line[0][1:] 
    if line[0][0] in byteVocab and int(byteVocab[line[0][0]]) > 255: output.writelines("##")
    for i in range(len(line[0])):
        token = line[0][i]
        if token not in byteVocab: continue
#        print(token)
        token16 = token
    #    if len(getChinese(token[0])) > 0:
    #        token16 =str(chr(int(byteVocab[token[0]]))) + token[1:]
        token256 = int(byteVocab[token]) 
        if token256 > 255: token256-=256
        token16 =str(chr(token256)) #+ token[1:]
#        print(token16)
        tk = (str(base16encode((b256tob16[token16]))))
        output.writelines("{}".format(tk))
#        print(tk)
        if i == len(line[0]) - 1: output.writelines("\t{}".format(str(line[1])))
#    print(line[1])
    

for i in range(16):
    for j in range(16):
#        if i < 4: 
#            output.writelines("{}\t{}{}\t{}\n".format("C"+str(10*i+j), b16[i], b16[j], "1"))
#            continue
#        output.writelines("{}\t{}{}\t{}\n".format(chr(10*i+j), b16[i], b16[j], "1"))
        output.writelines("{}{}\t{}\n".format( b16[i], b16[j], "1"))
        output.writelines("##{}{}\t{}\n".format( b16[i], b16[j], "1"))
