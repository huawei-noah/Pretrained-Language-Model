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
    pair = line[0].split(' ')
    output.writelines("{}\n".format(pair[0]+pair[1]))
