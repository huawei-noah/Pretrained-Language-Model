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
protectList = {}

Bvocab = open("byteVocab.txt", 'r', encoding = 'utf-8')

for line in Bvocab:
    tokens = line.strip().split('\t')
    print(tokens[0])
    print(tokens[1])
    byteVocab[tokens[0]] = tokens[1]

Pvocab = open("protectList.txt", 'r', encoding = 'utf-8')

for line in Pvocab:
    tokens = str(line.strip())#.split('\t')
    print(tokens)
#    print(tokens[1])
    protectList[tokens] = 0 #tokens[1]
def getPunc(context):
#    context = context.decode("utf-8") # convert context from str to unicode
    filtrate = re.compile(u'[^\u0020-\u002f\u003A-\u0040\u005B-\u0060\u007B-\u007E\u00A0-\u00BF\u2000-\u206f\u3000-\u303f\uff00-\uffef]') # non-Chinese unicode range
    context = filtrate.sub(r'', context) # remove all non-Chinese characters
#    context = context.encode("utf-8") # convert unicode back to str
    return context

def getChinese(context):
#    context = context.decode("utf-8") # convert context from str to unicode
    filtrate = re.compile(u'[^\u4E00-\u9FA5]') # non-Chinese unicode range
    context = filtrate.sub(r'', context) # remove all non-Chinese characters
#    context = context.encode("utf-8") # convert unicode back to str
    return context

def ifLatin(text):
    length = 0
    context = text
    filtrate = re.compile(u'[^\u0041-\u005A]') 
    context = filtrate.sub(r'', context) 
    length += len(context)

    context = text
    filtrate = re.compile(u'[^\u0061-\u007A]') 
    context = filtrate.sub(r'', context) 
    length += len(context)

    context = text
    filtrate = re.compile(u'[^\u00C0-\u00D6]') 
    context = filtrate.sub(r'', context) 
    length += len(context)

    context = text
    filtrate = re.compile(u'[^\u00D8-\u00F6]') 
    context = filtrate.sub(r'', context) 
    length += len(context)

    context = text
    filtrate = re.compile(u'[^\u00F8-\u00FF]') 
    context = filtrate.sub(r'', context) 
    length += len(context)

    if length == 0: return False
    
    return True

for i in range(10):
    b16[str(i)] = i 

b16['A'] = 10
b16['B'] = 11
b16['C'] = 12
b16['D'] = 13
b16['E'] = 14
b16['F'] = 15

b256tob16 = {}
def base16decode(s):
    result = 0
    for c in s:
        result = result * 16 + b16[c]
    return result

def base256encode(n):
    return chr(n)
    result = ''
    while n > 0:
        n = int(n)
        result = chr(n%256) + result
        n /= 256
    return result

bytechars = {}

for line in sys.stdin:
    line = line.strip().split(' ')#.split() #bytes(line.strip(), encoding="utf-8")
    for tokens in line:
        tokens = str(tokens)
        if tokens in protectList: 
            output.write(tokens) 
            output.write(" ")
            continue
        tk = (str(base64.b16encode(tokens.encode("utf-8")))[2:-1])
        output.write(tk)
        output.write(" ")
    output.write("\n")
