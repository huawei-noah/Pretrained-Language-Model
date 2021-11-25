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

Bvocab = open("byteVocab.txt", 'r', encoding = 'utf-8')

for line in Bvocab:
    tokens = line.strip().split('\t')
    print(tokens[0])
    print(tokens[1])
    byteVocab[tokens[0]] = tokens[1]

def getPunc(context):
#    context = context.decode("utf-8") # convert context from str to unicode
    filtrate = re.compile(u'[^\u0020-\u002f\u003A-\u0040\u005B-\u0060\u007B-\u007E\u00A0-\u00BF\u2000-\u206f\u3000-\u303f\uff00-\uffef]') # non-Chinese unicode range
    context = filtrate.sub(r'', context) # remove all non-Chinese characters
#    context = context.encode("utf-8") # convert unicode back to str
    return context

def getCJK(context):
#    context = context.decode("utf-8") # convert context from str to unicode
    filtrate = re.compile(u'[^\u4E00-\u9FA5]') # non-Chinese unicode range
    contextch = filtrate.sub(r'', context) # remove all non-Chinese characters
    
    filtrate = re.compile(u'[^\uac00-\ud7ff]') # non-Chinese unicode range
    contextko = filtrate.sub(r'', context) # remove all non-Chinese characters
#    context = context.encode("utf-8") # convert unicode back to str
    filtrate = re.compile(u'[^\u30a0-\u30ff]') # non-Chinese unicode range
    contextja = filtrate.sub(r'', context) # remove all non-Chinese characters
    return contextch+contextko+contextja

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
#    if i%2==0:
#        question = getChinese(line.strip())
#    else:
#    line = line.decode("utf-8")
    line = line.strip()#.split() #bytes(line.strip(), encoding="utf-8")
    lasttoken = ' '
    for token in line:
        if token == ' ':
            output.write(' ')
            lasttoken = ' '
            continue
     #   if lasttoken != ' ' and not re.match(r'[+-]?\d+$', token) and not re.match(r'[a-z]+',token,re.I):
     #       output.write(" ")
     #       lasttoken = ' '
     #   if lasttoken != ' ' and re.match(r'[+-]?\d+$', token) and re.match(r'[a-z]+',lasttoken,re.I):
     #       output.write(" ")
     #       lasttoken = ' '
     #   if lasttoken != ' ' and re.match(r'[+-]?\d+$', lasttoken) and re.match(r'[a-z]+',token,re.I):
      #      output.write(" ")
       #     lasttoken = ' '
#        print(str(base64.b16encode(token.encode("utf-8"))))
#        output.write(token)
#        output.write(" ")
        if lasttoken != ' ': 
            if len(getCJK(token)) > 0 or len(getPunc(token)) > 0: 
                output.write(" ")
                lasttoken = ' '
        tk = (str(base64.b16encode(token.encode("utf-8")))[2:-1])
#        output.write(" base16:")
#        output.write(tk)
#        output.write(" ")
        num = len(tk)/2
        for i in range(int(num)):
            if lasttoken == ' ' and i == 0: 
                ch = str(byteVocab[str((base16decode(tk[2*i:2*i+2])))])
            else:
                ch = str(byteVocab[str(256+(base16decode(tk[2*i:2*i+2])))])
                #ch = str(base256encode(base16decode(tk[2*i:2*i+2])))
#            if ch not in bytechars:
#                bytechars[ch] = 0
#            bytechars[ch] += 1
            output.write(ch)
#            b256tob16[str(base256encode(base16decode(tk[2*i:2*i+2])))] = tk[2*i:2*i+2]
#            output.write(" ")
#        output.write((base64.b16decode(str(base64.b16encode(token.encode("utf-8")))[2:-1]).decode('utf-8')))
#        print((base64.b16decode(str(base64.b16encode(token.encode("utf-8")))[2:-1])).decode('utf-8'))
       # output.write(str(token))
#        output.write(token)
        #if len(getChinese(token)) > 0: #not re.match(r'[+-]?\d+$', token) and not re.match(r'[a-z]+',token,re.I):
        if len(getCJK(token)) > 0 or len(getPunc(token)) > 0: 
            output.write(" ")
            lasttoken = ' '
        else: lasttoken = token
      #  if len(getChinese(token)) > 0: output.write(' ')
    output.write("\n")
    count += len(line)
#charVocab = open("charVocab.txt", 'w')
#for ch in bytechars:
#    charVocab.writelines("{}\t{}\n".format(ch, bytechars[ch]))
#table = open("b256tob16.txt", "w")
#for i in b256tob16:
#    table.writelines("{} {}\n".format(i, b256tob16[i]))
