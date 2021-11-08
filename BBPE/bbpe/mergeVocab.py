# -*- coding: UTF-8 -*-
import re
import sys
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

numVocab = {}

for i in range(10):
#    print(str(i).encode("utf-8"))
    token = (str(base64.b16encode(str(i).encode("utf-8")))[2:-1])
    if token not in numVocab: numVocab[token] = 1
print(numVocab)
for line in Vocab:
    tokens = line.strip().split('\t')
    if tokens[0] in byteVocab: continue
    isNum = False
#    print(tokens[0])
#    print(int(len((tokens[0]))/2))
    
    for i in range(int(len((tokens[0]))/2)):
#      print(tokens[0][2*i:2*i+2])
      if i == 0 and tokens[0][0:2] == '##':
#        print('##')
        continue 
      if tokens[0][2*i:2*i+2] not in numVocab: break
      if i == int(len(tokens[0])/2)-1: 
        isNum = True
    if isNum:
      print(tokens[0])
      continue
 #   print(line)
#    if tokens[0] not in vocab:
        #vocab[tokens[0]+'\t'+tokens[1]] = int(tokens[2])
    vocab[tokens[0]] = int(tokens[1])


mergedVocab = sorted(vocab.items(), key=lambda item:item[1], reverse=True)

output = open(sys.argv[3], 'w')
for item in mergedVocab:
    output.writelines("{}\t{}\n".format(item[0], item[1]))
