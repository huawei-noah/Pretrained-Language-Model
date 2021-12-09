# coding=utf-8
# 2021.09.29-Modified slightly for JABER
# 2019.11.25-Changed for finetuning pruned model 
#            Huawei Technologies Co., Ltd. 

# Copyright 2018 The Google AI Language Team Authors.
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
"""The tokneization classs for BBPE."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import unicodedata
import six
# import tensorflow as tf
import base64
import re

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

b16 = {}
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

def print_3byte(start, token):
    count = 0
    if len(token) < start + 6: return '-1'
    tk = token[start:start+6]
    try:
        return (base64.b16decode(tk).decode('utf-8'))
        return 1
    except:
        count+=1
    finally:
        count+=1
    return '-1' 

def print_2byte(start, token):
    count = 0
    if len(token) < start + 4: return '-1'
    tk = token[start:start+4]
    try:
        return (base64.b16decode(tk).decode('utf-8'))
    except:
        count+=1
    finally:
        count+=1
    return '-1' 

def print_1byte(start, token):
    count = 0
    if len(token) < start + 2: return '-1'
    tk = token[start:start+2]
    try:
        return (base64.b16decode(tk).decode('utf-8'))
    except:
        count+=1
    finally:
        count+=1
    return '-1' 

def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
  """Returns text encoded in a way suitable for print or `tf.logging`."""

  # These functions want `str` for both Python2 and Python3, but in one case
  # it's a Unicode string and in the other it's a byte string.
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text
    elif isinstance(text, unicode):
      return text.encode("utf-8")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")

def printable_text_byte(text):
  """Returns text encoded in a way suitable for print or `tf.logging`."""

  # These functions want `str` for both Python2 and Python3, but in one case
  # it's a Unicode string and in the other it's a byte string.
  token = text
  start = 0 
  out = ''
  if len(text) > 2 and text[0:2] == "##": 
      token = text[2:] 
      out = "##" + out
  if six.PY3:
    if len(token) == 6 and print_3byte(start, token) != '-1': 
        out += print_3byte(start, token)
        return out
    if len(token) > 0:
        while True:
  #          print(tk)
            if start >= len(token): break
                #print(base64.b16decode(tk.encode("utf-8")).decode('utf-8'))
                #output.write(base64.b16decode(tk).decode('utf-8'))
            if print_1byte(start, token) != '-1': 
                out += print_1byte(start, token) 
                start += 2 
                continue
            elif print_2byte(start, token) != '-1': 
                out += print_2byte(start, token) 
                start += 4
                continue
            elif print_3byte(start, token) != '-1':
                out += print_3byte(start, token)
                start += 6
            else:
                out += ("0x"+str(token[start:start+2]))
                #out = (" 0x"+str(token)+" ")
                start +=2
        return out
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
  """Loads a vocabulary file into a dictionary."""
  vocab = collections.OrderedDict()
  index = 0
  with open(vocab_file, "r") as reader:
    while True:
      token = convert_to_unicode(reader.readline())
      if not token:
        break
      token = token.strip()
      vocab[token] = index
      index += 1
  return vocab


def convert_by_vocab(vocab, items):
  """Converts a sequence of [tokens|ids] using the vocab."""
  output = []
  for item in items:
    output.append(vocab[item])
  return output


def convert_tokens_to_ids(vocab, tokens):
  return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
  return convert_by_vocab(inv_vocab, ids)


def whitespace_tokenize(text):
  """Runs basic whitespace cleaning and splitting on a piece of text."""
  text = text.strip()
  if not text:
    return []
  tokens = text.split()
  return tokens


class FullTokenizer(object):
  """Runs end-to-end tokenziation."""

  def __init__(self, vocab_file, do_lower_case=False, CJK_tokenize=True):
    self.vocab = load_vocab(vocab_file)
    self.inv_vocab = {v: k for k, v in self.vocab.items()}
    self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case, CJK_tokenize = CJK_tokenize)
    self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

  def tokenize(self, text):
    split_tokens = []
    for token in self.basic_tokenizer.tokenize(text):
      for sub_token in self.wordpiece_tokenizer.tokenize(token):
        split_tokens.append(sub_token)

    return split_tokens

  def convert_tokens_to_ids(self, tokens):
    return convert_by_vocab(self.vocab, tokens)

  def convert_ids_to_tokens(self, ids):
    return convert_by_vocab(self.inv_vocab, ids)


class BasicTokenizer(object):
  """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

  def __init__(self, do_lower_case=False, CJK_tokenize=True):
    """Constructs a BasicTokenizer.
    Args:
      do_lower_case: Whether to lower case the input.
    """
    self.do_lower_case = do_lower_case
    self.CJK_tokenize = CJK_tokenize

  def tokenize(self, text):
    """Tokenizes a piece of text."""
    text = convert_to_unicode(text)
    text = self._clean_text(text)

    # This was added on November 1st, 2018 for the multilingual and Chinese
    # models. This is also applied to the English models now, but it doesn't
    # matter since the English models were not trained on any Chinese data
    # and generally don't have any Chinese data in them (there are Chinese
    # characters in the vocabulary because Wikipedia does have some Chinese
    # words in the English Wikipedia.).
    if self.CJK_tokenize == False: text = self._tokenize_chinese_chars(text)

    orig_tokens = whitespace_tokenize(text)
    split_tokens = []
    split_tokens_utf8 = []
    for token in orig_tokens:
      if self.do_lower_case:
        token = token.lower()
        token = self._run_strip_accents(token)
      split_tokens.extend(self._run_split_on_punc(token))
    for token in split_tokens:
      tk = (str(base64.b16encode(token.encode("utf-8")))[2:-1])
      split_tokens_utf8.append(tk)
    output_tokens = whitespace_tokenize(" ".join(split_tokens_utf8))
    

    return output_tokens

  def _run_strip_accents(self, text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
      cat = unicodedata.category(char)
      if cat == "Mn":
        continue
      output.append(char)
    return "".join(output)

  def _run_split_on_punc(self, text):
    """Splits punctuation on a piece of text."""
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
      char = chars[i]
      if _is_punctuation(char):
        output.append([char])
        start_new_word = True
      else:
        if start_new_word:
          output.append([])
        start_new_word = False
        output[-1].append(char)
      i += 1

    return ["".join(x) for x in output]

  def _tokenize_chinese_chars(self, text):
    """Adds whitespace around any CJK character."""
    output = []

    """   
    if self.CJK_tokenize:
      import MeCab
      tokens = MeCab.tokenize(text)
      for char in tokens:
        cp = ord(char)
        if self._is_chinese_char(cp):
          output.append(" ")
          output.append(char)
          output.append(" ")
        else:
          output.append(char)
      return "".join(output) 
    """
    for char in text:
      cp = ord(char)
      if self._is_chinese_char(cp):
        output.append(" ")
        output.append(char)
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)

  def _is_chinese_char(self, cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
      return True

    return False

  def _clean_text(self, text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
      cp = ord(char)
      if cp == 0 or cp == 0xfffd or _is_control(char):
        continue
      if _is_whitespace(char):
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)


class WordpieceTokenizer(object):
  """Runs WordPiece tokenziation."""

  def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=1000):
    self.vocab = vocab
    self.unk_token = unk_token
    self.max_input_chars_per_word = max_input_chars_per_word

  def tokenize(self, text):
    """Tokenizes a piece of text into its word pieces.
    This uses a greedy longest-match-first algorithm to perform tokenization
    using the given vocabulary.
    For example:
      input = "unaffable"
      output = ["un", "##aff", "##able"]
    Args:
      text: A single token or whitespace separated tokens. This should have
        already been passed through `BasicTokenizer.
    Returns:
      A list of wordpiece tokens.
    """

    text = convert_to_unicode(text)

    output_tokens = []
    for token in whitespace_tokenize(text):
      chars = list(token)
      if len(chars) > self.max_input_chars_per_word:
        output_tokens.append(self.unk_token)
        continue

      is_bad = False
      start = 0
      sub_tokens = []
      while start < len(chars):
        end = len(chars)
        cur_substr = None
        while start < end:
          substr = "".join(chars[start:end])
          if start > 0:
            substr = "##" + substr
          if substr in self.vocab:
            cur_substr = substr
            break
          end -= 1
        if cur_substr is None:
          is_bad = True
          break
        sub_tokens.append(cur_substr)
        start = end

      if is_bad:
        output_tokens.append(self.unk_token)
      else:
        output_tokens.extend(sub_tokens)
    return output_tokens


def _is_whitespace(char):
  """Checks whether `chars` is a whitespace character."""
  # \t, \n, and \r are technically contorl characters but we treat them
  # as whitespace since they are generally considered as such.
  if char == " " or char == "\t" or char == "\n" or char == "\r":
    return True
  cat = unicodedata.category(char)
  if cat == "Zs":
    return True
  return False


def _is_control(char):
  """Checks whether `chars` is a control character."""
  # These are technically control characters but we count them as whitespace
  # characters.
  if char == "\t" or char == "\n" or char == "\r":
    return False
  cat = unicodedata.category(char)
  if cat.startswith("C"):
    return True
  return False


def getPunc(context):
#    context = context.decode("utf-8") # convert context from str to unicode
    filtrate = re.compile(u'[^\u0020-\u002f\u003A-\u0040\u005B-\u0060\u007B-\u007E\u00A0-\u00BF\u2000-\u206f\u3000-\u303f\uff00-\uffef]') # non-Chinese unicode range
    context = filtrate.sub(r'', context) # remove all non-Chinese characters
#    context = context.encode("utf-8") # convert unicode back to str
    return context

def _is_punctuation(char):
  """Checks whether `chars` is a punctuation character."""
 # if len(getPunc(char)) == 0: return False
 # return True
  cp = ord(char)
  # We treat all non-letter/number ASCII as punctuation.
  # Characters such as "^", "$", and "`" are not in the Unicode
  # Punctuation class but we treat them as punctuation anyways, for
  # consistency.
  if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
      (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
    return True
  cat = unicodedata.category(char)
  if cat.startswith("P"):
    return True
  return False