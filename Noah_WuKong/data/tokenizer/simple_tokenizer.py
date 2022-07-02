#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 2022.01.03 - Make tokenizer support Chinese language.
#     Huawei Technologies Co., Ltd.
# Copyright (c) 2022, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
from functools import lru_cache
from pathlib import Path
from typing import Union, List

import torch
import unicodedata


def is_control(char):
    """Checks whether `char` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def is_whitespace(char):
    """Checks whether `char` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if (
            (0x4E00 <= cp <= 0x9FFF)
            or (0x3400 <= cp <= 0x4DBF)  #
            or (0x20000 <= cp <= 0x2A6DF)  #
            or (0x2A700 <= cp <= 0x2B73F)  #
            or (0x2B740 <= cp <= 0x2B81F)  #
            or (0x2B820 <= cp <= 0x2CEAF)  #
            or (0xF900 <= cp <= 0xFAFF)
            or (0x2F800 <= cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


def is_punctuation(char):
    """Checks whether `char` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (33 <= cp <= 47) or (58 <= cp <= 64) \
            or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def strip_accents(text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)


class WordpieceTokenizer(object):
    def __init__(self, vocab_path, sot_text, eot_text):
        with open(vocab_path) as vocab_file:
            vocab = [line.strip() for line in vocab_file]
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.max_input_chars_per_word = 100
        self.tokenize_chinese_chars = True
        self.unk_token = "[UNK]"
        self.never_split = [self.unk_token, sot_text, eot_text]

    @staticmethod
    def __whitespace_tokenize(text):
        """Runs basic whitespace cleaning and splitting on a piece of text."""
        text = text.strip()
        if not text:
            return []
        tokens = text.split()
        return tokens

    def __split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        if self.never_split and text in self.never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    @staticmethod
    def __clean_text(text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or is_control(char):
                continue
            if is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    @staticmethod
    def __tokenize_chinese_chars(text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def __wordpiece_tokenize(self, text):
        output_tokens = []
        for token in self.__whitespace_tokenize(text):
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
                    if substr in self.encoder:
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

    def __basic_tokenize(self, text):
        # union() returns a new set by concatenating the two sets.
        text = self.__clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        if self.tokenize_chinese_chars:
            text = self.__tokenize_chinese_chars(text)
        orig_tokens = self.__whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if token not in self.never_split:
                token = token.lower()
                token = strip_accents(token)
            split_tokens.extend(self.__split_on_punc(token))
        output_tokens = self.__whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def text_tokenize(self, text):
        split_tokens = []
        for token in self.__basic_tokenize(text):
            if token in self.never_split:
                split_tokens.append(token)
            else:
                split_tokens += self.__wordpiece_tokenize(token)
        return split_tokens

    def encode(self, text):
        tokens = self.text_tokenize(text)
        return [self.encoder.get(token, self.unk_token) for token in tokens]

    def decode(self, tokens):
        segments = [self.decoder.get(token, self.unk_token) for token in tokens]
        text = ""
        for segment in segments:
            if segment in self.never_split:
                text += segment
            else:
                text += segment.lstrip("##")
        return text


class SimpleTokenizer:
    def __init__(self, context_len=None):
        self.sot_text = "[CLS]"
        self.eot_text = "[SEP]"
        self.context_len = context_len if context_len else 32
        vocab_path = "res/vocab.txt"
        vocab = Path(__file__).parent.joinpath(vocab_path).as_posix()
        self.tokenizer = WordpieceTokenizer(
            vocab, self.sot_text, self.eot_text)

    @lru_cache()
    def get_sot_token(self):
        return self.tokenizer.encoder[self.sot_text]

    @lru_cache()
    def get_eot_token(self):
        return self.tokenizer.encoder[self.eot_text]

    def tokenize(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Returns the tokenized representation of given input string(s)

        Parameters
        ----------
        texts : Union[str, List[str]]
            An input string or a list of input strings to tokenize

        Returns
        -------
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, CONTEXT_LEN]
        """
        if isinstance(texts, str):
            texts = [texts]
        sot_token = self.tokenizer.encoder[self.sot_text]
        eot_token = self.tokenizer.encoder[self.eot_text]
        all_tokens = [[sot_token] + self.tokenizer.encode(text) +
                      [eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), self.context_len, dtype=torch.long)
        for i, tokens in enumerate(all_tokens):
            if len(tokens) > self.context_len:
                tokens = tokens[:self.context_len - 1] + [eot_token]
            result[i, : len(tokens)] = torch.tensor(tokens)
        return result
