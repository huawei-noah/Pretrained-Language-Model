# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import gzip
import html
from functools import lru_cache
from pathlib import Path
from typing import Union, List

import ftfy
import numpy as np
import regex as re

from .utils import is_control, is_whitespace, is_chinese_char, \
    is_punctuation, strip_accents

SOT_TEXT = "<|startoftext|>"
EOT_TEXT = "<|endoftext|>"
CONTEXT_LEN = 77

vocab_path_en = "bpe_simple_vocab_16e6.txt.gz"
vocab_path_zh = "vocab_zh.txt"


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


class BpeTokenizer:
    def __init__(self, bpe_path):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]

        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = merges[1: 49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]

        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend([SOT_TEXT, EOT_TEXT])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {
            SOT_TEXT: SOT_TEXT,
            EOT_TEXT: EOT_TEXT,
        }
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except IndexError:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(
                self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" ")
            )
        return bpe_tokens

    def decode(self, tokens):
        text = "".join([self.decoder[token] for token in tokens])
        text = (
            bytearray([self.byte_decoder[c] for c in text])
            .decode("utf-8", errors="replace")
            .replace("</w>", " ")
        )
        return text


class WordpieceTokenizer:
    def __init__(self, vocab_path):
        with open(vocab_path) as vocab_file:
            vocab = [line.strip() for line in vocab_file]
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.max_input_chars_per_word = 100
        self.tokenize_chinese_chars = True
        self.unk_token = "[UNK]"
        self.never_split = [self.unk_token, SOT_TEXT, EOT_TEXT]

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


# default tokenizer for 'en'
_tokenizer = BpeTokenizer(Path(__file__).with_name(vocab_path_en).as_posix())


def set_tokenizer_lang(lang="en", context_length=77):
    global _tokenizer, SOT_TEXT, EOT_TEXT, CONTEXT_LEN
    CONTEXT_LEN = context_length
    if lang == "en":
        vocab_en = Path(__file__).with_name(vocab_path_en).as_posix()
        _tokenizer = BpeTokenizer(vocab_en)
    elif lang == "zh":
        vocab_zh = Path(__file__).with_name(vocab_path_zh).as_posix()
        SOT_TEXT = "[CLS]"
        EOT_TEXT = "[SEP]"
        _tokenizer = WordpieceTokenizer(vocab_zh)
    else:
        raise RuntimeError("Tokenizer for language \"{}\" is not supported."
                           .format(lang))


@lru_cache()
def get_sot_token():
    return _tokenizer.encoder[SOT_TEXT]


@lru_cache()
def get_eot_token():
    return _tokenizer.encoder[EOT_TEXT]


def tokenize(texts: Union[str, List[str]]) -> np.ndarray:
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

    sot_token = _tokenizer.encoder[SOT_TEXT]
    eot_token = _tokenizer.encoder[EOT_TEXT]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = np.zeros((len(all_tokens), CONTEXT_LEN), dtype=np.int64)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > CONTEXT_LEN:
            tokens = tokens[:CONTEXT_LEN - 1] + [eot_token]

        result[i, : len(tokens)] = np.array(tokens)

    return result
