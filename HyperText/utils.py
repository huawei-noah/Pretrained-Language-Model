# -*- coding:utf-8 -*-
#The MIT License (MIT)
#Copyright (c) 2021 Huawei Technologies Co., Ltd.

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import torch
import time
import random
from datetime import timedelta
import torch.utils.data as Data

MAX_VOCAB_SIZE = 5000000
UNK, PAD = '<UNK>', '<PAD>'


def hash_str(gram_str):
    gram_bytes = bytes(gram_str, encoding='utf-8')
    hash_size = 18446744073709551616
    h = 2166136261
    for gram in gram_bytes:
        h = h ^ gram
        h = (h * 1677619) % hash_size
    return h


def addWordNgrams(hash_list, n, bucket):
    ngram_hash_list = []
    len_hash_list = len(hash_list)
    for index, hash_val in enumerate(hash_list):
        bound = min(len_hash_list, index + n)

        for i in range(index + 1, bound):
            hash_val = hash_val * 116049371 + hash_list[i]
            ngram_hash_list.append(hash_val % bucket)

    return ngram_hash_list


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    label_set = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line_splits = line.split("\t")
            if len(line_splits) != 2:
                print(line)
            content, label = line_splits
            label_set.add(label.strip())

            for word in tokenizer(content.strip()):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1

        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[
                     :max_size]

        vocab_list = [[PAD, 111101], [UNK, 111100]] + vocab_list
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}

    base_datapath = os.path.dirname(file_path)
    with open(os.path.join(base_datapath, "vocab.txt"), "w", encoding="utf-8") as f:
        for w, c in vocab_list:
            f.write(str(w) + " " + str(c) + "\n")
    with open(os.path.join(base_datapath, "labels.txt"), "w", encoding="utf-8") as fr:
        labels_list = list(label_set)
        labels_list.sort()
        for l in labels_list:
            fr.write(l + "\n")
    return vocab_dic, list(label_set)


def _pad(data, pad_id, width=-1):
    if (width == -1):
        width = max(len(d) for d in data)
    rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
    return rtn_data


def load_vocab(vocab_path, max_size, min_freq):
    vocab = {}
    with open(vocab_path, 'r', encoding="utf-8") as fhr:
        for line in fhr:
            line = line.strip()
            line = line.split(' ')
            if len(line) != 2:
                continue
            token, count = line
            vocab[token] = int(count)

    sorted_tokens = sorted([item for item in vocab.items() if item[1] >= min_freq], key=lambda x: x[1], reverse=True)
    sorted_tokens = sorted_tokens[:max_size]
    all_tokens = [[PAD, 0], [UNK, 0]] + sorted_tokens
    vocab = {item[0]: i for i, item in enumerate(all_tokens)}
    return vocab


def load_labels(label_path):
    labels = []
    with open(label_path, 'r', encoding="utf-8") as fhr:
        for line in fhr:
            line = line.strip()
            if line not in labels:
                labels.append(line)
    return labels


def build_dataset(config, use_word, min_freq=5):
    print("use min words freq:%d" % (min_freq))
    if use_word:
        tokenizer = lambda x: x.split(' ')  # word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level

    _ = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=min_freq)

    vocab = load_vocab(config.vocab_path, max_size=MAX_VOCAB_SIZE, min_freq=min_freq)
    print(f"Vocab size: {len(vocab)}")
    labels = load_labels(config.labels_path)
    print(f"label size: {len(labels)}")

    train = TextDataset(
        file_path=config.train_path,
        vocab=vocab,
        labels=labels,
        tokenizer=tokenizer,
        wordNgrams=config.wordNgrams,
        buckets=config.bucket,
        device=config.device,
        max_length=config.max_length,
        nraws=80000,
        shuffle=True
    )

    dev = TextDataset(
        file_path=config.dev_path,
        vocab=vocab,
        labels=labels,
        tokenizer=tokenizer,
        wordNgrams=config.wordNgrams,
        buckets=config.bucket,
        device=config.device,
        max_length=config.max_length,
        nraws=80000,
        shuffle=False
    )

    test = TextDataset(
        file_path=config.test_path,
        vocab=vocab,
        labels=labels,
        tokenizer=tokenizer,
        wordNgrams=config.wordNgrams,
        buckets=config.bucket,
        device=config.device,
        max_length=config.max_length,
        nraws=80000,
        shuffle=False
    )

    config.class_list = labels
    config.num_classes = len(labels)

    return vocab, train, dev, test


class TextDataset(Data.Dataset):
    def __init__(self, file_path, vocab, labels, tokenizer, wordNgrams,
                 buckets, device, max_length=32, nraws=80000, shuffle=False):

        file_raws = 0
        with open(file_path, 'r', encoding="utf-8") as f:
            for _ in f:
                file_raws += 1
        self.file_path = file_path
        self.file_raws = file_raws
        if file_raws < 200000:
            self.nraws = file_raws
        else:
            self.nraws = nraws
        self.shuffle = shuffle
        self.vocab = vocab
        self.labels = labels
        self.tokenizer = tokenizer
        self.wordNgrams = wordNgrams
        self.buckets = buckets
        self.max_length = max_length
        self.device = device

    def process_oneline(self, line):
        line = line.strip()
        content, label = line.split('\t')
        if len(content.strip()) == 0:
            content = "0"
        tokens = self.tokenizer(content.strip())
        seq_len = len(tokens)
        if seq_len > self.max_length:
            tokens = tokens[:self.max_length]

        token_hash_list = [hash_str(token) for token in tokens]
        ngram = addWordNgrams(token_hash_list, self.wordNgrams, self.buckets)
        ngram_pad_size = int((self.wordNgrams - 1) * (self.max_length - self.wordNgrams / 2))

        if len(ngram) > ngram_pad_size:
            ngram = ngram[:ngram_pad_size]
        tokens_to_id = [self.vocab.get(token, self.vocab.get(UNK)) for token in tokens]
        y = self.labels.index(label.strip())

        return tokens_to_id, ngram, y

    def initial(self):
        self.finput = open(self.file_path, 'r', encoding="utf-8")
        self.samples = list()

        for _ in range(self.nraws):
            line = self.finput.readline()
            if line:
                preprocess_data = self.process_oneline(line)
                self.samples.append(preprocess_data)
            else:

                break
        self.current_sample_num = len(self.samples)
        self.index = list(range(self.current_sample_num))
        if self.shuffle:
            random.shuffle(self.samples)

    def __len__(self):
        return self.file_raws

    def __getitem__(self, item):
        idx = self.index[0]
        one_sample = self.samples[idx]
        self.index = self.index[1:]
        self.current_sample_num -= 1

        if self.current_sample_num <= 0:

            for _ in range(self.nraws):
                line = self.finput.readline()
                if line:
                    preprocess_data = self.process_oneline(line)
                    self.samples.append(preprocess_data)
                else:
                    break
            self.current_sample_num = len(self.samples)
            self.index = list(range(self.current_sample_num))
            if self.shuffle:
                random.shuffle(self.samples)

        return one_sample


def text_collate_fn(batch_data):
    x = torch.LongTensor(_pad([_[0] for _ in batch_data], pad_id=0))
    y = torch.LongTensor([_[2] for _ in batch_data])
    wordNgrams = torch.LongTensor(_pad([_[1] for _ in batch_data], pad_id=0))

    return (x, wordNgrams), y

def build_dataloader(dataset, batch_size, shuffle=False):
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        collate_fn=text_collate_fn
    )

    return dataloader

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
