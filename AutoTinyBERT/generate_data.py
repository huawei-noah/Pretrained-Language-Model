# coding=utf-8
# 2019.12.2-Changed for data preprocessor
#      Huawei Technologies Co., Ltd. <yinyichun@huawei.com>
# Copyright 2020 Huawei Technologies Co., Ltd.
# Copyright 2018 The Google AI Language Team Authors, The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

import json
import logging
import os
import shelve
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm, trange
from tempfile import TemporaryDirectory

import numpy as np
from random import randrange

from transformer.tokenization import BertTokenizer

# This is used for running on Huawei Cloud.
oncloud = True
try:
    import moxing as mox
except:
    oncloud = False

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentDatabase:
    def __init__(self, reduce_memory=False):
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(self.temp_dir.name)
            self.document_shelf_filepath = self.working_dir / 'shelf.db'
            self.document_shelf = shelve.open('/cache/shelf.db',
                                              flag='n', protocol=-1)
            self.documents = None
        else:
            self.documents = []
            self.document_shelf = None
            self.document_shelf_filepath = None
            self.temp_dir = None
        self.doc_lengths = []
        self.doc_cumsum = None
        self.cumsum_max = None
        self.reduce_memory = reduce_memory

    def add_document(self, document):
        if not document:
            return
        if self.reduce_memory:
            current_idx = len(self.doc_lengths)
            self.document_shelf[str(current_idx)] = document
        else:
            self.documents.append(document)
        self.doc_lengths.append(len(document))

    def _precalculate_doc_weights(self):
        self.doc_cumsum = np.cumsum(self.doc_lengths)
        self.cumsum_max = self.doc_cumsum[-1]

    def sample_doc(self, current_idx, sentence_weighted=True):
        # Uses the current iteration counter to ensure we don't sample the same doc twice
        if sentence_weighted:
            # With sentence weighting, we sample docs proportionally to their sentence length
            if self.doc_cumsum is None or len(self.doc_cumsum) != len(self.doc_lengths):
                self._precalculate_doc_weights()
            rand_start = self.doc_cumsum[current_idx]
            rand_end = rand_start + self.cumsum_max - self.doc_lengths[current_idx]
            sentence_index = randrange(rand_start, rand_end) % self.cumsum_max
            sampled_doc_index = np.searchsorted(self.doc_cumsum, sentence_index, side='right')
        else:
            # If we don't use sentence weighting, then every doc has an equal chance to be chosen
            sampled_doc_index = (current_idx + randrange(1, len(self.doc_lengths))) % len(self.doc_lengths)
        assert sampled_doc_index != current_idx
        if self.reduce_memory:
            return self.document_shelf[str(sampled_doc_index)]
        else:
            return self.documents[sampled_doc_index]

    def __len__(self):
        return len(self.doc_lengths)

    def __getitem__(self, item):
        if self.reduce_memory:
            return self.document_shelf[str(item)]
        else:
            return self.documents[item]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        if self.document_shelf is not None:
            self.document_shelf.close()
        if self.temp_dir is not None:
            self.temp_dir.cleanup()


def main():
    parser = ArgumentParser()
    parser.add_argument('--train_corpus', type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bert_model", type=str, required=True)
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--reduce_memory", action="store_true",
                        help="Reduce memory usage for large datasets by keeping data on disc rather than in memory")

    parser.add_argument("--num_workers", type=int, default=1,
                        help="The number of workers to use to write the files")

    # add 1. for huawei yun.
    parser.add_argument("--data_url", type=str, default="", help="s3 url")
    parser.add_argument("--train_url", type=str, default="", help="s3 url")
    parser.add_argument("--init_method", default='', type=str)

    args = parser.parse_args()

    # add 2. for huawei yun.
    if oncloud:
        os.environ['DLS_LOCAL_CACHE_PATH'] = "/cache"
        local_data_dir = os.environ['DLS_LOCAL_CACHE_PATH']
        assert mox.file.exists(local_data_dir)
        logging.info("local disk: " + local_data_dir)
        logging.info("copy data from s3 to local")
        logging.info(mox.file.list_directory(args.data_url, recursive=True))
        mox.file.copy_parallel(args.data_url, local_data_dir)
        logging.info("copy finish...........")

        args.train_corpus = Path(os.path.join(local_data_dir, args.train_corpus))
        args.bert_model = os.path.join(local_data_dir, args.bert_model)
        
        args.train_url = os.path.join(args.train_url, args.output_dir)
        args.output_dir = Path(os.path.join(local_data_dir, args.output_dir))

    if args.num_workers > 1 and args.reduce_memory:
        raise ValueError("Cannot use multiple workers while reducing memory")

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    doc_num = 0
    with DocumentDatabase(reduce_memory=args.reduce_memory) as docs:
        with args.train_corpus.open() as f:
            doc = []
            for line in tqdm(f, desc="Loading Dataset", unit=" lines"):
                line = line.strip()
                if line == "":
                    docs.add_document(doc)
                    doc = []
                    doc_num += 1
                    if doc_num % 100 == 0:
                        logger.info('loaded {} docs!'.format(doc_num))
                else:
                    tokens = tokenizer.tokenize(line)
                    doc.append(tokens)
            if doc:
                docs.add_document(doc)  # If the last doc didn't end on a newline, make sure it still gets added
        if len(docs) <= 1:
            exit("ERROR: No document breaks were found in the input file! These are necessary to allow the script to "
                 "ensure that random NextSentences are not sampled from the same document. Please add blank lines to "
                 "indicate breaks between documents in your input file. If your dataset does not contain multiple "
                 "documents, blank lines can be inserted at any natural boundary, such as the ends of chapters, "
                 "sections or paragraphs.")

        args.output_dir.mkdir(exist_ok=True)
        file_num = 28

        fouts = []
        for i in range(file_num):
            file_name = os.path.join(str(args.output_dir), 'train_doc_tokens_ngrams_{}.json'.format(i))
            fouts.append(open(file_name, 'w'))

        cnt = 0
        for doc_idx in trange(len(docs), desc="Document"):
            document = docs[doc_idx]
            i = 0
            tokens = []
            while i < len(document):
                segment = document[i]
                if len(tokens) + len(segment) > args.max_seq_len:
                    instance = {"tokens": tokens}

                    file_idx = cnt % file_num
                    fouts[file_idx].write(json.dumps(instance) + '\n')

                    cnt += 1
                    if cnt % 100000 == 0:
                        logger.info('loaded {} examples!'.format(cnt))

                    if cnt <= 10:
                        logger.info('instance: {}'.format(instance))

                    tokens = []
                    tokens += segment
                else:
                    tokens += segment

                i += 1

            if tokens:
                instance = {"tokens": tokens}
                file_idx = cnt % file_num
                fouts[file_idx].write(json.dumps(instance) + '\n')

        for fout in fouts:
            fout.close()

        if oncloud:
            logging.info(mox.file.list_directory(str(args.output_dir), recursive=True))
            mox.file.copy_parallel(str(args.output_dir), args.train_url)


if __name__ == '__main__':
    main()
