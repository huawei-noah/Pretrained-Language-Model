# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import hashlib
import copy
from tqdm import tqdm
import argparse
from fairseq.data import Dictionary

def word_starts(tokens):
    '''
    :param tokens: src or tgt sen-tokens
    :return: is start of current token.
    '''
    is_word_start = []
    has_next = False
    for token in tokens:
        is_word_start.append(1 if has_next == False else 0)
        if token.endswith("@@"):
            has_next = True
        else:
            has_next = False
    return is_word_start

def get_align_word(raw_tokens,tgt_sents,word_trans_dict,align_word):
    '''
    :param raw_tokens: source words.
    :tgt_sents: target sentence.
    :return: update align_word of current sent pair.
    '''

    if raw_tokens in word_trans_dict:
        candi_word = word_trans_dict[raw_tokens]
        for tgt_tokens in candi_word:
            if tgt_tokens in tgt_sents:
                align_word.append([sent2idx(raw_tokens.split(" ")),sent2idx(tgt_tokens.split(" "))])
                return align_word
    return align_word

def sent2idx(tokens):
    con = []
    for token in tokens:
        try:
            tok_id = str( Src_dict.index(token))
        except:
            tok_id = "3"
        con.append(tok_id)
    return "-".join(con)

def add_bound_token(sent,bound_token='-'):
    return bound_token + sent + bound_token

def get_replace_ratio(src_file,tgt_file):
    md5_alignSets = {}
    md5_sents = {}

    with open(src_file,"r",encoding="utf-8") as f:
        src_con = f.readlines()
    with open(tgt_file,"r",encoding="utf-8") as f:
        tgt_con = f.readlines()
    assert len(src_con) == len(tgt_con), print("length not match. src={},tgt={}".format(len(src_con),len(tgt_con)))
    pairs_count = 0
    for i in tqdm(range(len(src_con))):
        source = src_con[i].strip("\n")
        target = tgt_con[i].strip("\n")
        raw_source = copy.deepcopy(source)
        raw_target = copy.deepcopy(target)

        source = source.split(" ")
        src_word_start = word_starts(source)
        assert len(source) == len(src_word_start)
        indexs = 0
        align_word = []
        while indexs < len(source):
            word_start = src_word_start[indexs]
            if word_start == 0:
                continue
            start_indexs = indexs
            indexs += 1
            while indexs < len(src_word_start) and src_word_start[indexs] == 0:
                indexs += 1
            be_replaced_word = " ".join(source[start_indexs:indexs])
            align_word = get_align_word(be_replaced_word,target,word_trans_dict,align_word)
        
        source2id = sent2idx(source)
        target2id = sent2idx(raw_target.split(" "))
        align_con_idx = []
        for items in align_word:
            word_src = add_bound_token(items[0])
            word_tgt = add_bound_token(items[1])
            src_ids = add_bound_token(source2id)
            tgt_ids = add_bound_token(target2id)
            if word_src in src_ids and word_tgt in tgt_ids:
                align_con_idx.append(items)

        if len(align_con_idx) > 0:
            pairs_count += len(align_con_idx)
            key_2id = source2id + "| | |" + target2id
            key_2id_md5 = hashlib.md5(key_2id.encode('utf8')).hexdigest()
            md5_alignSets[key_2id_md5]=align_con_idx
            md5_sents[key_2id_md5]=[raw_source,raw_target,key_2id]

    return md5_alignSets, md5_sents,pairs_count

def get_args():
    parser = argparse.ArgumentParser(description="extracting aligned pairs from train/valid/test data")
    parser.add_argument('--vocab-path',type=str, help='vocab path when you training model')
    parser.add_argument('--data-path', type=str, help='training data')
    parser.add_argument('--trans-path', type=str, help='(multi-lingual) bilingual word translation dicts')
    parser.add_argument('--output-path', type=str, help='output path')
    parser.add_argument('--prefix', type=str, help='prefix of file')
    parser.add_argument('--langs', type=str, help='language pairs')
    parser.add_argument('--add-mask', action='store_true', help='adding mask tokens for dictionary')
    parser.add_argument('--merge', action='store_true', help='merge all translation dicts')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    with open(args.trans_path, "r", encoding="utf-8") as f:
        word_trans_dict = json.load(f)

    Src_dict = Dictionary.load(args.vocab_path)
    Tgt_dic = Dictionary.load(args.vocab_path)
    print("Ensure matching your pre-training (need add '<mask>' token):{}, ".format(args.add_mask))
    if args.add_mask:
        Src_dict.add_symbol("<mask>")
        Tgt_dic.add_symbol("<mask>")

    merge_alignsets = {}
    for lang_pair in args.langs.split(","):
        src,tgt = lang_pair.split("-")
        src_file = "{}/{}.{}.spm.{}".format(args.data_path, args.prefix, lang_pair, src)
        tgt_file = "{}/{}.{}.spm.{}".format(args.data_path, args.prefix, lang_pair, tgt)
        md5_alignSets,md5_sents,pairs_count = get_replace_ratio(src_file, tgt_file)
        print("finding {} translation aligned pairs between lang_pair:{}".format(pairs_count,lang_pair))
        with open("{}/{}.md5_align.json".format(args.output_path,lang_pair), "w", encoding="utf-8") as f:
            json.dump(md5_alignSets,f,ensure_ascii=False,indent=4)
        with open("{}/{}.md5_sent.json".format(args.output_path,lang_pair), "w", encoding="utf-8") as f:
            json.dump(md5_sents,f,ensure_ascii=False,indent=4)
        if args.merge:
            merge_alignsets.update(md5_alignSets)
    if args.merge:
        with open("{}/{}.merge.json".format(args.output_path,args.prefix), "w", encoding="utf-8") as f:
            json.dump(merge_alignsets,f,ensure_ascii=False,indent=4)


