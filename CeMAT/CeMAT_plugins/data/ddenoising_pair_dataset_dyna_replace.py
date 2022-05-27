# 2022 - Added code for CeMAT
#        Huawei Technologies Co., Ltd. <lipengfei111@huawei.com>
# Copyright 2022 Huawei Technologies Co., Ltd.
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


import math
import torch
import copy
import logging
import random
from random import choice
import numpy as np
from typing import Dict, List, Tuple
from fairseq.data import FairseqDataset, data_utils
import json

logger = logging.getLogger(__name__)
import hashlib


def collate(
        samples,
        pad_idx,
        eos_idx,
        left_pad_source=True,
        left_pad_target=False,
        input_feeding=True,
        pad_to_length=None,
        pad_to_multiple=1,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):

        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )

    def check_alignment(alignment, src_len, tgt_len):
        if alignment is None or len(alignment) == 0:
            return False
        if (
                alignment[:, 0].max().item() >= src_len - 1
                or alignment[:, 1].max().item() >= tgt_len - 1
        ):
            logger.warning("alignment size mismatch found, skipping alignment!")
            return False
        return True

    def compute_alignment_weights(alignments):
        """
        Given a tensor of shape [:, 2] containing the source-target indices
        corresponding to the alignments, a weight vector containing the
        inverse frequency of each target index is computed.
        For e.g. if alignments = [[5, 7], [2, 3], [1, 3], [4, 2]], then
        a tensor containing [1., 0.5, 0.5, 1] should be returned (since target
        index 3 is repeated twice)
        """
        align_tgt = alignments[:, 1]
        _, align_tgt_i, align_tgt_c = torch.unique(
            align_tgt, return_inverse=True, return_counts=True
        )
        align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
        return 1.0 / align_weights.float()

    id = torch.LongTensor([s["id"] for s in samples])

    src_tokens = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )
    # sort by descending source length
    src_lengths = torch.LongTensor(
        [s["source"].ne(pad_idx).long().sum() for s in samples]
    )
    src_lengths, sort_order = src_lengths.sort(descending=True)

    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    # for plus_encoder_loss:1 ,basic source_output is has?
    src_out_lengths = 0
    plus_encoder_loss = None
    source_output = None
    if samples[0].get("source_output", None) is not None:
        source_output = merge("source_output", left_pad=left_pad_source)
        plus_encoder_loss = True
        # 统计encoder output 对应的非padding长度
        src_out_lengths = torch.LongTensor(
            [s["source_output"].ne(pad_idx).long().sum() for s in samples]
        ).index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    all_tokens = 0
    src_ntokens = 0
    if samples[0].get("target", None) is not None:
        target = merge(
            "target",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
        target = target.index_select(0, sort_order)
        tgt_lengths = []
        tgt_input_lengths = []
        for s in samples:
            tgt_lengths.append(s["target"].ne(pad_idx).long().sum())
            tgt_input_lengths.append(len(s["target"]))
        tgt_lengths = torch.LongTensor(tgt_lengths).index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()
        all_tokens = sum(tgt_input_lengths)

        # for encoder_loss:2 update ntokens for computer loss
        if plus_encoder_loss:
            ntokens += src_out_lengths.sum().item()
            src_ntokens = src_out_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens", left_pad=left_pad_target)
        elif input_feeding:  # add one offset for auto-regressive
            ## !!!version v5:move_eos_to_beginning=False, 虽然这一步不会执行
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                "target",
                left_pad=left_pad_target,
                move_eos_to_beginning=False,
                pad_to_length=pad_to_length["target"]
                if pad_to_length is not None
                else None,
            )
    else:
        ntokens = src_lengths.sum().item()

    batch = {
        "id": id,
        "nsentences": len(samples),
        "src_ntokens": src_ntokens,
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
        },
        "target": target,
        "tgt_ntokens": all_tokens,
    }
    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens.index_select(
            0, sort_order
        )

    # for encoder_loss:3 add encoder output predict
    if source_output is not None:
        batch["source"] = source_output.index_select(
            0, sort_order
        )

    if samples[0].get("alignment", None) is not None:
        bsz, tgt_sz = batch["target"].shape
        src_sz = batch["net_input"]["src_tokens"].shape[1]

        offsets = torch.zeros((len(sort_order), 2), dtype=torch.long)
        offsets[:, 1] += torch.arange(len(sort_order), dtype=torch.long) * tgt_sz
        if left_pad_source:
            offsets[:, 0] += src_sz - src_lengths
        if left_pad_target:
            offsets[:, 1] += tgt_sz - tgt_lengths

        alignments = [
            alignment + offset
            for align_idx, offset, src_len, tgt_len in zip(
                sort_order, offsets, src_lengths, tgt_lengths
            )
            for alignment in [samples[align_idx]["alignment"].view(-1, 2)]
            if check_alignment(alignment, src_len, tgt_len)
        ]

        if len(alignments) > 0:
            alignments = torch.cat(alignments, dim=0)
            align_weights = compute_alignment_weights(alignments)

            batch["alignments"] = alignments
            batch["align_weights"] = align_weights

    if samples[0].get("constraints", None) is not None:
        # Collate the packed constraints across the samples, padding to
        # the length of the longest sample.
        lens = [sample.get("constraints").size(0) for sample in samples]
        max_len = max(lens)
        constraints = torch.zeros((len(samples), max(lens))).long()
        for i, sample in enumerate(samples):
            constraints[i, 0: lens[i]] = samples[i].get("constraints")
        batch["constraints"] = constraints

    return batch





class DDenoisingPairDatasetDynaReplace(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        constraints (Tensor, optional): 2d tensor with a concatenated, zero-
            delimited list of constraints for each sentence.
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
        num_buckets (int, optional): if set to a value greater than 0, then
            batches will be bucketed into the given number of batch shapes.
        src_lang_id (int, optional): source language ID, if set, the collated batch
            will contain a field 'src_lang_id' in 'net_input' which indicates the
            source language of the samples.
        tgt_lang_id (int, optional): target language ID, if set, the collated batch
            will contain a field 'tgt_lang_id' which indicates the target language
             of the samples.
    """

    def __init__(
            self,
            src,
            src_sizes,
            src_dict,
            tgt=None,
            tgt_sizes=None,
            tgt_dict=None,
            left_pad_source=True,
            left_pad_target=False,
            shuffle=True,
            input_feeding=True,
            remove_eos_from_source=False,
            append_eos_to_target=False,
            align_dataset=None,
            constraints=None,
            append_bos=False,
            eos=None,
            num_buckets=0,
            src_lang_id=None,
            tgt_lang_id=None,
            pad_to_multiple=1,
            plus_encoder_loss=False,
            add_langs=None,
            shuffle_lang_pair=None,
            args=None,
            word_trans_dict=None,
            word_align_dict=None,
            policy_ratio_dicts=None,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()

        if tgt is not None:
            assert len(src) == len(
                tgt
            ), "Source and target must contain the same number of examples"
        self.src = src
        self.tgt = tgt
        self.args = args
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.src_mask_index = self.src_dict.index("<mask>")
        self.tgt_mask_index = self.tgt_dict.index("<mask>")
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.align_dataset = align_dataset
        self.src_lang_id = src_lang_id
        self.tgt_lang_id = tgt_lang_id

        self.plus_encoder_loss = plus_encoder_loss
        self.add_langs_raw = add_langs
        self.add_langs_id = [self.tgt_dict.index("[{}]".format(add_langs[0])),
                             self.tgt_dict.index("[{}]".format(add_langs[1]))]
        self.shuffle_lang_pair = shuffle_lang_pair
        self.word_trans_dict = word_trans_dict

        self.mono_word_replace_ratio_alpha = policy_ratio_dicts["mono_word_replace_ratio_alpha"]
        self.mono_word_replace_ratio_beta = policy_ratio_dicts["mono_word_replace_ratio_beta"]
        self.mono_masking_ratio_alpha = policy_ratio_dicts["mono_masking_ratio_tgt_alpha"]
        self.mono_masking_ratio_beta = policy_ratio_dicts["mono_masking_ratio_tgt_beta"]
        self.para_word_replace_ratio = policy_ratio_dicts["para_word_replace_ratio"]
        self.para_tgt_masking_ratio_alpha = policy_ratio_dicts["para_tgt_masking_ratio_alpha"]
        self.para_tgt_masking_ratio_beta = policy_ratio_dicts["para_tgt_masking_ratio_beta"]
        self.para_src_masking_ratio_alpha = policy_ratio_dicts["para_src_masking_ratio_alpha"]
        self.para_src_masking_ratio_beta = policy_ratio_dicts["para_src_masking_ratio_beta"]

        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.sizes = (
            np.vstack((self.src_sizes, self.tgt_sizes)).T
            if self.tgt_sizes is not None
            else self.src_sizes
        )

        if self.align_dataset is not None:
            assert (
                    self.tgt_sizes is not None
            ), "Both source and target needed when alignments are provided"
        self.constraints = constraints
        self.append_bos = append_bos
        self.eos = eos if eos is not None else src_dict.eos()
        if num_buckets > 0:
            from fairseq.data import BucketPadLengthDataset

            self.src = BucketPadLengthDataset(
                self.src,
                sizes=self.src_sizes,
                num_buckets=num_buckets,
                pad_idx=self.src_dict.pad(),
                left_pad=self.left_pad_source,
            )
            self.src_sizes = self.src.sizes
            logger.info("bucketing source lengths: {}".format(list(self.src.buckets)))
            if self.tgt is not None:
                self.tgt = BucketPadLengthDataset(
                    self.tgt,
                    sizes=self.tgt_sizes,
                    num_buckets=num_buckets,
                    pad_idx=self.tgt_dict.pad(),
                    left_pad=self.left_pad_target,
                )
                self.tgt_sizes = self.tgt.sizes
                logger.info(
                    "bucketing target lengths: {}".format(list(self.tgt.buckets))
                )

            # determine bucket sizes using self.num_tokens, which will return
            # the padded lengths (thanks to BucketPadLengthDataset)
            num_tokens = np.vectorize(self.num_tokens, otypes=[np.long])
            self.bucketed_num_tokens = num_tokens(np.arange(len(self.src)))
            self.buckets = [
                (None, num_tokens) for num_tokens in np.unique(self.bucketed_num_tokens)
            ]
        else:
            self.buckets = None
        self.pad_to_multiple = pad_to_multiple

        self.md5_align = word_align_dict
        self.word_tras2id_dict = word_trans_dict

    def get_batch_shapes(self):
        return self.buckets

    def set_epoch(self, epoch, **unused):
        self.epoch = epoch

    def word_starts(self, tokens):
        '''
        :param tokens: src or tgt sen-tokens
        :return: if the sub-word is the begining of whole word
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

    def replace_mono_word(self, src_item):
        '''
        :param source:
        :return:  list
        '''
        src_list = src_item.numpy().tolist()
        core_src = "-".join([str(x) for x in src_list])
        core_tgt = copy.deepcopy(core_src)

        if len(src_list) == 2:
            return src_item, src_item, list(range(len(src_item))), [0] * len(src_item)

        source = " ".join([self.src_dict[vocab_i] for vocab_i in src_list]).split(" ")
        is_word_start = self.word_starts(source)
        self.word_replace_ratio = round(
            np.random.uniform(self.mono_word_replace_ratio_alpha, self.mono_word_replace_ratio_beta),
            2)

        src_item_pad = []
        indexs = 0
        curr_core_len = 0
        while indexs < len(source):
            # special token
            if is_word_start[indexs] == 0 or src_list[indexs] < self.src_dict.nspecial or src_list[indexs] in self.add_langs_id:
                src_item_pad.append(0)
                indexs += 1
                curr_core_len += 1
                continue

            # get the whole word
            whole_word = [str(src_list[indexs])]
            indexs += 1
            while indexs < len(is_word_start) and is_word_start[indexs] == 0:
                whole_word.append(str(src_list[indexs]))
                indexs += 1
                curr_core_len += 1
            item = "-".join(whole_word)

            # 1. if no align
            if item not in self.word_tras2id_dict:
                src_item_pad += ([0] * len(item.split("-")))
                continue
            replace_list = self.word_tras2id_dict[item]
            if item in replace_list:
                replace_list.remove(item)
            # 2.if no useful align
            if len(replace_list) == 0:
                src_item_pad += ([0] * len(item.split("-")))
                continue
            # 3. if rand show no replace
            rand = np.random.random()
            if rand > self.word_replace_ratio:
                src_item_pad += ([0] * len(item.split("-")))
                continue

            lens_befor_src = len("-".join((core_src.split("-"))[:len(src_item_pad)]))
            lens_befor_tgt = len("-".join((core_tgt.split("-"))[:indexs - len(item.split("-"))]))

            # print(indexs,core_src[lens_befor_src:],core_tgt[lens_befor_tgt:])
            tgt_replace_word = self.add_bound_token(
                "-".join([str(self.tgt_mask_index)] * len(item.split("-"))))
            item = self.add_bound_token(item)
            replace_word = self.add_bound_token(random.choice(replace_list))
            core_src = core_src[:lens_befor_src] + core_src[lens_befor_src:].replace(item, replace_word, 1)
            core_tgt = core_tgt[:lens_befor_tgt] + core_tgt[lens_befor_tgt:].replace(item, tgt_replace_word, 1)
            src_item_pad += ([1] * len(replace_word[1:-1].split("-")))
            # print("align",item,replace_word,tgt_replace_word)
            # print("align:"," ".join([self.src_dict[int(vocab_i)] for vocab_i in item[1:-1].split("-") ]),"\t",
            #      " ".join([self.tgt_dict[int(vocab_i)] for vocab_i in replace_word[1:-1].split("-")]),"\t",
            #      " ".join([self.tgt_dict[int(vocab_i)] for vocab_i in tgt_replace_word[1:-1].split("-")]))
            # print("align",core_src,"| | |",core_tgt)
        # print(core_src,"| | |",core_tgt)
        src_itemn = list(map(int, core_src.split("-")))
        tgt_itemn = list(map(int, core_tgt.split("-")))

        # assert.
        if len(src_itemn) != len(src_item_pad):
            print(len(src_itemn), len(src_item_pad))
            print(src_itemn, src_item_pad)
            exit(0)

        idx = 1
        n_idx = 1
        tgt_align_src = [0]
        while (idx < len(tgt_itemn) and n_idx < len(src_itemn)):
            if tgt_itemn[idx] == self.tgt_mask_index:
                tgt_align_src.append(-1)
                idx += 1
                continue
            if src_item_pad[n_idx] == True:
                n_idx += 1
                continue
            if tgt_itemn[idx] == src_itemn[n_idx]:
                tgt_align_src.append(n_idx)
                n_idx += 1
            idx += 1
        assert len(tgt_align_src) == len(tgt_itemn), print(len(tgt_align_src), len(tgt_itemn))
        return src_itemn, tgt_itemn, tgt_align_src, src_item_pad


    def add_bound_token(self, sent, bound_token='-'):
        return bound_token + sent + bound_token

    def replace_para_word(self, src_item, tgt_item):
        '''
        :param source:
        :return:  list
        '''

        src_list = src_item.numpy().tolist()
        tgt_list = tgt_item.numpy().tolist()
        src_lang_id = src_list[0]
        tgt_lang_id = tgt_list[0]
        core_src = "-".join([str(x) for x in src_list][1:-1])
        core_tgt = "-".join([str(x) for x in tgt_list][1:-1])

        # get aligned pairs if has.
        shuf_pair = False
        core_md5 = hashlib.md5((core_src + "| | |" + core_tgt).encode('utf8')).hexdigest()
        if core_md5 not in self.md5_align:
            shuf_pair = True
            core_md5 = hashlib.md5((core_tgt + "| | |" + core_src).encode('utf8')).hexdigest()
            if core_md5 not in self.md5_align:
                return src_item, tgt_item, None
        # if shuf_pair == True:
        #     logger.info("need shuf src,tgt")

        core_src = self.add_bound_token(core_src)
        core_tgt = self.add_bound_token(core_tgt)
        align_info = copy.deepcopy(self.md5_align[core_md5])
        max_replace_len = len(core_src) * self.para_word_replace_ratio
        random.shuffle(align_info)
        cur_replace_len = 0
        idx = 0
        while cur_replace_len <= max_replace_len and idx < len(align_info):
            # align_item: translation pairs.
            align_item = align_info[idx]
            if shuf_pair:
                align_item[0], align_item[1] = align_item[1], align_item[0]

            # get candi pairs.
            try:
                replace_list = self.word_tras2id_dict[align_item[0]]
            except:
                replace_list = self.word_tras2id_dict[align_item[1]]
            if align_item[1] in replace_list:
                replace_list.remove(align_item[1])
            # if align_item[0] in replace_list:
            #    replace_list.remove(align_item[0])
            if len(replace_list) == 0:
                idx += 1
                continue

            # 1.bound new replace tokens.
            src_replace_word = self.add_bound_token(choice(replace_list))
            tgt_replace_word = self.add_bound_token(
                "-".join([str(self.tgt_mask_index)] * len(align_item[1].split("-"))))

            # 2.bound translation pair.
            align_item[0] = self.add_bound_token(align_item[0])
            align_item[1] = self.add_bound_token(align_item[1])

            # 3. replacing
            core_src = core_src.replace(align_item[0], src_replace_word)
            core_tgt = core_tgt.replace(align_item[1], tgt_replace_word)

            cur_replace_len += len(align_item[0].split("-"))
            idx += 1

        src_itemn = [src_lang_id] + list(map(int, core_src[1:-1].split("-"))) + [self.src_dict.eos()]
        tgt_itemn = [tgt_lang_id] + list(map(int, core_tgt[1:-1].split("-"))) + [self.tgt_dict.eos()]
        if max(src_itemn) > 64904:
            print(core_src, core_md5)

        # 4. get src_replace_flag.
        src_item_idx = 0
        src_itemn_idx = 0
        src_replace_flag = []
        while src_itemn_idx < len(src_itemn):
            if src_item[src_item_idx] == src_itemn[src_itemn_idx]:
                src_replace_flag.append(0)
                src_item_idx += 1
            else:
                src_replace_flag.append(1)
            src_itemn_idx += 1
        assert len(src_replace_flag) == len(src_itemn)
        assert len(tgt_item) == len(tgt_itemn), print("lens of raw:{} and new:{}".format(len(tgt_item),len(tgt_itemn)))
        return torch.tensor(src_itemn), torch.tensor(tgt_itemn), src_replace_flag

    def _mask_block_para(
            self,
            sentence: np.ndarray,
            mask_idx: int,
            pad_idx: int,
            dictionary_token_range: Tuple,
            sub_mask_ratio: list,
            masking_ratio: int,
            raw_tgt_item: np.ndarray,
    ):
        """
        Mask tokens for Masked Language Model training
        Samples mask_ratio tokens that will be predicted by LM.

        Note:This function may not be efficient enough since we had multiple
        conversions between np and torch, we can replace them with torch
        operators later.

        Args:
            sentence: 1d tensor to be masked
            mask_idx: index to use for masking the sentence
            pad_idx: index to use for masking the target for tokens we aren't
                predicting
            dictionary_token_range: range of indices in dictionary which can
                be used for random word replacement
                (e.g. without special characters)
        Return:
            masked_sent: masked sentence
            target: target with words which we are not predicting replaced
                by pad_idx
        """

        self.masking_ratio = masking_ratio
        self.masking_prob = 0.8
        self.random_token_prob = 0.1
        masked_sent = copy.deepcopy(sentence)
        if raw_tgt_item != None:
            target = copy.deepcopy(raw_tgt_item)
        else:
            target = copy.deepcopy(sentence)

        sent_length = len(sentence)
        mask_num = math.ceil(sent_length * self.masking_ratio)

        try:
            mask = np.random.choice(sent_length, mask_num, replace=False, p=sub_mask_ratio)
        except:
            mask = []

        for i in range(sent_length):
            if i in mask:
                rand = np.random.random()
                # replace with mask if probability is less than masking_prob
                # (Eg: 0.8)
                if rand < self.masking_prob:
                    masked_sent[i] = mask_idx

                # replace with random token if probability is less than
                # masking_prob + random_token_prob (Eg: 0.9)
                elif rand < (self.masking_prob + self.random_token_prob):
                    # sample random token from dictionary
                    masked_sent[i] = np.random.randint(
                        dictionary_token_range[0], dictionary_token_range[1]
                    )
            else:
                if masked_sent[i] != mask_idx:
                    target[i] = pad_idx

        return masked_sent, target

    def _mask_block_mono(
            self,
            src_sentence: np.ndarray,
            tgt_sentence: np.ndarray,
            mask_idx: int,
            pad_idx: int,
            dictionary_token_range: Tuple,
            sub_mask_ratio: list,
            masking_ratio: int,
            can_mask_num: int,
            tgt_item: np.ndarray,
            tgt_align_src: list,
            src_item_pad: list,
    ):
        """
        Mask tokens for Masked Language Model training
        Samples mask_ratio tokens that will be predicted by LM.

        Note:This function may not be efficient enough since we had multiple
        conversions between np and torch, we can replace them with torch
        operators later.

        Args:
            sentence: 1d tensor to be masked
            mask_idx: index to use for masking the sentence
            pad_idx: index to use for masking the target for tokens we aren't
                predicting
            dictionary_token_range: range of indices in dictionary which can
                be used for random word replacement
                (e.g. without special characters)
        Return:
            masked_sent: masked sentence
            target: target with words which we are not predicting replaced
                by pad_idx
        """

        # ensure the masking consistent between src and tgt.
        self.masking_ratio = masking_ratio
        self.masking_prob = 0.8
        self.random_token_prob = 0.1

        source = np.copy(src_sentence)
        masked_src = np.copy(src_sentence)
        masked_tgt = np.copy(tgt_sentence)
        # raw_input target
        target = np.copy(tgt_item)

        sent_length = len(tgt_sentence)
        mask_num = min(math.ceil(can_mask_num * self.masking_ratio), sent_length - sub_mask_ratio.count(0))
        if mask_num == 0:
            mask = []
        else:
            mask = np.random.choice(sent_length, mask_num, replace=False, p=sub_mask_ratio)

        for i in range(sent_length):
            src_index = tgt_align_src[i]
            if i in mask:
                rand = np.random.random()
                # replace with mask if probability is less than masking_prob
                # (Eg: 0.8)
                if rand < self.masking_prob:
                    masked_tgt[i] = mask_idx
                    # 将source 的mask比例调低１
                    if rand > 0.1:
                        masked_src[src_index] = mask_idx
                    else:
                        source[src_index] = pad_idx

                # replace with random token if probability is less than
                # masking_prob + random_token_prob (Eg: 0.9)
                elif rand < (self.masking_prob + self.random_token_prob):
                    # sample random token from dictionary
                    masked_tgt[i] = np.random.randint(
                        dictionary_token_range[0], dictionary_token_range[1]
                    )
                    masked_src[src_index] = np.random.randint(
                        dictionary_token_range[0], dictionary_token_range[1]
                    )
            else:
                if masked_tgt[i] != mask_idx:
                    target[i] = pad_idx
                source[src_index] = pad_idx

        source = [self.tgt_dict.pad_index if src_item_pad[i] == 1 else source[i] for i in
                  range(len(source))]

        return masked_tgt, target, masked_src, source

    def masking_para(self, sen_tokens, token_range, mask_index, pad_index, sub_mask_ratio, masking_ratio,
                        raw_tgt_item=None):

        # mask according to specified probabilities.
        masked_sen, ignored_sen = self._mask_block_para(
            sen_tokens,
            mask_index,
            pad_index,
            token_range,
            sub_mask_ratio,
            masking_ratio,
            raw_tgt_item,
        )
        return torch.tensor(masked_sen), torch.tensor(ignored_sen)

    def masking_mono(self, sen_tokens, tgt_tokens, token_range, mask_index, pad_index, sub_mask_ratio, masking_ratio,
                    can_mask_num,
                    tgt_item, tgt_align_src, src_item_pad):

        # mask according to specified probabilities.
        masked_tgt, target, masked_src, source = self._mask_block_mono(
            sen_tokens,
            tgt_tokens,
            mask_index,
            pad_index,
            token_range,
            sub_mask_ratio,
            masking_ratio,
            can_mask_num,
            tgt_item,
            tgt_align_src,
            src_item_pad,
        )

        return torch.tensor(masked_tgt), torch.tensor(target), torch.tensor(masked_src), torch.tensor(source)

    def process_for_mono(self, src_item):
        '''
        :param src_item: replace,second_mask
        :return:
        '''
        tgt_item_raw = copy.deepcopy(src_item)
        # if replace
        if self.args.no_replace_first:
            src_item_pad = [0] * len(src_item)
            tgt_align_src = list(range(len(src_item)))
            tgt_item = copy.deepcopy(src_item)
        elif self.args.no_CSM:
            tgt_item = copy.deepcopy(src_item)
            src_item, _, tgt_align_src, src_item_pad = self.replace_mono_word(src_item)
        else:
            src_item, tgt_item, tgt_align_src, src_item_pad = self.replace_mono_word(src_item)

        token_range = (self.tgt_dict.nspecial, len(self.tgt_dict))
        can_mask_num, sub_mask_ratio = self.get_mask_ratio(tgt_item, self.tgt_dict)

        if can_mask_num == 0:
            src_mask = [self.tgt_dict.pad_index for _ in range(len(src_item))]
            tgt_mask = [tgt_item_raw[i] if tgt_item[i] == self.tgt_mask_index else self.tgt_dict.pad_index for i in
                        range(len(tgt_item))]
            return torch.tensor(tgt_item), torch.tensor(tgt_mask), torch.tensor(src_item), torch.tensor(src_mask)
        masking_ratio = round(np.random.uniform(self.mono_masking_ratio_alpha, self.mono_masking_ratio_beta), 2)
        return self.masking_mono(src_item, tgt_item, token_range, self.tgt_mask_index,
                                self.tgt_dict.pad_index, sub_mask_ratio, masking_ratio, can_mask_num, tgt_item_raw,
                                tgt_align_src,
                                src_item_pad)

    def selfbound(self, a, maxk):
        return float(str(a).split('.')[0] + '.' + str(a).split('.')[1][:maxk])

    def get_mask_ratio(self, tokens, dicts, src_replace_flag=None):
        '''
            get tokens's mask ratio and total mask numbers.
        '''
        no_mask = []
        for vocabi in tokens:
            if vocabi == self.src_mask_index:
                no_mask.append(1)
            elif vocabi < dicts.nspecial:
                no_mask.append(1)
            elif vocabi in self.add_langs_id:
                no_mask.append(1)
            else:
                no_mask.append(0)

        if src_replace_flag != None:
            for i in range(len(no_mask)):
                no_mask[i] = no_mask[i] or src_replace_flag[i]

        can_mask_num = len(tokens) - sum(no_mask)
        if can_mask_num == 0:
            return can_mask_num, []

        item_p = self.selfbound(1/can_mask_num, 3)
        add_extr = round(1.0 - item_p * can_mask_num, 5)
        max_item_p = item_p + add_extr

        sub_mask_ratio = []
        add_max = True
        for item in no_mask:
            if item == 1:
                sub_mask_ratio.append(0)
            elif add_max:
                sub_mask_ratio.append(max_item_p)
                add_max = False
            else:
                sub_mask_ratio.append(item_p)
        return can_mask_num, sub_mask_ratio

    def process_for_para(self, src_item, tgt_item):
        is_swap = random.random()
        self.add_langs = copy.deepcopy(self.add_langs_raw)

        if is_swap > 0.5 and self.shuffle_lang_pair:
            src_item, tgt_item = tgt_item, src_item
            self.add_langs = (self.add_langs[1], self.add_langs[0])
            eos = self.src_dict.eos()
        else:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()

        if self.append_eos_to_target:
            if len(tgt_item) > 0 and tgt_item[-1] != eos:
                tgt_item = torch.cat([tgt_item, torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if len(tgt_item) > 0 and tgt_item[0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), tgt_item])
            bos = self.src_dict.bos()
            if src_item[0] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), src_item])

        raw_tgt_item = copy.deepcopy(tgt_item)

        # replace word in src and mask in target, for para corpus
        # src_rep_flag : flag that src was replaced(0:no,1:yes)
        if self.args.no_replace_first:
            src_rep_flag = None
        elif self.args.no_CSM:      # only code-switching
            src_item, _, src_rep_flag = self.replace_para_word(src_item, tgt_item)
        else:
            src_item, tgt_item, src_rep_flag = self.replace_para_word(src_item, tgt_item)

        if self.plus_encoder_loss or (self.args.ENC_DM):
            token_range = (self.src_dict.nspecial, len(self.src_dict))
            can_mask_num, sub_mask_ratio = self.get_mask_ratio(src_item, self.src_dict, src_rep_flag)
            if can_mask_num > 0:
                masking_ratio = round(
                    np.random.uniform(self.para_src_masking_ratio_alpha, self.para_src_masking_ratio_beta), 2)
                mask_src, ignored_src = self.masking_para(src_item, token_range, self.src_mask_index,
                                                             self.src_dict.pad_index, sub_mask_ratio, masking_ratio)
            else:
                src_mask = [self.src_dict.pad_index for _ in range(len(src_item))]
                mask_src, ignored_src = torch.tensor(src_item).clone().detach(), torch.tensor(src_mask)
        else:
            src_mask = [self.src_dict.pad_index] * len(src_item)
            mask_src, ignored_src = torch.tensor(src_item), torch.tensor(src_mask)

        token_range = (self.tgt_dict.nspecial, len(self.tgt_dict))
        can_mask_num, sub_mask_ratio = self.get_mask_ratio(tgt_item, self.tgt_dict)
        if can_mask_num > 0:
            masking_ratio = round(np.random.uniform(self.para_tgt_masking_ratio_alpha, self.para_tgt_masking_ratio_beta), 2)
            mask_tgt, ignored_tgt = self.masking_para(tgt_item, token_range, self.tgt_mask_index,
                                                         self.tgt_dict.pad_index, sub_mask_ratio, masking_ratio,
                                                         raw_tgt_item)
        else:
            tgt_mask = [raw_tgt_item[i] if tgt_item[i] == self.tgt_mask_index else self.tgt_dict.pad_index for i in
                        range(len(tgt_item))]
            mask_tgt, ignored_tgt = torch.tensor(tgt_item).clone().detach(), torch.tensor(tgt_mask)
        return mask_tgt, ignored_tgt, mask_src, ignored_src

    def __getitem__(self, index):
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        try:
            tgt_item = self.tgt[index] if self.tgt is not None else None
            src_item = self.src[index]
        except:
            print("all length.{}, and index is {}".format(len(self.tgt), index))
            src_item = self.src[index]
        if self.add_langs_raw[0] != self.add_langs_raw[1]:
            mask_tgt, ignored_tgt, mask_src, ignored_src = self.process_for_para(src_item, tgt_item)
        else:
            mask_tgt, ignored_tgt, mask_src, ignored_src = self.process_for_mono(src_item)

        if self.plus_encoder_loss:
            example = {
                "id": index,
                "source": mask_src,
                "source_output": ignored_src,
                "target": ignored_tgt,
                "prev_output_tokens": mask_tgt,
            }

            # print("src_raw:{}\nsrc_input:{}\nsrc_output:{}\ntgt_raw:{}\ntgt_input:{}\ntgt_output:{}\n\n".format(
            #      " ".join([self.src_dict[vocab_i] for vocab_i in src_item]),
            #      " ".join([self.src_dict[vocab_i] for vocab_i in mask_src]),
            #      " ".join([self.src_dict[vocab_i] for vocab_i in ignored_src]),
            #      " ".join([self.tgt_dict[vocab_i] for vocab_i in tgt_item]),
            #      " ".join([self.tgt_dict[vocab_i] for vocab_i in mask_tgt]),
            #      " ".join([self.tgt_dict[vocab_i] for vocab_i in ignored_tgt])))
        else:
            example = {
                "id": index,
                "source": src_item,
                "target": ignored_tgt,
                "prev_output_tokens": mask_tgt,
            }
        if self.align_dataset is not None:
            example["alignment"] = self.align_dataset[index]
        if self.constraints is not None:
            example["constraints"] = self.constraints[index]
        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'source': source_pad_to_length, 'target': target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language
                   IDs of each sample in the batch
        """
        res = collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
        )
        if self.src_lang_id is not None or self.tgt_lang_id is not None:
            src_tokens = res["net_input"]["src_tokens"]
            bsz = src_tokens.size(0)
            if self.src_lang_id is not None:
                res["net_input"]["src_lang_id"] = (
                    torch.LongTensor([[self.src_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
            if self.tgt_lang_id is not None:
                res["tgt_lang_id"] = (
                    torch.LongTensor([[self.tgt_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
        return res

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        if self.buckets is None:
            # sort by target length, then source length
            if self.tgt_sizes is not None:
                indices = indices[np.argsort(self.tgt_sizes[indices], kind="mergesort")]
            return indices[np.argsort(self.src_sizes[indices], kind="mergesort")]
        else:
            # sort by bucketed_num_tokens, which is:
            #   max(padded_src_len, padded_tgt_len)
            return indices[
                np.argsort(self.bucketed_num_tokens[indices], kind="mergesort")
            ]

    @property
    def supports_prefetch(self):
        return getattr(self.src, "supports_prefetch", False) and (
                getattr(self.tgt, "supports_prefetch", False) or self.tgt is None
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)

    def filter_indices_by_size(self, indices, max_sizes):
        """Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        return data_utils.filter_paired_dataset_indices_by_size(
            self.src_sizes,
            self.tgt_sizes,
            indices,
            max_sizes,
        )

