# 2022 - Added code for CeMAT
#        Huawei Technologies Co., Ltd. <lipengfei111@huawei.com>
# Copyright 2022 Huawei Technologies Co., Ltd.
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import itertools
import json
import logging
import os
import copy
from argparse import Namespace

import numpy as np
from fairseq import metrics, options, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    ResamplingDataset,
    encoders,
    indexed_dataset,
)
from ..data import ConcatPairDataset, DDenoisingPairDatasetDynaReplace
from fairseq.tasks import register_task,LegacyFairseqTask


EVAL_BLEU_ORDER = 4

logger = logging.getLogger(__name__)

def load_langpair_dataset(
        data_path,
        split,
        src,
        src_dict,
        tgt,
        tgt_dict,
        combine,
        dataset_impl,
        upsample_primary,
        left_pad_source,
        left_pad_target,
        max_source_positions,
        max_target_positions,
        prepend_bos=False,
        load_alignments=False,
        truncate_source=False,
        append_source_id=False,
        num_buckets=0,
        shuffle=True,
        pad_to_multiple=1,
        plus_encoder_loss=False,
        add_langs=None,
        shuffle_lang_pair=None,
        args=None,
        word_trans_dict=None,
        word_align_dict=None,
        policy_ratio_dicts=None,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    def split_exists_valid(split, lang, data_path):
        filename = os.path.join(data_path, "{}.{}".format(split, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")
        if not "-" in split_k:
            # infer langcode
            if split_exists(split_k, src, tgt, src, data_path):
                prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
            elif split_exists(split_k, tgt, src, src, data_path):
                prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
            else:
                if k > 0:
                    break
                else:
                    raise FileNotFoundError(
                        "Dataset not found: {} ({}) {} {}".format(split, data_path, src, tgt)
                    )
        else:
            # for multi-valid
            if split_exists_valid( split_k, src, data_path):
                prefix = os.path.join(data_path, split_k+".")
            else:
                if k > 0:
                    break
                else:
                    raise FileNotFoundError(
                        "Dataset not found: {} ({}) ".format(split, data_path)
                    )
        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        # for monolingual instances.
        if src == tgt:
            tgt_dataset = copy.deepcopy(src_dataset)
        else:
            tgt_dataset = data_utils.load_indexed_dataset(
                prefix + tgt, tgt_dict, dataset_impl
            )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    eos = None
    # add src and tag lang id on the biganing of sens.
    if add_langs:
        src_dataset = PrependTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )


    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return DDenoisingPairDatasetDynaReplace(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
        plus_encoder_loss=plus_encoder_loss,
        add_langs=add_langs,
        shuffle_lang_pair=shuffle_lang_pair,
        args=args ,
        word_trans_dict=word_trans_dict ,
        word_align_dict=word_align_dict,
        policy_ratio_dicts= policy_ratio_dicts,
    )


@register_task("cemat_pretraining")
class CematPretraining(LegacyFairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner; \
                            however, valid and test data are always in the first directory to \
                            avoid the need for repeating them in all directories')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--load-alignments', action='store_true',
                            help='load the binarized alignments')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--truncate-source', action='store_true', default=False,
                            help='truncate source to max-source-positions')
        parser.add_argument('--num-batch-buckets', default=0, type=int, metavar='N',
                            help='if >0, then bucket source and target lengths into N '
                                 'buckets and pad accordingly; this is useful on TPUs '
                                 'to minimize the number of compilations')

        # options for reporting BLEU during validation
        parser.add_argument('--eval-bleu', action='store_true',
                            help='evaluation with BLEU scores')
        # options for reporting BLEU during validation
        parser.add_argument('--eval-bleu-detok', type=str, default="space",
                            help='detokenize before computing BLEU (e.g., "moses"); '
                                 'required if using --eval-bleu; use "space" to '
                                 'disable detokenization; see fairseq.data.encoders '
                                 'for other options')
        parser.add_argument('--eval-bleu-detok-args', type=str, metavar='JSON',
                            help='args for building the tokenizer, if needed')
        parser.add_argument('--eval-tokenized-bleu', action='store_true', default=False,
                            help='compute tokenized BLEU instead of sacrebleu')
        parser.add_argument('--eval-bleu-remove-bpe', nargs='?', const='@@ ', default=None,
                            help='remove BPE before computing BLEU')
        parser.add_argument('--eval-bleu-args', type=str, metavar='JSON',
                            help='generation args for BLUE scoring, '
                                 'e.g., \'{"beam": 4, "lenpen": 0.6}\'')
        parser.add_argument('--eval-bleu-print-samples', action='store_true',
                            help='print sample generations during validation')
        # fmt: on

        parser.add_argument(
            "--tokens-per-sample",
            default=512,
            type=int,
            help="max number of total tokens over all segments"
                 " per sample for dataset",
        )
        # build block model
        parser.add_argument(
            "--sample-break-mode",
            default="complete_doc",
            type=str,
            help="mode for breaking sentence",
        )
        # mask prob.
        parser.add_argument(
            "--mask",
            default=0.0,
            type=float,
            help="fraction of words/subwords that will be masked",
        )
        parser.add_argument(
            "--mask-random",
            default=0.0,
            type=float,
            help="instead of using [MASK], use random token this often",
        )
        parser.add_argument(
            "--insert",
            default=0.0,
            type=float,
            help="insert this percentage of additional random tokens",
        )

        parser.add_argument(
            "--permute",
            default=0.0,
            type=float,
            help="take this proportion of subwords and permute them",
        )
        parser.add_argument(
            "--rotate",
            default=0.5,
            type=float,
            help="rotate this proportion of inputs",
        )
        parser.add_argument(
            "--poisson-lambda",
            default=3.0,
            type=float,
            help="randomly shuffle sentences for this proportion of inputs",
        )
        parser.add_argument(
            "--permute-sentences",
            default=0.0,
            type=float,
            help="shuffle this proportion of sentences in all inputs",
        )
        # mask尺度
        parser.add_argument(
            "--mask-length",
            default="subword",
            type=str,
            choices=["subword", "word", "span-poisson"],
            help="mask length to choose",
        )
        parser.add_argument(
            "--replace-length",
            default=-1,
            type=int,
            help="when masking N tokens, replace with 0, 1, or N tokens (use -1 for N)",
        )
        parser.add_argument(
            "--max-source-positions",
            default=1024,
            type=int,
            metavar="N",
            help="max number of tokens in the source sequence",
        )
        parser.add_argument(
            "--max-target-positions",
            default=1024,
            type=int,
            metavar="N",
            help="max number of tokens in the target sequence",
        )
        parser.add_argument(
            "--multilang-sampling-alpha",
            type=float,
            default=1.0,
            help="smoothing alpha for sample ratios across multiple datasets",
        )
        parser.add_argument("--add-lang-token", default=False, action="store_true")
        parser.add_argument(
            "--langs", type=str, help="language ids we are considering", default=None
        )
        parser.add_argument(
            "--no-whole-word-mask-langs",
            type=str,
            default="",
            metavar="N",
            help="languages without spacing between words dont support whole word masking",
        )
        parser.add_argument(
            "--s3-dir",
            type=str,
            default="",
            metavar="N",
            help="path to s3 checkpoints",
        )
        # add extra param
        parser.add_argument('--share-dict', action='store_true',
                            help='share encoder,decoder vocab')
        parser.add_argument('--shuffle-lang-pair', action='store_true',
                            help='random shuffle the lang within lang-pair')
        # parser.add_argument('--add-mono', action='store_true',
        #                     help='adding mono data')
        parser.add_argument('--trans-dict', type=str, default="",
                    help='replace word translations dict.')
        parser.add_argument("--is-exo-infer", default=False, action="store_true")
        parser.add_argument("--src-exo-lang", default=False, action="store_true")
        parser.add_argument("--tgt-exo-lang", default=False, action="store_true")
        parser.add_argument("--for-NAT", default=False, action="store_true")
        parser.add_argument('--no-CSM', action='store_true',
                            help='without CSM operations')
        parser.add_argument('--ENC-DM', action='store_true',
                            help='without CSM operations')
        parser.add_argument("--no-replace-first", default=False, action='store_true',
                            help='first running replace ,then mask..')
        parser.add_argument("--mono-word-replace-ratio-alpha", type=float, default=0.25,
                            help=" ",)
        parser.add_argument("--mono-word-replace-ratio-beta", type=float, default=0.35,
                            help=" ",)
        parser.add_argument("--mono-masking-ratio-src-alpha", type=float, default=0.15,
                            help=" ",)
        parser.add_argument("--mono-masking-ratio-src-beta", type=float, default=0.25,
                            help=" ",)
        parser.add_argument("--mono-masking-ratio-tgt-alpha", type=float, default=0.3,
                            help=" ",)
        parser.add_argument("--mono-masking-ratio-tgt-beta", type=float, default=0.4,
                            help=" ",)
        parser.add_argument("--para-word-replace-ratio", type=float, default=0.15,
                            help=" ",)
        parser.add_argument("--para-src-masking-ratio-alpha", type=float, default=0.1,
                            help=" ",)
        parser.add_argument("--para-src-masking-ratio-beta", type=float, default=0.2,
                            help=" ",)
        parser.add_argument("--para-tgt-masking-ratio-alpha", type=float, default=0.2,
                            help=" ",)
        parser.add_argument("--para-tgt-masking-ratio-beta", type=float, default=0.5,
                            help=" ",)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        # get padding...
        args.left_pad_source = utils.eval_bool(args.left_pad_source)
        args.left_pad_target = utils.eval_bool(args.left_pad_target)
        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(
                paths[0]
            )

        print("path:",os.path.join(paths[0], "/Dicts/dict.txt"))
        dictionary = cls.load_dictionary(
            os.path.join(paths[0]+"/Dicts/", "dict.txt")
        )

        return cls(args, dictionary,paths)

    def init_dict(self,dictionary,paths):
        word_trans_dict={}
        self.word_align = {}
        if len(self.args.trans_dict) != 0:
            word_trans_dict = self.load_dict(os.path.join(paths[0] + "/Dicts/", self.args.trans_dict))
            self.word_align = self.load_dict(os.path.join(paths[0]+"/Dicts/",'train.merge.json'))
            self.word_align.update( self.load_dict(os.path.join(paths[0]+"/Dicts/",'valid.merge.json')))
        else:
            logger.info("word_trans and word_align didn't been add,please check...")
        logger.info("word_trans_dict.length.{}".format(len(word_trans_dict)))
        logger.info("word_align.length.{}".format(len(self.word_align)))

        # add lang token to dict.
        if self.args.add_lang_token:
            languages = self.args.langs.split(",")
            for lang_pair in languages:
                logger.info("{} was add to dictionary".format(lang_pair))
                lang = lang_pair.split("-")
                dictionary.add_symbol("[{}]".format(lang[0]))
                dictionary.add_symbol("[{}]".format(lang[1]))
        return dictionary,word_trans_dict

    def load_dict(self, file):
        with open(file,"r",encoding="utf-8") as f:
            return json.load(f)

    def add_dict(self,dicts,key, value):
        assert key not in dicts, print(dicts)
        dicts[key] = value
        return dicts

    def update_policy_dicts(self):
        self.policy_ratio_dicts = {}
        self.add_dict(self.policy_ratio_dicts, "mono_word_replace_ratio_alpha", self.args.mono_word_replace_ratio_alpha)
        self.add_dict(self.policy_ratio_dicts, "mono_word_replace_ratio_beta", self.args.mono_word_replace_ratio_beta)
        self.add_dict(self.policy_ratio_dicts, "mono_masking_ratio_src_alpha", self.args.mono_masking_ratio_src_alpha)
        self.add_dict(self.policy_ratio_dicts, "mono_masking_ratio_src_beta", self.args.mono_masking_ratio_src_beta)
        self.add_dict(self.policy_ratio_dicts, "mono_masking_ratio_tgt_alpha", self.args.mono_masking_ratio_tgt_alpha)
        self.add_dict(self.policy_ratio_dicts, "mono_masking_ratio_tgt_beta", self.args.mono_masking_ratio_tgt_beta)
        self.add_dict(self.policy_ratio_dicts, "para_word_replace_ratio", self.args.para_word_replace_ratio)
        self.add_dict(self.policy_ratio_dicts, "para_tgt_masking_ratio_alpha", self.args.para_tgt_masking_ratio_alpha)
        self.add_dict(self.policy_ratio_dicts, "para_tgt_masking_ratio_beta", self.args.para_tgt_masking_ratio_beta)
        self.add_dict(self.policy_ratio_dicts, "para_src_masking_ratio_alpha", self.args.para_src_masking_ratio_alpha)
        self.add_dict(self.policy_ratio_dicts, "para_src_masking_ratio_beta", self.args.para_src_masking_ratio_beta)
        logger.info("data policy ratios of raplacing and masking:{}".format(self.policy_ratio_dicts))


    def __init__(self, args, dictionary,paths):
        super().__init__(args)
        self.langs = args.langs
        self.seed = args.seed
        self.args = args
        self.dictionary,self.word_trans_dict = self.init_dict(dictionary,paths)
        self.mask_idx = self.dictionary.add_symbol("<mask>")
        self.src_dict = self.dictionary
        self.tgt_dict = self.dictionary
        self.update_policy_dicts()
        logger.info("[{}] dictionary: {} types".format(args.langs, len(self.tgt_dict)))


    def _get_sample_prob(self, dataset_lens):
        """
        Get smoothed sampling porbability by languages. This helps low resource
        languages by upsampling them.
        """
        prob = dataset_lens / dataset_lens.sum()
        smoothed_prob = prob ** self.args.multilang_sampling_alpha
        smoothed_prob = smoothed_prob / smoothed_prob.sum()
        return smoothed_prob

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0

        # if not training data set, use the first shard for valid and test
        if split != getattr(self.args, "train_subset", None):
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        lang_datasets = []
        for lang_pair in self.args.langs.split(","):
            [src, tgt] = lang_pair.split("-")

            # if multi-valid : valid.zh-en
            if "valid" in split and split != "valid":
                split_lang = split.split(".")[-1]
                if ( (lang_pair != split_lang) and (tgt + "-" + src != split_lang) ):
                    continue
                # special for (fil and fi) langs.
                if (lang_pair != split_lang and (tgt + "-" + src == split_lang and "fil" in split_lang)):
                    continue

            data_path_temp = data_path
            if self.args.add_lang_token:
                add_langs = (src, tgt)
            else:
                add_langs = None
            dataset = load_langpair_dataset(
                data_path_temp,
                split,
                src,
                self.src_dict,
                tgt,
                self.tgt_dict,
                combine=combine,
                dataset_impl=self.args.dataset_impl,
                upsample_primary=self.args.upsample_primary,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
                load_alignments=self.args.load_alignments,
                truncate_source=self.args.truncate_source,
                num_buckets=self.args.num_batch_buckets,
                shuffle=(split != "test"),
                pad_to_multiple=self.args.required_seq_len_multiple,
                plus_encoder_loss=self.args.plus_encoder_loss,
                add_langs=add_langs,
                shuffle_lang_pair=self.args.shuffle_lang_pair,
                args=self.args,
                word_trans_dict=self.word_trans_dict,
                word_align_dict=self.word_align,
                policy_ratio_dicts=self.policy_ratio_dicts
            )
            lang_datasets.append(dataset)

        dataset_lengths = np.array(
            [len(d) for d in lang_datasets],
            dtype=float,
        )
        logger.info(
            "loaded total {} blocks for all languages".format(
                int(dataset_lengths.sum()),
            )
        )
        if split == self.args.train_subset:
            # For train subset, additionally up or down sample languages.
            sample_probs = self._get_sample_prob(dataset_lengths)
            logger.info(
                "Sample probability by language: {}".format(
                    {
                        lang: "{0:.4f}".format(sample_probs[id])
                        for id, lang in enumerate(self.args.langs.split(","))
                    }
                )
            )
            size_ratio = (sample_probs * dataset_lengths.sum()) / dataset_lengths
            logger.info(
                "Up/Down Sampling ratio by language: {}".format(
                    {
                        lang: "{0:.2f}".format(size_ratio[id])
                        for id, lang in enumerate(self.args.langs.split(","))
                    }
                )
            )

            resampled_lang_datasets = [
                ResamplingDataset(
                    lang_datasets[i],
                    size_ratio=size_ratio[i],
                    seed=self.args.seed,
                    epoch=epoch,
                    replace=size_ratio[i] >= 1.0,
                )
                for i, d in enumerate(lang_datasets)
            ]
            dataset = ConcatPairDataset(
                resampled_lang_datasets,
            )
        else:
            dataset = ConcatPairDataset(lang_datasets)
        self.datasets[split] = dataset


    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

