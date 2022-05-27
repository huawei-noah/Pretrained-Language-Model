# 2021 - Added code for CeMAT
#        Huawei Technologies Co., Ltd. <lipengfei111@huawei.com>
# Copyright 2022 Huawei Technologies Co., Ltd.
#
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import os
from fairseq import utils
from fairseq import options

from fairseq.data import (
    data_utils, Dictionary, ConcatDataset,PrependTokenDataset,
    IndexedRawTextDataset, IndexedCachedDataset, IndexedDataset,
)
from ..data import LanguagePairSelfDatasetMask

from fairseq.tasks import register_task, LegacyFairseqTask
import logging
logger = logging.getLogger(__name__)


class CeMATDictionary(Dictionary):
    def __init__(
            self,
            *,  # begin keyword-only arguments
            bos="<s>",
            pad="<pad>",
            eos="</s>",
            unk="<unk>",
            extra_special_symbols=None,
    ):
        super().__init__(bos=bos,pad=pad,eos=eos,unk=unk,extra_special_symbols=extra_special_symbols)

    def add_mask(self,mask='<mask>'):
        self.mask_idx = self.add_symbol(mask)
        return self.mask_idx

    def mask(self):
        """Helper to get index of pad symbol"""
        return self.mask_idx


@register_task('translation_self_from_pt')
class TranslationSelfTaskFromPT(LegacyFairseqTask):
    """
    Translate from one (source) language to another (target) language.
    Args:
        src_dict (Dictionary): dictionary for the source language
        tgt_dict (Dictionary): dictionary for the target language
    .. note::
        The translation task is compatible with :mod:`train.py <train>`,
        :mod:`generate.py <generate>` and :mod:`interactive.py <interactive>`.
    The translation task provides the following additional command-line
    arguments:
    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', nargs='+', help='path(s) to data directorie(s)')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        # parser.add_argument('--patience', default=10, type=int, metavar='N',
        #                     help='patience with early -stoping')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--self-target', action='store_true',
                            help='include self target')
        parser.add_argument('--dynamic-length', action='store_true',
                            help='dynamic length')
        parser.add_argument('--mask-range', action='store_true',
                            help='dynamic length')
        #add extra param
        parser.add_argument('--langs', metavar='LANG',
                            help='comma-separated list of monolingual language, '
                                 'for example, "en,de,fr". These should match the '
                                 'langs from pretraining (and be in the same order). '
                                 'You should always add all pretraining language idx '
                                 'during finetuning.')
        parser.add_argument("--add-lang-token", default=False, action="store_true")
        # parser.add_argument("--plus-encoder-loss", default=False, action="store_true")
        parser.add_argument('--share-dict', action='store_true',
                            help='share encoder,decoder vocab')
        parser.add_argument('--from-pt', action='store_true',
                            help='init,with pt')
        parser.add_argument('--dicts', type=str, default="dict.txt",
                    help='dicts name')

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.args = args
        self.src_mask_idx = src_dict.add_mask("<mask>")
        self.tgt_mask_idx = tgt_dict.add_mask("<mask>")
        logger.info("ensure tgt_mask_idx:{}".format(self.tgt_mask_idx))
        logger.info("share dict {}".format(self.args.share_dict))
        logger.info("[{}] dictionary: {} types".format(args.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(args.target_lang, len(tgt_dict)))
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).
        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)
        paths = utils.split_paths(args.data[0])
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(args.data[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        logger.info("loading dicts from.{}".format(args.dicts))
        dictionary = cls.load_dictionary(
            os.path.join(paths[0], args.dicts)
        )

        logger.info("args.add_lang_token: {} ".format(args.add_lang_token))
        if args.add_lang_token:
            languages = args.langs.split(",")
            for lang_pair in languages:
                print("{} was add to dictionary".format(lang_pair))
                lang = lang_pair.split("-")
                dictionary.add_symbol("[{}]".format(lang[0]))
                dictionary.add_symbol("[{}]".format(lang[1]))
        return cls(args, dictionary,dictionary)

    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        return CeMATDictionary.load(filename)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def split_exists(split, src, tgt, lang, data_path):
            filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
            if self.args.dataset_impl == 'raw' and IndexedRawTextDataset.exists(filename):
                return True
            elif self.args.dataset_impl != 'raw' and IndexedDataset.exists(filename):
                return True
            return False

        src_datasets = []
        tgt_datasets = []

        data_paths = self.args.data
        src = None
        tgt = None
        for dk, data_path in enumerate(data_paths):
            for k in itertools.count():
                split_k = split + (str(k) if k > 0 else '')

                # infer langcode
                src, tgt = self.args.source_lang, self.args.target_lang
                if split_exists(split_k, src, tgt, src, data_path):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
                elif split_exists(split_k, tgt, src, src, data_path):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
                else:
                    if k > 0 or dk > 0:
                        break
                    else:
                        raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

                src_datasets.append(data_utils.load_indexed_dataset(prefix + src, self.src_dict, self.args.dataset_impl))
                tgt_datasets.append(data_utils.load_indexed_dataset(prefix + tgt, self.tgt_dict, self.args.dataset_impl))

                # print('{}: {} examples'.format(prefix+src, len(src_datasets[-1])))
                # print('{}: {} examples'.format(prefix+tgt, len(tgt_datasets[-1])))

                if not combine:
                    break
        assert len(src_datasets) == len(tgt_datasets)
        if len(src_datasets) == 1:
            src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
        else:
            sample_ratios = [1] * len(src_datasets)
            sample_ratios[0] = self.args.upsample_primary
            src_dataset = ConcatDataset(src_datasets, sample_ratios)
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

        if split == "train":
            train = True
            seed = None
        elif split == "valid":
            train = True
            seed = 1
        elif split == "test":
            train = False
            seed = 1
        else:
            raise Exception('No such split: ' + str(split))

        if self.args.add_lang_token:
            src_dataset = PrependTokenDataset(
                src_dataset, self.src_dict.index("[{}]".format(src))
            )
            if tgt_dataset is not None:
                tgt_dataset = PrependTokenDataset(
                    tgt_dataset, self.tgt_dict.index("[{}]".format(tgt))
                )

        self.datasets[split] = LanguagePairSelfDatasetMask(
            src_dataset, src_dataset.sizes, self.src_dict,
            tgt_dataset, tgt_dataset.sizes, self.tgt_dict,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            shuffle=False,
            dynamic_length=self.args.dynamic_length,
            mask_range=self.args.mask_range,
            train=train,
            seed=seed,
            mask_idx=self.tgt_mask_idx
        )

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
