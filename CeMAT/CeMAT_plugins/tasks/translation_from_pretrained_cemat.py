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
import torch
from argparse import Namespace
import json
import logging
import argparse
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
)

import os
import numpy as np
from fairseq import metrics, options, utils
from fairseq.tasks import FairseqTask
from fairseq.tasks import register_task, LegacyFairseqTask

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
        add_lang_token=False,
):
    def split_exists(split, src, tgt, lang, data_path):
        logger.info(os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang)))
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    def split_exists_self(split, src, data_path):
        logger.info(os.path.join(data_path, "{}.{}-{}.{}".format(split, src, src, src)))
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, src, src))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    def split_exists_valid(split, lang, data_path):
        logger.info(os.path.join(data_path, "{}.{}".format(split, lang)))
        filename = os.path.join(data_path, "{}.{}".format(split, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")
        # print(split_k, src, tgt, src, data_path)
        prefix_src = None
        prefix_tgt = None
        if not "-" in split_k:
            # infer langcode
            if split_exists(split_k, src, tgt, src, data_path) :
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
            # infer langcode
            if split_exists_valid( split_k, src, data_path):
                prefix = os.path.join(data_path, split_k+".")
            else:
                if k > 0:
                    break
                else:
                    raise FileNotFoundError(
                        "Dataset not found: {} ({}) ".format(split, data_path)
                    )
        if prefix_src != None:
            prefix = prefix_src

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

        if prefix_tgt != None:
            prefix = prefix_tgt
        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        logger.info("::::data sample_ratios:{}".format(sample_ratios))
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
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    eos = None
    if add_lang_token:
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
    return LanguagePairDataset(
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
    )

@register_task("translation_from_pretrained_cemat")
class TranslationFromPretrainedCemat(LegacyFairseqTask):
    """
    Translate from source language to target language with a model initialized with a multilingual pretrain.

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

    @staticmethod
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
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
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
        # fmt: off
        parser.add_argument('--langs', default="",metavar='LANG',
                            help='comma-separated list of monolingual language, '
                                 'for example, "en,de,fr". These should match the '
                                 'langs from pretraining (and be in the same order). '
                                 'You should always add all pretraining language idx '
                                 'during finetuning.')
        parser.add_argument('--prepend-bos', action='store_true',
                            help='prepend bos token to each sentence, which matches '
                                 'mBART pretraining')
        # fmt: on
        #add extra param
        parser.add_argument(
            "--s3-dir",
            type=str,
            default="",
            metavar="N",
            help="path to s3 checkpoints",
        )
        parser.add_argument("--add-lang-token", default=False, action="store_true")
        parser.add_argument('--share-dict', action='store_true',
                            help='share encoder,decoder vocab')
        parser.add_argument('--freeze-encoder-layer',default=False, action="store_true",
                    help='free all encoder layer')
        parser.add_argument('--freeze-encoder-emb',default=False, action="store_true",
                    help='free encoder embeding ')
        parser.add_argument('--freeze-cross-attention',default=False, action="store_true",
                    help='free cross attention modules')


    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.args = args
        for d in [src_dict, tgt_dict]:
            d.add_symbol("<mask>")
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        logger.info("share dict {}".format(self.args.share_dict))
        logger.info("[{}] dictionary: {} types".format(args.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(args.target_lang, len(tgt_dict)))


    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        args.left_pad_source = utils.eval_bool(args.left_pad_source)
        args.left_pad_target = utils.eval_bool(args.left_pad_target)

        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(
                paths[0]
            )
        if args.source_lang is None or args.target_lang is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )

        dictionary = cls.load_dictionary(
            os.path.join(paths[0], "dict.txt")
        )

        # langs:"en-zh,my-en"
        logger.info("args.add_lang_token: {} ".format(args.add_lang_token))
        if args.add_lang_token and len(args.langs) > 0:
            languages = args.langs.split(",")
            for lang_pair in languages:
                if lang_pair == "-": continue
                logger.info("{} was add to dictionary".format(lang_pair))
                lang = lang_pair.split("-")
                dictionary.add_symbol("[{}]".format(lang[0]))
                dictionary.add_symbol("[{}]".format(lang[1]))
        return cls(args, dictionary,dictionary)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        src, tgt = self.args.source_lang, self.args.target_lang
        self.datasets[split] = load_langpair_dataset(
            data_path,
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
            max_source_positions=getattr(self.args, "max_source_positions", 1024),
            max_target_positions=getattr(self.args, "max_target_positions", 1024),
            load_alignments=self.args.load_alignments,
            prepend_bos=getattr(self.args, "prepend_bos", False),
            append_source_id=False,
            add_lang_token=self.args.add_lang_token,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        return LanguagePairDataset(
            src_tokens,
            src_lengths,
            self.source_dictionary,
            tgt_dict=self.target_dictionary,
            constraints=constraints,
        )

    def build_model(self, args, from_checkpoint=False):
        print("args:{}".format(args))
        model = super().build_model(args, from_checkpoint)
        if getattr(args, "eval_bleu", False):
            assert getattr(args, "eval_bleu_detok", None) is not None, (
                "--eval-bleu-detok is required if using --eval-bleu; "
                "try --eval-bleu-detok=moses (or --eval-bleu-detok=space "
                "to disable detokenization, e.g., when using sentencepiece)"
            )
            detok_args = json.loads(getattr(args, "eval_bleu_detok_args", "{}") or "{}")
            self.tokenizer = encoders.build_tokenizer(
                Namespace(
                    tokenizer=getattr(args, "eval_bleu_detok", None), **detok_args
                )
            )

            gen_args = json.loads(getattr(args, "eval_bleu_args", "{}") or "{}")
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.args.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args.eval_bleu:

            def sum_logs(key):
                return sum(log.get(key, 0) for log in logging_outputs)

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))
            # print(type(counts),counts[0])
            # print(type(totals))
            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                # metrics.log_scalar("_bleu_counts", np.array(counts.cpu().))
                # metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_counts", np.array(torch.tensor(counts)))
                metrics.log_scalar("_bleu_totals", np.array(torch.tensor(totals)))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu

                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=meters["_bleu_sys_len"].sum,
                        ref_len=meters["_bleu_ref_len"].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)

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

    def remove_sentencepiece(self, item, sentpiece=None):
        if not sentpiece:
            return item
        return item.replace(" ", "").replace("▁", " ")

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False, dicts=self.tgt_dict):
            s = dicts.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        prefix_tokens = None
        # if self.args.prefix_lang > 0:
        #     prefix_tokens = sample["target"][:, :1]
        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=prefix_tokens)

        hyps, refs = [], []
        srcs = []
        for i in range(len(gen_out)):
            hyps_item = decode(gen_out[i][0]["tokens"])
            hyps_item_bpe = self.remove_sentencepiece(hyps_item, (self.args.bpe == 'sentencepiece'))
            if hyps_item_bpe[0] == "[": # rm lang-tok id.
                hyps_item_bpe_split = " ".join( hyps_item_bpe.split(" ")[1:] )
            else:
                hyps_item_bpe_split = hyps_item_bpe
            hyps.append(hyps_item_bpe_split)

            lang_tokens = ""
            ref_item = decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            ref_item_bpe = self.remove_sentencepiece(ref_item, (self.args.bpe == 'sentencepiece'))

            if ref_item_bpe[0] == "[": # rm lang-tok id.
                ref_item_bpe = ref_item_bpe.split(" ")
                ref_item_bpe_split = " ".join( ref_item_bpe[1:] )
                lang_tokens=ref_item_bpe[0]
            else:
                ref_item_bpe_split = ref_item_bpe
            refs.append(ref_item_bpe_split)

            src_item = decode(
                    utils.strip_pad(sample["net_input"]["src_tokens"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                    dicts=self.src_dict,
                )
            src_item_bpe = self.remove_sentencepiece(src_item, (self.args.bpe == 'sentencepiece'))
            srcs.append(lang_tokens + "| |" + src_item_bpe)


        if self.args.eval_bleu_print_samples:
            logger.info("example sources: " + srcs[0])
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])

        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            if hasattr(self.args, "target_lang") and self.args.target_lang == 'zh':
                return sacrebleu.corpus_bleu(hyps, [refs], tokenize="zh")
            return sacrebleu.corpus_bleu(hyps, [refs])

