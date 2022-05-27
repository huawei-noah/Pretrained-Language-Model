# 2022 - Added code for CeMAT
#        Huawei Technologies Co., Ltd. <lipengfei111@huawei.com>
# Copyright 2022 Huawei Technologies Co., Ltd.
#
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import argparse

import torch
import sys

from fairseq import utils
from fairseq.data.indexed_dataset import get_available_dataset_impl
from CeMAT_maskPredict.strategies import STRATEGY_REGISTRY
from fairseq.options import add_common_eval_args, get_parser, add_dataset_args, add_interactive_args
from fairseq.options import add_distributed_training_args
from fairseq.options import add_checkpoint_args

def add_generation_args(parser):
    group = parser.add_argument_group('Generation')
    add_common_eval_args(group)
    # fmt: off
    #### bert sampling options ####
    group.add_argument('--decoding-strategy', default='left_to_right', choices=STRATEGY_REGISTRY.keys())
    group.add_argument('--gold-target-len', action='store_true', help='use gold target length')
    group.add_argument('--dehyphenate', action='store_true', help='turn hyphens into independent tokens')
    parser.add_argument('--decoding-iterations', default=None, type=int, metavar='N', help='number of decoding iterations in mask-predict')
    group.add_argument('--length-beam', default=5, type=int, metavar='N',
                       help='length beam size')
    #### other generation options ####
    group.add_argument('--beam', default=5, type=int, metavar='N',
                       help='beam size')
    group.add_argument('--nbest', default=1, type=int, metavar='N',
                       help='number of hypotheses to output')
    group.add_argument('--max-len-a', default=0, type=float, metavar='N',
                       help=('generate sequences of maximum length ax + b, '
                             'where x is the source length'))
    group.add_argument('--max-len-b', default=200, type=int, metavar='N',
                       help=('generate sequences of maximum length ax + b, '
                             'where x is the source length'))
    group.add_argument('--min-len', default=1, type=float, metavar='N',
                       help=('minimum generation length'))
    group.add_argument('--match-source-len', default=False, action='store_true',
                       help=('generations should match the source length'))
    group.add_argument('--no-early-stop', action='store_true',
                       help=('continue searching even after finalizing k=beam '
                             'hypotheses; this is more correct, but increases '
                             'generation time by 50%%'))
    group.add_argument('--unnormalized', action='store_true',
                       help='compare unnormalized hypothesis scores')
    group.add_argument('--no-beamable-mm', action='store_true',
                       help='don\'t use BeamableMM in attention layers')
    group.add_argument('--lenpen', default=1, type=float,
                       help='length penalty: <1.0 favors shorter, >1.0 favors longer sentences')
    group.add_argument('--unkpen', default=0, type=float,
                       help='unknown word penalty: <0 produces more unks, >0 produces fewer')
    group.add_argument('--replace-unk', nargs='?', const=True, default=None,
                       help='perform unknown replacement (optionally with alignment dictionary)')
    group.add_argument('--sacrebleu', action='store_true',
                       help='score with sacrebleu')
    group.add_argument('--score-reference', action='store_true',
                       help='just score the reference translation')
    group.add_argument('--prefix-size', default=0, type=int, metavar='PS',
                       help='initialize generation by target prefix of given length')
    group.add_argument('--no-repeat-ngram-size', default=0, type=int, metavar='N',
                       help='ngram blocking such that this size ngram cannot be repeated in the generation')
    group.add_argument('--sampling', action='store_true',
                       help='sample hypotheses instead of using beam search')
    group.add_argument('--sampling-topk', default=-1, type=int, metavar='PS',
                       help='sample from top K likely next words instead of all words')
    group.add_argument('--sampling-topp', default=-1.0, type=float, metavar='PS',
                       help='sample from the smallest set whose cumulative probability mass exceeds p for next words')
    group.add_argument('--temperature', default=1., type=float, metavar='N',
                       help='temperature for generation')
    group.add_argument('--diverse-beam-groups', default=-1, type=int, metavar='N',
                       help='number of groups for Diverse Beam Search')
    group.add_argument('--diverse-beam-strength', default=0.5, type=float, metavar='N',
                       help='strength of diversity penalty for Diverse Beam Search')
    group.add_argument('--print-alignment', action='store_true',
                       help='if set, uses attention feedback to compute and print alignment to source tokens')
    # fmt: on
    return group


def get_generation_parser(interactive=False, default_task='translation'):
    parser = get_parser('Generation', default_task)
    add_dataset_args(parser, gen=True)
    add_distributed_training_args(parser, default_world_size=1)
    add_generation_args(parser)
    add_checkpoint_args(parser)
    if interactive:
        add_interactive_args(parser)
    return parser