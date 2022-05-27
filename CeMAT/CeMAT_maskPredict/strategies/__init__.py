# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in # the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import argparse
import importlib
import os

from .decoding_strategy import DecodingStrategy


STRATEGY_REGISTRY = {}
STRATEGY_CLASS_NAMES = set()


def setup_strategy(args):
    return STRATEGY_REGISTRY[args.decoding_strategy](args)


def register_strategy(name):
    def register_strategy_cls(cls):
        if name in STRATEGY_REGISTRY:
            raise ValueError('Cannot register duplicate strategy ({})'.format(name))
        if not issubclass(cls, DecodingStrategy):
            raise ValueError('Strategy ({}: {}) must extend DecodingStrategy'.format(name, cls.__name__))
        if cls.__name__ in STRATEGY_CLASS_NAMES:
            raise ValueError('Cannot register strategy with duplicate class name ({})'.format(cls.__name__))
        STRATEGY_REGISTRY[name] = cls
        STRATEGY_CLASS_NAMES.add(cls.__name__)
        return cls

    return register_strategy_cls


# automatically import any Python files in the strategies/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        strategy_name = file[:file.find('.py')]
        importlib.import_module('CeMAT_maskPredict.strategies.' + strategy_name)
