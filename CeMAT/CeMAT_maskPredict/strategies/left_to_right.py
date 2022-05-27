# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch.nn.functional as F

from . import register_strategy
from .easy_first import EasyFirst
from .strategy_utils import duplicate_encoder_out, generate_step_with_prob, assign_single_value_long, assign_single_value_byte, assign_multi_value_long, convert_tokens


@register_strategy('left_to_right')
class LeftToRight(EasyFirst):
    
    def __init__(self, args):
        super().__init__(args)
    
    def generate(self, model, encoder_out, tokens, tgt_dict):
        bsz, seq_len = tokens.size()
        duplicate_encoder_out(encoder_out, bsz, self.beam_size)
        tokens = tokens.unsqueeze(1).repeat(1, self.beam_size, 1)
        lprobs = tokens.new(bsz, self.beam_size).float().fill_(float('-inf'))
        lprobs[:, 0] = 0

        """
        for batch in range(bsz):
            for beam in range(self.beam_size):
                print("Initialization: ", convert_tokens(tgt_dict, tokens[batch, beam]))
        print()
        """

        for position in range(seq_len):
            tokens = tokens.view(bsz * self.beam_size, seq_len) # merge beam with batch
            decoder_out = model.decoder(tokens, encoder_out)
            candidate_lprobs = self.generate_candidates(decoder_out, tokens, tgt_dict.mask(), position)
            tokens = tokens.view(bsz, self.beam_size, seq_len) # separate beam from batch
            candidate_lprobs = candidate_lprobs.view(bsz, self.beam_size, seq_len, -1) # separate beam from batch
            tokens, lprobs = self.select_best(tokens, lprobs, candidate_lprobs)

            """
            for batch in range(bsz):
                for beam in range(self.beam_size):
                    print("Prediction: ", convert_tokens(tgt_dict, tokens[batch, beam]))
            print()
            """

        return tokens[:, 0, :], lprobs[:, 0]
    
    def generate_candidates(self, decoder_out, tokens, mask, position):
        candidate_probs = F.softmax(decoder_out[0], dim=-1)
        candidate_probs = candidate_probs * tokens.eq(mask).float().unsqueeze(-1)
        candidate_probs[:, :, mask] = 0
        candidate_probs[:, position + 1:, :] = 0
        return candidate_probs.log()

