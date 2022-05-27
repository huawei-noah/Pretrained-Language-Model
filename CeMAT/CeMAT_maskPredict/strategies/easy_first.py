# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import torch.nn.functional as F

from . import DecodingStrategy, register_strategy
from .strategy_utils import duplicate_encoder_out, generate_step_with_prob, assign_single_value_long, assign_single_value_byte, assign_multi_value_long, convert_tokens


@register_strategy('easy_first')
class EasyFirst(DecodingStrategy):
    
    def __init__(self, args):
        super().__init__()
        self.beam_size = args.beam
    
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

        while tokens.eq(tgt_dict.mask()).sum() > 0:
            tokens = tokens.view(bsz * self.beam_size, seq_len) # merge beam with batch
            decoder_out = model.decoder(tokens, encoder_out)
            candidate_lprobs = self.generate_candidates(decoder_out, tokens, tgt_dict.mask())
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
    
    def generate_candidates(self, decoder_out, tokens, mask):
        candidate_probs = F.softmax(decoder_out[0], dim=-1)
        candidate_probs = candidate_probs * tokens.eq(mask).float().unsqueeze(-1)
        candidate_probs[:, :, mask] = 0
        return candidate_probs.log()
    
    def select_best(self, tokens, lprobs, candidate_lprobs):
        bsz, beam_size, seq_len, vocab_size = candidate_lprobs.size()
        max_candidates = (beam_size + 1) * beam_size // 2
        scores = lprobs.unsqueeze(-1).unsqueeze(-1) + candidate_lprobs
        candidate_lprobs, candidate_tokens = scores.view(bsz, -1).topk(max_candidates, dim=-1)

        new_token_id = candidate_tokens % vocab_size
        new_token_pos = (candidate_tokens // vocab_size) % seq_len
        new_token_beam = (candidate_tokens // vocab_size) // seq_len

        new_tokens = tokens.new(tokens.size()).fill_(0)
        new_lprobs = lprobs.new(lprobs.size()).fill_(float('-inf'))
        for batch in range(bsz):
            if torch.isinf(candidate_lprobs[batch, 0]).item():
                new_tokens[batch] = tokens[batch]
                new_lprobs[batch] = lprobs[batch]
            else:
                candidate = 0
                beam = 0
                while beam < beam_size:
                    new_tokens[batch, beam] = tokens[batch, new_token_beam[batch, candidate]]
                    new_tokens[batch, beam, new_token_pos[batch, candidate]] = new_token_id[batch, candidate]
                    if self.is_unique(new_tokens[batch, beam], new_tokens[batch, :beam]):
                        new_lprobs[batch, beam] = candidate_lprobs[batch, candidate]
                        beam += 1
                    candidate += 1

        return new_tokens, new_lprobs
        
    def is_unique(self, seq, seqs):
        if len(seqs) == 0:
            return True
        return (seqs - seq.unsqueeze(0)).ne(0).sum(-1).eq(0).sum().item() == 0

