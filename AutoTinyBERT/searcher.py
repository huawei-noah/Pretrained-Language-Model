# coding=utf-8
# Copyright 2021 Huawei Technologies Co., Ltd.
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
import random
import argparse


from latency_predictor import LatencyPredictor
from collections import OrderedDict
from operator import itemgetter


def is_efficient_arch(config):
    d_m = config['sample_hidden_size']
    d_f = config['sample_intermediate_sizes'][0]
    d_qkv = config['sample_qkv_sizes'][0]

    if 1.6 * d_m <= d_f <= 1.9 * d_m:
        if 0.7 * d_m <= d_qkv <= d_m:
            return True
    return False


def load_arch_perfs(file_name):
    one_shot_results = dict()
    arch = ''
    perfs = []
    with open(file_name, 'r') as fin:
        for line in fin:
            line = line.strip()
            if line.startswith('architecture = '):
                arch = line[len('architecture = '):]
            elif line.startswith('best_acc = f1: '):
                perf = float(line[len('best_acc = f1: '): len('best_acc = f1: ') + 7])
                perfs.append(perf)
            elif line.startswith('best_acc = '):
                if len(line[len('best_acc = '):]) <= 2:
                    perfs.append(0.0)
                else:
                    perfs.append(float(line[len('best_acc = '):]) * 100)
            elif line.startswith('best_acc ='):
                perfs.append(0.0)

            # we use the performance of SQuAD and MNLI as fitness
            if len(perfs) == 2:
                one_shot_results[arch] = sum(perfs) / len(perfs)
                arch = ''
                perfs = []
    return one_shot_results


def arch_str_format(arch):
    if isinstance(arch, dict):
        return str(arch)
    elif isinstance(arch, str):
        return arch
    else:
        raise TypeError


class Evolver(object):
    # both the explore_rate and mutation rate are set to 0.5
    def __init__(self, args, all_arches=None, popularity=25, explore_rate=0.5, prob_m=0.25,
                 latency_scale=None, candidates=None,
                 latency_predictor=None, search_space=None):
        self.args = args
        self.popularity = popularity
        self.prob_m = prob_m
        self.explore_rate = explore_rate
        self.candidates = candidates
        self.latency_predictor = latency_predictor
        self.all_arches = all_arches
        self.latency_scale = latency_scale
        self.search_space = search_space

    def generate_next_generation(self):
        arch_lis = []
        arch_str_lis = []
        all_arches = self.all_arches
        popularity = self.popularity
        explore_rate = self.explore_rate
        candidates = self.candidates

        all_arches = OrderedDict(sorted(all_arches.items(), key=itemgetter(1), reverse=True))

        parent_arches = list(all_arches.keys())[:popularity]
        parent_perfs = list(all_arches.values())[:popularity]

        cur_popularity = 0

        while cur_popularity < popularity:
            if random.random() < explore_rate:
                new_arch = random.choice(candidates)
                while str(new_arch) in self.all_arches:
                    new_arch = random.choice(candidates)

                if str(new_arch) not in arch_str_lis:
                    arch_str_lis.append(str(new_arch))
                else:
                    continue

                arch_lis.append(str(new_arch))
                cur_popularity += 1
            else:
                parent_arch = self.roulette(parent_arches, parent_perfs)
                new_arch = self.mutation(parent_arch)

                if str(new_arch) not in arch_str_lis:
                    arch_str_lis.append(str(new_arch))
                else:
                    continue

                arch_lis.append(str(new_arch))
                cur_popularity += 1
        return arch_lis[:popularity]

    def mutation(self, arch):
        args = self.args
        prob_m = self.prob_m
        latency_scale = self.latency_scale
        model = args.model
        new_arch_latency = 1e5
        latency_predictor = self.latency_predictor
        layer_numbers, hidden_sizes, ffn_sizes, qkv_sizes, qkv_sizes_mlm, head_sizes = generate_search_space(args)

        max_layer_index = len(layer_numbers)
        max_ffn_index = len(ffn_sizes)
        max_head_index = len(head_sizes)
        max_hid_index = len(hidden_sizes)
        max_qkv_index = len(qkv_sizes)
        arch = json.loads(json.dumps(eval(arch)))
        layer_num_idx = layer_numbers.index(arch['sample_layer_num'])
        hidden_size_idx = hidden_sizes.index(arch['sample_hidden_size'])

        ffn_size_idxes = []
        for ffn_size in arch['sample_intermediate_sizes']:
            ffn_size_idxes.append(ffn_sizes.index(ffn_size))

        head_numbers = arch['sample_num_attention_heads']

        qkv_size_idxes = []
        for qkv_size in arch['sample_qkv_sizes']:
            if model == 'MLM':
                qkv_size_idxes.append(qkv_sizes_mlm.index(qkv_size))
            else:
                qkv_size_idxes.append(qkv_sizes.index(qkv_size))

        new_arch = dict()

        latency_min, latency_max = latency_scale
        while new_arch_latency < latency_min or new_arch_latency > latency_max or str(
                new_arch) in self.all_arches:
            ### layer number
            new_layer_idx = layer_num_idx
            if prob_m > random.random():
                layer_offset = random.choice([-1, 0, 1])
                if 0 <= layer_num_idx + layer_offset < max_layer_index:
                    new_layer_idx = layer_num_idx + layer_offset

            new_arch['sample_layer_num'] = layer_numbers[new_layer_idx]
            new_layer_num = layer_numbers[new_layer_idx]

            new_size_idx = hidden_size_idx
            if prob_m * 2 > random.random():
                size_offset = random.choice([-1, 0, 1])
                if 0 <= hidden_size_idx + size_offset < max_hid_index:
                    new_size_idx = hidden_size_idx + size_offset

            ffn_size_idx = ffn_size_idxes[0]
            new_ffn_size_idx = ffn_size_idx
            if prob_m * 2 > random.random():
                ffn_offset = random.choice([-2, -1, 0, 1, 2])
                if 0 <= ffn_size_idx + ffn_offset < max_ffn_index:
                    new_ffn_size_idx = ffn_size_idx + ffn_offset

            new_arch['sample_hidden_size'] = hidden_sizes[new_size_idx]
            new_arch['sample_intermediate_sizes'] = [ffn_sizes[new_ffn_size_idx]] * new_layer_num

            if model == 'MLM':
                num_attention_head = head_numbers[0]
                new_num_attention_head = num_attention_head
                if prob_m > random.random():
                    num_attention_offset = random.choice([-2, -1, 0, 1, 2])
                    if 0 < num_attention_head + num_attention_offset <= max_head_index:
                        new_num_attention_head = num_attention_head + num_attention_offset
                new_arch['sample_num_attention_heads'] = [new_num_attention_head] * new_layer_num
                new_arch['sample_qkv_sizes'] = [new_num_attention_head * 64] * new_layer_num
            else:
                new_qkv_idx = qkv_size_idxes[0]
                if prob_m > random.random():
                    qkv_offset = random.choice([-2, -1, 0, 1, 2])
                    if 0 <= qkv_size_idxes[0] + qkv_offset < max_qkv_index:
                        new_qkv_idx = qkv_size_idxes[0] + qkv_offset

                new_arch['sample_num_attention_heads'] = [12] * new_layer_num
                new_arch['sample_qkv_sizes'] = [qkv_sizes[new_qkv_idx]] * new_layer_num

            new_arch_latency = latency_predictor.predict_lat(new_arch)

            print('old arch: {}'.format(arch))
            print('after mutation, the new arch is : {}'.format(new_arch))
            print('new_arch_latency: {}'.format(new_arch_latency))
        # input('pause!')
        return new_arch

    def roulette(self, arches, perfs):
        '''
        Input: a list of N fitness values (list or tuple)
        Output: selected index
        '''
        mini = min(perfs)
        new_fit = [perfs[i] - mini for i in range(len(perfs))]
        sum_fit = sum(new_fit)
        rnd_point = random.uniform(0, sum_fit)
        accumulator = 0.0
        for ind, val in enumerate(new_fit):
            accumulator += val
            if accumulator >= rnd_point:
                return arches[ind]


def generate_search_space(args):
    # build arch space
    min_hidden_size, max_hidden_size = args.hidden_size_space
    min_ffn_size, max_ffn_size = args.intermediate_size_space
    min_qkv_size, max_qkv_size = args.qkv_size_space
    min_head_size, max_head_size = args.head_num_space

    # both hidden_step and ffn_step are 4 in original paper.
    # these settings are for efficiency.
    hidden_step = 16
    ffn_step = 32
    qkv_step = 12
    head_step = 1

    number_hidden_step = int((max_hidden_size - min_hidden_size) / hidden_step)
    number_ffn_step = int((max_ffn_size - min_ffn_size) / ffn_step)
    number_qkv_step = int((max_qkv_size - min_qkv_size) / qkv_step)
    number_head_step = int((max_head_size - min_head_size) / head_step)

    layer_numbers = list(range(args.layer_num_space[0], args.layer_num_space[1] + 1))
    hidden_sizes = [i * hidden_step + min_hidden_size for i in range(number_hidden_step + 1)]
    ffn_sizes = [i * ffn_step + min_ffn_size for i in range(number_ffn_step + 1)]
    qkv_sizes = [i * qkv_step + min_qkv_size for i in range(number_qkv_step + 1)]
    qkv_sizes_mlm = [(i + 1) * 64 for i in range(number_head_step)]
    head_sizes = [i * head_step + min_head_size for i in range(number_head_step + 1)]
    return (layer_numbers, hidden_sizes, ffn_sizes, qkv_sizes, qkv_sizes_mlm, head_sizes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Arguments for Predictor Lat(*)
    parser.add_argument('--ckpt_path', type=str, default='latency_dataset/ckpts/time.pt',
                        help='path to save latency predictor weights')
    parser.add_argument('--candidate_file', type=str, default='')
    parser.add_argument('--arch_perfs_file', type=str, default='')
    parser.add_argument('--output_file', type=str, default='cands/archs.txt')

    parser.add_argument('--feature_norm', type=float, nargs='+', default=[768, 12, 3072, 768],
                        help='normalizing factor for each feature')
    parser.add_argument('--lat_norm', type=float, default=200, help='normalizing factor for latency')
    parser.add_argument('--feature_dim', type=int, default=4, help='dimension of feature vector')
    parser.add_argument('--hidden_dim', type=int, default=2000, help='hidden dimension of '
                                                                     'FC layers in latency predictor')
    parser.add_argument('--hidden_layer_num', type=int, default=3, help='number of FC layers')

    # Arguments for Searching
    parser.add_argument('--latency_constraint', type=float, default=7)
    parser.add_argument('--method', type=str, default='Random')
    parser.add_argument('--model', type=str, default='MLM')
    parser.add_argument('--gen_size', type=int, default=25)

    parser.add_argument('--layer_num_space', nargs='+', type=int, default=[1, 8])
    parser.add_argument('--hidden_size_space', nargs='+', type=int, default=[128, 768])
    parser.add_argument('--qkv_size_space', nargs='+', type=int, default=[180, 768])
    parser.add_argument('--head_num_space', nargs='+', type=int, default=[1, 12])
    parser.add_argument('--intermediate_size_space', nargs='+', type=int, default=[128, 3072])

    args = parser.parse_args()
    print(args)

    assert args.method in ['Random', 'Fast', 'Evolved', 'Candidate'], 'method must be in [Random, Fast,' \
                                                                      ' Evolved, Candidate]'
    assert args.model in ['MLM', 'KD'], 'model must be in [MLM, KD]'

    predictor = LatencyPredictor(feature_norm=args.feature_norm,
                                 lat_norm=args.lat_norm, feature_dim=args.feature_dim,
                                 hidden_dim=args.hidden_dim, ckpt_path=args.ckpt_path)
    predictor.load_ckpt()

    # this latency is evaluated on Intel(R) Xeon(R) CPU E7-4850 v2 @ 2.30GHz
    # should be changed for specific hardware
    if args.model == 'KD':
        bert_base_lat = 1063
    elif args.model == 'MLM':
        bert_base_lat = 886

    latency = bert_base_lat / args.latency_constraint

    # you can design different upper and lower value in here
    latency_min, latency_max = 0.85 * latency, 1.1 * latency

    candidates = []
    fast_candidates = []

    search_space = generate_search_space(args)
    layer_numbers, hidden_sizes, ffn_sizes, qkv_sizes, qkv_sizes_mlm, head_sizes = search_space

    # Get the candidates
    if args.method == 'Candidate':
        assert args.candidate_file, 'candidate file must be set!'
        for layer_num in layer_numbers:
            config = dict()
            config['sample_layer_num'] = layer_num

            if args.model == 'KD':
                config['sample_num_attention_heads'] = [12] * layer_num

                for hidden_size in hidden_sizes:
                    config['sample_hidden_size'] = hidden_size

                    for ffn_size in ffn_sizes:
                        config['sample_intermediate_sizes'] = [ffn_size] * layer_num

                        for qkv_size in qkv_sizes:
                            config['sample_qkv_sizes'] = [qkv_size] * layer_num
                            lat_ = predictor.predict_lat(config)

                            if latency_min <= lat_ <= latency_max:
                                candidates.append(dict(config))

            else:
                for head_size in head_sizes:
                    config['sample_num_attention_heads'] = [head_size] * layer_num
                    config['sample_qkv_sizes'] = [head_size * 64] * layer_num

                    for hidden_size in hidden_sizes:
                        config['sample_hidden_size'] = hidden_size

                        for ffn_size in ffn_sizes:
                            config['sample_intermediate_sizes'] = [ffn_size] * layer_num
                            lat_ = predictor.predict_lat(config)

                            if latency_min <= lat_ <= latency_max:
                                candidates.append(dict(config))

        with open(args.candidate_file, 'w') as fout:
            for cand in candidates:
                fout.write(str(cand) + '\n')

        print('Size of candidates: {}'.format(len(candidates)))
        exit()
    else:
        def load_candidates(file_name):
            candidates_ = []
            with open(file_name, 'r') as fin:
                for line in fin:
                    line = line.strip()
                    candidate_ = json.loads(json.dumps(eval(line)))
                    candidates_.append(candidate_)
            return candidates_

        candidates = load_candidates(args.candidate_file)

    print('Size of candidates: {}'.format(len(candidates)))
    for candidate in candidates:
        if is_efficient_arch(candidate):
            fast_candidates.append(dict(candidate))
    print('Size of fast candidates: {}'.format(len(fast_candidates)))

    if args.method == 'Random':
        import random
        with open(args.output_file, 'w') as fout:
            cand_arches = random.sample(candidates, args.gen_size)
            for cand in cand_arches:
                fout.write(str(cand) + '\n')
    elif args.method == 'Fast':
        import random
        with open(args.output_file, 'w') as fout:
            cand_arches = random.sample(fast_candidates, args.gen_size)
            for cand in cand_arches:
                fout.write(str(cand) + '\n')
    elif args.method == 'Evolved':
        assert args.candidate_file
        all_arches = load_arch_perfs(args.arch_perfs_file)
        evolver = Evolver(args, all_arches=all_arches,
                          candidates=candidates,
                          latency_predictor=predictor,
                          search_space=search_space,
                          latency_scale=[latency_min, latency_max])
        cand_arches = evolver.generate_next_generation()
        with open(args.output_file, 'w') as fout:
            for cand in cand_arches:
                fout.write(str(cand) + '\n')
    else:
        raise NotImplementedError

