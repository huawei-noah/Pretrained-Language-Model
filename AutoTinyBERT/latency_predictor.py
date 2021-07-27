# HAT: Hardware-Aware Transformers for Efficient Natural Language Processing
# Hanrui Wang, Zhanghao Wu, Zhijian Liu, Han Cai, Ligeng Zhu, Chuang Gan and Song Han
# The 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020.
# Paper: https://arxiv.org/abs/2005.14187
# Project page: https://hanruiwang.me/project_pages/hat/

import random
import argparse
import numpy as np
import json

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, feature_dim, hidden_dim, hidden_layer_num):
        super(Net, self).__init__()

        self.first_layer = nn.Linear(feature_dim, hidden_dim)

        self.layers = nn.ModuleList()

        for i in range(hidden_layer_num):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.predict = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.first_layer(x))

        for i in range(len(self.layers)):
            x = F.relu(self.layers[i](x))

        x = self.predict(x)

        return x


class LatencyPredictor(object):
    def __init__(self, feature_norm, lat_norm, ckpt_path, lat_dataset_path='./latency_dataset/lat.tmp', feature_dim=10,
                 hidden_dim=400, hidden_layer_num=3, train_steps=5000, bsz=128, lr=1e-5):
        self.dataset_path = lat_dataset_path
        self.feature_norm = np.array(feature_norm)
        self.lat_norm = lat_norm
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.hidden_layer_num = hidden_layer_num
        self.ckpt_path = ckpt_path

        self.dataset = None

        self.train_x = None
        self.train_y = None

        self.valid_x = None
        self.valid_y = None

        self.test_x = None
        self.test_y = None

        self.model = Net(self.feature_dim, self.hidden_dim, self.hidden_layer_num)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

        self.train_steps = train_steps
        self.bsz = bsz

    def train(self):
        for i in range(self.train_steps):
            sample_ind = random.sample(range(len(self.train_x)), k=self.bsz)
            sample_x = [self.train_x[sample_ind[k]] for k in range(self.bsz)]
            sample_y = [self.train_y[sample_ind[k]] for k in range(self.bsz)]

            sample_x_tensor = torch.Tensor(sample_x)
            sample_y_tensor = torch.Tensor(sample_y)

            prediction = self.model(sample_x_tensor).squeeze()

            loss = self.criterion(prediction, sample_y_tensor)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # validation
            if i % 100 == 0:
                with torch.no_grad():
                    sample_x_tensor = torch.Tensor(self.valid_x)
                    sample_y_tensor = torch.Tensor(self.valid_y)

                    prediction = self.model(sample_x_tensor).squeeze()
                    loss = self.criterion(prediction, sample_y_tensor)
                    print(f"Validation loss at {i} steps: {loss}")

        # test
        with torch.no_grad():
            sample_x_tensor = torch.Tensor(self.test_x)
            sample_y_tensor = torch.Tensor(self.test_y)
            prediction = self.model(sample_x_tensor).squeeze()
            loss = self.criterion(prediction, sample_y_tensor)
            print(f"Predicted latency: {prediction}")
            print(f"Real latency: {self.test_y}")
            print(f"Loss: {loss}")

            print(f"RMSE: {np.sqrt(self.criterion(self.lat_norm*sample_y_tensor, self.lat_norm*prediction))}")
            print(f"MAPD: {torch.mean(torch.abs((sample_y_tensor - prediction) / sample_y_tensor))}")

        torch.save(self.model.state_dict(), self.ckpt_path)

    def load_ckpt(self):
        self.model.load_state_dict(torch.load(self.ckpt_path))

    def predict_lat(self, config):
        with torch.no_grad():
            def config_2_feature(config):
                features = []
                features.append(config['sample_hidden_size'])
                features.append(config['sample_layer_num'])
                features.append(sum(config['sample_intermediate_sizes']) /
                                (1.0 * len(config['sample_intermediate_sizes'])))
                features.append(sum(config['sample_qkv_sizes']) / (1.0 * len(config['sample_qkv_sizes'])))
                return features

            features = config_2_feature(config)
            features_norm = np.array(features) / self.feature_norm

            prediction = self.model(torch.Tensor(features_norm)).item() * self.lat_norm

        return prediction

    def split(self):
        sample_num = len(self.dataset['x'])
        train_num = int(np.floor(0.8 * sample_num))
        valid_num = int(np.floor(0.1 * sample_num))

        self.train_x = self.dataset['x'][:train_num]
        self.train_y = self.dataset['y'][:train_num]

        self.valid_x = self.dataset['x'][train_num:(train_num+valid_num)]
        self.valid_y = self.dataset['y'][train_num:(train_num+valid_num)]

        self.test_x = self.dataset['x'][(train_num+valid_num):]
        self.test_y = self.dataset['y'][(train_num+valid_num):]

    def read_dataset(self):
        features_norm_all = []
        lats_all = []
        cnt = 0
        with open(self.dataset_path, 'r') as fid:
            # next(fid) # skip first line of CSV
            for line in fid:
                line = line.strip()

                try:
                    subbert_config, inf_time = line.split('\t')
                    subbert_config = json.loads(json.dumps(eval(subbert_config)))
                except:
                    print('Got error! when parsing {}!'.format(line))

                def config_2_feature(config):
                    features = []
                    features.append(config['sample_hidden_size'])
                    features.append(config['sample_layer_num'])
                    features.append(sum(config['sample_intermediate_sizes']) /
                                    (1.0 * len(config['sample_intermediate_sizes'])))
                    features.append(sum(config['sample_qkv_sizes']) / (1.0 * len(config['sample_qkv_sizes'])))
                    return features

                features_eval = config_2_feature(subbert_config)
                features_norm = np.array(features_eval) / self.feature_norm
                features_norm_all.append(features_norm)
                lats_all.append(float(inf_time) / self.lat_norm)

                cnt += 1
                if cnt % 100000 == 0:
                    print('Loaded {} structures!'.format(cnt))

        tmp = list(zip(features_norm_all, lats_all))
        random.shuffle(tmp)
        features_norm_all, lats_all = zip(*tmp)
        self.dataset = {'x': features_norm_all, 'y': lats_all}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lat_dataset_path', type=str, default='./latency_dataset/ckpts/tmp.pt',
                        help='the path to read latency dataset')
    parser.add_argument('--ckpt_path', type=str, default='latency_dataset/ckpts/time.pt',
                        help='path to save latency predictor weights')
    parser.add_argument('--feature_norm', type=float, nargs='+', default=[768, 12, 3072, 768],
                        help='normalizing factor for each feature')
    parser.add_argument('--lat_norm', type=float, default=200, help='normalizing factor for latency')
    parser.add_argument('--feature_dim', type=int, default=4, help='dimension of feature vector')
    parser.add_argument('--hidden_dim', type=int, default=2000, help='hidden dimension of FC layers in latency predictor')
    parser.add_argument('--hidden_layer_num', type=int, default=3, help='number of FC layers')

    parser.add_argument('--train_steps', type=int, default=5000, help='latency predictor training steps')
    parser.add_argument('--bsz', type=int, default=128, help='latency predictor training batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='latency predictor training learning rate')

    # Arguments for getting candidates according to the latency constraint.
    parser.add_argument('--get_candidates', action='store_true')
    parser.add_argument('--model', type=str, default='MLM')
    parser.add_argument('--candidate_file', type=str, default='')
    parser.add_argument('--latency_constraint', type=float, default=7)
    parser.add_argument('--layer_num_space', nargs='+', type=int, default=[1, 8])
    parser.add_argument('--hidden_size_space', nargs='+', type=int, default=[128, 768])
    parser.add_argument('--qkv_size_space', nargs='+', type=int, default=[180, 768])
    parser.add_argument('--head_num_space', nargs='+', type=int, default=[1, 12])
    parser.add_argument('--intermediate_size_space', nargs='+', type=int, default=[128, 3072])

    args = parser.parse_args()
    print(args)

    assert args.get_candidates and args.candidate_file, 'get_candidates and candidate_file must be set simultaneously'
    assert args.model in ['MLM', 'KD']

    predictor = LatencyPredictor(lat_dataset_path=args.lat_dataset_path, feature_norm=args.feature_norm,
                                 lat_norm=args.lat_norm, feature_dim=args.feature_dim,
                                 hidden_dim=args.hidden_dim,
                                 hidden_layer_num=args.hidden_layer_num,
                                 ckpt_path=args.ckpt_path,
                                 train_steps=args.train_steps,
                                 bsz=args.bsz,
                                 lr=args.lr)

    if not args.get_candidates:
        predictor.read_dataset()
        predictor.split()
        predictor.train()
        print('Latency predictor training finished!\nThe model has been saved!')
    else:
        predictor.load_ckpt()

        bert_base_lat = 1063
        latency = bert_base_lat / args.latency_constraint
        latency_min, latency_max = 0.85 * latency, 1.1 * latency

        candidates = []
        fast_candidates = []

        # build arch space
        min_hidden_size, max_hidden_size = args.hidden_size_space
        min_ffn_size, max_ffn_size = args.intermediate_size_space
        min_qkv_size, max_qkv_size = args.qkv_size_space
        min_head_size, max_head_size = args.head_num_space

        # both hidden_step and ffn_step are set to 4 in original paper
        hidden_step = 32  # 8 # 4
        ffn_step = 64  # 16 # 4
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
        head_sizes = [i * head_step + min_head_size for i in range(number_head_step + 1)]

        # Get the candidates
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

        print('Size of candidates: {}'.format(len(candidates)))

        with open(args.candidate_file, 'w') as fout:
            for candidate in candidates:
                fout.write(json.dumps(candidate) + '\n')


    # configs.append({'sample_layer_num': 4, 'sample_num_attention_heads': [12]*4, 'sample_hidden_size': 512,
    #                 'sample_intermediate_sizes': [2048]*4, 'sample_qkv_sizes': [516]*4})
    # configs.append({'sample_layer_num': 5, 'sample_num_attention_heads': [12]*5, 'sample_hidden_size': 564,
    #                 'sample_intermediate_sizes': [1024]*5, 'sample_qkv_sizes': [528]*5})
    # configs.append({'sample_layer_num': 4, 'sample_num_attention_heads': [12]*4, 'sample_hidden_size': 312,
    #                 'sample_intermediate_sizes': [1200]*4, 'sample_qkv_sizes': [312]*4})
    # configs.append({'sample_layer_num': 5, 'sample_num_attention_heads': [12]*5, 'sample_hidden_size': 324,
    #                 'sample_intermediate_sizes': [600]*5, 'sample_qkv_sizes': [324]*5})
    # configs.append({'sample_layer_num': 4, 'sample_num_attention_heads': [12]*4, 'sample_hidden_size': 264,
    #                 'sample_intermediate_sizes': [1056]*4, 'sample_qkv_sizes': [264]*4})
    # configs.append({'sample_layer_num': 5, 'sample_num_attention_heads': [12]*5, 'sample_hidden_size': 280,
    #                 'sample_intermediate_sizes': [512]*5, 'sample_qkv_sizes': [276]*5})
    # configs.append({'sample_layer_num': 4, 'sample_num_attention_heads': [12]*4, 'sample_hidden_size': 192,
    #                 'sample_intermediate_sizes': [768]*4, 'sample_qkv_sizes': [192]*4})
    # configs.append({'sample_layer_num': 4, 'sample_num_attention_heads': [12]*4, 'sample_hidden_size': 256,
    #                 'sample_intermediate_sizes': [480]*4, 'sample_qkv_sizes': [192]*4})

    # configs.append({'sample_layer_num': 12, 'sample_num_attention_heads': [12] * 12, 'sample_hidden_size': 768,
    #                 'sample_intermediate_sizes': [3072] * 12, 'sample_qkv_sizes': [768] * 12})
    # configs.append({'sample_layer_num': 4, 'sample_num_attention_heads': [8]*4, 'sample_hidden_size': 512,
    #                 'sample_intermediate_sizes': [2048]*4, 'sample_qkv_sizes': [512]*4})
    # configs.append({'sample_layer_num': 5, 'sample_num_attention_heads': [8]*5, 'sample_hidden_size': 564,
    #                 'sample_intermediate_sizes': [1054]*5, 'sample_qkv_sizes': [512]*5})
    # configs.append({'sample_layer_num': 4, 'sample_num_attention_heads': [5]*4, 'sample_hidden_size': 320,
    #                 'sample_intermediate_sizes': [1280]*4, 'sample_qkv_sizes': [320]*4})
    # configs.append({'sample_layer_num': 4, 'sample_num_attention_heads': [6]*4, 'sample_hidden_size': 396,
    #                 'sample_intermediate_sizes': [624]*4, 'sample_qkv_sizes': [384]*4})
    # configs.append({'sample_layer_num': 4, 'sample_num_attention_heads': [4]*4, 'sample_hidden_size': 256,
    #                 'sample_intermediate_sizes': [1024]*4, 'sample_qkv_sizes': [256]*4})
    # configs.append({'sample_layer_num': 4, 'sample_num_attention_heads': [4]*4, 'sample_hidden_size': 432,
    #                 'sample_intermediate_sizes': [384]*4, 'sample_qkv_sizes': [256]*4})
    # configs.append({'sample_layer_num': 4, 'sample_num_attention_heads': [3]*4, 'sample_hidden_size': 192,
    #                 'sample_intermediate_sizes': [768]*4, 'sample_qkv_sizes': [192]*4})
    # configs.append({'sample_layer_num': 3, 'sample_num_attention_heads': [4]*3, 'sample_hidden_size': 320,
    #                 'sample_intermediate_sizes': [608]*3, 'sample_qkv_sizes': [256]*3})
    #
    # for config in configs:
    #     print(f'Example config: {config}')
    #     print(f'Example latency: {predictor.predict_lat(config)}')

