import random


def sample_arch_4_kd(layer_numbers, hidden_sizes, ffn_sizes, qkv_sizes,
                     reset_rand_seed=False, rand_seed=0):

    if reset_rand_seed:
        random.seed(rand_seed)

    config = dict()

    layer_num = random.choice(layer_numbers)

    config['sample_layer_num'] = layer_num
    config['sample_hidden_size'] = random.choice(hidden_sizes)
    config['sample_intermediate_sizes'] = [random.choice(ffn_sizes)] * layer_num
    config['sample_num_attention_heads'] = [12] * layer_num
    config['sample_qkv_sizes'] = [random.choice(qkv_sizes)] * layer_num
    return config


def sample_arch_4_mlm(layer_numbers, hidden_sizes, ffn_sizes,
                      head_numbers, reset_rand_seed=False, rand_seed=0):

    if reset_rand_seed:
        random.seed(rand_seed)

    config = dict()

    layer_num = random.choice(layer_numbers)
    head_num = random.choice(head_numbers)

    config['sample_layer_num'] = layer_num
    config['sample_hidden_size'] = random.choice(hidden_sizes)
    config['sample_intermediate_sizes'] = [random.choice(ffn_sizes)] * layer_num
    config['sample_num_attention_heads'] = [head_num] * layer_num
    config['sample_qkv_sizes'] = [head_num * 64] * layer_num
    return config

