"""
PANGUALPHA train script
"""


import os
import numpy as np
import time
from mindspore import context, Tensor
from mindspore.train.model import Model
import mindspore.communication.management as D
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net, load_distributed_checkpoint
import mindspore.common.dtype as mstype
from mindspore.parallel._cost_model_context import _set_multi_subgraphs
from mindspore.parallel import set_algo_parameters
from pangu_alpha import PANGUALPHAPipeline, PANGUALPHA, EvalNet
from pangu_alpha_config import PANGUALPHAConfig


def run_predict_pipeline(args_opt):
    device_id = int(os.getenv("DEVICE_ID"))
    rank_id_str = os.getenv('RANK_ID', '0')
    rank_id = int(
        rank_id_str[rank_id_str.rfind('-') +
                    1:])
    print('rank_id:{}'.format(rank_id), "rank_id str:{}".format(rank_id_str))
    device_id = int(os.getenv('DEVICE_ID'))
    local_rank = rank_id
    print('local_rank:{}, device id:{} start to run...'.format(
        local_rank, device_id),
        flush=True)
    context.set_context(save_graphs=False,
                        mode=context.GRAPH_MODE,
                        device_target="Ascend",
                        device_id=device_id)
    context.set_context(variable_memory_max_size="30GB")
    if args_opt.distribute == "true":
        D.init()
        device_num = D.get_group_size()
        rank = D.get_rank()
        print("device_id is {}, rank_id is {}, device_num is {}".format(
            device_id, rank, device_num))
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
            gradients_mean=False,
            device_num=device_num,
            full_batch=True,
            loss_repeated_mean=True,
            enable_parallel_optimizer=False,
            pipeline_stages=args_opt.stage_num)
        set_algo_parameters(elementwise_op_strategy_follow=True)
        _set_multi_subgraphs()

    else:
        rank = 0
        device_num = 1

    model_parallel_num = args_opt.tensor_model_parallel_num
    stage_device_num = int(device_num / args_opt.stage_num)
    data_parallel_num = int(stage_device_num / model_parallel_num)
    per_batch_size = args_opt.per_batch_size
    batch_size = per_batch_size * data_parallel_num * args_opt.micro_size
    config = PANGUALPHAConfig(
        data_parallel_num=data_parallel_num,
        model_parallel_num=model_parallel_num,
        batch_size=batch_size,
        seq_length=args_opt.seq_length,
        vocab_size=args_opt.vocab_size,
        embedding_size=args_opt.embedding_size,
        num_layers=args_opt.num_layers,
        num_heads=args_opt.num_heads,
        expand_ratio=4,
        post_layernorm_residual=False,
        dropout_rate=0.0,
        compute_dtype=mstype.float16,
        use_past=False,
        self_layernorm=True,
        forward_reduce_scatter=True,
        stage_num=args_opt.stage_num,
        micro_size=args_opt.micro_size,
        word_emb_dp=False)
    print("===config is: ", config, flush=True)
    print("=====args_opt is: ", args_opt, flush=True)

    per_stage_layers = config.num_layers // config.stage_num
    per_stage_devices = device_num // config.stage_num
    self_stage = rank_id // per_stage_devices

    # all cards will save ckpt
    train_stage_num = 16
    train_device_num = 1024
    train_mp = 16
    ckpt_name = args_opt.load_ckpt_name
    train_per_stage_num = train_device_num // train_stage_num
    if config.mp != train_mp:
        raise ValueError("the model parallel num is not equal to training model parallel num")
    concat_stage_num = train_stage_num // config.stage_num
    pangu_alpha = PANGUALPHAPipeline(config)
    eval_net = EvalNet(pangu_alpha)
    eval_net.set_train(False)
    model_predict = Model(eval_net)
    inputs_np = Tensor(np.ones(shape=(1, config.seq_length)), mstype.int32)
    model_predict.infer_predict_layout(inputs_np)
    print("======start load_distributed checkpoint", flush=True)
    for i in range(self_stage * concat_stage_num, (self_stage + 1) * concat_stage_num):
        stage_position = local_rank % (config.mp * config.dp)
        ckpt_rank = i * train_per_stage_num + stage_position  # 訓練時候的rank號
        ckpt_dir = os.path.join(args_opt.load_ckpt_path, f"rank_{(ckpt_rank)}")  # 命名還是以訓練時候的rank號命名
        local_ckpt_file = os.path.join(ckpt_dir, ckpt_name)
        if not os.path.exists(local_ckpt_file):
            raise ValueError("Ckpt file not exits,", local_ckpt_file)
        params_dict = load_checkpoint(local_ckpt_file, filter_prefix="adam")
        load_param_into_net(eval_net, params_dict)
    print("================load param ok=================", flush=True)
    # here predict with fake input
    model_predict.predict(inputs_np)


def run_predict_no_pipeline(args_opt):
    device_id = int(os.getenv("DEVICE_ID"))
    rank_id_str = os.getenv('RANK_ID', '0')
    rank_id = int(
        rank_id_str[rank_id_str.rfind('-') +
                    1:])
    print('rank_id:{}'.format(rank_id), "rank_id str:{}".format(rank_id_str))
    device_id = int(os.getenv('DEVICE_ID'))
    local_rank = rank_id
    print('local_rank:{}, device id:{} start to run...'.format(
        local_rank, device_id),
        flush=True)
    context.set_context(save_graphs=False,
                        mode=context.GRAPH_MODE,
                        device_target="Ascend",
                        device_id=device_id)
    context.set_context(variable_memory_max_size="30GB")
    if args_opt.distribute == "true":
        D.init()
        device_num = D.get_group_size()
        rank = D.get_rank()
        print("device_id is {}, rank_id is {}, device_num is {}".format(
            device_id, rank, device_num))
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
            gradients_mean=False,
            device_num=device_num,
            full_batch=True,
            loss_repeated_mean=True,
            enable_parallel_optimizer=False,
            strategy_ckpt_load_file=args_opt.strategy_load_ckpt_path,
            pipeline_stages=args_opt.stage_num)
        set_algo_parameters(elementwise_op_strategy_follow=True)
        _set_multi_subgraphs()

    else:
        rank = 0
        device_num = 1

    model_parallel_num = args_opt.tensor_model_parallel_num
    data_parallel_num = int(device_num / model_parallel_num)
    per_batch_size = args_opt.per_batch_size
    batch_size = per_batch_size * data_parallel_num
    config = PANGUALPHAConfig(
        data_parallel_num=data_parallel_num,
        model_parallel_num=model_parallel_num,
        batch_size=batch_size,
        seq_length=args_opt.seq_length,
        vocab_size=args_opt.vocab_size,
        embedding_size=args_opt.embedding_size,
        num_layers=args_opt.num_layers,
        num_heads=args_opt.num_heads,
        expand_ratio=4,
        post_layernorm_residual=False,
        dropout_rate=0.0,
        compute_dtype=mstype.float16,
        use_past=False,
        self_layernorm=True,
        forward_reduce_scatter=True,
        stage_num=args_opt.stage_num,
        micro_size=args_opt.micro_size,
        eod_reset=False,
        word_emb_dp=True,
        load_ckpt_path=args_opt.load_ckpt_path)
    print("===config is: ", config, flush=True)
    print("=====args_opt is: ", args_opt, flush=True)

    ckpt_name = args_opt.load_ckpt_name
    pangu_alpha = PANGUALPHA(config)
    eval_net = EvalNet(pangu_alpha)
    eval_net.set_train(False)
    model_predict = Model(eval_net)
    inputs_np = Tensor(np.ones(shape=(config.batch_size, config.seq_length)), mstype.int32)
    predict_layout = model_predict.infer_predict_layout(inputs_np)
    print("======start load_distributed checkpoint", flush=True)
    # For 2.6B and 13B models, the number of ckpt files is 512.
    
    ckpt_name = 'filerted'
    ckpt_file_list = [os.path.join(args_opt.load_ckpt_path, f"{ckpt_name}_{ckpt_rank}.ckpt") for ckpt_rank in range(0, 512)]
    print(f"Loading from path {ckpt_file_list[0]}", flush=True)
    load_distributed_checkpoint(eval_net, ckpt_file_list, predict_layout)
    print("================load param ok=================", flush=True)

    from tokenization_jieba import JIEBATokenizer
    from generate import generate
    tokenizer = JIEBATokenizer(os.path.join(args_opt.tokenizer_path, 'vocab.vocab'),
                               os.path.join(args_opt.tokenizer_path, 'vocab.model'))

    sample = "今天是一个好天气"
    tokenized_token = tokenizer.tokenize(sample)
    start_sentence = tokenizer.convert_tokens_to_ids(tokenized_token)
    input_ids = np.array(start_sentence).reshape(1, -1)
    output_ids = generate(model_predict, input_ids, config.seq_length, 9)
    output_samples = tokenizer.convert_ids_to_tokens(output_ids.tolist())
    print('Output is:', output_samples, flush=True)


def run_predict(args_opt):
    if args_opt.stage_num > 1:
        run_predict_pipeline(args_opt)
    else:
        run_predict_no_pipeline(args_opt)

