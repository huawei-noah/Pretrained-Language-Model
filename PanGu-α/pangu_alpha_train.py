"""
PanGu train script
"""


import os
import math
from pathlib2 import Path
from mindspore import context
from mindspore.train.model import Model
import mindspore.communication.management as D
from mindspore.context import ParallelMode
import mindspore.nn as nn
from mindspore.train.callback import TimeMonitor, ModelCheckpoint, CheckpointConfig, Callback
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
import mindspore.common.dtype as mstype
from mindspore.parallel._cost_model_context import _set_multi_subgraphs
from mindspore.parallel import set_algo_parameters
import mindspore.dataset as de
from dataset import create_dataset, create_dataset_dp
from pangu_alpha import PANGUALPHAPipeline, PANGUALPHA, PANGUALPHAWithLossPipeline, PANGUALPHAWithLoss, CrossEntropyLoss
from pangu_alpha_wrapcell import PANGUALPHATrainPipelineWithLossScaleCell, PANGUALPHATrainOneStepWithLossScaleCell, \
    VirtualDatasetOneInputCell
from utils import LearningRate
from pangu_alpha_config import PANGUALPHAConfig, set_parse

class LossCallBack(Callback):
    """
    Monitor the loss in training.
    """
    def __init__(self, dataset_size=-1, local_rank=0, scale=1):
        super(LossCallBack, self).__init__()
        self._dataset_size = dataset_size
        self.local_rank = local_rank
        self.scale = scale

    def step_end(self, run_context):
        """
        Print loss after each step
        """
        cb_params = run_context.original_args()
        # NOTE: We send the data after sending twice sink size
        # where sink size is equal to the dataset_size (a fake one) here
        de.config.set_sending_batches(cb_params.cur_step_num + 2*self._dataset_size)
        if self._dataset_size > 0 and self.local_rank % 8 == 0:
            percent, epoch_num = math.modf(cb_params.cur_step_num /
                                           self._dataset_size)
            if percent == 0:
                percent = 1
                epoch_num -= 1
            print(
                "local_rank: {}, epoch: {}, step: {}, output is {}, overflow is {}, scale is {}"
                    .format(int(self.local_rank), int(epoch_num),
                            cb_params.cur_step_num,
                            cb_params.net_outputs[0].asnumpy() / self.scale,
                            cb_params.net_outputs[1].asnumpy(),
                            cb_params.net_outputs[2].asnumpy()))
            if len(cb_params.net_outputs) > 3:
                print("global norm is: ", cb_params.net_outputs[3].asnumpy())


def run_train_pipeline(args_opt):
    device_id = int(os.getenv("DEVICE_ID"))
    rank_id = int(os.getenv("RANK_ID"))
    local_rank = rank_id
    print('local_rank:{}, device id:{} start to run...'.format(
        local_rank, device_id),
        flush=True)
    context.set_context(save_graphs=False,
                        mode=context.GRAPH_MODE,
                        device_target="Ascend",
                        device_id=device_id)
    context.set_context(variable_memory_max_size="31GB")
    strategy_ckpt_save_file = "/cache/" + "strategy" + str(local_rank) + ".ckpt"
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
            enable_parallel_optimizer=bool(args_opt.optimizer_shard),
            pipeline_stages=args_opt.stage_num,
            strategy_ckpt_save_file=strategy_ckpt_save_file)
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
        dropout_rate=0.1,
        compute_dtype=mstype.float16,
        use_past=False,
        self_layernorm=True,
        forward_reduce_scatter=True,
        stage_num=args_opt.stage_num,
        micro_size=args_opt.micro_size,
        word_emb_dp=False)
    print("===config is: ", config, flush=True)
    pangu_alpha = PANGUALPHAPipeline(config)
    loss = CrossEntropyLoss(config)
    pangu_alpha_with_loss = PANGUALPHAWithLossPipeline(config, pangu_alpha, loss)
    pangu_alpha_with_loss = VirtualDatasetOneInputCell(pangu_alpha_with_loss)

    print("=====args_opt is: ", args_opt, flush=True)
    lr = LearningRate(learning_rate=args_opt.start_lr,
                      end_learning_rate=args_opt.end_lr,
                      warmup_steps=args_opt.warmup_step,
                      decay_steps=args_opt.decay_steps)

    per_stage_layers = config.num_layers // config.stage_num
    per_stage_devices = device_num // config.stage_num
    self_stage = rank_id // per_stage_devices
    range_min = self_stage * per_stage_layers
    range_max = range_min + per_stage_layers
    if self_stage == 0:
        params = [pangu_alpha.embedding_table]
        params.extend(pangu_alpha.backbone.pangu_alpha_embedding.position_embedding.trainable_params())
    elif self_stage == config.stage_num - 1:
        params = [pangu_alpha.embedding_table]
        params.extend(pangu_alpha.backbone.layernorm.trainable_params())
        params.extend(pangu_alpha.backbone.top_query_embedding.trainable_params())
    else:
        params = []
    for i in range(range_min, range_max):
        params.extend(pangu_alpha.backbone.blocks[i].trainable_params())

    decay_filter = lambda x: 'layernorm' not in x.name.lower() and "bias" not in x.name.lower()

    decay_params = list(filter(decay_filter, params))
    other_params = list(filter(lambda x: not decay_filter(x), params))
    group_params = [{
        'params': decay_params,
        'weight_decay': args_opt.weight_decay
    }, {
        'params': other_params,
        'weight_decay': 0.0
    }, {
        'order_params': params
    }]
    if args_opt.optimizer == "lamb":
        optimizer = nn.Lamb(group_params, learning_rate=lr)
    else:
        optimizer = nn.AdamWeightDecay(group_params, learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)

    save_steps = args_opt.save_steps
    ckpt_dir = os.path.join(args_opt.ckpt_save_sir, f"rank_{str(local_rank)}")
    if not os.path.exists(ckpt_dir):
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    ds = create_dataset(config.batch_size, data_path=args_opt.data_url, data_start_index=0)
    
    epoch_num = args_opt.epoch_size
    step_per_epoch = ds.get_dataset_size()
    callback_size = args_opt.sink_size
    actual_epoch_num = int(epoch_num * step_per_epoch / callback_size)
    callback = [
        TimeMonitor(callback_size),
        LossCallBack(callback_size, local_rank, config.stage_num)
    ]
    config_ck = CheckpointConfig(save_checkpoint_steps=save_steps,
                                 keep_checkpoint_max=1,
                                 integrated_save=False,
                                 filter_prefix="accu_grads")
    ckpoint_cb = ModelCheckpoint(prefix="PanguAlpha",
                                 directory=ckpt_dir,
                                 config=config_ck)
    callback.append(ckpoint_cb)
    loss_scale_value = math.pow(2, 32)
    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=loss_scale_value,
                                             scale_factor=2,
                                             scale_window=1000)

    pangu_alpha_with_grads = PANGUALPHATrainPipelineWithLossScaleCell(
        pangu_alpha_with_loss, optimizer=optimizer, config=config, scale_update_cell=update_cell)

    model = Model(pangu_alpha_with_grads)
    de.config.set_sending_batches(2*args_opt.sink_size)
    model.train(actual_epoch_num,
                ds,
                callbacks=callback,
                sink_size=callback_size,
                dataset_sink_mode=True)

def run_train_no_pipeline(args_opt):

    device_id = int(os.getenv("DEVICE_ID"))
    rank_id = int(os.getenv("RANK_ID"))
    local_rank = rank_id
    print('local_rank:{}, device id:{} start to run...'.format(
        local_rank, device_id),
        flush=True)
    save_graphs_path = "/var/log/npu/slog/device-" + str(local_rank) + "/"
    context.set_context(save_graphs=False,
                        save_graphs_path=save_graphs_path,
                        mode=context.GRAPH_MODE,
                        device_target="Ascend",
                        device_id=device_id)
    context.set_context(variable_memory_max_size="31GB")
    strategy_ckpt_save_file = "/cache/" + "strategy" + str(local_rank) + ".ckpt"
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
            full_batch=False,
            loss_repeated_mean=True,
            enable_parallel_optimizer=bool(args_opt.optimizer_shard),
            pipeline_stages=args_opt.stage_num,
            strategy_ckpt_save_file=strategy_ckpt_save_file)
        set_algo_parameters(elementwise_op_strategy_follow=True)
        _set_multi_subgraphs()
    else:
        rank = 0
        device_num = 1

    model_parallel_num = args_opt.tensor_model_parallel_num
    data_parallel_num = int(device_num / model_parallel_num)
    per_batch_size = args_opt.per_batch_size
    batch_size = per_batch_size * device_num
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
        dropout_rate=0.1,
        compute_dtype=mstype.float16,
        use_past=False,
        self_layernorm=True,
        forward_reduce_scatter=True,
        stage_num=args_opt.stage_num,
        micro_size=args_opt.micro_size,
        word_emb_dp=True)
    print("===config is: ", config, flush=True)
    pangu_alpha = PANGUALPHA(config)
    loss = CrossEntropyLoss(config)
    pangu_alpha_with_loss = PANGUALPHAWithLoss(config, pangu_alpha, loss)
    pangu_alpha_with_loss = VirtualDatasetOneInputCell(pangu_alpha_with_loss)

    print("=====args_opt is: ", args_opt, flush=True)
    lr = LearningRate(learning_rate=args_opt.start_lr,
                      end_learning_rate=args_opt.end_lr,
                      warmup_steps=args_opt.warmup_step,
                      decay_steps=args_opt.decay_steps)

    decay_filter = lambda x: 'layernorm' not in x.name.lower() and "bias" not in x.name.lower()
    params = pangu_alpha.trainable_params()
    decay_params = list(filter(decay_filter, params))
    other_params = list(filter(lambda x: not decay_filter(x), params))
    group_params = [{
        'params': decay_params,
        'weight_decay': args_opt.weight_decay
    }, {
        'params': other_params,
        'weight_decay': 0.0
    }, {
        'order_params': params
    }]
    if args_opt.optimizer == "lamb":
        optimizer = nn.Lamb(group_params, learning_rate=lr)
    else:
        optimizer = nn.AdamWeightDecay(group_params, learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)

    save_steps = args_opt.save_steps
    ckpt_dir = os.path.join(args_opt.ckpt_save_sir, f"rank_{str(local_rank)}")
    if not os.path.exists(ckpt_dir):
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    ds = create_dataset_dp(config.batch_size, data_path=args_opt.data_url, data_start_index=0, device_num=device_num, rank=rank)
    epoch_num = args_opt.epoch_size
    step_per_epoch = ds.get_dataset_size()
    callback_size = args_opt.sink_size
    actual_epoch_num = int(epoch_num * step_per_epoch / callback_size)
    callback = [
        TimeMonitor(callback_size),
        LossCallBack(callback_size, local_rank)
    ]
    config_ck = CheckpointConfig(save_checkpoint_steps=save_steps,
                                 keep_checkpoint_max=1,
                                 integrated_save=False)
    ckpoint_cb = ModelCheckpoint(prefix="PanguAlpha",
                                 directory=ckpt_dir,
                                 config=config_ck)
    callback.append(ckpoint_cb)
    loss_scale_value = math.pow(2, 32)
    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=loss_scale_value,
                                             scale_factor=2,
                                             scale_window=1000)

    pangu_alpha_with_grads = PANGUALPHATrainOneStepWithLossScaleCell(
        pangu_alpha_with_loss, optimizer=optimizer, config=config, scale_update_cell=update_cell)

    model = Model(pangu_alpha_with_grads)
    de.config.set_sending_batches(2*args_opt.sink_size)
    model.train(actual_epoch_num,
                ds,
                callbacks=callback,
                sink_size=callback_size,
                dataset_sink_mode=True)

def run_train(args_opt):
    if args_opt.stage_num > 1:
        run_train_pipeline(args_opt)
    else:
        run_train_no_pipeline(args_opt)



