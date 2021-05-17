"""
network config setting, gradient clip function and dynamic learning rate function
"""

import numpy as np
from multiprocessing import Process
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.nn.learning_rate_schedule import LearningRateSchedule, PolynomialDecayLR, WarmUpLR, CosineDecayLR
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_rank, get_group_size, create_group

class PANGUALPHAConfig:
    """
    PANGUALPHA config class which defines the model size
    """
    def __init__(self,
                 data_parallel_num,
                 model_parallel_num,
                 batch_size=32,
                 seq_length=1024,
                 vocab_size=50257,
                 embedding_size=768,
                 num_layers=12,
                 num_heads=12,
                 expand_ratio=4,
                 post_layernorm_residual=False,
                 dropout_rate=0.1,
                 compute_dtype=mstype.float16,
                 use_past=False,
                 self_layernorm=True,
                 forward_reduce_scatter=True,
                 word_emb_dp=True,
                 stage_num=16,
                 micro_size=32):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.expand_ratio = expand_ratio
        self.post_layernorm_residual = post_layernorm_residual
        self.dropout_rate = dropout_rate
        self.compute_dtype = compute_dtype
        self.use_past = use_past
        self.dp = data_parallel_num
        self.mp = model_parallel_num
        self.self_layernorm = self_layernorm
        self.forward_reduce_scatter = forward_reduce_scatter
        self.stage_num = stage_num
        self.micro_size = micro_size
        self.word_emb_dp = word_emb_dp

    def __str__(self):
        info = "[PANGUALPHAConfig]" + '===' * 10 + '\n'
        for k, v in self.__dict__.items():
            var_info = "{}:{}\n".format(k, v)
            info += var_info
        info += '=' * 10
        return info


get_square_sum = C.MultitypeFuncGraph("get_square_sum")


@get_square_sum.register("Tensor", "Number")
def _get_square_sum(grad, value):
    norm = P.ReduceSum(False)(F.square(grad), ()) / value
    norm = F.expand_dims(F.cast(norm, mstype.float32), 0)
    return norm

apply_global_norm = C.MultitypeFuncGraph("apply_global_norm")


@apply_global_norm.register("Tensor", "Tensor", "Tensor")
def _apply_global_norm(clip_norm, global_norm, grad):
    grad = grad * clip_norm / global_norm
    return grad


class GlobalNormPipline(nn.Cell):
    """
    Calculate the global norm value of given tensors
    """
    def __init__(self, params, config):
        super(GlobalNormPipline, self).__init__()
        self.norm = nn.Norm()
        self.hyper_map = C.HyperMap()
        self.allreduce_filter = tuple("projection.bias" not in x.name and "layernorm" not in x.name and "position_embedding.embedding_table" not in x.name for x in params)
        self.allreduce_group_size = ()
        for item in self.allreduce_filter:
            if item:
                self.allreduce_group_size = self.allreduce_group_size + (1.0, )
            else:
                self.allreduce_group_size = self.allreduce_group_size + (config.mp * 1.0, )
        self.length = len(params)
        group_list ,group_name = _get_model_parallel_group(config.mp)
        print("rank_list", group_name)
        print("group_size_list", self.allreduce_group_size)
        create_group(group_name, group_list)
        self.allreduce = P.AllReduce(group=group_name)
        pipeline_group_list, pipeline_group_name = _get_pipeline_group()
        print("pipeline_group_name", pipeline_group_name)
        create_group(pipeline_group_name, pipeline_group_list)
        self.allreduce2 = P.AllReduce(group=pipeline_group_name)

    def construct(self, grads):
        square_sum = self.hyper_map(get_square_sum, grads, self.allreduce_group_size)
        square_reduce_sum = F.addn(square_sum)
        stage_square_reduce_sum = self.allreduce(square_reduce_sum)
        global_square_reduce_sum = self.allreduce2(stage_square_reduce_sum)
        global_norms = F.sqrt(global_square_reduce_sum)
        return global_norms

class GlobalNorm(nn.Cell):
    """

    Calculate the global norm value of given tensors

    """
    def __init__(self, params, config):
        super(GlobalNorm, self).__init__()
        self.norm = nn.Norm()
        self.hyper_map = C.HyperMap()
        self.config = config
        self.allreduce_filter = tuple("projection.bias" not in x.name and "layernorm" not in x.name and "embedding_table"
                                      not in x.name for x in params)
        self.length = len(params)
        self.values = []
        self.group_size = get_group_size()
        for item in self.allreduce_filter:
            if item:
                self.values.append(1.0)
            else:
                self.values.append(self.group_size*1.0)
        self.values = tuple(self.values)
    def construct(self, grads):
        square_sum_dp = self.hyper_map(get_square_sum, grads, self.values)
        global_norms = F.sqrt(P.AllReduce()(F.addn(square_sum_dp)))
        return global_norms



class ClipByGlobalNorm(nn.Cell):
    """
    Clip grads by global norm
    """
    def __init__(self, params, config, clip_norm=1.0, pipeline=True):
        super(ClipByGlobalNorm, self).__init__()
        if pipeline:
            self.global_norm = GlobalNormPipline(params, config)
        else:
            self.global_norm = GlobalNorm(params, config)
        self.clip_norm = Tensor([clip_norm], mstype.float32)
        self.hyper_map = C.HyperMap()

    def construct(self, grads):
        global_norm_origin = self.global_norm(grads)
        cond = P.GreaterEqual()(global_norm_origin, self.clip_norm)
        global_norm = F.select(cond, global_norm_origin, self.clip_norm)
        grads = self.hyper_map(
            F.partial(apply_global_norm, self.clip_norm, global_norm), grads)
        return grads, global_norm_origin


def _get_model_parallel_group(mp):
    rank = get_rank()
    stage_nums = auto_parallel_context().get_pipeline_stages()
    device_nums = get_group_size()
    per_stage_device_nums = device_nums // stage_nums
    stage_id = rank // per_stage_device_nums
    local_stage_rank_id = rank % per_stage_device_nums
    index = local_stage_rank_id // mp
    group = range(0, mp)
    rank_str_list = [str(x + index * mp + stage_id * per_stage_device_nums)  for x in group]
    rank_list_str = "-".join(rank_str_list)
    rank_list = [x + index * mp + stage_id * per_stage_device_nums for x in group]
    return rank_list, rank_list_str

def _get_pipeline_group():
    rank = get_rank()
    stage_nums = auto_parallel_context().get_pipeline_stages()
    device_nums = get_group_size()
    per_stage_device_nums = device_nums // stage_nums
    stage_id = rank // per_stage_device_nums
    local_stage_rank_id = rank % per_stage_device_nums
    group = range(0, stage_nums)
    rank_list = [local_stage_rank_id + x * per_stage_device_nums for x in group]
    rank_str_list = [str(local_stage_rank_id + x * per_stage_device_nums) for x in group]
    rank_list_str = "-".join(rank_str_list)
    return rank_list, rank_list_str

class LearningRate(LearningRateSchedule):
    """
    Warmup-decay learning rate for PANGUALPHA network.
    """
    def __init__(self,
                 learning_rate,
                 end_learning_rate,
                 warmup_steps,
                 decay_steps,
                 power=1.0,
                 use_cosine=True):
        super(LearningRate, self).__init__()
        self.warmup_flag = False
        if warmup_steps > 0:
            self.warmup_flag = True
            self.warmup_lr = WarmUpLR(learning_rate, warmup_steps)
        self.decay_lr = PolynomialDecayLR(learning_rate, end_learning_rate,
                                          decay_steps, power)
        self.cosine_decay_lr = CosineDecayLR(end_learning_rate, learning_rate,
                                             decay_steps)
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))

        self.greater = P.Greater()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.cast = P.Cast()
        self.use_cosine = use_cosine

    def construct(self, global_step):
        """dynamic learning rate"""
        if not self.use_cosine:
            decay_lr = self.decay_lr(global_step)
        else:
            decay_lr = self.cosine_decay_lr(global_step)
        if self.warmup_flag:
            is_warmup = self.cast(self.greater(self.warmup_steps, global_step),
                                  mstype.float32)
            warmup_lr = self.warmup_lr(global_step)
            lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        else:
            lr = decay_lr
        return lr

