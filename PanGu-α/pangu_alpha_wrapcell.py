"""PANGUALPHA training wrapper"""

import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.communication.management import get_group_size, create_group
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore.common.parameter import Parameter
from mindspore.ops.operations.comm_ops import _VirtualDataset
from utils import ClipByGlobalNorm


GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0
clip_grad = C.MultitypeFuncGraph("clip_grad")

@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """
    Clip gradients.

    Inputs:
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor], clipped gradients.
    """
    if clip_type not in [0, 1]:
        return grad
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = C.clip_by_value(grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
                                   F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad

grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()
update_accu_grads = C.MultitypeFuncGraph("update_accu_grads")


@update_accu_grads.register("Tensor", "Tensor")
def _update_accu_grads(accu_grad, grad):
    return F.depend(accu_grad, grad)



@grad_scale.register("Tensor", "Tensor", "Tensor")
def tensor_grad_scale(scale, grad, accu_grad):
    #mul = P.Mul()
    new_grad = accu_grad * reciprocal(scale)
    zeros = F.tensor_mul(accu_grad, 0.0)
    clear = F.assign(accu_grad, zeros)
    F.control_depend(new_grad, clear)
    F.control_depend(grad, new_grad)
    return new_grad

@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * reciprocal(scale)

class VirtualDatasetOneInputCell(nn.Cell):
    def __init__(self, backbone):
        super(VirtualDatasetOneInputCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._virtual_dataset = _VirtualDataset()

    def construct(self, *data):
        data_ = self._virtual_dataset(*data)
        return self._backbone(*data_)


class PANGUALPHATrainPipelineWithLossScaleCell(nn.Cell):
    """
    Encapsulation class of PANGUALPHA network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
    """
    def __init__(self, network, optimizer, config, scale_update_cell=None, enable_global_norm=True):
        super(PANGUALPHATrainPipelineWithLossScaleCell, self).__init__(auto_prefix=False)
        self.config = config
        self.network = network
        self.network.add_flags(defer_inline=True)
        self.weights = optimizer.parameters
        self.accu_grads = self.weights.clone(prefix="accu_grads", init="zeros")
        self.optimizer = optimizer
        self.enable_global_norm = enable_global_norm
        self.grad = C.GradOperation(get_by_list=True,
                                    sens_param=True)
        self.reducer_flag = False
        self.allreduce = P.AllReduce()
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = F.identity
        self.degree = 1
        if self.reducer_flag:
            self.degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, False, self.degree)
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.cast = P.Cast()
        self.alloc_status = P.NPUAllocFloatStatus(False)
        self.get_status = P.NPUGetFloatStatus(False)
        self.clear_before_grad = P.NPUClearFloatStatus(False)
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.depend_parameter_use = P.ControlDepend(depend_mode=1)
        self.base = Tensor(1, mstype.float32)
        self.less_equal = P.LessEqual()
        self.hyper_map = C.HyperMap()
        self.loss_scale = None
        self.reshape = P.Reshape()
        self.control = P.ControlDepend(1)
        self.clip_norm = Tensor(1000.0, mstype.float32)
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32),
                                        name="loss_scale")
        self.clip = ClipByGlobalNorm(self.weights, self.config)
        self.micro_size = config.micro_size

    @C.add_flags(has_effect=True)
    def construct(self,
                  input_ids,
                  input_position,
                  attention_mask,
                  past=None,
                  sens=None):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(input_ids, input_position, attention_mask)
        if sens is None:
            scaling_sens = self.loss_scale
            scaling_sens = self.reshape(scaling_sens, (1,))
        else:
            scaling_sens = sens
        # alloc status and clear should be right before gradoperation
        init = self.alloc_status()
        status_clear = self.clear_before_grad(init)
        #clear_depend = self.control(status_clear, self.weights)
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_position,
                                                 attention_mask,
                                                 self.cast(scaling_sens / self.micro_size,
                                                           mstype.float32))
        get_status = self.get_status(init)
        get_status_depend = F.control_depend(grads, get_status)
        flag_sum = self.reduce_sum(init, (0,))
        flag_sum_depend = F.control_depend(get_status, flag_sum)
        loss = F.depend(loss, status_clear)
        loss = F.depend(loss, get_status_depend)
        loss = F.depend(loss, flag_sum_depend)
        # apply grad reducer on grads
        accu_grads = self.grad_reducer(self.accu_grads)
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens * self.degree), grads, accu_grads)

        grads, global_norms = self.clip(grads)
        global_norm = P.Reshape()(global_norms, (()))
        if self.is_distributed:
            # sum overflow flag over devices
            flag_reduce = self.allreduce(flag_sum)
            cond = self.less_equal(self.base, flag_reduce)
        else:
            cond = self.less_equal(self.base, flag_sum)
        overflow = cond
        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, cond)
        if overflow:
            succ = False
        else:
            succ = self.optimizer(grads)
        ret = (loss, overflow, scaling_sens, global_norm)
        return F.depend(ret, succ)


class PANGUALPHATrainOneStepWithLossScaleCell(nn.Cell):
    """
    Encapsulation class of PANGUALPHA network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
    """

    def __init__(self,
                 network,
                 optimizer,
                 scale_update_cell=None,
                 enable_global_norm=True,
                 config=None):
        super(PANGUALPHATrainOneStepWithLossScaleCell,
              self).__init__(auto_prefix=False)
        self.network = network
        self.config = config
        self.network.add_flags(defer_inline=True)
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.enable_global_norm = enable_global_norm
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.reducer_flag = False
        self.allreduce = P.AllReduce()
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [
            ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL
        ]:
            self.reducer_flag = True
        self.grad_reducer = F.identity
        self.degree = 1
        if self.reducer_flag:
            self.degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters,
                                                       False, self.degree)
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.cast = P.Cast()
        self.alloc_status = P.NPUAllocFloatStatus()
        self.get_status = P.NPUGetFloatStatus()
        self.clear_before_grad = P.NPUClearFloatStatus()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.depend_parameter_use = P.ControlDepend(depend_mode=1)
        self.base = Tensor(1, mstype.float32)
        self.less_equal = P.LessEqual()
        self.hyper_map = C.HyperMap()
        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(
                scale_update_cell.get_loss_scale(), dtype=mstype.float32),
                name="loss_scale")
        self.clip = ClipByGlobalNorm(self.weights, self.config, pipeline=False)

    @C.add_flags(has_effect=True)
    def construct(self, input_ids, input_position=None, attention_mask=None, layer_past=None, sens=None):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(input_ids, input_position, attention_mask)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens
        # alloc status and clear should be right before gradoperation
        init = self.alloc_status()
        self.clear_before_grad(init)
        grads = self.grad(self.network,
                          weights)(input_ids,
                                   input_position, attention_mask,
                                   self.cast(scaling_sens, mstype.float32))
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        grads = self.hyper_map(
            F.partial(grad_scale, scaling_sens * self.degree), grads)

        grads, global_norms = self.clip(grads)
        global_norm = P.Reshape()(global_norms, (()))

        self.get_status(init)
        flag_sum = self.reduce_sum(init, (0,))
        if self.is_distributed:
            # sum overflow flag over devices
            flag_reduce = self.allreduce(flag_sum)
            cond = self.less_equal(self.base, flag_reduce)
        else:
            cond = self.less_equal(self.base, flag_sum)

        overflow = cond
        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, cond)
        if overflow:
            succ = False
        else:
            succ = self.optimizer(grads)
        ret = (loss, cond, scaling_sens, global_norm)
        return F.depend(ret, succ)
