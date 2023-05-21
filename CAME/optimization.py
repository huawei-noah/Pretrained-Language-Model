# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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

"""PyTorch optimization for BERT model."""

import math
import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
from torch.nn.utils import clip_grad_norm_
#from fused_adam_local import FusedAdam
from apex.optimizers import FusedAdam
from apex.multi_tensor_apply import multi_tensor_applier
import amp_C
from utils import is_main_process

multi_tensor_l2norm = amp_C.multi_tensor_l2norm
lamb_compute_update = amp_C.multi_tensor_lamb_stage1_cuda
lamb_apply_update = amp_C.multi_tensor_lamb_stage2_cuda
scale = amp_C.multi_tensor_scale


def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))

def warmup_constant(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return max((x - 1. )/ (warmup - 1.), 0.)
    
def warmup_poly(x, warmup=0.002, degree=0.5):
    if x < warmup:
        return x/warmup
    return (1.0 - x)**degree


SCHEDULES = {
    'warmup_cosine':warmup_cosine,
    'warmup_constant':warmup_constant,
    'warmup_linear':warmup_linear,
    'warmup_poly':warmup_poly,
}

class BertAdam(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix.
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """
    def __init__(self, params, lr=required, warmup=-1, t_total=-1, schedule='warmup_linear',
                 b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01,
                 max_grad_norm=1.0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= b1 < 1.0:
            raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        b1=b1, b2=b2, e=e, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        super(BertAdam, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group['e'])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay'] > 0.0:
                    update += group['weight_decay'] * p.data

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state['step'] += 1

        return loss

class SM3(Optimizer):
    """Implements SM3 algorithm.
    It has been proposed in `Memory-Efficient Adaptive Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): coefficient that scale delta before it is applied
            to the parameters (default: 0.1)
        momentum (float, optional): coefficient used to scale prior updates
            before adding. This drastically increases memory usage if
            `momentum > 0.0`. This is ignored if the parameter's gradient
            is sparse. (default: 0.0)
        beta (float, optional): coefficient used for exponential moving
            averages (default: 0.0)
        eps (float, optional): Term added to square-root in denominator to
            improve numerical stability (default: 1e-30)
    .. _Memory-Efficient Adaptive Optimization:
        https://arxiv.org/abs/1901.11150
    """
    def __init__(self, params, lr=0.1, momentum=0.0, beta=0.0, eps=1e-30, clip_threshold=1.0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {0}".format(lr))
        if not 0.0 <= momentum < 1.0:
            raise ValueError("Invalid momentum: {0}".format(momentum))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta: {0}".format(beta))
        if not 0.0 <= eps:
             raise ValueError("Invalid eps: {0}".format(eps))

        defaults = {'lr': lr, 'momentum': momentum, 'beta': beta, 'eps': eps, 'clip_threshold': clip_threshold}
        super(SM3, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            beta = group['beta']
            eps = group['eps']
            clip_threshold = group['clip_threshold']
            for p in group['params']:
                if p is None:
                    continue
                grad = p.grad

                state = self.state[p]
                shape = grad.shape
                rank = len(shape)

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = 0.
                    _add_initial_accumulators(state, grad)

                if grad.is_sparse:
                    # the update is non-linear so indices must be unique
                    grad.coalesce()
                    grad_indices = grad._indices()
                    grad_values = grad._values()

                    # Transform update_values into sparse tensor
                    def make_sparse(values):
                        constructor = grad.new
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor().resize_as_(grad)
                        return constructor(grad_indices, values, grad.size())

                    acc = state[_key(0)]
                    update_values = _compute_sparse_update(beta, acc, grad_values, grad_indices)
                    self._update_sparse_accumulator(beta, acc, make_sparse(update_values))

                    # Add small amount for numerical stability
                    update_values.add_(eps).rsqrt_().mul_(grad_values)

                    update = make_sparse(update_values)
                else:
                    # Get previous accumulators mu_{t-1}
                    if rank > 1:
                        acc_list = [state[_key(i)] for i in range(rank)]
                    else:
                        acc_list = [state[_key(0)]]

                    # Get update from accumulators and gradients
                    update = _compute_update(beta, acc_list, grad, clip_threshold)

                    # Update accumulators.
                    self._update_accumulator(beta, acc_list, update)

                    # Add small amount for numerical stability
                    update.add_(eps).rsqrt_().mul_(grad)

                    if momentum > 0.:
                        m = state['momentum_buffer']
                        update.mul_(1. - momentum).add_(m, alpha=momentum)
                        state['momentum_buffer'] = update.detach()

                p.sub_(update, alpha=group['lr'])
                state['step'] += 1
        return loss
        
    def _update_accumulator(self, beta, acc_list, update):
        for i, acc in enumerate(acc_list):
            nu_max = _max_reduce_except_dim(update, i)
            if beta > 0.:
                torch.max(acc, nu_max, out=acc)
            else:
                # No need to compare - nu_max is bigger because of grad ** 2
                acc.copy_(nu_max)

    def _update_sparse_accumulator(self, beta, acc, update):
        nu_max = _max_reduce_except_dim(update.to_dense(), 0).squeeze()
        if beta > 0.:
            torch.max(acc, nu_max, out=acc)
        else:
            # No need to compare - nu_max is bigger because of grad ** 2
            acc.copy_(nu_max)

def _compute_sparse_update(beta, acc, grad_values, grad_indices):
    # In the sparse case, a single accumulator is used.
    update_values = torch.gather(acc, 0, grad_indices[0])
    if beta > 0.:
        update_values.mul_(beta)
    update_values.addcmul_(grad_values, grad_values, value=1. - beta)
    return update_values

def _compute_update(beta, acc_list, grad, clip_threshold):
    rank = len(acc_list)
    update = acc_list[0].clone()
    for i in range(1, rank):
        # We rely on broadcasting to get the proper end shape.
        update = torch.min(update, acc_list[i])
    if beta > 0.:
        update.mul_(beta)
    grad_residual = update - grad * grad
    update.addcmul_(grad, grad, value=1. - beta)
    update.div_((grad_residual / clip_threshold).clamp_(min=0.4))

    return update

def _key(i):
    # Returns key used for accessing accumulators
    return 'accumulator_' + str(i)

def _add_initial_accumulators(state, grad):
    # Creates initial accumulators. For a dense tensor of shape (n1, n2, n3),
    # then our initial accumulators are of shape (n1, 1, 1), (1, n2, 1) and
    # (1, 1, n3). For a sparse tensor of shape (n, *), we use a single
    # accumulator of shape (n,).
    shape = grad.shape
    rank = len(shape)
    defaults = {'device': grad.device, 'dtype': grad.dtype}
    acc = {}

    if grad.is_sparse:
        acc[_key(0)] = torch.zeros(shape[0], **defaults)
    elif rank == 0:
        # The scalar case is handled separately
        acc[_key(0)] = torch.zeros(shape, **defaults)
    else:
        for i in range(rank):
            acc_shape = [1] * i + [shape[i]] + [1] * (rank - 1 - i)
            acc[_key(i)] = torch.zeros(acc_shape, **defaults)

    state.update(acc)

def _max_reduce_except_dim(tensor, dim):
    # Computes max along all dimensions except the given dim.
    # If tensor is a scalar, it returns tensor.
    rank = len(tensor.shape)
    result = tensor
    if rank > 0:
        assert dim < rank
        for d in range(rank):
            if d != dim:
                result = result.max(dim=d, keepdim=True).values
    return result