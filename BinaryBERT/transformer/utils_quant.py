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
import torch
import torch.nn as nn
import logging
import math

class SymQuantizer(torch.autograd.Function):
    """
        uniform quantization
    """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        ctx.num_bits = num_bits
        if num_bits < 32:
            input = torch.where(input < clip_val[1], input, clip_val[1])
            input = torch.where(input > clip_val[0], input, clip_val[0])
            # NOTE: dynamic scaling (max_input).
            if layerwise:
                max_input = torch.max(torch.abs(input)).expand_as(input)
            else:
                if input.ndimension() <= 3:
                    # weight & hidden layer
                    max_input = torch.max(torch.abs(input), dim=-1, keepdim=True)[0].expand_as(input).detach()
                elif input.ndimension() == 4:
                    # TODO: attention score matrix, calculate alpha / beta per head
                    tmp = input.view(input.shape[0], input.shape[1], -1)
                    max_input = torch.max(torch.abs(tmp), dim=-1, keepdim=True)[0].unsqueeze(-1).expand_as(
                        input).detach()
                else:
                    raise ValueError
            s = (2 ** (num_bits - 1) - 1) / max_input
            output = torch.round(input * s).div(s)
        else:
            output = input

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        num_bits = ctx.num_bits
        grad_input = grad_output.clone()
        grad_clip = None
        if num_bits < 32:
            grad_input[input.ge(clip_val[1])] = 0
            grad_input[input.le(clip_val[0])] = 0
            # refer to PACT
            grad_clip_pos = (grad_output * input.ge(clip_val[1]).float()).sum()
            grad_clip_neg = (grad_output * (input.le(clip_val[0]).float())).sum()
            grad_clip = torch.tensor([grad_clip_neg, grad_clip_pos]).to(input.device)
        return grad_input, grad_clip, None, None


class AsymQuantizer(torch.autograd.Function):
    """
        min-max quantization
    """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        ctx.num_bits = num_bits
        if num_bits < 32:
            input = torch.where(input < clip_val[1], input, clip_val[1])
            input = torch.where(input > clip_val[0], input, clip_val[0])
            # NOTE: dynamic scaling gives better performance than static
            if layerwise:
                alpha = (input.max() - input.min()).detach()
                beta = input.min().detach()
                # alpha = clip_val[1] - clip_val[0]
                # beta = clip_val[0]
            else:
                if input.ndimension() <= 3:
                    # weight & hidden layer
                    alpha = (input.max(dim=-1, keepdim=True)[0] - input.min(dim=-1, keepdim=True)[0]).expand_as(
                        input).detach()
                    beta = input.min(dim=-1, keepdim=True)[0].expand_as(input).detach()
                elif input.ndimension() == 4:
                    # TODO: attention score matrix, calculate alpha / beta per head
                    tmp = input.view(input.shape[0], input.shape[1], -1)
                    alpha = (tmp.max(dim=-1, keepdim=True)[0].unsqueeze(-1) - \
                             tmp.min(dim=-1, keepdim=True)[0].unsqueeze(-1)).expand_as(input).detach()
                    beta = tmp.min(dim=-1, keepdim=True)[0].unsqueeze(-1).expand_as(input).detach()
                else:
                    raise ValueError
            input_normalized = (input - beta) / (alpha + 1e-8)
            s = (2 ** num_bits - 1)
            quant_input = torch.round(input_normalized * s).div(s)
            output = quant_input * (alpha + 1e-8) + beta
        else:
            output = input

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        num_bits = ctx.num_bits
        grad_input = grad_output.clone()
        grad_clip = None
        if num_bits < 32:
            grad_input[input.ge(clip_val[1])] = 0
            grad_input[input.le(clip_val[0])] = 0
            # refer to PACT
            grad_clip_pos = (grad_output * input.ge(clip_val[1]).float()).sum()
            grad_clip_neg = (grad_output * (input.le(clip_val[0]).float())).sum()
            grad_clip = torch.tensor([grad_clip_neg, grad_clip_pos]).to(input.device)
        return grad_input, grad_clip, None, None


class LaqQuantizer(torch.autograd.Function): # can only be used for weight
    """Loss-aware binarization and quantization
    Ref: https://arxiv.org/abs/1611.01600, https://arxiv.org/abs/1802.08635
    """
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        """
        :param input: tensor to be ternarized
        :return: quantized tensor
        """
        ctx.save_for_backward(input)
        if num_bits < 32:
            # TODO: currently only support identity approximate hessian
            D = torch.ones_like(input)
            # if len(optimizer.state[optimizer.param_groups[0]['params'][0]]) == 0:
            #     v = torch.ones_like(input)
            # else:
            #     v = optimizer.state[input]['exp_avg_sq']
            # D = v.sqrt() + 1e-8  # to be consistent with the setting in the optimizer
            if layerwise:
                if num_bits == 1:  # lab
                    alpha = (D * input).abs().sum() / D.sum()
                    b = input.sign()
                    output = alpha * b
                else:
                    n = 2 ** (num_bits - 1) - 1
                    b = input.sign()
                    # compute the threshold, converge within 10 iterations
                    alpha = (b * D * input).abs().sum() / (b * D).abs().sum()
                    b = ((input / alpha).clamp(-1., 1.) * n).round() / n
                    for i in range(10):
                        alpha = (b * D * input).abs().sum() / (b * D).abs().sum()
                        b = ((input / alpha).clamp(-1., 1.) * n).round() / n
                    output = alpha * b

            else:
                if num_bits == 1:
                    alpha = (D * input).abs().sum(dim=1, keepdim=True) / D.sum(dim=1, keepdim=True)
                    b = input.sign()
                    output = alpha * b
                else:
                    n = 2 ** (num_bits - 1) - 1
                    b = input.sign()
                    # compute the threshold, converge within 10 iterations
                    alpha = (b * D * input).abs().sum(dim=1, keepdim=True) / (b * D).abs().sum(dim=1, keepdim=True)
                    b = ((input / alpha).clamp(-1., 1.) * n).round() / n
                    for i in range(10):
                        alpha = (b * D * input).abs().sum(dim=1, keepdim=True) / (b * D).abs().sum(dim=1, keepdim=True)
                        b = ((input / alpha).clamp(-1., 1.) * n).round() / n
                    output = alpha * b
        else:
            output = input

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        grad_input = grad_output.clone()
        return grad_input, None, None, None


class SymLsqQuantizer(torch.autograd.Function):
    """
        Learned Step-size Quantization.
        https://arxiv.org/abs/1902.08153
    """
    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        if not layerwise:
            # TODO
            raise NotImplementedError
        ctx.num_bits = num_bits
        if num_bits == 32:
            return input

        Qn = -2 ** (num_bits - 1)
        Qp = 2 ** (num_bits - 1) - 1

        assert alpha > 0, 'alpha = {:.6f} becomes non-positive'.format(alpha)
        if alpha.item() == 1.0 and (not alpha.initialized):
            alpha.initialize_wrapper(input, num_bits, symmetric=True, init_method='default')

        grad_scale = 1.0 / math.sqrt(input.numel() * Qp)
        # grad_scale = 1.0
        ctx.save_for_backward(input, alpha)
        ctx.other = grad_scale, Qn, Qp
        q_w = (input / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits == 32:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp = ctx.other
        q_w = input_ / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big # this is more cpu-friendly than torch.ones(input_.shape)
        grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
                -q_w + q_w.round())) * grad_output * grad_scale).sum().unsqueeze(dim=0)
        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None


class AsymLsqQuantizer(torch.autograd.Function):
    """
        Asymetric LSQ quantization. Modified from LSQ.
    """
    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        if not layerwise:
            # TODO
            raise NotImplementedError
        ctx.num_bits = num_bits
        if num_bits == 32:
            return input

        Qn = 0
        Qp = 2 ** (num_bits) - 1
        # asymmetric: make sure input \in [0, +\inf], remember to add it back
        min_val = input.min().item()
        input_ = input - min_val

        assert alpha > 0, 'alpha = {:.6f} becomes non-positive'.format(alpha)
        if alpha.item() == 1.0 and (not alpha.initialized):
            alpha.initialize_wrapper(input, num_bits, symmetric=False, init_method='default')

        grad_scale = 1.0 / math.sqrt(input.numel() * Qp)
        # grad_scale = 1.0
        ctx.save_for_backward(input_, alpha)
        ctx.other = grad_scale, Qn, Qp
        q_w = (input_ / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        w_q = w_q + min_val
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits == 32:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp = ctx.other
        q_w = input_ / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big   # this is more cpu-friendly than torch.ones(input_.shape)
        grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
                -q_w + q_w.round())) * grad_output * grad_scale).sum().unsqueeze(dim=0)
        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None

class LsqStepSize(nn.Parameter):
    def __init__(self, tensor):
        super(LsqStepSize, self).__new__(nn.Parameter, data=tensor)
        self.initialized = False

    def _initialize(self, init_tensor):
        assert not self.initialized, 'already initialized.'
        self.data.copy_(init_tensor)
        # print('Stepsize initialized to %.6f' % self.data.item())
        self.initialized = True

    def initialize_wrapper(self, tensor, num_bits, symmetric, init_method='default'):
        # input: everthing needed to initialize step_size
        Qp = 2 ** (num_bits - 1) - 1 if symmetric else 2 ** (num_bits) - 1
        if init_method == 'default':
            init_val = 2 * tensor.abs().mean() / math.sqrt(Qp) if symmetric \
                else 4 * tensor.abs().mean() / math.sqrt(Qp)
        elif init_method == 'uniform':
            init_val = 1./(2*Qp+1) if symmetric else 1./Qp

        self._initialize(init_val)


class BwnQuantizer(torch.autograd.Function):
    """Binary Weight Network (BWN)
     Ref: https://arxiv.org/abs/1603.05279
     """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        """
        :param input: tensor to be bianrized
        :return: quantized tensor
        """
        ctx.save_for_backward(input)
        if layerwise:
            s = input.size()
            m = input.norm(p=1).div(input.nelement())
            result = input.sign().mul(m.expand(s))
        else:
            n = input[0].nelement()  # W of size axb, return a vector of  ax1
            s = input.size()
            m = input.norm(1, 1, keepdim=True).div(n)
            result = input.sign().mul(m.expand(s))

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        grad_input = grad_output.clone()
        return grad_input, None, None, None


class TwnQuantizer(torch.autograd.Function):
    """Ternary Weight Networks (TWN)
    Ref: https://arxiv.org/abs/1605.04711
    """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        """
        :param input: tensor to be ternarized
        :return: quantized tensor
        """
        ctx.save_for_backward(input)
        if layerwise:
            m = input.norm(p=1).div(input.nelement())
            thres = 0.7 * m
            pos = (input > thres).float()
            neg = (input < -thres).float()
            mask = (input.abs() > thres).float()
            alpha = (mask * input).abs().sum() / mask.sum()
            result = alpha * pos - alpha * neg
        else:  # row-wise only for embed / weight
            n = input[0].nelement()
            m = input.data.norm(p=1, dim=1).div(n)
            thres = (0.7 * m).view(-1, 1).expand_as(input)
            pos = (input > thres).float()
            neg = (input < -thres).float()
            mask = (input.abs() > thres).float()
            alpha = ((mask * input).abs().sum(dim=1) / mask.sum(dim=1)).view(-1, 1)
            result = alpha * pos - alpha * neg

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        grad_input = grad_output.clone()
        return grad_input, None, None, None


def act_quant_fn(input, clip_val, num_bits, symmetric, quant_method, layerwise):
    if num_bits == 32:
        return input
    elif quant_method == "bwn" and num_bits == 1:
        raise NotImplementedError("Too bad performance, please dont")
        # quant_fn = BwnQuantizer
    elif quant_method == "twn" and num_bits == 2:
        raise NotImplementedError("Too bad performance, please dont")
        # quant_fn = TwnQuantizer
    elif quant_method=="uniform" and num_bits >= 2 and symmetric:
        quant_fn = SymQuantizer
    elif quant_method == "uniform" and num_bits >= 2 and not symmetric:
        quant_fn = AsymQuantizer
    elif quant_method == "lsq" and num_bits >= 2 and symmetric:
        quant_fn = SymLsqQuantizer
    elif quant_method == "lsq" and num_bits >= 2 and not symmetric:
        quant_fn = AsymLsqQuantizer
    else:
        raise ValueError("Unknownquant_method")

    input = quant_fn.apply(input, clip_val, num_bits, layerwise)

    return input


def weight_quant_fn(weight,  clip_val,  num_bits,  symmetric, quant_method, layerwise):
    if num_bits == 32:
        return weight

    # play with different variants of t2b quantizer
    elif quant_method == "bwn" and num_bits == 1:
        quant_fn = BwnQuantizer
    elif quant_method == "twn" and num_bits == 2:
        quant_fn = TwnQuantizer
    elif num_bits >= 2 and symmetric and quant_method == "uniform":
        quant_fn = SymQuantizer
    elif quant_method == "uniform" and num_bits >= 2 and not symmetric:
        quant_fn = AsymQuantizer
    elif quant_method == "lsq" and num_bits >= 2 and symmetric:
        quant_fn = SymLsqQuantizer
    elif quant_method == "lsq" and num_bits >= 2 and not symmetric:
        quant_fn = AsymLsqQuantizer
    elif quant_method == "laq":
        quant_fn = LaqQuantizer
    else:
        raise ValueError("Unknown quant_method")

    weight = quant_fn.apply(weight, clip_val,  num_bits, layerwise)
    return weight


class QuantizeLinear(nn.Linear):

    def __init__(self, *kargs, clip_val=2.5, weight_bits=8, input_bits=8, learnable=False, symmetric=True,
                 weight_layerwise=True, input_layerwise=True, weight_quant_method="twn", input_quant_method="uniform",
                 **kwargs):
        super(QuantizeLinear, self).__init__(*kargs, **kwargs)
        self.weight_bits = weight_bits
        self.input_bits = input_bits
        self.learnable = learnable
        self.symmetric = symmetric
        self.weight_layerwise = weight_layerwise
        self.input_layerwise = input_layerwise
        self.weight_quant_method = weight_quant_method
        self.input_quant_method = input_quant_method
        self._build_weight_clip_val(weight_quant_method, learnable, init_val=clip_val)
        self._build_input_clip_val(input_quant_method, learnable, init_val=clip_val)

    def _build_weight_clip_val(self, quant_method, learnable, init_val):
        if quant_method == 'uniform':
            # init_val = self.weight.mean().item() + 3 * self.weight.std().item()
            self.register_buffer('weight_clip_val', torch.tensor([-init_val, init_val]))
            if learnable:
                self.weight_clip_val = nn.Parameter(self.weight_clip_val)
        elif quant_method == 'lsq':
            # TODO: for now we abuse the name for consistent reference in learner.
            assert learnable, 'LSQ must use leranable step size!'
            self.weight_clip_val = LsqStepSize(torch.tensor(1.0)) # stepsize will be initialized in the first quantization
        else:
            self.register_buffer('weight_clip_val', None)

    def _build_input_clip_val(self, quant_method, learnable, init_val):
        if quant_method == 'uniform':
            self.register_buffer('input_clip_val', torch.tensor([-init_val, init_val]))
            if learnable:
                self.input_clip_val = nn.Parameter(self.input_clip_val)
        elif quant_method == 'lsq':
            # TODO: for now we abuse the name for consistent reference in learner.
            assert learnable, 'LSQ must use leranable step size!'
            self.input_clip_val = LsqStepSize(torch.tensor(1.0))  # stepsize will be initialized in the first quantization
        else:
            self.register_buffer('input_clip_val', None)

    def forward(self, input):
        # quantize weight
        weight = weight_quant_fn(self.weight, self.weight_clip_val, num_bits=self.weight_bits, symmetric=self.symmetric,
                                 quant_method=self.weight_quant_method, layerwise=self.weight_layerwise)
        # quantize input
        input = act_quant_fn(input, self.input_clip_val, num_bits=self.input_bits, symmetric=self.symmetric,
                             quant_method=self.input_quant_method, layerwise=self.input_layerwise)
        out = nn.functional.linear(input, weight)
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out


class QuantizeEmbedding(nn.Embedding):

    def __init__(self, *kargs, clip_val=2.5, weight_bits=8, input_bits=8, learnable=False, symmetric=True,
                 embed_layerwise=False, weight_quant_method="twn", input_quant_method="uniform", **kwargs):
        super(QuantizeEmbedding, self).__init__(*kargs, **kwargs)
        self.weight_bits = weight_bits
        self.input_bits = input_bits
        self.learnable = learnable
        self.symmetric = symmetric
        self.embed_layerwise = embed_layerwise
        self.weight_quant_method = weight_quant_method
        self.input_quant_method = input_quant_method
        self._build_embed_clip_val(weight_quant_method, learnable, init_val=clip_val)

    def _build_embed_clip_val(self, quant_method, learnable, init_val):
        if quant_method == 'uniform':
            self.register_buffer('embed_clip_val', torch.tensor([-init_val, init_val]))
            if learnable:
                self.embed_clip_val = nn.Parameter(self.embed_clip_val)
        elif quant_method == 'lsq':
            # TODO: for now we abuse the name for consistent reference in learner.
            assert learnable, 'LSQ must use leranable step size!'
            self.embed_clip_val = LsqStepSize(torch.tensor(1.0)) # stepsize will be initialized in the first quantization
        else:
            self.register_buffer('embed_clip_val', None)

    def forward(self, input):
        weight = weight_quant_fn(self.weight, self.embed_clip_val, num_bits=self.weight_bits, symmetric=self.symmetric,
                                 quant_method=self.weight_quant_method, layerwise=self.embed_layerwise)

        out = nn.functional.embedding(
            input, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

        return out

