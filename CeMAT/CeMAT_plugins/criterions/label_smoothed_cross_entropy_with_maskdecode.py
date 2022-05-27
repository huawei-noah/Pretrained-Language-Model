# 2022 - Added code for CeMAT
#        Huawei Technologies Co., Ltd. <lipengfei111@huawei.com>
# Copyright 2022 Huawei Technologies Co., Ltd.
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import math
import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion("label_smoothed_cross_entropy_with_maskdecode")
class LabelSmoothedCrossEntropyCriterionWithMaskDecode(FairseqCriterion):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=0,
            report_accuracy=False,
            plus_encoder_loss=False,
            encoder_loss_lambda=0.,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.plus_encoder_loss = plus_encoder_loss
        self.encoder_loss_lambda = encoder_loss_lambda
        print("self.plus_encoder_loss:{},self.encoder_loss_lambda:{}".format(self.plus_encoder_loss,self.encoder_loss_lambda))
        if hasattr(task, "target_dictionary"):
            tgt_dict = task.target_dictionary
            src_dict = task.target_dictionary
            self.padding_idx = tgt_dict.pad() if tgt_dict is not None else -100
            self.src_padding_idx = src_dict.pad() if src_dict is not None else -100


    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--plus-encoder-loss', action='store_true',
                            help='plus bert loss for biBart task')
        parser.add_argument('--encoder-loss-lambda', default=0.,type=float,
                            help='weighting for encoder\'loss when add extra encoder\'loss ')
        parser.add_argument('--report-accuracy', action='store_true',
                            help='report accuracy metric')
        parser.add_argument('--ignore-prefix-size', default=0, type=int,
                            help='Ignore first N tokens')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training

        note: has been fix:
        model's output is encoder_output,decoder_output
        """
        if self.plus_encoder_loss and self.encoder_loss_lambda > 0:
            encoder_out, decoder_out = model(**sample["net_input"])
        else:
            decoder_out = model(**sample["net_input"])
            encoder_out = None

        loss, nll_loss = self.compute_loss(model, decoder_out, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        nll_loss_encoder = nll_loss
        if self.plus_encoder_loss and self.encoder_loss_lambda > 0 :
            loss_encoder, nll_loss_encoder = self.compute_loss(model, encoder_out, sample, reduce=reduce,
                                                               is_encoder=True)
            loss = (1.0 - self.encoder_loss_lambda)* loss + (self.encoder_loss_lambda)*loss_encoder
            nll_loss = (1.0 - self.encoder_loss_lambda)* nll_loss + (self.encoder_loss_lambda)*nll_loss_encoder

        logging_output = {
            "loss": loss.data,
            "enc_loss": nll_loss_encoder.data,
            "nll_loss": nll_loss.data,
            "enc_ntokens": sample["src_ntokens"],
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "all_ntokens":sample["tgt_ntokens"]
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, decoder_out, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample, is_encoder=False):
        if is_encoder:
            lprobs = model.get_normalized_probs_may_encoder(net_output, is_encoder=True, log_probs=True)
            target = model.get_sources(sample, net_output)
        else:
            lprobs = model.get_normalized_probs_may_encoder(net_output, is_encoder=False, log_probs=True)
            target = model.get_targets(sample, net_output)

        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
                target = target[:, self.ignore_prefix_size:].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size:, :, :].contiguous()
                target = target[self.ignore_prefix_size:, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True, is_encoder=False):
        if is_encoder:
            lprobs, target = self.get_lprobs_and_target(model, net_output, sample, is_encoder)
            padding_idx = self.src_padding_idx
        else:
            lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
            padding_idx = self.padding_idx
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        enc_ntokens = sum(log.get("enc_ntokens", 0) for log in logging_outputs)
        enc_nll_loss_sum = sum(log.get("enc_loss", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        if enc_ntokens != 0:
            metrics.log_scalar(
                "encnll_loss", enc_nll_loss_sum / enc_ntokens / math.log(2), enc_ntokens, round=3
            )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
