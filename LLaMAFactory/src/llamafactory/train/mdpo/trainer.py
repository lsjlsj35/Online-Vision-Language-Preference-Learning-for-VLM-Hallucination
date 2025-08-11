# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/dpo_trainer.py
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

import copy
import json
import os
import requests
import time
import warnings
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime
from filelock import FileLock
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Trainer
from transformers.training_args import OptimizerNames
from transformers.trainer_pt_utils import nested_detach
from transformers.utils import (
    is_apex_available,
    is_sagemaker_mp_enabled,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_npu_available,
    is_torch_xpu_available
)
# from transformers.trainer_pt_utils import smp_forward_backward
from trl import DPOTrainer
from trl.trainer import disable_dropout_in_model
from typing_extensions import override

from ...data.processors.processor_utils import infer_seqlen
from ...extras.constants import IGNORE_INDEX
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, get_batch_logps

if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

    from ...hparams import FinetuningArguments, VTCLArguments

if is_apex_available():
    from apex import amp


class CustomDPOTrainer(DPOTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        vtcl_args: "VTCLArguments",
        processor: Optional["ProcessorMixin"],
        disable_dropout: bool = True,
        **kwargs,
    ):
        if disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)

        self.finetuning_args = finetuning_args
        self.f_divergence_type = "reverse_kl"
        self.reference_free = False
        self.use_dpo_data_collator = True  # hack to avoid warning
        self.generate_during_eval = False  # disable at evaluation
        self.label_pad_token_id = IGNORE_INDEX
        self.padding_value = 0
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.precompute_ref_log_probs = False
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self._peft_has_been_casted_to_bf16 = False

        self.ref_model = ref_model
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # dpo hyperparams
        self.beta = finetuning_args.pref_beta
        self.loss_type = finetuning_args.pref_loss
        self.ftx_gamma = finetuning_args.pref_ftx
        self.label_smoothing = finetuning_args.dpo_label_smoothing
        self.simpo_gamma = finetuning_args.simpo_gamma

        self.strategy = vtcl_args.strategy
        self.vtcl_mode = vtcl_args.vtcl_mode
        self.anchor_ratio = vtcl_args.anchor_ratio
        self.neg_anchor_ratio = vtcl_args.neg_anchor_ratio
        self.half_dpo_loss = not vtcl_args.not_half_dpo_loss
        self._proc_idx_label = kwargs["args"].process_index

        Trainer.__init__(self, model=model, **kwargs)
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.pissa_convert:
            self.callback_handler.add_callback(PissaConvertCallback)

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)
    
    def get_time(self):
        timestamp = time.time()
        dt_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        output = f"Timestamp: {timestamp}\nFormatted: {dt_str}\n"
        return output

    def odds_ratio_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes ORPO's odds ratio (OR) loss for batched log probabilities of the policy model.
        """
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
        )
        sft_loss = -chosen_logps
        odds_ratio_loss = -F.logsigmoid(log_odds)
        orpo_loss = sft_loss + self.beta * odds_ratio_loss
        return orpo_loss

    def simpo_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes SimPO loss for batched log probabilities of the policy model.
        """
        pi_logratios = chosen_logps - rejected_logps
        gamma_logratios = self.simpo_gamma / self.beta
        logits = pi_logratios - gamma_logratios
        simpo_loss = -F.logsigmoid(self.beta * logits)
        return simpo_loss

    def compute_preference_loss(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        reference_chosen_logps: Optional["torch.Tensor"],
        reference_rejected_logps: Optional["torch.Tensor"],
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes loss for preference learning.
        """
        if self.vtcl_mode == "sameBatch":
            if self.strategy == "vtcl":
                cc = policy_chosen_logps[::2]
                rr = policy_chosen_logps[1::2]
                cr = policy_rejected_logps[::2]
                rc = policy_rejected_logps[1::2]
                ccref = reference_chosen_logps[::2]
                rrref = reference_chosen_logps[1::2]
                crref = reference_rejected_logps[::2]
                rcref = reference_rejected_logps[1::2]

                dpo1, creward1, rreward1 = self.dpo_loss(cc, cr, ccref, crref)
                dpo2, creward2, rreward2 = self.dpo_loss(rr, rc, rrref, rcref)
                mdpo1, mcreward1, mrreward1 = self.dpo_loss(cc, rc, ccref, rcref)
                mdpo2, mcreward2, mrreward2 = self.dpo_loss(rr, cr, rrref, crref)
                if self.anchor_ratio >= 1e-6:
                    anc_losses = self.dpo_loss(
                        policy_chosen_logps, reference_rejected_logps, reference_chosen_logps, reference_rejected_logps
                    )[0]
                else:
                    anc_losses = None
                return (dpo1+dpo2) / 2, (mdpo1+mdpo2) / 2, anc_losses, (creward1+creward2) / 2, (rreward1+rreward2) / 2, (mcreward1+mcreward2) / 2, (mrreward1+mrreward2) / 2
            elif self.strategy == 'svco-noref':
                cc = policy_chosen_logps[::2]
                rc = policy_chosen_logps[1::2]
                cr = policy_rejected_logps[::2]
                rr = policy_rejected_logps[1::2]
                ccref = reference_chosen_logps[::2]
                rcref = reference_chosen_logps[1::2]
                crref = reference_rejected_logps[::2]
                rrref = reference_rejected_logps[1::2]
                dpo1, creward1, rreward1 = self.dpo_loss(cc, rc, ccref, rcref)
                dpo2, creward2, rreward2 = self.dpo_loss(rr, cr, rrref, crref)
                if self.anchor_ratio >= 1e-6:
                    anc_losses1 = self.dpo_loss(
                        cc, rcref, ccref, rcref
                    )[0]
                    anc_losses2 = self.dpo_loss(
                        rr, crref, rrref, crref
                    )[0]
                    anc_losses = (anc_losses1+anc_losses2) / 2
                else:
                    anc_losses = None
                return (dpo1+dpo2) / 2, anc_losses, (creward1+creward2) / 2, (rreward1+rreward2) / 2
            elif self.strategy in ["svco-ref", "svco-textref"]:
                r, no_img = torch.chunk(policy_rejected_logps, 2)
                rref, no_img_ref = torch.chunk(reference_rejected_logps, 2)
                cdpo, creward, noImgreward = self.dpo_loss(policy_chosen_logps, no_img, reference_chosen_logps, no_img_ref)
                rdpo, _, rreward = self.dpo_loss(no_img, r, no_img_ref, rref)
                if self.anchor_ratio >= 1e-6:
                    anc_losses1 = self.dpo_loss(policy_chosen_logps, no_img_ref, reference_chosen_logps, no_img_ref)[0]
                    anc_losses2 = self.dpo_loss(no_img, rref, no_img_ref, rref)[0]
                    anc_losses = (anc_losses1+anc_losses2) / 2
                else:
                    anc_losses = None
                return cdpo, rdpo, anc_losses, creward, noImgreward, rreward
            elif self.strategy in ["orm"]:
                rc, cr, no_img = torch.chunk(policy_rejected_logps, 3)
                rcref, crref, no_img_ref = torch.chunk(reference_rejected_logps, 3)
                cdpo, ccreward, noImgreward = self.dpo_loss(policy_chosen_logps, no_img, reference_chosen_logps, no_img_ref)
                rdpo, _, rcreward = self.dpo_loss(no_img, rc, no_img_ref, rcref)
                dpo, _, crreward = self.dpo_loss(policy_chosen_logps, cr, reference_chosen_logps, crref)
                if self.anchor_ratio >= 1e-6:
                    anc_losses = self.anchor_ratio * self.dpo_loss(policy_chosen_logps, no_img_ref, reference_chosen_logps, no_img_ref)[0]
                    if self.neg_anchor_ratio > 0:
                        anc_losses -= self.neg_anchor_ratio * self.dpo_loss(cr, reference_chosen_logps, crref, reference_chosen_logps)[0]
                else:
                    anc_losses = None
                return cdpo, rdpo, dpo, ccreward, rcreward, crreward, noImgreward, anc_losses
            elif self.strategy == "dpo":
                raise NotImplementedError
            else:
                raise KeyError
        else:
            if self.strategy in ["orm"]:
                rc, cr, no_img = torch.chunk(policy_rejected_logps, 3)
                rcref, crref, no_img_ref = torch.chunk(reference_rejected_logps, 3)
                cdpo, ccreward, noImgreward = self.dpo_loss(policy_chosen_logps, no_img, reference_chosen_logps, no_img_ref)
                rdpo, _, rcreward = self.dpo_loss(no_img, rc, no_img_ref, rcref)
                dpo, _, crreward = self.dpo_loss(policy_chosen_logps, cr, reference_chosen_logps, crref)
                if self.anchor_ratio >= 1e-6:
                    anc_losses = self.anchor_ratio * self.dpo_loss(policy_chosen_logps, no_img_ref, reference_chosen_logps, no_img_ref)[0]
                    if self.neg_anchor_ratio > 0:
                        anc_losses -= self.neg_anchor_ratio * self.dpo_loss(cr, reference_chosen_logps, crref, reference_chosen_logps)[0]
                else:
                    anc_losses = None
                return cdpo, rdpo, dpo, ccreward, rcreward, crreward, noImgreward, anc_losses
            elif self.strategy in ["svco-ref", "svco-textref"]:
                r, no_img = torch.chunk(policy_rejected_logps, 2)
                rref, no_img_ref = torch.chunk(reference_rejected_logps, 2)
                cdpo, creward, noImgreward = self.dpo_loss(policy_chosen_logps, no_img, reference_chosen_logps, no_img_ref)
                rdpo, _, rreward = self.dpo_loss(no_img, r, no_img_ref, rref)
                if self.anchor_ratio >= 1e-6:
                    anc_losses = self.anchor_ratio * self.dpo_loss(policy_chosen_logps, no_img_ref, reference_chosen_logps, no_img_ref)[0]
                else:
                    anc_losses = None
                return cdpo, rdpo, anc_losses, creward, noImgreward, rreward
            # single / difBatch -> single
            elif self.strategy in ["vtcl", "mdpo", "ori-mdpo"]:
                cr = policy_rejected_logps[::2]
                rc = policy_rejected_logps[1::2]
                crref = reference_rejected_logps[::2]
                rcref = reference_rejected_logps[1::2]

                dpo, creward, rreward = self.dpo_loss(policy_chosen_logps, cr, reference_chosen_logps, crref)
                mdpo, mcreward, mrreward = self.dpo_loss(policy_chosen_logps, rc, reference_chosen_logps, rcref)
                if self.anchor_ratio >= 1e-6:
                    anc_losses = self.anchor_ratio * self.dpo_loss(policy_chosen_logps, crref, reference_chosen_logps, crref)[0]
                    if self.neg_anchor_ratio > 0:
                        anc_losses -= self.neg_anchor_ratio * self.dpo_loss(cr, reference_chosen_logps, crref, reference_chosen_logps)[0]
                else:
                    anc_losses = None
                return dpo, mdpo, anc_losses, creward, rreward, mcreward, mrreward
            elif self.strategy in ["dpo", "svco-noref"]:
                dpo, creward, rreward = self.dpo_loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps)
                if self.anchor_ratio >= 1e-6:
                    anc_losses = self.anchor_ratio * self.dpo_loss(policy_chosen_logps, reference_rejected_logps, reference_chosen_logps, reference_rejected_logps)[0]
                    if self.neg_anchor_ratio > 0:
                        anc_losses -= self.neg_anchor_ratio * self.dpo_loss(policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, reference_chosen_logps)[0]
                else:
                    anc_losses = None
                return dpo, creward, rreward, anc_losses
            else:
                raise KeyError

    @override
    def concatenated_forward(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes the sum log probabilities of the labels under given logits if loss_type is not IPO, ORPO or SimPO.

        Otherwise the average log probabilities.
        """
        if self.finetuning_args.use_ref_model:
            batch = {k: v.detach().clone() for k, v in batch.items()}  # avoid error
        # length = batch["input_ids"].shape[0] // 2
        # chosen_batch = {k: v[:length] for k, v in batch.items()}
        # rejected_batch = {k: v[length:] for k, v in batch.items()}
        # chosen_logits = model(**chosen_batch, return_dict=True, use_cache=False).logits.to(torch.float32)
        # chosen_logps, chosen_length = get_batch_logps(logits=chosen_logits, labels=chosen_batch["labels"])
        
        # rejected_logits = model(**rejected_batch, return_dict=True, use_cache=False).logits.to(torch.float32)
        # rejected_logps, _ = get_batch_logps(logits=rejected_logits, labels=rejected_batch["labels"])
        # for k, v in batch.items():
        #     print(k, v.shape)
        all_logits: "torch.Tensor" = model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)
        all_logps, valid_length = get_batch_logps(logits=all_logits, labels=batch["labels"])
        if self.loss_type in ["ipo", "orpo", "simpo"]:
            all_logps = all_logps / valid_length
        if self.strategy in ["orm"]:
            batch_size = batch["input_ids"].size(0) // 4
            split_size = [batch_size, 3*batch_size]
        elif self.strategy in ["svco-ref", "svco-textref"]:
            batch_size = batch["input_ids"].size(0) // 3
            split_size = [batch_size, 2*batch_size]
        elif self.strategy in ["vtcl", "mdpo", "ori-mdpo"] and self.vtcl_mode != "sameBatch":
            batch_size = batch["input_ids"].size(0) // 3
            split_size = [batch_size, 2*batch_size]
        else:
            batch_size = batch["input_ids"].size(0) // 2
            split_size = [batch_size, batch_size]
        chosen_logps, rejected_logps = all_logps.split(split_size=split_size, dim=0)
        chosen_logits, rejected_logits = all_logits.split(split_size=split_size, dim=0)
        chosen_length, _ = valid_length.split(split_size=split_size, dim=0)
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps / chosen_length
    
    def concatenated_forward_for_chosen_sft(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple["torch.Tensor"]:
        r"""
        Computes the average log probabilities for the given data
        """
        all_logits: "torch.Tensor" = model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)
        all_logps, valid_length = get_batch_logps(logits=all_logits, labels=batch["labels"])
        return all_logps / valid_length

    @override
    def compute_reference_log_probs(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple[Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Computes log probabilities of the reference model.
        """
        if not self.finetuning_args.use_ref_model:
            return None, None

        if self.ref_model is None:
            ref_model = model
            ref_context = self.accelerator.unwrap_model(model).disable_adapter()
        else:
            ref_model = self.ref_model
            ref_context = nullcontext()

        with torch.no_grad(), ref_context:
            reference_chosen_logps, reference_rejected_logps, *_ = self.concatenated_forward(ref_model, batch)

        return reference_chosen_logps, reference_rejected_logps

    @override
    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
        metrics = {},
        use_normal_dpo = False,
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        r"""
        Computes the DPO loss and other metrics for the given batch of inputs for train or test.
        """

        if self.strategy in ["sft", "grpo"]:
            policy_logps_avg = self.concatenated_forward_for_chosen_sft(model, batch)
            losses = -policy_logps_avg

            prefix = "eval_" if train_eval == "eval" else ""
            return losses.mean(), metrics
        elif self.strategy in ["vtcl", "mdpo", "ori-mdpo"]:
            (
                policy_chosen_logps,
                policy_rejected_logps,
                policy_chosen_logits,
                policy_rejected_logits,
                policy_chosen_logps_avg,
            ) = self.concatenated_forward(model, batch)

            reference_chosen_logps, reference_rejected_logps = self.compute_reference_log_probs(model, batch)
            dpo, mdpo, anc_losses, cr, rr, cocr, corr = self.compute_preference_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
            )
            sft_loss = -policy_chosen_logps_avg

            prefix = "eval_" if train_eval == "eval" else ""
            if not use_normal_dpo:
                metrics["{}loss/mdpo".format(prefix)] = mdpo.mean().cpu()
            metrics["{}loss/text_dpo".format(prefix)] = dpo.mean().cpu()
            if use_normal_dpo:
                losses = dpo
            else:
                losses = mdpo + dpo
            if anc_losses is not None:
                metrics["{}losses/AncPO".format(prefix)] = anc_losses.mean().cpu()
                losses +=  + anc_losses.mean()

            reward_accuracies = (cr > rr).float()
            if self.vtcl_mode != "sameBatch":
                policy_rejected_logps = policy_rejected_logps[::2]
            logps_accuracies = (policy_chosen_logps > policy_rejected_logps).float()
            co_reward_accuracies = (cocr > corr).float()

            metrics["{}COrewards/chosen".format(prefix)] = cocr.mean().cpu()
            metrics["{}COrewards/rejected".format(prefix)] = corr.mean().cpu()
            metrics["{}COrewards/accuracies".format(prefix)] = co_reward_accuracies.mean().cpu()
            metrics["{}COrewards/margins".format(prefix)] = (cocr - corr).mean().cpu()
            metrics["{}rewards/chosen".format(prefix)] = cr.mean().cpu()
            metrics["{}rewards/rejected".format(prefix)] = rr.mean().cpu()
            metrics["{}rewards/accuracies".format(prefix)] = reward_accuracies.mean().cpu()
            metrics["{}rewards/margins".format(prefix)] = (cr - rr).mean().cpu()
            metrics["{}logps/rejected".format(prefix)] = policy_rejected_logps.detach().mean().cpu()
            metrics["{}logps/chosen".format(prefix)] = policy_chosen_logps.detach().mean().cpu()
            metrics["{}logps/accuracies".format(prefix)] = logps_accuracies.mean().cpu()
            metrics["{}logits/rejected".format(prefix)] = policy_rejected_logits.detach().mean().cpu()
            metrics["{}logits/chosen".format(prefix)] = policy_chosen_logits.detach().mean().cpu()

            return losses.mean(), metrics
        elif self.strategy in ["orm"]:
            (
                policy_chosen_logps,
                policy_rejected_logps,
                policy_chosen_logits,
                policy_rejected_logits,
                policy_chosen_logps_avg,
            ) = self.concatenated_forward(model, batch)
            
            reference_chosen_logps, reference_rejected_logps = self.compute_reference_log_probs(model, batch)
            cdpo, rdpo, dpo, ccreward, rcreward, crreward, noImgreward, anchor_loss = self.compute_preference_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
            )
            prefix = "eval_" if train_eval == "eval" else ""
            metrics["{}loss/cdpo".format(prefix)] = cdpo.mean().cpu()
            metrics["{}loss/rdpo".format(prefix)] = rdpo.mean().cpu()
            metrics["{}loss/text_dpo".format(prefix)] = dpo.mean().cpu()
            if self.half_dpo_loss:
                losses = (cdpo + rdpo) / 2 + dpo
            else:
                losses = cdpo + rdpo + dpo
            if anchor_loss is not None:
                metrics["{}losses/AncPO".format(prefix)] = anchor_loss.mean().cpu()
                losses += anchor_loss.mean()

            sft_loss = -policy_chosen_logps_avg

            if self.ftx_gamma > 1e-6:
                losses += self.ftx_gamma * sft_loss

            p_rejected = torch.chunk(policy_rejected_logps, 3)[1]
            logps_accuracies = (policy_chosen_logps > p_rejected).float()

            metrics["{}rewards/chosen".format(prefix)] = ccreward.detach().mean().cpu()
            metrics["{}rewards/noImage".format(prefix)] = noImgreward.detach().mean().cpu()
            metrics["{}rewards/rejected".format(prefix)] = crreward.detach().mean().cpu()
            metrics["{}rewards/neg_img".format(prefix)] = rcreward.detach().mean().cpu()
            metrics["{}rewards/margins".format(prefix)] = (ccreward - crreward).detach().mean().cpu()
            metrics["{}logps/rejected".format(prefix)] = p_rejected.detach().mean().cpu()
            metrics["{}logps/chosen".format(prefix)] = policy_chosen_logps.detach().detach().mean().cpu()
            metrics["{}logps/accuracies".format(prefix)] = logps_accuracies.mean().cpu()
            
            return losses.mean(), metrics
        elif self.strategy in ["svco-ref", "svco-textref"]:
            (
                policy_chosen_logps,
                policy_rejected_logps,
                policy_chosen_logits,
                policy_rejected_logits,
                policy_chosen_logps_avg,
            ) = self.concatenated_forward(model, batch)
            
            reference_chosen_logps, reference_rejected_logps = self.compute_reference_log_probs(model, batch)
            cdpo, rdpo, anchor_loss, cr, noImgr, rr = self.compute_preference_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
            )
            prefix = "eval_" if train_eval == "eval" else ""
            metrics["{}loss/cdpo".format(prefix)] = cdpo.mean().cpu()
            metrics["{}loss/rdpo".format(prefix)] = rdpo.mean().cpu()
            losses = (cdpo + rdpo) / 2
            if anchor_loss is not None:
                metrics["{}loss/dpo".format(prefix)] = losses.mean().cpu()
                metrics["{}losses/AncPO".format(prefix)] = anchor_loss.mean().cpu()
                losses += anchor_loss.mean()

            sft_loss = -policy_chosen_logps_avg

            if self.ftx_gamma > 1e-6:
                losses += self.ftx_gamma * sft_loss

            reward_accuracies = (cr > rr).float()
            p_rejected = torch.chunk(policy_rejected_logps, 2)[0]
            logps_accuracies = (policy_chosen_logps > p_rejected).float()

            metrics["{}rewards/chosen".format(prefix)] = cr.detach().mean().cpu()
            metrics["{}rewards/noImage".format(prefix)] = noImgr.detach().mean().cpu()
            metrics["{}rewards/rejected".format(prefix)] = rr.detach().mean().cpu()
            metrics["{}rewards/accuracies".format(prefix)] = reward_accuracies.detach().mean().cpu()
            metrics["{}rewards/margins".format(prefix)] = (cr - rr).detach().mean().cpu()
            metrics["{}logps/rejected".format(prefix)] = p_rejected.detach().mean().cpu()
            metrics["{}logps/chosen".format(prefix)] = policy_chosen_logps.detach().detach().mean().cpu()
            metrics["{}logps/accuracies".format(prefix)] = logps_accuracies.mean().cpu()
            
            return losses.mean(), metrics
        elif self.strategy in ["dpo", "svco-noref"]:
            (
                policy_chosen_logps,
                policy_rejected_logps,
                policy_chosen_logits,
                policy_rejected_logits,
                policy_chosen_logps_avg,
            ) = self.concatenated_forward(model, batch)
            
            reference_chosen_logps, reference_rejected_logps = self.compute_reference_log_probs(model, batch)
            losses, cr, rr, anchor_loss = self.compute_preference_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
            )

            prefix = "eval_" if train_eval == "eval" else ""
            if anchor_loss is not None:
                metrics["{}loss/dpo".format(prefix)] = losses.mean().cpu()
                metrics["{}losses/AncPO".format(prefix)] = anchor_loss.mean().cpu()
                losses += anchor_loss.mean()

            sft_loss = -policy_chosen_logps_avg

            if self.ftx_gamma > 1e-6:
                losses += self.ftx_gamma * sft_loss

            reward_accuracies = (cr > rr).float()
            logps_accuracies = (policy_chosen_logps > policy_rejected_logps).float()

            metrics["{}rewards/chosen".format(prefix)] = cr.mean().cpu()
            metrics["{}rewards/rejected".format(prefix)] = rr.mean().cpu()
            metrics["{}rewards/accuracies".format(prefix)] = reward_accuracies.mean().cpu()
            metrics["{}rewards/margins".format(prefix)] = (cr - rr).mean().cpu()
            metrics["{}logps/rejected".format(prefix)] = policy_rejected_logps.detach().mean().cpu()
            metrics["{}logps/chosen".format(prefix)] = policy_chosen_logps.detach().mean().cpu()
            metrics["{}logps/accuracies".format(prefix)] = logps_accuracies.mean().cpu()
            metrics["{}logits/rejected".format(prefix)] = policy_rejected_logits.detach().mean().cpu()
            metrics["{}logits/chosen".format(prefix)] = policy_chosen_logits.detach().mean().cpu()
            
            return losses.mean(), metrics
        else:
            raise NotImplementedError
        

class OnlineCustomDPOTrainer(CustomDPOTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        vtcl_args: "VTCLArguments",
        processor: Optional["ProcessorMixin"],
        disable_dropout: bool = True,
        **kwargs,
    ):
        processed_vtcl_args = copy.deepcopy(vtcl_args)
        processed_vtcl_args.strategy = vtcl_args.strategy.replace("online-", '')
        super().__init__(model, ref_model, finetuning_args, processed_vtcl_args, processor, disable_dropout, **kwargs)
        self.processor = processor
        self.PADDING_TOKEN_ID = self.tokenizer.pad_token_id
        # to absolute path
        self.online_data_save_dir = os.path.abspath(vtcl_args.online_data_save_dir)
        if not os.path.exists(self.online_data_save_dir):
            os.makedirs(self.online_data_save_dir)
        self._sample_data_lock = FileLock(self.online_data_save_dir.rstrip('/') + "/online_sample_data.lock")

    def _prepare_for_sampling(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
    ) -> List[Dict[str, "torch.Tensor"]]:
        # input_ids attention_mask labels pixel_values
        total = batch["input_ids"].shape[0]
        if self.strategy in ["dpo"]:
            total = batch["input_ids"].shape[0] // 2
        elif self.strategy in ["orm"]:
            total = batch["input_ids"].shape[0] // 4
        elif self.strategy in ["sft", "grpo"]:
            total = batch["input_ids"].shape[0]
        else:
            raise NotImplementedError

        all_batch = []
        for idx in range(total):
            input_ids = batch["input_ids"][idx]
            attention_mask = batch["attention_mask"][idx]
            labels = batch["labels"][idx]
            mm_inputs = {k: batch[k][idx][None, :] for k in batch.keys() if k not in ["input_ids", "attention_mask", "labels"]}
            mask = labels == -100
            input_ids = input_ids[mask]
            ground_truth = self.tokenizer.decode(labels[~mask], skip_special_tokens=True)
            attention_mask = attention_mask[mask]
            mask = input_ids == self.PADDING_TOKEN_ID
            input_ids = input_ids[~mask]
            attention_mask = attention_mask[~mask]
            all_batch.append({
                "input_ids": input_ids[None, :],
                "attention_mask": attention_mask[None, :],
                "ground_truth": ground_truth,
                **mm_inputs,
            })
        return all_batch
    
    def _get_critic_score(self, prompt, response, ground_truth):
        while True:
            try:
                instruct = requests.post("http://localhost:8002/critic", json={"prompt": prompt, "response": response, "answer": ground_truth})
                instruct.raise_for_status()
                return instruct.json()["critic"]
            except Exception as e:
                print(f"[INFO-rank{self._proc_idx_label}]: An error occurred: {e}")
                time.sleep(1)

    def _get_img_generation_instruction(self, prompt, response, ground_truth):
        # json's response should be the correct one while gt is the wrong one.
        while True:
            try:
                instruct = requests.post("http://localhost:8002/describe", json={"prompt": prompt, "response": response, "ground_truth": ground_truth})
                instruct.raise_for_status()
                return instruct.json()["description"]
            except Exception as e:
                print(f"[INFO-rank{self._proc_idx_label}]: An error occurred: {e}")
                time.sleep(3)
    
    def _get_img_from_instruction(self, prompt: str, info: Dict[str, str], height=384, width=384, num_inference_steps=40, guidance_scale=7.5):
        port = '8001' if int(self._proc_idx_label) <=1 else '8003'
        while True:
            try:
                img_path = requests.post(f"http://localhost:{port}/generate", json={"prompt": prompt, "h": height, "w": width, "save_dir": self.online_data_save_dir, "info": info, "num_inference_steps": num_inference_steps, "guidance_scale": guidance_scale})
                img_path.raise_for_status()
                img_path = img_path.json()["path"]
                # open image
                neg_mm_inputs = self.data_collator.template.mm_plugin.get_mm_inputs([img_path], [], [1], [0], [], self.data_collator.processor)
                return neg_mm_inputs
            except Exception as e:
                print(f"[INFO-rank{self._proc_idx_label}]: An error occurred: {e}")
                time.sleep(3)
    
    def _sample_responses(self, input_batch, model, k=16):
        sampled_responses = []
        model.eval()
        for inputs in input_batch:
            ground_truth = inputs.pop("ground_truth")
            with torch.no_grad():
                outputs = self.tokenizer.batch_decode(self.accelerator.unwrap_model(model).generate(**inputs, num_return_sequences=k, temperature=1.2, max_new_tokens=256, do_sample=True, top_p=0.8), skip_special_tokens=True)
            new_outputs = []
            for output in outputs:
                # llava1.5
                split_output = output.split("ASSISTANT:", 1)
                if len(split_output) > 1:
                    new_outputs.append(split_output[1].strip())
                else:
                    new_outputs.append(output)
            outputs = list(set(new_outputs))
            prompt = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
            # for llava1.5
            prompt = prompt.split("USER:", 1)[1].split("ASSISTANT:", 1)[0].strip()
            sampled_responses.extend([{
                "prompt": prompt,
                "response": output,
                "ground_truth": ground_truth,
                "pixel_values": inputs["pixel_values"],
            } for output in outputs])
        model.train()
        return sampled_responses
        
    def _encode_text(
        self,
        prompt,
        chosen,
        rejected
    ):
        prompt_ids, chosen_ids = self.data_collator.template.encode_oneturn(self.tokenizer, [{"role": "user", "content": "<image>"*576+"\n"+prompt}, {"role": "assistant", "content": chosen}])
        _, rejected_ids = self.data_collator.template.encode_oneturn(self.tokenizer, [{"role": "user", "content": "<image>"*576+"\n"+prompt}, {"role": "assistant", "content": rejected}])
        chosen_ids += [self.tokenizer.eos_token_id]
        rejected_ids += [self.tokenizer.eos_token_id]
        source_len, target_len = infer_seqlen(len(prompt_ids), max(len(chosen_ids), len(rejected_ids)), 2048)
        prompt_ids = prompt_ids[:source_len]
        chosen_ids = chosen_ids[:target_len]
        rejected_ids = rejected_ids[:target_len]

        chosen_input_ids = prompt_ids + chosen_ids
        chosen_labels = [IGNORE_INDEX] * source_len + chosen_ids
        rejected_input_ids = prompt_ids + rejected_ids
        rejected_labels = [IGNORE_INDEX] * source_len + rejected_ids
        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": [1] * len(chosen_input_ids),
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": [1] * len(rejected_input_ids),
            "rejected_labels": rejected_labels,
        }
    
    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
        metrics={},
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        r"""
        Computes the DPO loss and other metrics for the given batch of inputs for train or test.
        """
        # breakpoint()
        use_dpo = True
        USE_DUMMY = False
        if train_eval == "train":
            input_batch = self._prepare_for_sampling(model, batch)
            sampled_batch = self._sample_responses(input_batch, model, k=16)
            print(f"[INFO-rank{self._proc_idx_label}]: Sampling completed.")
            sampled_batch = [{**item, "score": self._get_critic_score(item["prompt"], item["response"], item["ground_truth"])} for item in sampled_batch]
            sampled_batch = [item for item in sampled_batch if item["score"] != "none"]
            if len(sampled_batch) == 0:
                print(f"[INFO-rank{self._proc_idx_label}]: No valid sample.")
                USE_DUMMY = True
            else:
                print(f"[INFO-rank{self._proc_idx_label}]: Scoring completed. Average score: {sum([item['score'] for item in sampled_batch])*1.0/len(sampled_batch):.2f}.")
                with self._sample_data_lock:
                    with open(f"{self.online_data_save_dir}/online_sample_data.jsonl", "a") as f:
                        f.write(json.dumps({"sample": [{"score": item["score"], "response": item["response"], "prompt": item["prompt"], "ground_truth": item["ground_truth"]} for item in sampled_batch], "num_sampled": len(sampled_batch), "steps": self.state.global_step, "node_rank": self._proc_idx_label}) + "\n")
                print(f"[INFO-rank{self._proc_idx_label}]: Data saved.")
                prefix = "eval_" if train_eval == "eval" else ""
                metrics["{}generation_score".format(prefix)] = torch.tensor([item["score"] for item in sampled_batch], dtype=torch.float16).mean().cpu()

                aggregated_batch = defaultdict(list)
                for item in sampled_batch:
                    aggregated_batch[item["prompt"]].append(item)
                sampled_batch = list(aggregated_batch.values())  # [[same prompt], []]

                training_batch = []
                for minibatch in sampled_batch:
                    # ordering
                    minibatch = sorted(minibatch, key=lambda x: x["score"], reverse=True)
                    if 1 <= len(minibatch) <= 4:
                        if minibatch[0]["score"] - minibatch[-1]["score"] >= 4 and minibatch[0]["score"] >= 6:
                            training_batch.append({"chosen": minibatch[0], "rejected": minibatch[-1]})
                        elif minibatch[-1]["score"] <= 5:
                            training_batch.append({"chosen": None, "rejected": minibatch[-1]})
                    else:
                        mean = sum([item["score"] for item in minibatch]) / len(minibatch)
                        var = sum([(item["score"] - mean) ** 2 for item in minibatch]) / (len(minibatch)-1)
                        std = max(1.5, var ** 0.5)
                        idx = 0
                        while minibatch[idx]["score"] - minibatch[-1-idx]["score"] >  2*std and minibatch[idx]["score"] >= 6:
                            training_batch.append({"chosen": minibatch[idx], "rejected": minibatch[-1-idx]})
                            idx += 1
                        if idx == 0 and minibatch[-1]["score"] <= 5:
                            training_batch.append({"chosen": None, "rejected": minibatch[-1]})
                training_batch = training_batch[:3]
                if training_batch != []:
                    use_dpo = False
                    print(f"[INFO-rank{self._proc_idx_label}]: Get training text preference data.")
                    for it in training_batch:
                        if it["chosen"] is None:
                            it["chosen"] = copy.deepcopy(it["rejected"])
                            it["chosen"]["response"] = it["chosen"]["ground_truth"]
                            it["chosen"]["score"] = 11
                    training_batch = [{"prompt": item["chosen"]["prompt"], "chosen_response": item["chosen"]["response"], "rejected_response": item["rejected"]["response"], "chosen_pixel_values": item["chosen"]["pixel_values"], "rejected_pixel_values": None, "chosen_score": item["chosen"]["score"], "rejected_score": item["rejected"]["score"]} for item in training_batch]
                    if self.strategy not in ["dpo", "sft"]:
                        training_batch = [{**item, "desc": self._get_img_generation_instruction(item["prompt"], item["chosen_response"], item["rejected_response"])} for item in training_batch]
                        print(f"[INFO-rank{self._proc_idx_label}]: Get neg image instruction.")
                        training_batch = [{**item, "rejected_pixel_values": self._get_img_from_instruction(item["desc"], {"step": self.state.global_step, **{k: item[k] for k in ["prompt", "chosen_response", "rejected_response", "chosen_score", "rejected_score"]}}, 384, 384)["pixel_values"]} for item in training_batch]
                        print(f"[INFO-rank{self._proc_idx_label}]: Get neg image pixel values.")
                    else:
                        training_batch = [{**item, "rejected_pixel_values": item["chosen_pixel_values"]} for item in training_batch]
                    # breakpoint()
                    training_batch = [{"images": [item["chosen_pixel_values"], item["rejected_pixel_values"]], "videos": [], **self._encode_text(item["prompt"], item["chosen_response"], item["rejected_response"])} for item in training_batch]
                    batch = self.data_collator(training_batch, skip_mm_inputs=True)
                    for k, v in batch.items():
                        batch[k] = v.to(model.device)
                    metrics = {k: v for k, v in metrics.items() if not k.startswith("eval_")}
                else:
                    print(f"[INFO-rank{self._proc_idx_label}]: No training data.")
                    USE_DUMMY = True
        loss, metrics = super().get_batch_loss_metrics(model, batch, train_eval, metrics, use_dpo)
        if USE_DUMMY:
            return loss * 0.0, {"skip": 1.0}
        metrics["skip"] = 0.0
        return loss, metrics


class OnlineCustomDPOTrainerWithBuffer(OnlineCustomDPOTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        vtcl_args: "VTCLArguments",
        processor: Optional["ProcessorMixin"],
        disable_dropout: bool = True,
        **kwargs,
    ):
        import json
        import random
        ds_path = vtcl_args.ds_path
        super().__init__(
            model,
            ref_model,
            finetuning_args,
            vtcl_args,
            processor,
            disable_dropout,
            **kwargs,
        )
        self._use_experience_buffer = vtcl_args.experience_buffer
        self._exp_buffer = []
        self.exp_per_batch = vtcl_args.exp_per_batch
        self.online_only = vtcl_args.online_only
        self.online_only_strategy = vtcl_args.online_only_strategy

        random.seed(0)
        with open(ds_path) as f:
            data = json.load(f)
            random.shuffle(data)
        self._real_dataset = []
        for item in data:
            # prompt+chosen
            value = {"chosen_img_path": item["images"][0], "prompt": item["conversations"][0]["value"].split("<image>", 1)[-1].strip(), "chosen": item["chosen"]["value"], "rejected": item["rejected"]["value"]}
            self._real_dataset.append(value)
        self._ds_len = len(self._real_dataset)
        self._load_new_sample_data_lock = FileLock(self.online_data_save_dir.rstrip('/') + "/load_new_sample_data.lock")
        if self._proc_idx_label == 0:
            self._set_global_ds_idx(0)
        
    def _set_global_ds_idx(self, idx):
        with open(self.online_data_save_dir.rstrip('/') + "/real_ds_idx.txt", 'w') as f:
            f.write(str(idx))

    def _get_global_ds_idx(self):
        with open(self.online_data_save_dir.rstrip('/') + "/real_ds_idx.txt", 'r') as f:
            return int(f.read())
        
    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
        metrics={},
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        r"""
        Computes the DPO loss and other metrics for the given batch of inputs for train or test.
        """
        # breakpoint()
        should_epoch_stop = torch.tensor(False, dtype=torch.bool, device=model.device)
        if train_eval == "train":
            with self._sample_data_lock:
                with open(f"{self.online_data_save_dir}/time_log.jsonl", "a") as f:
                    f.write(f"[INFO-rank{self._proc_idx_label}][backward][end]: "+self.get_time() + "\n")
            while len(self._exp_buffer) < self.exp_per_batch:
                with self._load_new_sample_data_lock:
                    idx = self._get_global_ds_idx()
                    if idx >= self._ds_len:
                        should_epoch_stop = torch.tensor(True, dtype=torch.bool, device=model.device)
                        idx -= self._ds_len
                    idx += 1
                    self._set_global_ds_idx(idx)
                data_item = self._real_dataset[(idx-1)%self._ds_len]
                real_batch = [{"images": [data_item["chosen_img_path"], data_item["chosen_img_path"]], "videos": [], **self._encode_text(data_item["prompt"], data_item["chosen"], data_item["rejected"])}]
                batch = self.data_collator(real_batch)
                for k, v in batch.items():
                    batch[k] = v.to(model.device)
                input_batch = self._prepare_for_sampling(model, batch)
                with self._sample_data_lock:
                    with open(f"{self.online_data_save_dir}/time_log.jsonl", "a") as f:
                        f.write(f"[INFO-rank{self._proc_idx_label}][sample][start]: "+self.get_time() + "\n")
                sampled_batch = self._sample_responses(input_batch, model, k=16)
                with self._sample_data_lock:
                    with open(f"{self.online_data_save_dir}/time_log.jsonl", "a") as f:
                        f.write(f"[INFO-rank{self._proc_idx_label}][sample][end]: "+self.get_time() + "\n")
                print(f"[INFO-rank{self._proc_idx_label}]: Sampling completed.")
                with self._sample_data_lock:
                    with open(f"{self.online_data_save_dir}/time_log.jsonl", "a") as f:
                        f.write(f"[INFO-rank{self._proc_idx_label}][score][start]: "+self.get_time() + "\n")
                sampled_batch = [{**item, "score": self._get_critic_score(item["prompt"], item["response"], item["ground_truth"])} for item in sampled_batch]
                with self._sample_data_lock:
                    with open(f"{self.online_data_save_dir}/time_log.jsonl", "a") as f:
                        f.write(f"[INFO-rank{self._proc_idx_label}][score][end]: "+self.get_time() + "\n")
                sampled_batch = [item for item in sampled_batch if item["score"] != "none"]
                if len(sampled_batch) == 0:
                    print(f"[INFO-rank{self._proc_idx_label}]: No valid sample.")
                    continue
                print(f"[INFO-rank{self._proc_idx_label}]: Scoring completed. Average score: {sum([item['score'] for item in sampled_batch])*1.0/len(sampled_batch):.2f}.")
                with self._sample_data_lock:
                    with open(f"{self.online_data_save_dir}/online_sample_data.jsonl", "a") as f:
                        f.write(json.dumps({"sample": [{"score": item["score"], "response": item["response"], "prompt": item["prompt"], "ground_truth": item["ground_truth"]} for item in sampled_batch], "num_sampled": len(sampled_batch), "steps": self.state.global_step, "node_rank": self._proc_idx_label}) + "\n")
                print(f"[INFO-rank{self._proc_idx_label}]: Data saved.")
                prefix = "eval_" if train_eval == "eval" else ""
                metrics["{}generation_score".format(prefix)] = torch.tensor([item["score"] for item in sampled_batch], dtype=torch.float16).mean().cpu()

                aggregated_batch = defaultdict(list)
                for item in sampled_batch:
                    aggregated_batch[item["prompt"]].append(item)
                sampled_batch = list(aggregated_batch.values())  # [[same prompt], []]

                training_batch = []
                for minibatch in sampled_batch:
                    # ordering
                    minibatch = sorted(minibatch, key=lambda x: x["score"], reverse=True)
                    if 1 <= len(minibatch) <= 4:
                        if minibatch[0]["score"] - minibatch[-1]["score"] >= 4 and minibatch[0]["score"] >= 6:
                            training_batch.append({"chosen": minibatch[0], "rejected": minibatch[-1]})
                        elif minibatch[-1]["score"] <= 5:
                            if not self.online_only:
                                training_batch.append({"chosen": None, "rejected": minibatch[-1]})
                            elif self.online_only_strategy == "addtional":
                                if minibatch[0]["score"] - minibatch[-1]["score"] >= 2:
                                    training_batch.append({"chosen": minibatch[0], "rejected": minibatch[-1]})
                    else:
                        mean = sum([item["score"] for item in minibatch]) / len(minibatch)
                        var = sum([(item["score"] - mean) ** 2 for item in minibatch]) / (len(minibatch)-1)
                        std = max(1.5, var ** 0.5)
                        idx = 0
                        while minibatch[idx]["score"] - minibatch[-1-idx]["score"] >  2*std and minibatch[idx]["score"] >= 6:
                            training_batch.append({"chosen": minibatch[idx], "rejected": minibatch[-1-idx]})
                            idx += 1
                        if idx == 0 and minibatch[-1]["score"] <= 5:
                            if not self.online_only:
                                training_batch.append({"chosen": None, "rejected": minibatch[-1]})
                            elif self.online_only_strategy == "addtional":
                                if minibatch[0]["score"] - minibatch[-1]["score"] >= 2:
                                    training_batch.append({"chosen": minibatch[0], "rejected": minibatch[-1]})
                    self._exp_buffer.extend(training_batch)

            training_batch, self._exp_buffer = self._exp_buffer[:self.exp_per_batch], self._exp_buffer[self.exp_per_batch:]
            print(f"[INFO-rank{self._proc_idx_label}]: Get training text preference data.")
            for it in training_batch:
                if it["chosen"] is None:
                    it["chosen"] = copy.deepcopy(it["rejected"])
                    it["chosen"]["response"] = it["chosen"]["ground_truth"]
                    it["chosen"]["score"] = 11
            training_batch = [{"prompt": item["chosen"]["prompt"], "chosen_response": item["chosen"]["response"], "rejected_response": item["rejected"]["response"], "chosen_pixel_values": item["chosen"]["pixel_values"], "rejected_pixel_values": None, "chosen_score": item["chosen"]["score"], "rejected_score": item["rejected"]["score"]} for item in training_batch]
            if self.strategy not in ["dpo", "sft"]:
                training_batch = [{**item, "desc": self._get_img_generation_instruction(item["prompt"], item["chosen_response"], item["rejected_response"])} for item in training_batch]
                with self._sample_data_lock:
                    with open(f"{self.online_data_save_dir}/time_log.jsonl", "a") as f:
                        f.write(f"[INFO-rank{self._proc_idx_label}][desc]: "+self.get_time() + "\n")
                print(f"[INFO-rank{self._proc_idx_label}]: Get neg image instruction.")
                training_batch = [{**item, "rejected_pixel_values": self._get_img_from_instruction(item["desc"], {"step": self.state.global_step, **{k: item[k] for k in ["prompt", "chosen_response", "rejected_response", "chosen_score", "rejected_score"]}}, 384, 384)["pixel_values"]} for item in training_batch]
                with self._sample_data_lock:
                    with open(f"{self.online_data_save_dir}/time_log.jsonl", "a") as f:
                        f.write(f"[INFO-rank{self._proc_idx_label}][neg_img]: "+self.get_time() + "\n")
                print(f"[INFO-rank{self._proc_idx_label}]: Get neg image pixel values.")
            else:
                training_batch = [{**item, "rejected_pixel_values": item["chosen_pixel_values"]} for item in training_batch]
            # breakpoint()
            training_batch = [{"images": [item["chosen_pixel_values"], item["rejected_pixel_values"]], "videos": [], **self._encode_text(item["prompt"], item["chosen_response"], item["rejected_response"])} for item in training_batch]
            batch = self.data_collator(training_batch, skip_mm_inputs=True)
            for k, v in batch.items():
                batch[k] = v.to(model.device)
            metrics = {k: v for k, v in metrics.items() if not k.startswith("eval_")}
        loss, metrics = super(OnlineCustomDPOTrainer, self).get_batch_loss_metrics(model, batch, train_eval, metrics, use_normal_dpo=train_eval=="eval")
        with self._sample_data_lock:
            with open(f"{self.online_data_save_dir}/time_log.jsonl", "a") as f:
                f.write(f"[INFO-rank{self._proc_idx_label}][forward]: "+self.get_time() + "\n")
        if train_eval == "train":
            gathered = self.accelerator.gather(should_epoch_stop)
            if gathered.any():
                self.control.should_epoch_stop = True
        return loss, metrics


class GRPOTrainerWithBuffer(OnlineCustomDPOTrainerWithBuffer):
    def __init__(self, *args, **kwargs):
        self.num_samples_per_minibatch = 6
        self.step_per_sample = 3  # 6 samples to train per minibatch, 3 steps = 18samples > 17
        super().__init__(*args, **kwargs)
        self.grpo_update_step = 0

    def _get_batch_pertoken_logps(self, logits, labels, label_pad_token_id=IGNORE_INDEX):
        if logits.shape[:-1] != labels.shape:
            raise ValueError
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id
        labels[labels == label_pad_token_id] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        return per_token_logps, loss_mask

    def _grpo_get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
        score_batch,
        train_eval: Literal["train", "eval"] = "train",
        metrics={},
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        all_logits = model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)
        per_token_logps, loss_mask = self._get_batch_pertoken_logps(all_logits, batch["labels"])
        score = torch.tensor(score_batch, dtype=torch.float32, device=per_token_logps.device).unsqueeze(1)
        with torch.no_grad(), self.accelerator.unwrap_model(model).disable_adapter():
            all_logits_ref = model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)
            per_token_logps_ref, _ = self._get_batch_pertoken_logps(all_logits_ref, batch["labels"])
        logp_dif = per_token_logps_ref - per_token_logps
        per_token_KL_loss = logp_dif.exp() - logp_dif - 1
        KL_loss = per_token_KL_loss * loss_mask / loss_mask.sum(-1, keepdim=True)
        J = score * (per_token_logps - per_token_logps.detach()).exp() * loss_mask / loss_mask.sum(-1, keepdim=True) - self.beta * KL_loss
        J = J.sum(-1)
        print(f"[INFO-rank{self._proc_idx_label}]: J={J.detach().cpu().numpy().tolist()}")
        metrics[f"{'eval_' if train_eval=='eval' else ''}KL_loss"] = KL_loss.detach().mean().cpu()
        return -J.mean(), metrics
        
    def get_batch_loss_metrics(
        self,
        model,
        batch,
        train_eval,
        metrics={}
    ):
        should_epoch_stop = torch.tensor(False, dtype=torch.bool, device=model.device)
        if train_eval == "train":
            self.grpo_update_step += 1
            print(f"[INFO-rank{self._proc_idx_label}]: Step {self.grpo_update_step} starts!")
            if self.grpo_update_step % self.step_per_sample == 1:
                with self._load_new_sample_data_lock:
                    idx = self._get_global_ds_idx()
                    if idx >= self._ds_len:
                        should_epoch_stop = torch.tensor(True, dtype=torch.bool, device=model.device)
                        idx -= self._ds_len
                    idx += 1
                    self._set_global_ds_idx(idx)
                data_item = self._real_dataset[(idx-1)%self._ds_len]
                real_batch = [{"images": [data_item["chosen_img_path"], data_item["chosen_img_path"]], "videos": [], **self._encode_text(data_item["prompt"], data_item["chosen"], data_item["rejected"])}]
                batch = self.data_collator(real_batch)
                for k, v in batch.items():
                    batch[k] = v.to(model.device)
                input_batch = self._prepare_for_sampling(model, batch)
                sampled_batch = self._sample_responses(input_batch, model, k=16)
                print(f"[INFO-rank{self._proc_idx_label}]: Sampling completed.")
                sampled_batch = [{**item, "score": self._get_critic_score(item["prompt"], item["response"], item["ground_truth"])} for item in sampled_batch]
                sampled_batch = [item for item in sampled_batch if item["score"] != "none"]
                sampled_batch.append({"prompt": sampled_batch[-1]["prompt"], "response": sampled_batch[-1]["ground_truth"], "ground_truth": sampled_batch[-1]["ground_truth"], "pixel_values": sampled_batch[-1]["pixel_values"], "score": 10})
                if len(sampled_batch) == 0:
                    print(f"[INFO-rank{self._proc_idx_label}]: No valid sample.")
                else:
                    print(f"[INFO-rank{self._proc_idx_label}]: Scoring completed. Average score: {sum([item['score'] for item in sampled_batch])*1.0/len(sampled_batch):.2f}.")
                    with self._sample_data_lock:
                        with open(f"{self.online_data_save_dir}/online_sample_data.jsonl", "a") as f:
                            f.write(json.dumps({"sample": [{"score": item["score"], "response": item["response"], "prompt": item["prompt"], "ground_truth": item["ground_truth"]} for item in sampled_batch], "num_sampled": len(sampled_batch), "steps": self.state.global_step, "node_rank": self._proc_idx_label}) + "\n")
                    print(f"[INFO-rank{self._proc_idx_label}]: Data saved.")
                    prefix = "eval_" if train_eval == "eval" else ""
                    metrics["{}generation_score".format(prefix)] = torch.tensor([item["score"] for item in sampled_batch], dtype=torch.float16).mean().cpu()

                    aggregated_batch = defaultdict(list)
                    for item in sampled_batch:
                        aggregated_batch[item["prompt"]].append(item)
                    sampled_batch = list(aggregated_batch.values())  # [[same prompt], []]

                    training_batch = []
                    for minibatch in sampled_batch:
                        # ordering
                        mini_mean = sum([item["score"] for item in minibatch]) / len(minibatch)
                        mini_std = (sum([(item["score"] - mini_mean) ** 2 for item in minibatch]) / (len(minibatch))) ** 0.5
                        minibatch = sorted(minibatch, key=lambda x: x["score"], reverse=True)
                        for item in minibatch:
                            item["score"] = (item["score"] - mini_mean) / (mini_std + 1e-6)
                        self._exp_buffer.extend(minibatch)
            training_batch, self._exp_buffer = self._exp_buffer[:self.num_samples_per_minibatch], self._exp_buffer[self.num_samples_per_minibatch:]
            
            gathered = self.accelerator.gather(should_epoch_stop)
            if gathered.any():
                self.control.should_epoch_stop = True
            
            if len(training_batch) == 0:
                ratio = 0.0
                metrics["num_samples"] = 0.0
                score_batch = [0.0 for _ in batch["input_ids"]]
            else:
                ratio = 1.0
                
                metrics["num_samples"] = len(training_batch)*1.0
                score_batch = [item["score"] for item in training_batch]
                training_batch = [{"prompt": item["prompt"], "chosen_response": item["response"], "rejected_response": item["response"], "chosen_pixel_values": item["pixel_values"], "rejected_pixel_values": item["pixel_values"]} for item in training_batch]
                training_batch = [{"images": [item["chosen_pixel_values"], item["rejected_pixel_values"]], "videos": [], **self._encode_text(item["prompt"], item["chosen_response"], item["rejected_response"])} for item in training_batch]
                
                batch = self.data_collator(training_batch, skip_mm_inputs=True)
                for k, v in batch.items():
                    batch[k] = v.to(model.device)
                metrics = {k: v for k, v in metrics.items() if not k.startswith("eval_")}
            loss, metrics = self._grpo_get_batch_loss_metrics(model, batch, score_batch, train_eval, metrics)
            loss = loss * ratio
        else:
            loss, metrics = super(OnlineCustomDPOTrainer, self).get_batch_loss_metrics(model, batch, train_eval, metrics, use_normal_dpo=True)
        return loss, metrics