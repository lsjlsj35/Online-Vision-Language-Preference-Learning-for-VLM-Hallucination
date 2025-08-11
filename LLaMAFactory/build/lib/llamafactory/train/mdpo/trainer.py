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

import warnings
from collections import defaultdict
from contextlib import nullcontext
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

from ...extras.constants import IGNORE_INDEX
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, get_batch_logps



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
            elif self.strategy == "dpo":
                raise NotImplementedError
            else:
                raise KeyError
        else:
            # single / difBatch -> single
            if self.strategy in ["vtcl", "mdpo"]:
                cr = policy_rejected_logps[::2]
                rc = policy_rejected_logps[1::2]
                crref = reference_rejected_logps[::2]
                rcref = reference_rejected_logps[1::2]

                dpo, creward, rreward = self.dpo_loss(policy_chosen_logps, cr, reference_chosen_logps, crref)
                mdpo, mcreward, mrreward = self.dpo_loss(policy_chosen_logps, rc, reference_chosen_logps, rcref)
                if self.anchor_ratio >= 1e-6:
                    anc_losses = self.dpo_loss(policy_chosen_logps, crref, reference_chosen_logps, crref)[0]
                else:
                    anc_losses = None
                return dpo, mdpo, anc_losses, creward, rreward, mcreward, mrreward
            elif self.strategy == "dpo":
                dpo, creward, rreward = self.dpo_loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps)
                return dpo, creward, rreward
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

        all_logits: "torch.Tensor" = model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)
        all_logps, valid_length = get_batch_logps(logits=all_logits, labels=batch["labels"])
        if self.loss_type in ["ipo", "orpo", "simpo"]:
            all_logps = all_logps / valid_length

        if self.strategy in ["vtcl", "mdpo"] and self.vtcl_mode != "sameBatch":
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
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        r"""
        Computes the DPO loss and other metrics for the given batch of inputs for train or test.
        """
        metrics = {}

        if self.strategy == "sft":
            policy_logps_avg = self.concatenated_forward_for_chosen_sft(model, batch)
            losses = -policy_logps_avg

            prefix = "eval_" if train_eval == "eval" else ""
            return losses.mean(), metrics
        elif self.strategy == "vtcl":
            (
                policy_chosen_logps,
                policy_rejected_logps,
                policy_chosen_logits,
                policy_rejected_logits,
                policy_chosen_logps_avg,
            ) = self.concatenated_forward(model, batch)

            reference_chosen_logps, reference_rejected_logps = self.compute_reference_log_probs(model, batch)
            losses, codpo, anc_losses, cr, rr, cocr, corr = self.compute_preference_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
            )
            sft_loss = -policy_chosen_logps_avg

            prefix = "eval_" if train_eval == "eval" else ""
            metrics["{}losses/DPO".format(prefix)] = losses.mean().cpu()
            if self.ftx_gamma > 1e-6:
                losses += self.ftx_gamma * sft_loss
                
            metrics["{}losses/CoDPO".format(prefix)] = codpo.mean().cpu()
            losses = losses.mean() + codpo.mean()
            if anc_losses is not None:
                metrics["{}losses/AncPO".format(prefix)] = anc_losses.mean().cpu()
                losses +=  + self.anchor_ratio * anc_losses.mean()

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
        elif self.strategy == "dpo":
            (
                policy_chosen_logps,
                policy_rejected_logps,
                policy_chosen_logits,
                policy_rejected_logits,
                policy_chosen_logps_avg,
            ) = self.concatenated_forward(model, batch)
            
            reference_chosen_logps, reference_rejected_logps = self.compute_reference_log_probs(model, batch)
            losses, cr, rr = self.compute_preference_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
            )
            sft_loss = -policy_chosen_logps_avg

            prefix = "eval_" if train_eval == "eval" else ""
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
    