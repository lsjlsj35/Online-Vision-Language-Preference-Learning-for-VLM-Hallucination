import argparse
from dataclasses import dataclass, field
from tqdm import tqdm
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
from datasets import Dataset, load_dataset, concatenate_datasets
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoModel, AutoConfig, HfArgumentParser, TrainingArguments

from ...data import get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.ploting import plot_loss
from ...hparams import ModelArguments
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push, create_ref_model
from .trainer import CustomDPOTrainer
from .dataloader import PairwiseDataCollatorWithPadding, TokenDifMMDataCollatorWithPadding, VTCLDataCollatorWithPadding


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, VTCLArguments

def get_batch_logps(
    logits: "torch.Tensor", labels: "torch.Tensor", label_pad_token_id: int = IGNORE_INDEX
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    r"""
    Computes the log probabilities of the given labels under the given logits.

    Returns:
        logps: A tensor of shape (batch_size,) containing the sum of log probabilities.
        valid_length: A tensor of shape (batch_size,) containing the number of non-masked tokens.
    """
    bs, seq, _ = logits.shape
    _, seq_no_pad = labels.shape
    assert seq == seq_no_pad + 575  # 24*24-1
    padded_label = torch.full((bs, seq), label_pad_token_id, device=labels.device, dtype=labels.dtype)
    padded_label[:, -seq_no_pad:] = labels
    # if logits.shape[:-1] != labels.shape:
    #     raise ValueError("Logits (batchsize x seqlen) and labels must have the same shape.")

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != label_pad_token_id
    labels[labels == label_pad_token_id] = 0  # dummy token
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    # breakpoint()
    return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)


def F(exp=None, cp=None):
    from ..callbacks import LogCallback
    from ...hparams import get_train_args
    callbacks = []
    callbacks.append(LogCallback())
    model_args, data_args, training_args, finetuning_args, generating_args, vtcl_args = get_train_args()
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="rm", **tokenizer_module)
    # model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train, retain_lora=True)
    model = AutoModelForVision2Seq.from_pretrained(model_args.model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, model_args.adapter_name_or_path[0], torch_dtype=torch.bfloat16, device_map="auto").eval()
    
    if vtcl_args.strategy == "vtcl":
        collator_cls = VTCLDataCollatorWithPadding
    elif vtcl_args.strategy == "tdpo":
        collator_cls = TokenDifMMDataCollatorWithPadding
    else:
        raise KeyError
    data_collator = collator_cls(
        mask_same=vtcl_args.mask_same_sequence,
        template=template,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        **tokenizer_module,
    )
    # breakpoint()
    batch = data_collator(dataset_module["eval_dataset"].select(range(3))).to("cuda")
    with torch.no_grad():
        outputs = model(**batch, return_dict=True, use_cache=False)
        logps, loss_mask = get_batch_logps(outputs.logits.to(torch.float32), batch["labels"])
    print(logps, loss_mask)
    with torch.no_grad(), model.disable_adapter():
        outputs = model(**batch, return_dict=True, use_cache=False)
        logps1, loss_mask = get_batch_logps(outputs.logits.to(torch.float32), batch["labels"])
    print(logps1)
    print(logps-logps1)

        

if __name__ == "__main__":
    F()
