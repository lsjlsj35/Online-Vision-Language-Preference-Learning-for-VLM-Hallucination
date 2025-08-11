import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ..data import get_template_and_fix_tokenizer
from ..eval.evaluator_new import Evaluator, as_dict, get_suffix, set_seed
from ..hparams import get_eval_args
from ..model import load_model, load_tokenizer
from .template import get_template


class Generator(Evaluator):
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        self.model_args, self.data_args, self.eval_args, self.finetuning_args, generation_args, self.vtcl_args = get_eval_args(args)
        set_seed(int(time.time()))
        pro_tok = load_tokenizer(self.model_args)
        self.processor = pro_tok["processor"]
        self.tokenizer = pro_tok["tokenizer"]
        if self.tokenizer.pad_token:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # self.tokenizer.padding_side = "right"  # avoid overflow issue in batched inference for llama2
        self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)
        self.model = load_model(self.tokenizer, self.model_args, self.finetuning_args).eval().to("cuda")
        self.eval_template = get_template(self.eval_args.task)
        self.generation_config = generation_args.to_dict()
        
        adapter_path = self.model_args.adapter_name_or_path[0] if self.model_args.adapter_name_or_path else self.model_args.adapter_name_or_path
        if self.model_args.adapter_name_or_path:
            trained_info = adapter_path[adapter_path.find("saves"):].split('/', 1)[1]
            self.save_path = f"/root/LLaMA-Factory-new/generation/{self.eval_args.task}/{trained_info.replace('/', '_')}"
        else:
            self.save_path = f"/root/LLaMA-Factory-new/generation/{self.eval_args.task}/{self.model_args.model_name_or_path.rsplit('/', 1)[-1]}"
        print("logging dir:", self.save_path)
        if not Path(self.save_path).exists():
            Path(self.save_path).mkdir(parents=True)
            total_args, filtered_args = as_dict(self.model_args, self.data_args, self.eval_args, self.finetuning_args, generation_args, self.vtcl_args)
            torch.save(total_args, Path(self.save_path) / "total_args.pt")
            with open(Path(self.save_path) / "config.txt", "w") as f:
                json.dump(filtered_args, f, indent=4)


def run_generate():
    Generator().eval()


if __name__ == "__main__":
    run_generate()