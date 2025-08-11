"""
CUDA_VISIBLE_DEVICES=0 python -m src.llamafactory.generate.generator --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --max_new_tokens 384 --stage mdpo --finetuning_type lora --temperature 1.0 --task base

cd ~/LLaMA-Factory-new ; conda activate llamafactorynew
"""

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, Optional

import PIL
import torch
from tqdm import tqdm
# from qwen_vl_utils import process_vision_info

from ..data import get_template_and_fix_tokenizer
from ..hparams import get_eval_args
from ..model import load_model, load_tokenizer
from .template import get_template


IMAGE_PLACEHOLDER = "<image>"


def get_suffix(p):
    idx = 0
    while Path(p + f"_Exp{idx}").exists():
        idx += 1
    return f"_Exp{idx}"


def as_dict(*args):
    total_args = {}
    for arg in args:
        total_args.update(dataclasses.asdict(arg))
    filtered_args = total_args.copy()
    for k, v in total_args.items():
        if type(v) not in [int, float, str, bool]:
            if type(v) is list:
                if v == [] or type(v[0]) not in [int, float, str, bool]:
                    del filtered_args[k]
            else:
                del filtered_args[k]
    return total_args, filtered_args


class Generator:
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        self.model_args, self.data_args, self.eval_args, self.finetuning_args, generation_args, self.vtcl_args = get_eval_args(args)
        pro_tok = load_tokenizer(self.model_args)
        self.processor = pro_tok["processor"]
        self.tokenizer = pro_tok["tokenizer"]
        # self.tokenizer.padding_side = "right"  # avoid overflow issue in batched inference for llama2
        self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)
        self.model = load_model(self.tokenizer, self.model_args, self.finetuning_args).eval().to("cuda")
        self.eval_template = get_template(self.eval_args.task)
        self.generation_config = generation_args.to_dict()
        
        self.save_path = f"/root/LLaMA-Factory/data"
        print("logging dir:", self.save_path)
        if not Path(self.save_path).exists():
            Path(self.save_path).mkdir(parents=True)
            total_args, filtered_args = as_dict(self.model_args, self.data_args, self.eval_args, self.finetuning_args, generation_args, self.vtcl_args)
            torch.save(total_args, Path(self.save_path) / "total_args.pt")
            with open(Path(self.save_path) / "config.txt", "w") as f:
                json.dump(filtered_args, f, indent=4)

    def encode(self, messages):
        encoded_messages = self.template._encode(self.tokenizer, messages, None, None)
        prompt_ids = []
        for encoded_ids in encoded_messages:
            prompt_ids += encoded_ids
        return prompt_ids

    def generate(self):
        dataset = self.eval_template.load_data()
        i = 1
        while (Path(self.save_path) / f"trial_{i}.jsonl").exists():
            i += 1
        saved_file_path = Path(self.save_path) / f"trial_{i}.jsonl"
        with open(saved_file_path, 'a') as f:
            pbar = tqdm(dataset, total=self.eval_template.length)
            for item in pbar:
                messages, images = self.eval_template.format_example(item)
                if images and images.size[0] * images.size[1] > 1800000:
                    scale = (images.size[0] * images.size[1] / 1800000.) ** 0.5
                    images = images.resize((int(images.size[0] / scale), int(images.size[1] / scale)))
                # print("shape", images.size)
                content = messages[0]["content"]
                if content.startswith("<image>"):
                    content = content[7:].strip()
                if images:
                    messages[0]["content"] = [
                        {"type": "image"}, {"type": "text", "text": content}
                    ]
                text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

                inputs = self.processor(
                    text=[text_prompt], images=[images] if images else None, padding=True, return_tensors="pt"
                )
                inputs = inputs.to("cuda")
                with torch.no_grad():
                    # output = self.model.generate(**inputs, **self.generation_config)[0][len(input_ids):-1]  # <eos>
                    output = self.model.generate(**inputs, **self.generation_config)[0][inputs["input_ids"].shape[-1]:-1]
                response = self.tokenizer.decode(output)
                item.pop("IMAGE", None)
                json.dump({**item, "result": response}, f)
                f.write('\n')
                torch.cuda.empty_cache()
        self.eval_template.post_process(Path(self.save_path), saved_file_path)


def run_generate() -> None:
    Generator().generate()


if __name__ == "__main__":
    run_generate()