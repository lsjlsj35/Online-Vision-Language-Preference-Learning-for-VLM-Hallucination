import base64
import os
from io import BytesIO
from PIL import Image

import torch
from llamafactory.hparams import FinetuningArguments, ModelArguments
from llamafactory.model.loader import load_model, load_tokenizer
from transformers import AutoProcessor

from .utils import init_dataclass_from_dict


class DummyModel:
    def __init__(self, **kwargs):
        pass

    def _test_images_loading(self, images):
        if images is not None and type(images) != list:
            images = [images]
        all_image_contents = []
        for im in images:
            if type(im) is str:
                try:
                    im = Image.open(os.path.expanduser(im)).convert("RGB")
                except Exception as e:
                    im = Image.open(BytesIO(base64.b64decode(im))).convert("RGB")
            elif type(im) is bytes:
                im = Image.open(BytesIO(im)).convert("RGB")
            all_image_contents.append(im)
        return all_image_contents

    def __call__(self, prompt="", images=None, **kwargs):
        _ = self._test_images_loading(images) 
        return f"Get prompt: {prompt}"
    

class LlavaBaseModel:
    def __init__(self, base_path="/opt/tiger/vtcl/model/llava-1.5-7b-hf", **kwargs):
        from transformers import LlavaForConditionalGeneration
        self.model = LlavaForConditionalGeneration.from_pretrained(base_path, torch_dtype=torch.float16)
        self.model.eval().cuda()
        self.processor = AutoProcessor.from_pretrained(base_path, add_eos_token=False)
        self.dtype = torch.float16
        self.device = "cuda"

    def _get_generation_kwargs(self, generation_config=None):
        generation_config["temperature"] = generation_config.get("temperature", 0.1)
        generation_config["top_p"] = generation_config.get("top_p", 0.8)
        generation_config["max_new_tokens"] = generation_config.get("max_new_tokens", 384)
        generation_config["do_sample"] = generation_config.get("do_sample", True)
        return generation_config

    def _regularize_images(self, images):
        if images is not None and type(images) != list:
            images = [images]
        all_image_contents = []
        for im in images:
            if type(im) is str:
                try:
                    im = Image.open(os.path.expanduser(im)).convert("RGB")
                except Exception as e:
                    if self._verbose_flag:
                        print(f"Error: {e}")
                        self._verbose_flag = False
                    im = Image.open(BytesIO(base64.b64decode(im))).convert("RGB")
            elif type(im) is bytes:
                im = Image.open(BytesIO(im)).convert("RGB")
            all_image_contents.append(im)
        return all_image_contents

    def __call__(self, prompt="", images=None, **generation_config):
        all_image_contents = self._regularize_images(images) 
        conversation = [{"role": "user", "content": [{"type": "image"} for _ in all_image_contents] + [{"type": "text", "text": prompt}]}]
        prompt = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=all_image_contents, return_tensors="pt").to(self.device, self.dtype)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **self._get_generation_kwargs(generation_config))
        output = self.processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        output = output.rsplit("ASSISTANT:", 1)[-1].strip()
        return output
    


class LlamafactoryModel:
    def __init__(self, base_path="/opt/tiger/vtcl/model/llava-1.5-7b-hf", lora_path=None, finetuning_type="lora", stage="mdpo", bf16=True, trust_remote_code=True, lora_target="all", cutoff_len=2048, **kwargs):
        param_dict = {
            'model_name_or_path': base_path,
            'adapter_name_or_path': lora_path,
            'finetuning_type': finetuning_type,
            'stage': stage,
            'trust_remote_code': trust_remote_code,
            'lora_target': lora_target,
            'cutoff_len': cutoff_len,
            **kwargs
        }
        finetuning_args = init_dataclass_from_dict(FinetuningArguments, param_dict)
        model_args = init_dataclass_from_dict(ModelArguments, param_dict)
        tokenizer_module = load_tokenizer(model_args)
        self.processor = tokenizer_module["processor"]
        self.tokenizer = tokenizer_module["tokenizer"]
        self.model = load_model(self.tokenizer, model_args, finetuning_args, False)
        self.model.eval().cuda()
        self.dtype = torch.bfloat16 if bf16 else torch.float16
        self.device = "cuda"
        self._verbose_flag = True

    def _get_generation_kwargs(self, generation_config=None):
        generation_config["temperature"] = generation_config.get("temperature", 0.1)
        generation_config["top_p"] = generation_config.get("top_p", 0.8)
        generation_config["max_new_tokens"] = generation_config.get("max_new_tokens", 384)
        generation_config["do_sample"] = generation_config.get("do_sample", True)
        return generation_config

    def _regularize_images(self, images):
        if images is not None and type(images) != list:
            images = [images]
        all_image_contents = []
        for im in images:
            if type(im) is str:
                try:
                    im = Image.open(os.path.expanduser(im)).convert("RGB")
                except Exception as e:
                    if self._verbose_flag:
                        print(f"Error: {e}")
                        self._verbose_flag = False
                    im = Image.open(BytesIO(base64.b64decode(im))).convert("RGB")
            elif type(im) is bytes:
                im = Image.open(BytesIO(im)).convert("RGB")
            all_image_contents.append(im)
        return all_image_contents

    def __call__(self, prompt="", images=None, **generation_config):
        all_image_contents = self._regularize_images(images) 
        conversation = [{"role": "user", "content": [{"type": "image"} for _ in all_image_contents] + [{"type": "text", "text": prompt}]}]
        prompt = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=all_image_contents, return_tensors="pt").to(self.device, self.dtype)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **self._get_generation_kwargs(generation_config))
        output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        output = output.rsplit("ASSISTANT:", 1)[-1].strip()
        return output