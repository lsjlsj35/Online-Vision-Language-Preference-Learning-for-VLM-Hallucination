from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from PIL import Image
import re
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
import torch
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
# bytesIO
from io import BytesIO
import os
import base64

class LlavaModel:
    def __init__(self, base_path="/opt/tiger/vtcl/model/llava-1.5-7b-hf", **kwargs):
        from transformers import LlavaForConditionalGeneration
        self.device = "cuda"
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path=base_path,
            model_base=None,
            model_name="llava_llama_7b",
            load_8bit=False,
            device=self.device
        )
        # self.model.eval().cuda()
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
        image_tensor = process_images(all_image_contents, self.image_processor, self.model.config).to(self.device, dtype=torch.float16)
        image_size = [image.size for image in all_image_contents]

        conv = conv_templates["llava_v1"].copy()
        conv.system = "You are a helpful assistant for understanding images."
        conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN} {prompt}")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # tokenizer 输入
        inputs = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs=inputs[None, :],
                images=image_tensor,
                image_sizes=image_size,
                **self._get_generation_kwargs(generation_config)
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


class LlavaModelPeft:
    def __init__(self, base_path="/opt/tiger/vtcl/model/llava-v1.5-7b", model_path="", **kwargs):
        from transformers import LlavaForConditionalGeneration
        self.device = "cuda"
        disable_torch_init()
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path=model_path,
            model_base=base_path,
            model_name="llava-1.5lora",
            device=self.device
        )
        # self.model.eval().cuda()
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


        image_tensor = process_images(all_image_contents, self.image_processor, self.model.config).to(self.device, dtype=torch.float16)
        image_size = [image.size for image in all_image_contents]

        qs = prompt
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv_mode = "llava_v0"

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # tokenizer 输入
        inputs = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs=inputs[None, :],
                images=image_tensor,
                image_sizes=image_size,
                **self._get_generation_kwargs(generation_config)
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()