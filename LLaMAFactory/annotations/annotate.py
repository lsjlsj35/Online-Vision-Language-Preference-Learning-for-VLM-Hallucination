import hashlib
import json
import os
import re
from datetime import datetime
from filelock import FileLock
from fastapi import FastAPI, Request
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

lock = FileLock("write.lock")

if not hasattr(torch.backends, "cusparselt"):
    class Dummy:
        enabled = False
        def is_available(self, *args, **kwargs):
            return False
    torch.backends.cusparselt = Dummy()

app = FastAPI()

desc_model = None

generation_model = None


class Generator:
    pass


def setup_desc_route():
    global desc_model
    desc_model = Generator()
    desc_model.model = AutoModelForCausalLM.from_pretrained(
        "/opt/tiger/vtcl/model/qwen2.5-7b-instruct",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    desc_model.model.cuda().eval()
    desc_model.tokenizer = AutoTokenizer.from_pretrained(
        "/opt/tiger/vtcl/model/qwen2.5-7b-instruct",
        use_fast=False,
        padding_side="left",
    )

    @app.post("/describe")
    async def describe(request: Request):
        data = await request.json()
        prompt = DESC_PROMPT.format(
            question=data["prompt"],
            answer=data["ground_truth"],
            response=data["response"],
        )
        messages = [{"role": "user", "content": prompt}]
        text_prompt = desc_model.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = desc_model.tokenizer([text_prompt], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            output = desc_model.model.generate(**inputs, pad_token_id=desc_model.tokenizer.pad_token_id, temperature=0.1, do_sample=True, max_new_tokens=128)[0][inputs["input_ids"].shape[-1]:-1]
        output = desc_model.tokenizer.decode(output, skip_special_tokens=True)
        return {"description": output}

    @app.post("/critic")
    async def critic(request: Request):
        data = await request.json()
        prompt = CRITIC_PROMPT.format(
            question=data["prompt"],
            answer=data["answer"],
            response=data["response"],
        )
        messages = [{"role": "user", "content": prompt}]
        text_prompt = desc_model.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = desc_model.tokenizer([text_prompt], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            output = desc_model.model.generate(**inputs, pad_token_id=desc_model.tokenizer.pad_token_id, temperature=0.1, do_sample=True, max_new_tokens=128)[0][inputs["input_ids"].shape[-1]:-1]
        output = desc_model.tokenizer.decode(output, skip_special_tokens=True)
        pattern = r'score[:：]\s*\[?([0-9]+)\]?[\s\.,]*'
        match = re.search(pattern, output.lower(), re.IGNORECASE)
        if match:
            return {"critic": int(match.group(1))}
        
        pattern = r'^(\d+)'
        match = re.search(pattern, output.lower(), re.IGNORECASE)
        if match:
            return {"critic": int(match.group(1))}
        return {"critic": "none"}


def setup_critic_route():
    global desc_model
    desc_model = Generator()
    desc_model.model = AutoModelForCausalLM.from_pretrained(
        "/opt/tiger/vtcl/model/qwen2.5-7b-instruct",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    desc_model.model.eval()
    desc_model.tokenizer = AutoTokenizer.from_pretrained(
        "/opt/tiger/vtcl/model/qwen2.5-7b-instruct",
        use_fast=False,
        padding_side="left",
    )

    @app.post("/critic")
    async def critic(request: Request):
        data = await request.json()
        prompt = CRITIC_PROMPT.format(
            question=data["prompt"],
            answer=data["answer"],
            response=data["response"],
        )
        messages = [{"role": "user", "content": prompt}]
        text_prompt = desc_model.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = desc_model.tokenizer([text_prompt], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            output = desc_model.model.generate(**inputs, pad_token_id=desc_model.tokenizer.pad_token_id, temperature=0.1, do_sample=True, max_new_tokens=128)[0][inputs["input_ids"].shape[-1]:-1]
        output = desc_model.tokenizer.decode(output, skip_special_tokens=True)
        pattern = r'score[:：]\s*\[?([0-9]+)\]?[\s\.,]*'
        match = re.search(pattern, output.lower(), re.IGNORECASE)
        if match:
            return {"critic": int(match.group(1))}
        
        pattern = r'^(\d+)'
        match = re.search(pattern, output.lower(), re.IGNORECASE)
        if match:
            return {"critic": int(match.group(1))}
        return {"critic": "none"}
    

def setup_vllm_text_route():
    from vllm import LLM, SamplingParams
    global desc_model

    desc_model = Generator()
    desc_model.model = LLM(
        "/opt/tiger/vtcl/model/qwen2.5-32b-instruct",
        tensor_parallel_size=2,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
    )
    desc_model.tokenizer = AutoTokenizer.from_pretrained(
        "/opt/tiger/vtcl/model/qwen2.5-32b-instruct",
        use_fast=False,
        padding_side="left",
    )
    desc_model.critic_samp_param = SamplingParams(
        temperature=0.1,
        top_p=0.9,
        max_tokens=128,
        presence_penalty=1.0,
        frequency_penalty=1.0,
    )
    # TODO: average of N sampling for scoring
    desc_model.instruct_samp_param = SamplingParams(
        temperature=0.1,
        top_p=0.9,
        max_tokens=128,
        presence_penalty=1.0,
        frequency_penalty=1.0,
    )

    @app.post("/describe")
    async def describe(request: Request):
        data = await request.json()
        prompt = DESC_PROMPT.format(
            question=data["prompt"],
            answer=data["ground_truth"],
            response=data["response"],
        )
        messages = [{"role": "user", "content": prompt}]
        text_prompt = desc_model.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        output = desc_model.model.generate(
            text_prompt,
            desc_model.instruct_samp_param,
            use_tqdm=False,
        )[0].outputs[0].text

        # parse score

        return {"description": output}
    
    @app.post("/critic")
    async def critic(request: Request):
        data = await request.json()
        prompt = CRITIC_PROMPT.format(
            question=data["question"],
            answer=data["answer"],
            response=data["response"],
        )
        messages = [{"role": "user", "content": prompt}]
        text_prompt = desc_model.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        output = desc_model.model.generate(
            text_prompt,
            desc_model.critic_samp_param,
            use_tqdm=False,
        )[0].outputs[0].text

        pattern = r'score[:：]\s*\[?([0-9]+)\]?[\s\.,]*'
        match = re.search(pattern, output.lower(), re.IGNORECASE)
        if match:
            return {"critic": int(match.group(1))}
        
        pattern = r'^(\d+)'
        match = re.search(pattern, output.lower(), re.IGNORECASE)
        if match:
            return {"critic": int(match.group(1))}
        return {"critic": "none"}


def setup_generation_route():
    from diffusers import FluxPipeline
    global generation_model

    generation_model = Generator()
    generation_model.model = FluxPipeline.from_pretrained(
        "/opt/tiger/vtcl/model/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    @app.post("/generate")
    async def generate(request: Request):
        data = await request.json()
        prompt = data["prompt"]
        h = data["h"]
        w = data["w"]
        info = data["info"]
        save_dir = data["save_dir"]
        num_inference_steps = data["num_inference_steps"]
        guidance_scale = data["guidance_scale"]
        img = generation_model.model(
            prompt,
            height=h,
            width=w,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(0)
        ).images[0]

        hash_result = hashlib.sha256((datetime.now().isoformat()+prompt).encode('utf-8')).hexdigest()
        # 创建文件夹
        Path(save_dir.rstrip('/') + f"/images").mkdir(parents=True, exist_ok=True)  
        img_save_dir = save_dir.rstrip('/') + f'/images/{hash_result}.png'
        with lock:
            with open(save_dir.rstrip('/') + f"/data.jsonl", 'a') as f:
                f.write(json.dumps({
                    "image_instruction": prompt,
                    "image": img_save_dir,
                    **info,
                }) + '\n')
        img.save(img_save_dir)
        return {"path": img_save_dir}
    

def setup_dummy_route():
    @app.post("/describe")
    async def describe(request: Request):
        return {"description": "Dummy description"}
    @app.post("/critic")
    async def critic(request: Request):
        return {"critic": -1}
    @app.post("/generate")
    async def generate(request: Request):
        return {"path": "/opt/tiger/vtcl/data/OPA_DPO/0a2e7e1ac12be6fe.jpg"}


DESC_PROMPT = """# Task  
Given an unknown image-related question, a correct answer, and an inaccurate response, carefully analyze the differences between the response and the answer. Then, provide a brief description of the image so that it aligns with the correct answer and differs from the incorrect responses. In other words, infer the content of the image.

# Example
[Example 1] 
**<Question>**  
What is on the sandwich?  

**<Answer>**
The sandwich has tomatoes and lettuce on it.  

**<Response>**  
The sandwich has a slice of egg and tomato on it.

**<Output Description>**
A sandwich with only tomatoes and lettuce on it.

**Explanation**: The answer mentions lettuce and tomato, while the incorrect response mentions tomato and egg. So there is no egg on the sandwich.

[Example 2]
**<Question>**  
Can you point out the details that make this image unique?  

**<Answer>**
In the image, there is a plate with a slice of pizza topped with tomatoes, herbs, and cheese. The distinctive detail about the image is that the pizza is missing two slices, leaving just one slice remaining on the plate. This suggests that someone has already started enjoying the pizza. The slice appears to be well-cooked and freshly served, creating an appetizing and mouthwatering scene for the viewer.  

**<Response>**  
In the image, there is a slice of pizza on a plate with tomatoes and cheese. The pizza appears to be homemade and has been cut into two pieces. The tomatoes are sliced in half, revealing their juicy interior. The cheese on top of the pizza is melted, creating a delicious-looking dish. Additionally, there is a fork nearby, suggesting that someone might be planning to enjoy this pizza soon.  

**<Output Description>**
A plate with a one-third remaining piece of pizza, topped with herbs, cheese, and tomatoes; someone has finished eating and left.

**Explanation**: The answer mentions that only one-third of the pizza remains and that someone has just finished eating and left, which is inconsistent with the response. Therefore, the image should include these two features.

[Example 3]
**<Question>**
Bird or cow?

**<Answer>**
Bird

**<Response>**
The bird in the image is a small, brown and white bird with a distinctive head shape and coloration. It is not a cow. The bird is perched on a branch, which is situated in front of a white building.

**<Output Description>**
A big, blue bird perched on a branch in front of a black building.

**Explanation**: Both the answer and the response mention the bird, but the response is more detailed. So the description should be contrastive to the features of the bird in the response.

# Requirements
- The description should be brief but precise.  
- If both the answer and the response are long, focus on describing the one or two most significant differences.
- Do not provide any analysis or explanation; only describe the image.
- A common approach is to describe what is present in the image and what is missing.


**<Question>**
{question}
**<Answer>**
{answer}
**<Response>**
{response}

**<Output Description>**
"""


CRITIC_PROMPT = """# Task
Your role is as a discerning assistant tasked with evaluating model responses for multimodal tasks (though you have no access with the image). \
Upon being presented with a question that requires the interpretation of both text and images, you will receive two distinct responses. \
The first is crafted by our sophisticated multimodal model, while the second represents an approximate ideal answer--it may be incomplete. \
Your objective is to meticulously and precisely assess the model-generated response (the former) based on the provided reference answer (the latter).

- Here's how you should approach the assessment process:
    1. The quality of the response depends on its accuracy and the degree of adherence to the correct answer. Therefore, if the response is much more detailed than the reference answer, it should not be considered a very good response (although it may still be considered a good one).
    2. Directly provide the score of the response, with a full score of 10. Your response should follow this format: "Score: [x]\n", where "[x]" represents the score you give, and "\n" is a line break.
    3. Please do not provide additional reasoning, just give the score directly.

# Question
{question}

# Response
{response}

# Correct answer
{answer}
"""


model = os.getenv('MODEL')
if model == "desc":
    setup_desc_route()
elif model == "generate":
    setup_generation_route()
elif model == "critic":
    setup_critic_route()
elif model == "desc_critic":
    setup_vllm_text_route()
elif model == "dummy":
    setup_dummy_route()
else:
    raise ValueError(f"Unknown model: {model}")