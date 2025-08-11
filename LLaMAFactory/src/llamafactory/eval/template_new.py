import json
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
from PIL import Image


eval_templates = {}


def register(name, klass="generation"):
    def decorate(cls):
        eval_templates[name] = cls
        cls.klass = klass
        return cls
    return decorate


@dataclass
class Eval_Template:
    customed_inference = False
    def format_example(self, item):
        pass

    def __len__(self):
        return self.length

    def load_data(self):
        pass

    @property
    def length(self):
        return None
    
    def post_process(self, dir_path):
        pass


# class Generation_Task(Eval_Template):
#     def generate(model, inputs):
#         pass
    

@register(name="HalluBench", klass="generation")
class HalluBench(Eval_Template):
    no_img_template = """Please answer the following question as either "Yes", "No".\nQuestion: {}"""
    img_template = """<image> Please answer the following question according to the image as either "Yes", "No". \nQuestion: {}"""

    @property
    def length(self):
        return self._len

    def load_data(self):
        def _loader():
            base_path = "/root/benchmark/HallusionBench/hallusion_bench/"
            for item in data:
                if item["visual_input"] == '0':
                    item["IMAGE"] = None
                    yield item
                else:
                    if item["category"] == "VD":
                        path = base_path + item["filename"][2:]
                        assert f"{item['category']}/{item['subcategory']}/{item['set_id']}_{item['figure_id']}.png" in path, str(item)
                    else:
                        path = f"{base_path}{item['category']}/{item['subcategory']}/{item['set_id']}_{int(item['figure_id'])}.png"
                    img = Image.open(path)
                    item["IMAGE"] = img
                    yield item

        input_file_name = "/root/benchmark/HallusionBench/HallusionBench.json"
        with open(input_file_name) as f:
            data = json.load(f)
            self._len = len(data)
        return _loader()  # If not using wrapper, the property "_len" will not be assigned immediately when this method is called.
        
    def format_example(self, item):
        template = self.no_img_template if item["IMAGE"] is None else self.img_template
        return [
            {"role": "user", "content": template.format(item["question"])}
        ], item["IMAGE"]
    

@register(name="RBench", klass="generation")
class RBench(Eval_Template):
    img_template = """<image> Please answer the following question according to the image as either "Yes", "No". \nQuestion: {}"""

    @property
    def length(self):
        return self._len

    def load_data(self):
        def _loader():
            base_path = "/root/benchmark/R-Bench/validation/"
            for item in data:
                path = f"{base_path}{item['image']}"
                img = Image.open(path)
                item["IMAGE"] = img
                yield item

        input_file_name = "/root/benchmark/R-Bench/data_filterd/image-level_filterd.json"
        with open(input_file_name) as f:
            data = json.load(f)
            self._len = len(data)
        return _loader()
        
    def format_example(self, item: Dict[str, Any]):
        template = self.img_template
        return [
            {"role": "user", "content": template.format(item["text"])}
        ], item["IMAGE"]
    

@register(name="COCOEval", klass="generation")
class COCOEval(Eval_Template):
    """max_new_tokens 256"""
    img_template = """<image> Please describe the image in detail."""

    @property
    def length(self):
        return self._len
    
    def load_data(self):
        def _loader():
            for item in data:
                for idx, key in enumerate(["chosen", "rejected"]):
                    path = item['images'][idx]
                    img = Image.open(path)
                    item["IMAGE"] = img
                    item["KEY"] = key
                    yield item

        input_file_name = "/root/LLaMA-Factory/data/coco-val2014_1k.json"
        with open(input_file_name) as f:
            data = json.load(f)
            self._len = 2 * len(data)
        return _loader()
        
    def format_example(self, item: Dict[str, Any]):
        return [
            {"role": "user", "content": self.img_template}
        ], item["IMAGE"]
    

@register(name="MMHalBench", klass="generation")
class MMHalBench(Eval_Template):
    """max_new_tokens 128"""
    img_template = """<image> {}"""

    @property
    def length(self):
        return self._len

    def load_data(self):
        def _loader():
            base_path = "/root/dataset/MMHal-Bench/MMHal-Bench/images"
            for item in data:
                img_name = item["image_src"].rsplit('/', 1)[1]
                path = f"{base_path}/{img_name}"
                img = Image.open(path)
                item["IMAGE"] = img
                yield item

        input_file_name = "/root/dataset/MMHal-Bench/MMHal-Bench/response_template.json"
        with open(input_file_name) as f:
            data = json.load(f)
            self._len = len(data)
        return _loader()
        
    def format_example(self, item: Dict[str, Any]):
        template = self.img_template
        return [
            {"role": "user", "content": template.format(item["question"])}
        ], item["IMAGE"]
    

@register(name="MMHalBench_new", klass="generation")
class MMHalBench_New(Eval_Template):
    """max_new_tokens 128"""
    img_template = """<image> Please describe the image in detail."""
    img_template2 = """<image> {}\nPlease answer the following question:\n{}"""
    customed_inference = True

    @property
    def length(self):
        return self._len

    def load_data(self):
        def _loader():
            base_path = "/root/dataset/MMHal-Bench/MMHal-Bench/images"
            for item in data:
                img_name = item["image_src"].rsplit('/', 1)[1]
                path = f"{base_path}/{img_name}"
                img = Image.open(path)
                item["IMAGE"] = img
                yield item

        input_file_name = "/root/dataset/MMHal-Bench/MMHal-Bench/response_template.json"
        with open(input_file_name) as f:
            data = json.load(f)
            self._len = len(data)
        return _loader()
        
    def format_example1(self, item: Dict[str, Any]):
        template = self.img_template
        return [
            {"role": "user", "content": template.format(item["question"])}
        ], item["IMAGE"]
    
    def format_example2(self, item: Dict[str, Any], desc):
        template = self.img_template2
        return [
            {"role": "user", "content": template.format(desc, item["question"])}
        ], item["IMAGE"]
    
    def inference(self, item, encode_func, template, model, tokenizer, processor, generation_config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        messages, images = self.format_example1(item)
        input_ids = encode_func(messages)
        if images is not None:
            if type(images) is not list:
                images = [images]
            mm_inputs = template.mm_plugin.get_mm_inputs(images, [], len(images), 0, len(input_ids), processor).to("cuda")
        else:
            mm_inputs = {}
        inputs = {
            "input_ids": torch.tensor([input_ids], dtype=torch.int64).to("cuda"),
            "attention_mask": torch.tensor([[1] * len(input_ids)], dtype=torch.int64).to("cuda"),
            **mm_inputs
        }
        if model is not None and hasattr(model, "get_rope_index"):  # for qwen2vl mrope
            inputs["position_ids"], inputs["rope_deltas"] = model.get_rope_index(
                input_ids=inputs["input_ids"],
                image_grid_thw=mm_inputs.get("image_grid_thw", None),
                video_grid_thw=mm_inputs.get("video_grid_thw", None),
                attention_mask=inputs["attention_mask"],
            )
        with torch.no_grad():
            output = model.generate(**inputs, **generation_config)[0][len(input_ids):-1]  # <eos>
        response = tokenizer.decode(output)

        messages, images = self.format_example2(item, response)
        input_ids = encode_func(messages)
        if images is not None:
            if type(images) is not list:
                images = [images]
            mm_inputs = template.mm_plugin.get_mm_inputs(images, [], len(images), 0, len(input_ids), processor).to("cuda")
        else:
            mm_inputs = {}
        inputs = {
            "input_ids": torch.tensor([input_ids], dtype=torch.int64).to("cuda"),
            "attention_mask": torch.tensor([[1] * len(input_ids)], dtype=torch.int64).to("cuda"),
            **mm_inputs
        }
        if self.model is not None and hasattr(self.model, "get_rope_index"):  # for qwen2vl mrope
            inputs["position_ids"], inputs["rope_deltas"] = self.model.get_rope_index(
                input_ids=inputs["input_ids"],
                image_grid_thw=mm_inputs.get("image_grid_thw", None),
                video_grid_thw=mm_inputs.get("video_grid_thw", None),
                attention_mask=inputs["attention_mask"],
            )
        with torch.no_grad():
            output = model.generate(**inputs, **generation_config)[0][len(input_ids):-1]  # <eos>
        final_response = tokenizer.decode(output)
        return final_response, {"desc": response}
    

@register(name="MMVetV1", klass="generation")
class MMVetV1(Eval_Template):
    """max_new_tokens 384"""
    img_template = """<image> {}"""

    @property
    def length(self):
        return self._len

    def load_data(self):
        def _loader():
            base_path = "/root/benchmark/MMVet-v1/images"
            for i in range(218):
                item = data[f"v1_{i}"]
                img_name = item["imagename"]
                path = f"{base_path}/{img_name}"
                img = Image.open(path)
                item["IMAGE"] = img
                yield item

        input_file_name = "/root/benchmark/MMVet-v1/mm-vet.json"
        with open(input_file_name) as f:
            data = json.load(f)
            self._len = len(data)
            assert self._len == 218
        return _loader()
        
    def format_example(self, item: Dict[str, Any]):
        template = self.img_template
        return [
            {"role": "user", "content": template.format(item["question"])}
        ], item["IMAGE"]
    

@register(name="MMVetV2", klass="generation")
class MMVetV2(Eval_Template):
    img_template = """{}"""

    @property
    def length(self):
        return self._len

    def load_data(self):
        import re

        pattern = r'<IMG>v2_\d{1,3}_\d{1,2}\.(?:jpg|png)'
        base_path = "/root/benchmark/MMVet-v2/images/"
        def _loader():
            
            for i in range(517):
                item = data[f"v2_{i}"]

                q = item["question"]
                matches = re.findall(pattern, q)
                remaining = re.split(pattern, q)
                assert len(matches) >= 1

                item["IMAGE"] = [Image.open(base_path+p[5:]) for p in matches]
                new_remaining = []
                for seq in remaining:
                    new_remaining.append(seq.replace("<image>", "\\<image\\>"))
                item["prompt"] = "<image> ".join(new_remaining)
                item["img_name"] = [p[5:] for p in matches]
                item["id"] = f"v2_{i}"
                yield item

        input_file_name = "/root/benchmark/MMVet-v2/mm-vet-v2.json"
        with open(input_file_name) as f:
            data = json.load(f)
            self._len = len(data)
            assert self._len == 517
        return _loader()
        
    def format_example(self, item: Dict[str, Any]):
        template = self.img_template
        return [
            {"role": "user", "content": template.format(item["prompt"])}
        ], item["IMAGE"]
    

@register(name="MMVetV2_SingleImg", klass="generation")
class MMVetV2_SingleImg(Eval_Template):
    img_template = """<image> {}"""

    @property
    def length(self):
        return self._len

    def load_data(self):
        import re

        pattern = r'<IMG>v2_\d{1,3}_\d{1,2}\.(?:jpg|png)'
        base_path = "/root/benchmark/MMVet-v2/images/"
        def _loader():
            
            for i in range(517):
                item = data[f"v2_{i}"]

                q = item["question"]
                matches = re.findall(pattern, q)
                remaining = re.split(pattern, q)
                assert len(matches) >= 1
                if len(matches) > 1:
                    continue

                item["IMAGE"] = [Image.open(base_path+p[5:]) for p in matches]
                new_remaining = []
                for seq in remaining:
                    new_remaining.append(seq.replace("<IMG>", "\\<IMG\\>"))
                item["prompt"] = "<IMG>".join(new_remaining)
                item["img_name"] = [p[5:] for p in matches]
                item["id"] = f"v2_{i}"
                yield item

        input_file_name = "/root/benchmark/MMVet-v2/mm-vet-v2.json"
        with open(input_file_name) as f:
            data = json.load(f)
            self._len = len(data)
            assert self._len == 517
        return _loader()
        
    def format_example(self, item: Dict[str, Any]):
        template = self.img_template
        return [
            {"role": "user", "content": template.format(item["prompt"])}
        ], item["IMAGE"]


@register(name="POPE", klass="generation")
class POPE(Eval_Template):
    """max_new_tokens 64"""
    img_template = """<image> {}"""

    @property
    def length(self):
        return 9000

    def load_data(self):
        import pyarrow.parquet as pq
        from io import BytesIO

        def load(path):
            pq_file = pq.ParquetFile(path)
            data = pq_file.read().to_pandas()
            return data
        
        def _loader():
            for _, item in d1.iterrows():
                ditem = {}
                ditem["IMAGE"] = Image.open(BytesIO(item["image"]["bytes"]))
                ditem.update({
                    "question": item["question"],
                    "question_id": item["question_id"],
                    "image_source": item["image_source"],
                    "category": item["category"],
                    "answer": item["answer"],
                })
                yield ditem
            for _, item in d2.iterrows():
                ditem = {}
                ditem["IMAGE"] = Image.open(BytesIO(item["image"]["bytes"]))
                ditem.update({
                    "question": item["question"],
                    "question_id": item["question_id"],
                    "image_source": item["image_source"],
                    "category": item["category"],
                    "answer": item["answer"],
                })
                yield ditem
            for _, item in d3.iterrows():
                ditem = {}
                ditem["IMAGE"] = Image.open(BytesIO(item["image"]["bytes"]))
                ditem.update({
                    "question": item["question"],
                    "question_id": item["question_id"],
                    "image_source": item["image_source"],
                    "category": item["category"],
                    "answer": item["answer"],
                })
                yield ditem

        d1 = load("/root/dataset/POPE/data/test-00000-of-00003.parquet")
        d2 = load("/root/dataset/POPE/data/test-00001-of-00003.parquet")
        d3 = load("/root/dataset/POPE/data/test-00002-of-00003.parquet")
        return _loader()
        
    def format_example(self, item: Dict[str, Any]):
        template = self.img_template
        return [
            {"role": "user", "content": template.format(item["question"])}
        ], item["IMAGE"]
    

@register(name="RealWorldQA", klass="generation")
class RealWorldQA(Eval_Template):
    """max_new_tokens 384"""
    img_template = """<image> {}"""

    @property
    def length(self):
        return 765

    def load_data(self):
        import pyarrow.parquet as pq
        from io import BytesIO

        def load(path):
            pq_file = pq.ParquetFile(path)
            data = pq_file.read().to_pandas()
            return data
        
        def _loader():
            for _, item in d1.iterrows():
                ditem = {}
                ditem["IMAGE"] = Image.open(BytesIO(item["image"]["bytes"]))
                ditem.update({
                    "question": item["question"],
                    "answer": item["answer"],
                })
                yield ditem
            for _, item in d2.iterrows():
                ditem = {}
                ditem["IMAGE"] = Image.open(BytesIO(item["image"]["bytes"]))
                ditem.update({
                    "question": item["question"],
                    "answer": item["answer"],
                })
                yield ditem

        d1 = load("/root/dataset/RealworldQA/data/test-00000-of-00002.parquet")
        d2 = load("/root/dataset/RealworldQA/data/test-00001-of-00002.parquet")
        return _loader()
        
    def format_example(self, item: Dict[str, Any]):
        template = self.img_template
        return [
            {"role": "user", "content": template.format(item["question"])}
        ], item["IMAGE"]


@register(name="POPE_new", klass="generation")
class POPE_New(Eval_Template):
    """max_new_tokens 128"""
    img_template = """<image> Please describe the image in detail."""
    img_template2 = """<image> {}\nPlease answer the following question according to the given image as either "Yes", "No".\nStatement: {}"""
    customed_inference = True
    buffer = {}

    @property
    def length(self):
        return 9000

    def load_data(self):
        import pyarrow.parquet as pq
        from io import BytesIO

        def load(path):
            pq_file = pq.ParquetFile(path)
            data = pq_file.read().to_pandas()
            return data
        
        def _loader():
            for _, item in d1.iterrows():
                ditem = {}
                ditem["IMAGE"] = Image.open(BytesIO(item["image"]["bytes"]))
                ditem.update({
                    "question": item["question"],
                    "question_id": item["question_id"],
                    "image_source": item["image_source"],
                    "category": item["category"],
                    "answer": item["answer"],
                })
                yield ditem
            for _, item in d2.iterrows():
                ditem = {}
                ditem["IMAGE"] = Image.open(BytesIO(item["image"]["bytes"]))
                ditem.update({
                    "question": item["question"],
                    "question_id": item["question_id"],
                    "image_source": item["image_source"],
                    "category": item["category"],
                    "answer": item["answer"],
                })
                yield ditem
            for _, item in d3.iterrows():
                ditem = {}
                ditem["IMAGE"] = Image.open(BytesIO(item["image"]["bytes"]))
                ditem.update({
                    "question": item["question"],
                    "question_id": item["question_id"],
                    "image_source": item["image_source"],
                    "category": item["category"],
                    "answer": item["answer"],
                })
                yield ditem

        d1 = load("/root/dataset/POPE/data/test-00000-of-00003.parquet")
        d2 = load("/root/dataset/POPE/data/test-00001-of-00003.parquet")
        d3 = load("/root/dataset/POPE/data/test-00002-of-00003.parquet")
        return _loader()
        
    def format_example1(self, item: Dict[str, Any]):
        return [
            {"role": "user", "content": self.img_template}
        ], item["IMAGE"]
    
    def format_example2(self, item: Dict[str, Any], desc):
        template = self.img_template2
        return [
            {"role": "user", "content": template.format(desc, item["question"])}
        ], item["IMAGE"]
    
    def inference(self, item, encode_func, template, model, tokenizer, processor, generation_config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        if item["image_source"] in self.buffer:
            response = self.buffer[item["image_source"]]
        else:
            messages, images = self.format_example1(item)
            input_ids = encode_func(messages)
            if images is not None:
                if type(images) is not list:
                    images = [images]
                mm_inputs = template.mm_plugin.get_mm_inputs(images, [], len(images), 0, len(input_ids), processor).to("cuda")
            else:
                mm_inputs = {}
            inputs = {
                "input_ids": torch.tensor([input_ids], dtype=torch.int64).to("cuda"),
                "attention_mask": torch.tensor([[1] * len(input_ids)], dtype=torch.int64).to("cuda"),
                **mm_inputs
            }
            if model is not None and hasattr(model, "get_rope_index"):  # for qwen2vl mrope
                inputs["position_ids"], inputs["rope_deltas"] = model.get_rope_index(
                    input_ids=inputs["input_ids"],
                    image_grid_thw=mm_inputs.get("image_grid_thw", None),
                    video_grid_thw=mm_inputs.get("video_grid_thw", None),
                    attention_mask=inputs["attention_mask"],
                )
            with torch.no_grad():
                output = model.generate(**inputs, **generation_config)[0][len(input_ids):-1]  # <eos>
            response = tokenizer.decode(output)
            self.buffer[item["image_source"]] = response

        messages, images = self.format_example2(item, response)
        input_ids = encode_func(messages)
        if images is not None:
            if type(images) is not list:
                images = [images]
            mm_inputs = template.mm_plugin.get_mm_inputs(images, [], len(images), 0, len(input_ids), processor).to("cuda")
        else:
            mm_inputs = {}
        inputs = {
            "input_ids": torch.tensor([input_ids], dtype=torch.int64).to("cuda"),
            "attention_mask": torch.tensor([[1] * len(input_ids)], dtype=torch.int64).to("cuda"),
            **mm_inputs
        }
        if model is not None and hasattr(model, "get_rope_index"):  # for qwen2vl mrope
            inputs["position_ids"], inputs["rope_deltas"] = model.get_rope_index(
                input_ids=inputs["input_ids"],
                image_grid_thw=mm_inputs.get("image_grid_thw", None),
                video_grid_thw=mm_inputs.get("video_grid_thw", None),
                attention_mask=inputs["attention_mask"],
            )
        new_generation_config = generation_config.copy()
        new_generation_config["max_new_tokens"] = 10
        with torch.no_grad():
            output = model.generate(**inputs, **new_generation_config)[0][len(input_ids):-1]  # <eos>
        final_response = tokenizer.decode(output)
        return final_response, {"desc": response}


class AMBER_TEMPLATE(Eval_Template):
    img_template = """<image> {}"""
    input_file_name = "/root/benchmark/AMBER-master/data/query/query_all.json"

    @property
    def length(self):
        return self._len

    def load_data(self):
        def _loader():
            base_path = "/root/benchmark/AMBER-master/image"
            for item in data:
                img_name = item["image"]
                path = f"{base_path}/{img_name}"
                img = Image.open(path)
                item["IMAGE"] = img
                yield item
 
        with open(self.input_file_name) as f:
            data = json.load(f)
            self._len = len(data)
        return _loader()
        
    def format_example(self, item: Dict[str, Any]):
        template = self.img_template
        if item["query"] == "Describe this image.":
            query = "Please describe the image in detail."
        else:
            query = 'Please answer the following question according to the given image as either "Yes", "No".\nQuestion: ' + item["query"]
        return [
            {"role": "user", "content": template.format(query)}
        ], item["IMAGE"]
    
    def post_process(self, dir_path):
        with open(dir_path / "results.jsonl") as f:
            data = [json.loads(line) for line in f]
        for item in data:
            item["response"] = item["result"]
            del item["result"]
        with open(dir_path / "results.json", 'w') as f:
            json.dump(data, f, indent=4)


@register(name="AMBER_discriminative", klass="generation")
class AMBER_DISCRIMINATIVE(AMBER_TEMPLATE):
    """max_new_tokens 10"""
    input_file_name = "/root/benchmark/AMBER-master/data/query/query_discriminative.json"


@register(name="AMBER_generative", klass="generation")
class AMBER_GENERATIVE(AMBER_TEMPLATE):
    """max_new_tokens 384"""
    input_file_name = "/root/benchmark/AMBER-master/data/query/query_generative.json"

@register(name="AMBER_gf", klass="generation")
class AMBER_GENERATIVE(AMBER_TEMPLATE):
    """max_new_tokens 384"""
    input_file_name = "/root/benchmark/AMBER-master/data/query/query_generative_half_F.json"

@register(name="AMBER_gb", klass="generation")
class AMBER_GENERATIVE(AMBER_TEMPLATE):
    """max_new_tokens 384"""
    input_file_name = "/root/benchmark/AMBER-master/data/query/query_generative_half_B.json"


@register(name="llavaBenchintheWild", klass="generation")
class LLaVABenchintheWild(Eval_Template):
    img_template = """<image> {}"""
    input_file_name = "/root/benchmark/llava-bench-in-the-wild/questions.jsonl"

    @property
    def length(self):
        return self._len

    def load_data(self):
        def _loader():
            base_path = "/root/benchmark/llava-bench-in-the-wild/images"
            for item in data:
                img_name = item["image"]
                path = f"{base_path}/{img_name}"
                img = Image.open(path)
                item["IMAGE"] = img
                yield item
 
        with open(self.input_file_name) as f:
            data = [json.loads(line) for line in f]
            self._len = len(data)
        return _loader()
        
    def format_example(self, item: Dict[str, Any]):
        template = self.img_template
        return [
            {"role": "user", "content": template.format(item["text"])}
        ], item["IMAGE"]
    

@register(name="GQA", klass="generation")
class GQA(Eval_Template):
    img_template = """<image> Please answer with a word or phrase: {}"""
    input_file_name = "/root/dataset/GQA/testdev_balanced_questions.json"

    @property
    def length(self):
        return self._len

    def load_data(self):
        def _loader():
            base_path = "/root/dataset/GQA/images"
            for item in data:
                img_name = item["imageId"] + ".jpg"
                path = f"{base_path}/{img_name}"
                img = Image.open(path).convert("RGB")
                item["IMAGE"] = img
                yield item
 
        with open(self.input_file_name) as f:
            data = json.load(f)
            data = [{
                "question_id": idx,
                "prompt": v["question"],
                "gt_answer": v["answer"],
                "imageId": v["imageId"],
            } for idx, v in data.items()]
            self._len = len(data)
        return _loader()
        
    def format_example(self, item: Dict[str, Any]):
        template = self.img_template
        return [
            {"role": "user", "content": template.format(item["prompt"])}
        ], item["IMAGE"]
    

def get_eval_template(name):
    eval_template = eval_templates.get(name, None)()
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template