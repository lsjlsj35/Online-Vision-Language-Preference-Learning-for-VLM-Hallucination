import os
from typing import Any, Dict, List
from datasets import load_dataset


TEMPLATE = {}


TEMPLATE = {}


def eval_class(name):
    def wrapper(cls):
        NAME = [name] if type(name) != list else name
        for n in NAME:
            TEMPLATE[n] = {k: getattr(cls, k, None) for k in ["loading_callback", "formatting_callback", "saving_callback"]}
        return cls
    return wrapper


class EvalBaseTemplate:
    @staticmethod
    def loading_callback(data: List[Dict[str, Any]], file_path: str):
        return data
    
    @staticmethod
    def formatting_callback(item: Dict[str, Any], args):
        return item.get("prompt", "Hello"), item.get("image", None), {"answer": item.get("answer", "Hello")}
    
    @staticmethod
    def saving_callback(item: Dict[str, Any]):
        # item={"prompt": xxx, "response": xxx, "answer_id": x, "model_id": x, **saving_dict, "meta_data": {}}
        return item
    

@eval_class(["rqa", "tqa"])
class DefaultVQA:
    @staticmethod
    def formatting_callback(item, args):
        q = item.pop("text")
        return q, os.path.join(args.image_folder, item["image"]), item
    

@eval_class("amber")
class AMBER:
    @staticmethod
    def formatting_callback(item, args):
        q = item.pop("text")
        return q, os.path.join(args.image_folder, item["image"]), item
    
    @staticmethod
    def saving_callback(item):
        response = item.pop("response")
        return {**item, "text": response}


@eval_class("cvbench")
class CVBench:
    @staticmethod
    def formatting_callback(item, args):
        q = item.pop("prompt")
        q = q + '\n' + "Answer with the option's letter from the given choices directly."
        return q, os.path.join(args.image_folder, item["image"]), {k: item[k] for k in ["answer", "source", "task", "idx"]}


@eval_class("llavabench")
class LLaVABench:
    @staticmethod
    def formatting_callback(item, args):
        q = item.pop("text")
        return q, os.path.join(args.image_folder, item["image"]), item
    
    @staticmethod
    def saving_callback(item):
        response = item.pop("response")
        return {**item, "text": response}


@eval_class("mmhal")
class MMHal:
    @staticmethod
    def formatting_callback(item, args):
        q = item.pop("question")
        item.pop("model_answer")
        image_src = item["image_src"]
        return q, os.path.join(args.image_folder, image_src.rsplit('/')[-1]), item
        
    @staticmethod
    def saving_callback(item):
        prompt = item.pop("prompt")
        response = item.pop("response")
        return {**item, "question": prompt, "model_answer": response}
    

@eval_class("mmstar")
class MMStar:
    @staticmethod
    def loading_callback(item, fp):
        ds = load_dataset(os.path.expanduser(fp), "val")["val"].to_list()
        return ds

    @staticmethod
    def formatting_callback(item, args):
        q = item.pop("question") + '\n' + "Answer with the option's letter from the given choices directly."
        t = item.pop("category")
        l2_type = item.pop("l2_category")
        img = item.pop("image")["bytes"]
        return q, img, {"type": t, "l2_type": l2_type, **item}
    

@eval_class("objecthal")
class ObjectHalBench:
    @staticmethod
    def formatting_callback(item, args):
        q = item.pop("question")
        img = item.pop("image")
        return q, img, item
    
    @staticmethod
    def saving_callback(item):
        prompt = item.pop("prompt")
        response = item.pop("response")
        return {**item, "question": prompt, "model_answer": response}


def get_template(name):
    return TEMPLATE[name]