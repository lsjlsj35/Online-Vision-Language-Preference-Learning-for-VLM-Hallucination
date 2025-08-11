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
class Template:
    customed_inference = False
    _len = None
    def format_example(self, item):
        pass

    def __len__(self):
        return self.length

    def load_data(self):
        pass

    @property
    def length(self):
        return self._len
    
    def post_process(self, dir_path="", path=""):
        pass


# class Generation_Task(Eval_Template):
#     def generate(model, inputs):
#         pass
    

# @register(name="base")
# class BaseTemplate(Template):

#     @property
#     def length(self):
#         return self._len

#     def load_data(self):
#         def _loader():
#             for item in data:
#                 img = Image.open(item["images"][0])
#                 item["IMAGE"] = img
#                 yield item

#         input_file_name = "/root/LLaMA-Factory/data/POVID-train-random.json"
#         with open(input_file_name) as f:
#             data = json.load(f)
#             self._len = len(data)
#         return _loader()  # If not using wrapper, the property "_len" will not be assigned immediately when this method is called.
        
#     def format_example(self, item):
#         return [
#             {"role": "user", "content": item["conversations"][0]["value"]}
#         ], item["IMAGE"]
    
#     def post_process(self, dir_path, path):
#         with open(path) as f:
#             data = [json.loads(line) for line in f]
#         for item in data:
#             item["rejected"]["value"] = item["result"]
#             del item["result"]
#         with open(dir_path / "POVID-train-random-nega_gen.json", 'w') as f:
#             json.dump(data, f, indent=4)


class LMStandardFormatTemplate(Template):
    FILE_NAME = ""
    POST_PROCESS = True

    @property
    def length(self):
        return self._len

    def load_data(self):
        def _loader():
            for item in data:
                img = Image.open(item["images"][0])
                item["IMAGE"] = img
                yield item

        input_file_name = f"/root/LLaMA-Factory/data/{self.FILE_NAME}.json"
        with open(input_file_name) as f:
            data = json.load(f)
            self._len = len(data)
        return _loader()  # If not using wrapper, the property "_len" will not be assigned immediately when this method is called.
        
    def format_example(self, item):
        return [
            {"role": "user", "content": item["conversations"][0]["value"]}
        ], item["IMAGE"]
    
    def post_process(self, dir_path, path):
        if self.POST_PROCESS:
            with open(path) as f:
                data = [json.loads(line) for line in f]
            for item in data:
                item["rejected"]["value"] = item["result"]
                del item["result"]
            with open(dir_path / f"{self.FILE_NAME}_negaGen_raw.json", 'w') as f:
                json.dump(data, f, indent=4)
    

@register("mDPO_train")
class MDPOTrainTemplate(LMStandardFormatTemplate):
    FILE_NAME = "mDPO_train"


@register("mDPO_val")
class MDPOValTemplate(LMStandardFormatTemplate):
    FILE_NAME = "mDPO_val"


@register("mvc_train")
class MVCTrainTemplate(LMStandardFormatTemplate):
    FILE_NAME = "mvc_train"


@register("mvc_val")
class MVCValTemplate(LMStandardFormatTemplate):
    FILE_NAME = "mvc_val"


@register("opadpo_train")
class MVCTrainTemplate(LMStandardFormatTemplate):
    FILE_NAME = "opadpo_train"


@register("opadpo_val")
class MVCValTemplate(LMStandardFormatTemplate):
    FILE_NAME = "opadpo_val"


@register("mvc_filtered_train")
class MVCFilteredTrain(Template):
    prompt = """Please answer directly whether the response is correct according to the ground truth response. Sometimes there is hallucinations in the response while sometimes it may offer a more detailed description. If they are not the same, please answer "No".
- Question
{question}
    
- Response to be judged
{response}

- Correct answer
{answer}

Please only respond with either "Yes" or "No".
Your Answer: 
"""
    def load_data(self):
        def _loader():
            for item in self.data:
                yield item

        with open("/root/LLaMA-Factory-new/generation/mvc_train/llava-1.5-7b-hf/mvc_train_negaGen_raw.json") as f:
            self.data = json.load(f)
        self._len = len(self.data)
        return _loader()
        
    def format_example(self, item):
        return [
            {"role": "user", "content": self.prompt.format(
                question=item["conversations"][0]["value"].replace("<image>", '').strip(),
                response=item["rejected"]["value"],
                answer=item["chosen"]["value"]
            )}
        ], None
    

@register("mvc_bo3_train")
class MVCBo3Train(Template):
    path = "/root/LLaMA-Factory-new/generation/mvc_train/llava-1.5-7b-hf"
    prompt = """Please rank the 3 responses according to the ground truth response. Sometimes there is hallucination in the response while sometimes the response may offer a more detailed description.
- Notes
1. The quality of the response depends on the accuracy and adherence to the ground truth answer. So if the response are far more detailed, it may not be considered as a perfect one.
2. Your response should have the following format: "Best: [x]\nWorst: [y]" where "[x]" and "[y]" are the index of the corresponding response (1, 2 or 3).
3. It's possible that all the 3 responses are incorrect, at that time you should also give your rankings as above. 
4. You should answer "N/A" **when** and **only when** the 3 responses are similar or even the same. At other situations, you should provide your response in Notes 2.
5. Your response should only include your answer to my request, you don't need to give any reasons to your answer.

- Question
{question}
    
- Response 1
{response1}

- Response 2
{response2}

- Response 3
{response3}

- Correct answer
{answer}

Your Answer: 
"""
    def load_data(self):
        def _loader():
            for item in self.data:
                yield item

        self.data = []
        for i in range(1, 4):
            with open(f"{self.path}/trial_{i}.jsonl") as f:
                self.data.append([json.loads(line) for line in f])
        self.data = [{'1': i, '2': j, '3': k} for i, j, k in zip(*self.data)]
        self._len = len(self.data)
        return _loader()
        
    def format_example(self, ITEM):
        item = ITEM['1']
        return [
            {"role": "user", "content": self.prompt.format(
                question=item["conversations"][0]["value"].replace("<image>", '').strip(),
                response1=ITEM['1']["rejected"]["value"],
                response2=ITEM['2']["rejected"]["value"],
                response3=ITEM['3']["rejected"]["value"],
                answer=item["chosen"]["value"]
            )}
        ], None
    

@register("mDPO_score_train")
class MDPOScoreTrain(Template):
    path = "/root/LLaMA-Factory-new/generation/mDPO_train/llava-1.5-7b-hf"
    prompt = """# Task
I want you to be a fair judge to evaluate the quality of a response. Please assess how good the response is based on the provided reference answer and the corresponding question.

# Requirements
1. The quality of the response depends on its accuracy and the degree of adherence to the correct answer. Therefore, if the response is much more detailed than the reference answer, it should not be considered a **very** good response (although it may still be considered a good one).
2. Please directly provide the score of the response, with a full score of 10. Your response should follow this format: "Score: [x]\n", where "[x]" represents the score you give, and "\n" is a line break.
3. Please do not provide additional reasoning, just give the score directly.

- Question
{question}
    
- Response
{response}

- Correct answer
{answer}
"""
    def load_data(self):
        def _loader():
            for item in self.data:
                yield item

        self.data = []
        for i in range(1, 4):
            with open(f"{self.path}/trial_{i}.jsonl") as f:
                data = [json.loads(line) for line in f]
                data = [{**item, "idx": i} for item in data]
                self.data.extend(data)
        self._len = len(self.data)
        return _loader()
        
    def format_example(self, item):
        return [
            {"role": "user", "content": self.prompt.format(
                question=item["conversations"][0]["value"].replace("<image>", '').strip(),
                response=item["result"],
                answer=item["chosen"]["value"]
            )}
        ], None
    

@register("mvc_supplementary_train1")
class mvc_supp_score_train1(Template):
    path = "/root/LLaMA-Factory-new/generation/mvc_train/mdpo_dpo_llava-1.5-7b-hf_Exp5"
    i = 1
    prompt = """# Task
I want you to be a fair judge to evaluate the quality of a response. Please assess how good the response is based on the provided reference answer and the corresponding question.

# Requirements
1. The quality of the response depends on its accuracy and the degree of adherence to the correct answer. Therefore, if the response is much more detailed than the reference answer, it should not be considered a **very** good response (although it may still be considered a good one).
2. Please directly provide the score of the response, with a full score of 10. Your response should follow this format: "Score: [x]\n", where "[x]" represents the score you give, and "\n" is a line break.
3. Please do not provide additional reasoning, just give the score directly.

- Question
{question}
    
- Response
{response}

- Correct answer
{answer}
"""
    def load_data(self):
        def _loader():
            for item in self.data:
                yield item

        self.data = []
        with open(f"{self.path}/trial_{self.i}.jsonl") as f:
            data = [json.loads(line) for line in f]
            data = [{**item, "idx": self.i} for item in data]
            self.data.extend(data)
        self._len = len(self.data)
        return _loader()
        
    def format_example(self, item):
        return [
            {"role": "user", "content": self.prompt.format(
                question=item["conversations"][0]["value"].replace("<image>", '').strip(),
                response=item["result"],
                answer=item["chosen"]["value"]
            )}
        ], None


@register("mvc_supplementary_train2")
class mvc_supp_score_train2(mvc_supp_score_train1):
    i = 2


@register("mvc_supplementary_train3")
class mvc_supp_score_train3(mvc_supp_score_train1):
    i = 3


@register("mvc_supplementary_train4")
class mvc_supp_score_train4(mvc_supp_score_train1):
    i = 4


@register("mDPO_bo3_train")
class MDPOBo3Train(MVCBo3Train):
    path = "/root/LLaMA-Factory-new/generation/mDPO_train/llava-1.5-7b-hf"


@register("response_dif")
class ResponseDif(Template):
    prompt = """**[Task]**  
Given an unknown image-related question, a correct answer, and an inaccurate response, carefully analyze the differences between the response and the answer. Then, provide a brief instruction describing how to modify the image so that the original inaccurate response becomes correct while the original correct answer becomes incorrect.  

**[Example 1]**  
**<Question>**  
What is on the sandwich?  
**<Answer>**  
The sandwich has a slice of egg and tomato on it.  
**<Response>**  
The sandwich has tomatoes and lettuce on it.  

**<Modification>**
Replace the egg on the sandwich with lettuce.

**Explanation**: The answer mentions egg and tomato, while the response mentions tomato and lettuce. To make the response correct, the egg must be replaced with lettuce in the image.

**[Example 2]**  
**<Question>**  
What time of day is it?  
**<Answer>**  
It is sunset time, as indicated by the warm-colored sky.  
**<Response>**  
It is the early morning on the beach.  

**<Modification>**
Make the sky blue.

**Explanation**: The answer and the response describe different times of day.

**[Example 3]**  
**<Question>**  
Can you point out the details that make this image unique?  
**<Answer>**  
In the image, there is a slice of pizza on a plate with tomatoes and cheese. The pizza appears to be homemade and has been cut into two pieces. The tomatoes are sliced in half, revealing their juicy interior. The cheese on top of the pizza is melted, creating a delicious-looking dish. Additionally, there is a fork nearby, suggesting that someone might be planning to enjoy this pizza soon.  
**<Response>**  
In the image, there is a plate with a slice of pizza topped with tomatoes, herbs, and cheese. The distinctive detail about the image is that the pizza is missing two slices, leaving just one slice remaining on the plate. This suggests that someone has already started enjoying the pizza. The slice appears to be well-cooked and freshly served, creating an appetizing and mouthwatering scene for the viewer.  

**<Modification>**
Add a person eating the pizza.

**Explanation**: The response describes the state of the pizza and implies that someone has already started eating it. Since the response is lengthy, focus on the most critical difference

**[Requirements]**  
- The description should be brief but precise.  
- If both the response and the answer are lengthy, focus on the most critical one or two aspects for modification.  
- Do not provide any additional analysis. Please respond only with the modification instruction.  
- You should add an item which exists in the response but does not show in the answer, or remove something that is only mentioned in the answer, or replace the answer-mentioned content with a response-mentioned content.


**<Question>**
{question}
**<Answer>**
{answer}
**<Response>**
{response}

**<Modification>**
"""

    def load_data(self):
        def _loader():
            for item in self.data:
                yield item
            
        self.data = []
        with open("/root/LLaMA-Factory-new/annotations/opa_skew_tmp.json") as f:
            self.data = json.load(f)
        self._len = len(self.data)
        return _loader()
    
    def format_example(self, item):
        return [
            {"role": "user", "content": self.prompt.format(
                question=item["prompt"],
                answer=item["chosen"],
                response=item["rejected"]
            )}
        ], None
    

@register("desc_infer")
class DescriptionInfer(Template):
    prompt = """# Task  
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

    def load_data(self):
        def _loader():
            for item in self.data:
                yield item
            
        self.data = []
        with open("/root/LLaMA-Factory-new/annotations/opa_skew_tmp.json") as f:
            self.data = json.load(f)
        self._len = len(self.data)
        return _loader()
    
    def format_example(self, item):
        return [
            {"role": "user", "content": self.prompt.format(
                question=item["prompt"],
                answer=item["rejected"],  ### important!!
                response=item["chosen"]
            )}
        ], None


class DescriptionInferDIST(DescriptionInfer):
    IDX = -1
    def load_data(self):
        def _loader():
            for item in self.data:
                yield item
            
        self.data = []
        with open("/root/LLaMA-Factory-new/annotations/opa_skew_tmp.json") as f:
            self.data = json.load(f)[self.IDX::4]
        self._len = len(self.data)
        return _loader()
    

@register("desc_infer1")
class DescriptionInfer1(DescriptionInferDIST):
    IDX=0


@register("desc_infer2")
class DescriptionInfer2(DescriptionInferDIST):
    IDX=1


@register("desc_infer3")
class DescriptionInfer3(DescriptionInferDIST):
    IDX=2


@register("desc_infer4")
class DescriptionInfer4(DescriptionInferDIST):
    IDX=3
        

def get_template(name):
    eval_template = eval_templates.get(name, None)()
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template