import base64
from io import BytesIO
from pathlib import Path
from PIL import Image
from typing import Union


def _encode_image(image_path: Union[Path, str]):
    with Image.open(image_path) as img:
        width, height = img.size
        if width > 1000 or height > 1000:
            scale = min(1000 / width, 1000 / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        buffered.seek(0)
        result_img = base64.b64encode(buffered.read()).decode('utf-8')
        
    return result_img


def _add_img(message, *imgs):
    imgs = [_encode_image(img) for img in imgs]
    for img in imgs:
        message[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img}"}
        })
    return message


"""
input format:
{
    "prompt": str,
    "response": str,
    "answer": str,
    **others**
}
"""
class VLMTask_OverallScore_GT:
    instruction = """# Task
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
    @classmethod
    def f(cls, prompt="", response="", answer="", **kwargs):
        message = [{
            "role": "user",
            "content": [
                {"type": "text", "text": cls.instruction.format(question=prompt, response=response, answer=answer)}
            ]
        }]
        return message


"""
input format:
{
    "image": List[Union[Path, str]],
    "prompt": str,
    "response": str,
    "answer": str
}
"""
class VLMTask_OverallScore_GTandImage:
    pass


class VLMTask_SentScoreandRevise_GT:
    pass


TaskList = {
    "VLMTask_OverallScore_GT": VLMTask_OverallScore_GT,
}


def get_task(name):
    return TaskList[name].f