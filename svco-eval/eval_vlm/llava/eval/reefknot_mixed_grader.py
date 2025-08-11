import argparse
import json
import re
import torch
from tqdm import tqdm
from transformers import DebertaV2ForSequenceClassification, AutoTokenizer


parser = argparse.ArgumentParser(description='Process JSONL file path.')
parser.add_argument('--answer_file', default = "answer.jsonl",help='Path to the JSONL file')
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("/opt/tiger/vtcl/model/deberta-v2-xlarge")
model = DebertaV2ForSequenceClassification.from_pretrained("/opt/tiger/vtcl/model/deberta-v2-xlarge").cuda()


def is_answer_correct_dis(question, correct_label, llm_answer):
    llm_answer_cleaned = re.sub(r'[\(\)\s\.]', '', llm_answer.strip().lower())
    return llm_answer_cleaned.startswith(correct_label.strip().lower())


def is_answer_correct_mcq(question, correct_label, llm_answer):
    llm_answer_cleaned = re.sub(r'[\(\)\s\.]', ' ', llm_answer.strip().lower()).strip()
    return llm_answer_cleaned.startswith(correct_label.lower())


def _predict_entailment(premise, hypothesis):
    inputs = tokenizer(premise, hypothesis, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    return prediction


def is_answer_correct_vqa(question, correct_label, llm_answer):
    label1 = _predict_entailment(correct_label, llm_answer)
    label2 = _predict_entailment(llm_answer, correct_label)
    return label1 == 2 and label2 == 2


num_correct, num_total = 0, 0
DICT = {
    "cognitive": {q: {"correct": 0, "total": 0} for q in ["YN", "MCQ", "VQA"]},
    "percetion": {q: {"correct": 0, "total": 0} for q in ["YN", "MCQ", "VQA"]},
}

overall_data = []
with open(args.answer_file) as file:
    lines = file.readlines()
    index = 0
    for line in tqdm(lines, total=len(lines)):
        data = json.loads(line)
        question, correct_answer, model_response, task_type, relation_type = data["prompt"], data["answer"], data["text"], data["type"], data["relation_type"]
        if task_type == "Yes/No":
            correct = is_answer_correct_dis(question, correct_answer, model_response)
            DICT[relation_type]["YN"]["correct"] += int(correct)
            DICT[relation_type]["YN"]["total"] += 1
        elif task_type == "Multichoice":
            correct = is_answer_correct_mcq(question, correct_answer, model_response)
            DICT[relation_type]["MCQ"]["correct"] += int(correct)
            DICT[relation_type]["MCQ"]["total"] += 1
        elif task_type == "VQA":
            if "roberta_judge" not in data:
                correct = is_answer_correct_vqa(question, correct_answer, model_response)
                data["roberta_judge"] = correct
            else:
                correct = data["roberta_judge"]
            DICT[relation_type]["VQA"]["correct"] += int(correct)
            DICT[relation_type]["VQA"]["total"] += 1
        overall_data.append(data)

with open(args.answer_file, 'w') as file:
    for data in overall_data:
        file.write(json.dumps(data) + '\n')
            

print(f"== RESULTS ==")
for relation_type in DICT:
    print(f"{relation_type}")
    for task_type in DICT[relation_type]:
        print(f"{task_type}: {DICT[relation_type][task_type]['correct']/(0.0001+DICT[relation_type][task_type]['total'])}")
    print()
