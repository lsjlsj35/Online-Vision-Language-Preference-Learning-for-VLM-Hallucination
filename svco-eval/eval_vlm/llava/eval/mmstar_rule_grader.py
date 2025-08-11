import argparse
import json
import re
from collections import defaultdict
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Process JSONL file path.')
parser.add_argument('--answer_file', default = "answer.jsonl",help='Path to the JSONL file')
args = parser.parse_args()


def is_answer_correct_mcq(question, correct_label, llm_answer):
    llm_answer_cleaned = re.sub(r'[\(\)\s\.]', ' ', llm_answer.strip().lower()).strip()
    return llm_answer_cleaned.startswith(correct_label.lower())


# num_correct, num_total = 0, 0
# DICT = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

# with open(args.answer_file) as file:
#     lines = file.readlines()
#     index = 0
#     for line in tqdm(lines, total=len(lines)):
#         data = json.loads(line)
#         question, correct_answer, model_response, task_type, relation_type = data["prompt"], data["answer"], data["response"], data["l2_type"], data["type"]
#         correct = is_answer_correct_mcq(question, correct_answer, model_response)
#         DICT[relation_type][task_type]["correct"] += int(correct)
#         DICT[relation_type][task_type]["total"] += 1
            
# print(f"== RESULTS ==")
# for relation_type in DICT:
#     print(f"{relation_type}")
#     for task_type in DICT[relation_type]:
#         print(f"{task_type}: {DICT[relation_type][task_type]['correct']/(0.0001+DICT[relation_type][task_type]['total'])}")
#         print(f"      Total: {DICT[relation_type][task_type]['total']}")
#     print()

num_correct, num_total = 0, 0
DICT = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

total_correct = 0
total_samples = 0

with open(args.answer_file) as file:
    index = 0
    lines = file.readlines()
    for line in lines:
        data = json.loads(line)
        question, correct_answer, model_response, task_type, relation_type = data["prompt"], data["answer"], data["response"], data["l2_type"], data["type"]
        correct = is_answer_correct_mcq(question, correct_answer, model_response)
        DICT[relation_type][task_type]["correct"] += int(correct)
        DICT[relation_type][task_type]["total"] += 1
        total_correct += int(correct)
        total_samples += 1

print(f"== RESULTS ==")

print("= OVERALL =")
print(f"{total_correct}/{total_samples} = {total_correct/total_samples}")
for relation_type in DICT: 
    print(f"{relation_type}")
    for task_type in DICT[relation_type]:
        print(f"{task_type}: {DICT[relation_type][task_type]['correct']/(DICT[relation_type][task_type]['total'])}")
        print(f"      Total: {DICT[relation_type][task_type]['total']}")
    print()
# print(file)