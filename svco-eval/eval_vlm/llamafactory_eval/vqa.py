import argparse
import json
import math
import os
import shortuuid
from datasets import Dataset, load_dataset
from PIL import Image
from tqdm import tqdm


def eval_model(args, model, loading_callback=None, formatting_callback=None, saving_callback=None):
    if type(args.question_file) != list:
        args.question_file = [args.question_file]
    questions = []
    for qf in args.question_file:
        if qf.endswith(".jsonl"):
            single_file_questions = [json.loads(q) for q in open(os.path.expanduser(qf))]
        elif qf.endswith(".json"):
            single_file_questions = json.load(open(os.path.expanduser(qf)))
        elif os.path.isdir(qf):
            single_file_questions = []
        else:
            raise RuntimeError("Unknown file type")
        if loading_callback is not None:
            single_file_questions = loading_callback(single_file_questions, qf)
        elif os.path.isdir(qf):
            raise RuntimeError("Please provide a loading_callback for directory-based question files")
        questions.extend(single_file_questions)

    answers_file_path = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file_path), exist_ok=True)
    
    save_as_json = False if args.answers_file.endswith(".jsonl") else True
    ALL_RESULTS = []
    with open(answers_file_path, "w") as ans_file:
        for line in tqdm(questions):
            if formatting_callback is not None:
                qs, im, saving_dict = formatting_callback(line, args)
            else:
                # cvbench format
                qs, im, saving_dict = line["prompt"], os.path.join(args.image_folder, line["image"]), {k: line[k] for k in ["answer", "source", "task", "idx"]}
            outputs = model(qs, images=im, temperature=args.temperature, top_p=args.top_p, num_beams=args.num_beams, max_new_tokens=args.max_new_tokens)
            ans_id = shortuuid.uuid()

            result_item = {"prompt": qs, "response": outputs, "answer_id": ans_id, **saving_dict, "metadata": {}}
            if saving_callback is not None:
                result_item = saving_callback(result_item)

            if save_as_json:
                ALL_RESULTS.append(result_item)
            else:
                ans_file.write(json.dumps(result_item) + "\n")
        if save_as_json:
            json.dump(ALL_RESULTS, ans_file, indent=4)