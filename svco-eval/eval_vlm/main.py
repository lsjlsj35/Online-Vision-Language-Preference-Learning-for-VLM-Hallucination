import argparse
import os

from llamafactory_eval.template import get_template
from llamafactory_eval.vqa import eval_model


def main(args):
    if args.llavalib:
        if args.model_path is None or args.model_path in ["none"]:
            from llamafactory_eval.model_llava import LlavaModel
            model = LlavaModel(base_path=args.model_base)
        else:
            from llamafactory_eval.model_llava import LlavaModelPeft
            model = LlavaModelPeft(base_path=args.model_base, model_path=args.model_path)
    else:
        from llamafactory_eval.model import DummyModel, LlavaBaseModel, LlamafactoryModel
        if args.test:
            model = DummyModel()
        elif args.model_path is None or args.model_path in ["base", "none"]:
            model = LlavaBaseModel(base_path=args.model_base)
        else:
            model = LlamafactoryModel(base_path=args.model_base, lora_path=args.model_path)
    callbacks = get_template(args.template)
    eval_model(args, model, **callbacks)


def set_test(args):
    tgt_path = args.answers_file
    if '/' not in tgt_path:
        tgt_path = os.path.join("/opt/tiger/vtcl/svco-eval/LOG_EVAL/test", tgt_path)
    else:
        tgt_path = "/opt/tiger/vtcl/svco-eval/LOG_EVAL/test/" + tgt_path.split('/')[-1]
    args.answers_file = tgt_path
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", type=str, default="/mnt/bn/lsj-yg/tasks/vtcl/saves/mdpo_online-dpo/llava-1.5-7b-hf_Exp1/checkpoint-1050")
    parser.add_argument("--model-path", type=str, default=None, required=False)
    parser.add_argument("--model-base", type=str, default="/opt/tiger/vtcl/model/llava-1.5-7b-hf")
    parser.add_argument("--bf16", type=bool, default=True)

    parser.add_argument("--template", type=str, default="mmhal")
    
    parser.add_argument("--image-folder", type=str, default="/opt/tiger/vtcl/benchmark/cvbench/img")
    parser.add_argument("--question-file", type=str, default="/opt/tiger/vtcl/svco-eval/eval_vlm/playground/data/eval/cvbench/cvbench_test.jsonl")
    parser.add_argument("--answers-file", type=str, default="/opt/tiger/vtcl/svco-eval/LOG_EVAL/test_cvbench_ckpt1050.jsonl")

    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)

    parser.add_argument("--llavalib", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        args = set_test(args)
    main(args)