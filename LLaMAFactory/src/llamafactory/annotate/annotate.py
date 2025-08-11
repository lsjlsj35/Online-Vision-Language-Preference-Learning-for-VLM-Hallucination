import json
import time
import threading
from pathlib import Path

from openai import OpenAI

from ..hparams import get_annotate_args, AnnotateArguments
from .task import get_task
from .utils import get_all_data_and_process, get_api_config


COUNTER = 0
lock = threading.Lock()

DST_DIR = Path.cwd() / "annotations"


class Annotator:
    def __init__(self, base, key, message_func=None, model_version="deepseek-v3", generation_config=None):
        self.model = model_version
        self.base = base
        self.key = key
        self.client = OpenAI(api_key=key, base_url=base)
        self.message_func = message_func
        self.generation_config = {} if generation_config is None else generation_config

    def annotate(self, **template_input):
        message = self.message_func(**template_input)

        retry_interval = 1
        retry = 3
        for _ in range(retry):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message,
                    **self.generation_config
                )
                res = response.choices[0].message.content
                return res
            except Exception as e:
                print(e)
                retry_interval *= 2
                time.sleep(retry_interval)
        return ""
    

def main():
    args: AnnotateArguments = get_annotate_args()[0]

    processed_data = get_all_data_and_process(args)
    if processed_data is None:
        return
    task_func = get_task(args.task)

    key, base = get_api_config(args)

    annotator = Annotator(base, key, message_func=task_func, model_version=args.model)

    if args.max_workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def F(annotator, n, item, f):
            global COUNTER
            nonlocal wait_for, final_results
            if args.max_eval == 0 or COUNTER < args.max_eval:
                item["annotation"] = annotator.annotate(**item)
                COUNTER = COUNTER + 1
                print(COUNTER)
            with lock:
                if n == wait_for:
                    json.dump({"_id": wait_for, **item}, f)
                    f.write('\n')
                    print(n, "saved.")
                    wait_for += 1
                    while wait_for in final_results:
                        json.dump({"_id": wait_for, **final_results[wait_for]}, f)
                        f.write('\n')
                        print(wait_for, "saved.")
                        del final_results[wait_for]
                        wait_for += 1
                else:
                    final_results[n] = item
            return n, item
        
        final_results = {}
        futures = []

        wait_for = 0
        output_path: Path = (DST_DIR / f"{args.output}.jsonl")
        if output_path.exists() and not args.yes:
            if input(str(output_path)+" exists!") != "yes":
                return
        with open(DST_DIR / f"{args.output}.jsonl", 'w') as f:
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                for idx, item in enumerate(processed_data):
                    futures.append(executor.submit(F, annotator, idx, item, f))
            for job in as_completed(futures):
                pass
            if final_results:
                print("FINAL save")
                for k, v in final_results.items():
                    json.dump({"_id": k, **v}, f)
                    f.write('\n')


if __name__ == "__main__":
    main()