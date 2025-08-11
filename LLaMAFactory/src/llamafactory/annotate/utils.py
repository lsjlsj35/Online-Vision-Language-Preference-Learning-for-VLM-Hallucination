import glob
import json
import yaml
from pathlib import Path
from typing import Dict, List


from ..hparams import AnnotateArguments
from .formatting import get_format


BASE_PATH = Path.cwd()
GENERATION_PATH = BASE_PATH / "generation"


class LoadTool:
    @staticmethod
    def json(f):
        return json.load(f)
    
    @staticmethod
    def jsonl(f):
        data = [json.loads(l) for l in f]
        return data
    

def _get_load_tool(p: Path):
    if str(p).endswith('json'):
        return LoadTool.json
    if str(p).endswith('jsonl'):
        return LoadTool.jsonl
    else:
        raise ValueError(f"Unsupported file format: {p}")


def get_all_data_and_process(args: AnnotateArguments) -> List[Dict]:
    # get all paths
    target = args.target.split(',')
    file_name = args.file_prefix

    all_files = []
    for t in target:
        all_files.extend(GENERATION_PATH.glob(t.rstrip('/') + '/' + file_name))
    print("[FILE]:", end='\n\t')
    print('\n\t'.join([str(f) for f in all_files]))
    print(f"[FORMAT]: {args.format}")
    if not args.yes and input().lower() != "yes":
        return None
    
    # load and process
    all_data = []
    for path in all_files:
        with open(path) as f:
            all_data.extend(_get_load_tool(path)(f))
    process_func = get_format(args.format)
    print(all_data[0])
    return list(map(process_func, all_data))


def get_api_config(args: AnnotateArguments):
    if args.is_local_model:
        raise NotImplementedError
    with open(BASE_PATH / args.api_config_file) as f:
        data = yaml.safe_load(f)
    return data["api_key"], data["base"]