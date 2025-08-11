from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional


@dataclass
class AnnotateArguments:
    output: str = field(
        default="",
    )
    target: str = field(
        default="",
        metadata={"help": "generation files to annotate, use comma to seperate"}
    )
    file_prefix: Optional[str] = field(
        default="trial_*.jsonl",
        metadata={"help": "target file prefix, defualt to 'trial_*.jsonl'. "}
    )
    format: Optional[str] = field(
        default="default",
        metadata={"help": "the data format, default: \{ *train data format,  'response': '' \}"}
    )
    task: Optional[str] = field(
        default="VLMTask_OverallScore_GT",
        metadata={"help": "the task which the annotator does, it determines the prompt."}
    )
    model: Optional[str] = field(
        default="deepseek-v3",
        metadata={"help": "annotator"}
    )
    max_workers: Optional[int] = field(
        default=10
    )
    max_eval: Optional[int] = field(
        default=0,
    )
    api_config_file: Optional[str] = "config.yaml"
    is_local_model: Optional[bool] = False
    yes: Optional[bool] = field(
        default=False,
        metadata={"help": "skip displaying all the files"}
    )
    test: Optional[bool] = field(
        default=False
    )

    def __post_init__(self):
        if self.test:
            self.max_eval = 5
            self.output = "test"
        assert self.output, "Please enter output file name"