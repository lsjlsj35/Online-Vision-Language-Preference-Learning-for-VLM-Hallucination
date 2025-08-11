from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class VTCLArguments:
    strategy: Optional[str] = field(
        default="sft",
        metadata={"help": "learning strategy: vtcl, mdpo, sft, dpo"}
    )
    anchor_ratio: Optional[float] = field(
        default=0.999,
        metadata={"help": "anchor loss ratio in mDPO training pipeline"}
    )
    neg_anchor_ratio: Optional[float] = field(
        default=0.0
    )
    iter_training: Optional[bool] = field(
        default=False,
        metadata={"help": "use iter training step"}
    )
    not_half_dpo_loss: Optional[bool] = field(
        default=False,
        metadata={"help": "not half dpo loss"}
    )
    vtcl_mode: Optional[str] = field(
        default="single",
        metadata={"help": "s, single[only positive data]; sameBatch[pos and nega data are in the same batch]; difBatch[pos and nega data in different batch]"},
    )
    mask_same_sequence: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to mask the same parts when calculating loss."}
    )
    experience_buffer: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to use experience buffer"}
    )
    ds_path: Optional[str] = field(
        default="",
        metadata={"help": "the path of dataset"}
    )
    test: Optional[bool] = field(
        default=False,
        metadata={"help": "test mode"}
    )
    test_dataset_name: Optional[str] = field(
        default="",
        metadata={"help": "test dataset"}
    )
    test_output_name: Optional[str] = field(
        default=None,
        metadata={"help": "the name of saved result file; default save path: test.json"}
    )
    online_data_save_dir: Optional[str] = field(
        default="",
        metadata={"help": "the dir of saved online data"}
    )
    exp_per_batch: Optional[int] = field(
        default=2,
        metadata={"help": "the number of experience per batch"}
    )
    online_only: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to use online data only"}
    )
    online_only_strategy: Optional[str] = field(
        default="",
        metadata={"help": "the strategy of using online data only. "}
    )

    def __post_init__(self):
        self.load_double_dataset = True if self.vtcl_mode == "difBatch" else False
        self.use_negative_data = True if self.vtcl_mode == "sameBatch" else False
        if self.experience_buffer:
            assert self.ds_path != "", "please specify the path of dataset"
        if self.strategy == "sft" and self.anchor_ratio == 0.999:
            self.anchor_ratio = 0.0