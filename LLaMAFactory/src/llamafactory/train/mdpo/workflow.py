# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/examples/scripts/dpo.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import dataclasses
import json
from copy import deepcopy
from functools import partial
from pathlib import Path
from PIL import Image
from typing import List, Optional

# import byteio
from io import BytesIO
from tqdm import tqdm

import torch
import torchvision

from ...data import get_dataset, get_template_and_fix_tokenizer
from ...data.formatter import EmptyFormatter, FunctionFormatter, StringFormatter, ToolFormatter
from ...data.mm_plugin import get_mm_plugin, Qwen2vlPlugin, LlavaPlugin
from ...data.template import _register_template
from ...extras.constants import IGNORE_INDEX, IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER
from ...extras.ploting import plot_loss
from ...hparams import ModelArguments
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push, create_ref_model
from .trainer import CustomDPOTrainer, OnlineCustomDPOTrainer, OnlineCustomDPOTrainerWithBuffer, GRPOTrainerWithBuffer
from .dataloader import SFTDifCollator, MDPODifCollator, DPODifCollator, NoRefSVCODifCollator, SVCODifCollator, SVCOTextRefDifCollator, ORMDifCollator, oriMDPODifCollator


TYPE_CHECKING = True
if TYPE_CHECKING:
    from typing import Dict, Sequence, TypedDict, Union

    from PIL.Image import Image as ImageObject
    from transformers import ProcessorMixin, Seq2SeqTrainingArguments, TrainerCallback
    from transformers.image_processing_utils import BaseImageProcessor

    from .dataloader import MetaDifCollator
    from ...hparams import DataArguments, FinetuningArguments, VTCLArguments

    class EncodedImage(TypedDict):
        path: Optional[str]
        bytes: Optional[bytes]

    ImageInput = Union[str, bytes, EncodedImage, ImageObject]
    VideoInput = str


COLLATOR_MAPPING = {
    "sft": SFTDifCollator,
    "grpo": SFTDifCollator,
    "vtcl": MDPODifCollator,
    "mdpo": MDPODifCollator,
    "dpo": DPODifCollator,
    "svco-noref": NoRefSVCODifCollator,
    "svco-ref": SVCODifCollator,
    "svco-textref": SVCOTextRefDifCollator,
    "orm": ORMDifCollator,
    "ori-mdpo": oriMDPODifCollator
}


MODULE_KEYWORDS: Dict[str, Dict[str, List]] = {
    "llava": {
        "vision_encoder": ["vision_tower"],
        "vision_projector": ["multi_modal_projector"],
        "llm": ["language_model"]
    },
}

class LlavaNewPlugin(LlavaPlugin):
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        print()
        self._validate_input(images, videos)
        num_image_tokens = 0
        image_seqlen = getattr(processor, "image_seqlen") if self.expand_mm_tokens else 1
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)
                num_image_tokens += 1

            message["content"] = content.replace("{{image}}", self.image_token)

        # if len(images) != num_image_tokens:
        #     raise ValueError(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens.")

        return messages


class LlavaMDPO_BasePlugin(LlavaNewPlugin):
    _CROP=(0.999, 1.0)

    def _regularize_images(self, images: Sequence["ImageInput"], **kwargs) -> List["ImageObject"]:
        r"""
        Regularizes images to avoid error. Including reading and pre-processing.
        """
        results = []
        l_img = len(images)
        assert l_img % 3 == 0
        no_process_img = images[:2*l_img//3]
        process_img = images[2*l_img//3:]
        for image in no_process_img:
            if isinstance(image, str):
                image = Image.open(image)
            elif isinstance(image, bytes):
                image = Image.open(BytesIO(image))
            elif isinstance(image, dict):
                if image["bytes"] is not None:
                    image = Image.open(BytesIO(image["bytes"]))
                else:
                    image = Image.open(image["path"])
            if not isinstance(image, ImageObject):
                raise ValueError(f"Expect input is a list of Images, but got {type(image)}.")
            results.append(self._preprocess_image(image, **kwargs))

        for images in process_img:
            if isinstance(images, str):
                images = Image.open(images)
            elif isinstance(images, bytes):
                images = Image.open(BytesIO(images))
            elif isinstance(images, dict):
                if images["bytes"] is not None:
                    images = Image.open(BytesIO(images["bytes"]))
                else:
                    images = Image.open(images["path"])
            if not isinstance(images, ImageObject):
                raise ValueError(f"Expect input is a list of Images, but got {type(images)}.")
            results.append(self._preprocess_image(torchvision.transforms.RandomResizedCrop(size=images.size, scale=self._CROP)(images), **kwargs))

        return results
    

class LlavaORM_BasePlugin(LlavaNewPlugin):
    _CROP=(0.999, 1.0)

    def _get_preprocessed_image(self, image, additional_process=False, **kwargs):
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))
        elif isinstance(image, dict):
            if image["bytes"] is not None:
                image = Image.open(BytesIO(image["bytes"]))
            else:
                image = Image.open(image["path"])
        if not isinstance(image, ImageObject):
            raise ValueError(f"Expect input is a list of Images, but got {type(image)}.")
        if additional_process:
            image = torchvision.transforms.RandomResizedCrop(size=image.size, scale=self._CROP)(image)
        return self._preprocess_image(image, **kwargs)

    def _regularize_images(self, images: Sequence["ImageInput"], **kwargs) -> List["ImageObject"]:
        r"""
        Regularizes images to avoid error. Including reading and pre-processing.
        """
        results = []
        l_img = len(images)
        assert l_img % 3 == 0
        # no_process_img = images[:2*l_img//3]
        # process_img = images[2*l_img//3:]
        for idx, image in enumerate(images):
            if idx < l_img//3 or idx >= 2*l_img//3:
                results.append(self._get_preprocessed_image(image, **kwargs))
            else:
                results.append(self._get_preprocessed_image(image, additional_process=True, **kwargs))
        return results


class LlavaMDPO_r80Plugin(LlavaMDPO_BasePlugin):
    _CROP_IMG = (0.8, 1.0)


class LlavaMDPO_r50Plugin(LlavaMDPO_BasePlugin):
    _CROP_IMG = (0.5, 0.8)

class LlavaORM_r80Plugin(LlavaORM_BasePlugin):
    _CROP_IMG = (0.8, 1.0)

class Qwen2vlNewPlugin(Qwen2vlPlugin):
    def process_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"],
    ) -> List[Dict[str, str]]:
        self._validate_input(images, videos)
        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        merge_length: int = getattr(image_processor, "merge_size") ** 2
        mm_inputs = self._get_mm_inputs(images, videos, processor)
        image_grid_thw = mm_inputs.get("image_grid_thw", [])
        video_grid_thw = mm_inputs.get("video_grid_thw", [])

        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if num_image_tokens >= len(image_grid_thw):
                    raise ValueError(f"`len(images)` is less than the number of {IMAGE_PLACEHOLDER} tokens.")

                image_seqlen = image_grid_thw[num_image_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    IMAGE_PLACEHOLDER, f"<|vision_start|>{self.image_token * image_seqlen}<|vision_end|>", 1
                )
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                if num_video_tokens >= len(video_grid_thw):
                    raise ValueError(f"`len(videos)` is less than the number of {VIDEO_PLACEHOLDER} tokens.")

                video_seqlen = video_grid_thw[num_video_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    VIDEO_PLACEHOLDER, f"<|vision_start|>{self.video_token * video_seqlen}<|vision_end|>", 1
                )
                num_video_tokens += 1

            message["content"] = content

        # if len(images) != num_image_tokens:
        #     raise ValueError(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens.")

        if len(videos) != num_video_tokens:
            raise ValueError(f"The number of videos does not match the number of {VIDEO_PLACEHOLDER} tokens.")

        return messages


_register_template(
    name="llava_new",
    format_user=StringFormatter(slots=["USER: {{content}} ASSISTANT:"]),
    default_system=(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ),
    mm_plugin=LlavaNewPlugin(image_token="<image>", video_token="<|video_pad|>"),
)

_register_template(
    name="llava_mdpo_r80",
    format_user=StringFormatter(slots=["USER: {{content}} ASSISTANT:"]),
    default_system=(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ),
    mm_plugin=LlavaMDPO_r80Plugin(image_token="<image>", video_token="<|video_pad|>"),
)

_register_template(
    name="llava_mdpo_r50",
    format_user=StringFormatter(slots=["USER: {{content}} ASSISTANT:"]),
    default_system=(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ),
    mm_plugin=LlavaMDPO_r50Plugin(image_token="<image>", video_token="<|video_pad|>"),
)

_register_template(
    name="llava_orm_r80",
    format_user=StringFormatter(slots=["USER: {{content}} ASSISTANT:"]),
    default_system=(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ),
    mm_plugin=LlavaORM_r80Plugin(image_token="<image>", video_token="<|video_pad|>"),
)

_register_template(
    name="qwen2_vl_new",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen"),
    default_system="You are a helpful assistant.",
    stop_words=["<|im_end|>"],
    mm_plugin=Qwen2vlNewPlugin(image_token="<|image_pad|>", video_token="<|video_pad|>")
)


def is_process_zero(training_args):
    return training_args.process_index == 0


# Deprecated
def generate_save_path(model_args, data_args, training_args, vtcl_args):
    from pathlib import Path
    model_name = model_args.model_name_or_path.rsplit('/', 1)[1].split('-')[0]
    if training_args.do_train:
        ds_name = '-' + '_'.join(data_args.dataset)
    else:
        ds_name = ''
    vtcl_strategy = vtcl_args.strategy
    if vtcl_args.mask_same_sequence:
        vtcl_suffix = "_mask"
    else:
        vtcl_suffix = "_nomask"
    lr = f'{training_args.learning_rate:g}'
    bs = f'{training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}'
    output_dir = f"saves/vtcl/{model_name}{ds_name}/{vtcl_strategy}{vtcl_suffix}_lr{lr}_b{bs}_Exp"
    idx = 1
    while True:
        if not Path(output_dir+str(idx)).exists():
            break
        idx += 1
    return output_dir + str(idx)


def get_suffix(p):
    idx = 1
    while Path(p + f"_Exp{idx}").exists():
        idx += 1
    return f"_Exp{idx}"


def as_dict(*args):
    total_args = {}
    for arg in args:
        total_args.update(dataclasses.asdict(arg))
    filtered_args = total_args.copy()
    for k, v in total_args.items():
        if type(v) not in [int, float, str, bool]:
            if type(v) is list:
                if v == [] or type(v[0]) not in [int, float, str, bool]:
                    del filtered_args[k]
            else:
                del filtered_args[k]
    return total_args, filtered_args


def extra_args_examination(data_args: DataArguments, vtcl_args: VTCLArguments, training_args: Seq2SeqTrainingArguments):

    if vtcl_args.load_double_dataset:
        if type(data_args.dataset) is str:
            assert data_args.dataset.endswith("single"), f"Training mode: difBatch, need customed dataset (now '{data_args.dataset}')"
    else:
        if type(data_args.dataset) is str:
            assert not data_args.dataset.endswith("single"), f"Training mode: {vtcl_args.vtcl_mode}, customed dataset not supported."
    assert vtcl_args.strategy != "mdpo" or not vtcl_args.use_negative_data, "No pairwise MDPO"


def run_mdpo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    vtcl_args: "VTCLArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    extra_args_examination(data_args, vtcl_args, training_args)
    if training_args.deepspeed:
        from deepspeed.comm import barrier
    else:
        from torch.distributed import barrier

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="rm", **tokenizer_module)
    # print(dataset_module["train_dataset"][0])
    # breakpoint()
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    if vtcl_args.test:  # for dubugging
        training_args.logging_dir = "saves/test"
        training_args.output_dir = "saves/test"
        vtcl_args.online_data_save_dir = "saves/test"
        training_args.report_to = 'none'
    elif training_args.do_train:
        if training_args.do_predict:
            raise KeyError("Not supported: Please run another program to test the trained model on test dataset")
        model_name = model_args.model_name_or_path.split('/')[-1]
        training_args.logging_dir = f"tensorboards/mdpo_{vtcl_args.strategy}/{model_name}{'_full' if finetuning_args.finetuning_type == 'full' else '' if finetuning_args.finetuning_type == 'lora' else '_auto'}"
        training_args.output_dir = f"saves/mdpo_{vtcl_args.strategy}/{model_name}{'_full' if finetuning_args.finetuning_type == 'full' else '' if finetuning_args.finetuning_type == 'lora' else '_auto'}"
        vtcl_args.online_data_save_dir = f"online_data/mdpo_{vtcl_args.strategy}/{model_name}{'_full' if finetuning_args.finetuning_type == 'full' else '' if finetuning_args.finetuning_type == 'lora' else '_auto'}"

        suffix = get_suffix(training_args.output_dir)
        training_args.logging_dir += suffix
        training_args.output_dir += suffix
        vtcl_args.online_data_save_dir += suffix
        if training_args.world_size > 1:
            barrier()
        if training_args.world_size == 1 or is_process_zero(training_args):
            print("logging dir:", training_args.logging_dir)
            Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
            Path(vtcl_args.online_data_save_dir).mkdir(parents=True, exist_ok=True)
            total_args, filtered_args = as_dict(model_args, data_args, training_args, finetuning_args, vtcl_args)
            torch.save(total_args, Path(training_args.output_dir) / "total_args.pt")
            with open(Path(training_args.output_dir) / "config.txt", "w") as f:
                json.dump(filtered_args, f, indent=4)
    elif training_args.do_predict:
        if model_args.adapter_name_or_path:
            training_args.output_dir = model_args.adapter_name_or_path[0]
        else:
            training_args.output_dir = model_args.model_name_or_path
    else:
        raise KeyError("'do_train' and 'do_predict' can't be False at the same time.")

    # if vtcl_args.strategy in "dpo":
    #     collator_cls = PairwiseDataCollatorWithPadding
    #     extra_config = {}
    # elif vtcl_args.strategy in ["mdpo", "vtcl"]:
    #     collator_cls = VTCLIterTrainCollatorWithPadding if vtcl_args.iter_training else MDPOSpecialCollatorWithPadding if vtcl_args.special_vtcl else MDPOFasterCollatorWithPadding if vtcl_args.faster_loader else MDPOPairwiseDataCollatorWithPadding
    #     extra_config = {"mask_same": vtcl_args.mask_same_sequence}
    # elif vtcl_args.strategy == "sft":
    #     collator_cls = SFTDifMMDataCollatorWithPadding
    #     extra_config = {"mask_same": vtcl_args.mask_same_sequence}
    # else:
    #     raise KeyError

    if vtcl_args.strategy.startswith("online"):
        collator_cls: MetaDifCollator = COLLATOR_MAPPING[vtcl_args.strategy.replace("online-", "")]
    else:
        collator_cls: MetaDifCollator = COLLATOR_MAPPING[vtcl_args.strategy]

    data_collator = collator_cls(
        template=template,
        model=model,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        use_negative_data=vtcl_args.use_negative_data,
        mask_same=vtcl_args.mask_same_sequence,
        **tokenizer_module,
    )

    # Create reference model
    if finetuning_args.use_ref_model:
        if finetuning_args.ref_model is None and (not training_args.do_train):  
            if model_args.adapter_name_or_path:
                ref_model = None  # use the base model as the reference model
            else:
                ref_model = model  # use the model itself
        else:
            ref_model = create_ref_model(model_args, finetuning_args)
            # raise NotImplementedError
    else:
        ref_model = None

    # Update arguments
    training_args.remove_unused_columns = False  # important for multimodal and pairwise dataset
    training_args.report_to = ["tensorboard"]
    # print(training_args.report_to)
    # br
    # Initialize our Trainer

    trainer_cls = GRPOTrainerWithBuffer if vtcl_args.strategy == "grpo" else CustomDPOTrainer if not vtcl_args.strategy.startswith("online") else OnlineCustomDPOTrainerWithBuffer if vtcl_args.experience_buffer else OnlineCustomDPOTrainer
    trainer = trainer_cls(
        model=model,
        ref_model=ref_model,
        args=training_args,
        finetuning_args=finetuning_args,
        vtcl_args=vtcl_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
        **tokenizer_module,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_model()
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "rewards/accuracies", "eval_rewards/accuracies"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        if id(model) == id(ref_model):  # unable to compute rewards if reference model is the model itself
            remove_keys = [key for key in metrics.keys() if "rewards" in key]
            for key in remove_keys:
                metrics.pop(key)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Testing
    if training_args.do_predict:
        metrics = trainer.evaluate(metric_key_prefix="test")
        if id(model) == id(ref_model):  # unable to compute rewards if reference model is the model itself
            remove_keys = [key for key in metrics.keys() if "rewards" in key]
            for key in remove_keys:
                metrics.pop(key)
        if vtcl_args.test_output_name is None:
            trainer.save_metrics("test", metrics)
        else:
            trainer.save_metrics(vtcl_args.test_output_name, metrics)

    # Create model card
    # only save when do_train = True
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
    return training_args.output_dir, vtcl_args.test_dataset_name


if __name__ == "__main__":
    from ..callbacks import LogCallback
    from ...hparams import get_train_args
    callbacks = []
    callbacks.append(LogCallback())
    model_args, data_args, training_args, finetuning_args, generating_args, vtcl_args = get_train_args()
    run_mdpo(model_args, data_args, training_args, finetuning_args, vtcl_args, callbacks)