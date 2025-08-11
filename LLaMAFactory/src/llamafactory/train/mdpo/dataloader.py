import difflib
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import DataCollatorForSeq2Seq

# from ...data import MultiModalDataCollatorForSeq2Seq
    
if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from ...data.template import Template
    from ...extras.constants import IGNORE_INDEX, IMAGE_PLACEHOLDER
############################################################
# 
# <image> should NEVER be masked !!!!!!!!!!!!!!!!!!
#
###############################################################
    

@dataclass
class SpecialMultiModalDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    r"""
    Data collator that supports VLMs.

    Features should contain input_ids, attention_mask, labels, and optionally contain images and videos.
    """

    template: Optional["Template"] = None
    processor: Optional["ProcessorMixin"] = None

    def __post_init__(self):
        if self.template is None:
            raise ValueError("Template is required for MultiModalDataCollator.")

    def __call__(self, features: Sequence[Dict[str, Any]], skip_mm_inputs=False) -> Dict[str, "torch.Tensor"]:
        batch_images, batch_videos, batch_imglens, batch_vidlens, batch_input_ids = [], [], [], [], []
        no_img_idx = []
        for idx, feature in enumerate(features):
            images = feature.pop("images", None) or []
            videos = feature.pop("videos", None) or []
            if images==[]:
                no_img_idx.append(idx)
            batch_images.extend(images)
            batch_videos.extend(videos)
            batch_imglens.append(len(images))
            batch_vidlens.append(len(videos))
            batch_input_ids.append(feature["input_ids"])

        if (
            self.processor is not None and sum(batch_imglens) == 0 and sum(batch_vidlens) == 0
        ):  # avoid process hanging in zero3/fsdp case
            fake_messages = [{"role": "user", "content": IMAGE_PLACEHOLDER}]
            fake_images = [Image.new("RGB", (64, 64), (255, 255, 255))]
            fake_messages = self.template.mm_plugin.process_messages(fake_messages, fake_images, [], self.processor)
            fake_input_ids = self.tokenizer.encode(fake_messages[0]["content"], add_special_tokens=False)
            fake_input_ids, _ = self.template.mm_plugin.process_token_ids(
                fake_input_ids, None, fake_images, [], self.tokenizer, self.processor
            )
            if self.tokenizer.padding_side == "right":
                features[0]["input_ids"] = features[0]["input_ids"] + fake_input_ids
                features[0]["attention_mask"] = features[0]["attention_mask"] + [0] * len(fake_input_ids)
                features[0]["labels"] = features[0]["labels"] + [IGNORE_INDEX] * len(fake_input_ids)
            else:
                features[0]["input_ids"] = fake_input_ids + features[0]["input_ids"]
                features[0]["attention_mask"] = [0] * len(fake_input_ids) + features[0]["attention_mask"]
                features[0]["labels"] = [IGNORE_INDEX] * len(fake_input_ids) + features[0]["labels"]

            batch_images = fake_images
            batch_imglens[0] = 1
            batch_input_ids[0] = features[0]["input_ids"]

        if skip_mm_inputs:
            features = super().__call__(features)
            features["pixel_values"] = torch.concat([i.to("cuda") for i in batch_images], dim=0)
            return features

        mm_inputs = self.template.mm_plugin.get_mm_inputs(
            batch_images, batch_videos, batch_imglens, batch_vidlens, batch_input_ids, self.processor
        )

        if "image_grid_thw" in mm_inputs:
            if no_img_idx != []:
                for idx in no_img_idx:
                    mm_inputs["image_grid_thw"].insert(idx, [])
            for i, (feat, grid_thw) in enumerate(zip(features, mm_inputs["image_grid_thw"])):
                feature = deepcopy(feat)
                loc = feature["input_ids"].index(self.IMAGE_TOKEN_ID)
                num = feature["input_ids"].count(self.IMAGE_TOKEN_ID)
                target_num = (grid_thw.prod() // (getattr(self.processor.image_processor, "merge_size") ** 2)).item()
                if target_num > num:
                    dif = target_num - num
                    features[i]["input_ids"][loc:loc] = [self.IMAGE_TOKEN_ID] * dif
                    features[i]["attention_mask"][loc:loc] = [1] * dif
                    features[i]["labels"][loc:loc] = [-100] * dif
                elif target_num < num:
                    features[i]["input_ids"] = feature["input_ids"][:loc+target_num] + feature["input_ids"][loc+num:]
                    features[i]["attention_mask"] = feature["attention_mask"][:loc+target_num] + feature["attention_mask"][loc+num:]
                    features[i]["labels"] = feature["labels"][:loc+target_num] + feature["labels"][loc+num:]
                else:
                    features[i]["input_ids"] = feature["input_ids"]
                    features[i]["attention_mask"] = feature["attention_mask"]
                    features[i]["labels"] = feature["labels"]

        if "token_type_ids" in mm_inputs:
            token_type_ids = mm_inputs.pop("token_type_ids")
            if no_img_idx != []:
                for idx in no_img_idx:
                    token_type_ids.insert(idx, None)
            for i, feature in enumerate(features):
                feature["token_type_ids"] = token_type_ids[i]
        features: Dict[str, "torch.Tensor"] = super().__call__(features)

        if self.model is not None and hasattr(self.model, "get_rope_index"):  # for qwen2vl mrope
            features["position_ids"], features["rope_deltas"] = self.model.get_rope_index(
                input_ids=features["input_ids"],
                image_grid_thw=mm_inputs.get("image_grid_thw", None),
                video_grid_thw=mm_inputs.get("video_grid_thw", None),
                attention_mask=features["attention_mask"],
            )

        if "cross_attention_mask" in mm_inputs:  # for mllama inputs when pad_to_multiple_of is enabled
            cross_attention_mask = mm_inputs.pop("cross_attention_mask")
            seq_len = features["input_ids"].size(1)
            orig_len = cross_attention_mask.size(1)
            mm_inputs["cross_attention_mask"] = F.pad(cross_attention_mask, (0, 0, 0, 0, 0, seq_len - orig_len))

        features.update(mm_inputs)
        if isinstance(features.get("pixel_values"), list):  # for pixtral inputs
            features = features.data  # use default_collate() instead of BatchEncoding.to()

        if "image_bound" in features:  # for minicpmv inputs
            bsz, seq_length = features["input_ids"].shape
            features["position_ids"] = torch.arange(seq_length).long().repeat(bsz, 1)
            return {"data": features, "input_ids": features["input_ids"], "labels": features["labels"]}

        return features


@dataclass
class MetaDifCollator(SpecialMultiModalDataCollatorForSeq2Seq):
    def __init__(self, mask_same=True, use_negative_data=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.IMAGE_TOKEN_ID = self.tokenizer.convert_tokens_to_ids("<image>")
        except Exception:
            self.IMAGE_TOKEN_ID = 151655


@dataclass
class SFTDifCollator(MetaDifCollator):
    def __init__(self, mask_same=True, use_negative_data=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_same = mask_same
        self.use_nega_data = use_negative_data  # the same batch

    def mask_same_sequence(self, batch):
        labels = batch["labels"]
        half_bs = labels.shape[0] // 2
        for i in range(0, half_bs):
            s1 = labels[i]
            s2 = labels[i+half_bs]
            idx_1 = s1 != self.label_pad_token_id
            idx_2 = s2 != self.label_pad_token_id
            slist1 = s1[idx_1].tolist()
            slist2 = s2[idx_2].tolist()
            diff = difflib.ndiff(list(map(str, slist1)), list(map(str, slist2)))

            OUTPUT1 = []
            OUTPUT2 = []
            for word in diff:
                if word.startswith('- '):
                    OUTPUT1.append(int(word[2:]))
                elif word.startswith('+ '):
                    OUTPUT2.append(int(word[2:]))
                elif word.startswith('  '):
                    OUTPUT1.append(self.label_pad_token_id)
                    OUTPUT2.append(self.label_pad_token_id)
            labels[i][idx_1] = torch.tensor(OUTPUT1, device=s1.device, dtype=s1.dtype)
        return batch

    def __call__(self, features: Sequence[Dict[str, Any]], skip_mm_inputs=False) -> Dict[str, "torch.Tensor"]:
        concatenated_features = []
        KEY = ("chosen", "rejected") if self.mask_same or self.use_nega_data else ["chosen"]
        for key in KEY:
            for feature in features:
                target_feature = {
                    "input_ids": feature["{}_input_ids".format(key)],
                    "attention_mask": feature["{}_attention_mask".format(key)],
                    "labels": feature["{}_labels".format(key)],
                    "images": feature["images"][0:1],
                    "videos": feature["videos"],
                }
                concatenated_features.append(target_feature)

        batch = super().__call__(concatenated_features, skip_mm_inputs)
        # for k, v in batch.items():
        #     print(k, v.shape)
        if self.mask_same:
            batch = self.mask_same_sequence(batch)
            if not self.use_nega_data:
                n_data = batch["labels"].shape[0] // 2
                for k in ["input_ids", "attention_mask", "labels", "rope_deltas", "image_grid_thw"]:
                # for k in batch.keys():
                    batch[k] = batch[k][:n_data]
                batch["position_ids"] = batch["position_ids"][:, :n_data]
                batch["pixel_values"] = batch["pixel_values"][:torch.prod(batch["image_grid_thw"], 1).sum()]
        return batch
    

@dataclass
class DPODifCollator(MetaDifCollator):
    r"""
    Data collator for pairwise data.
    """
    def __init__(self, mask_same=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_same = mask_same

    def mask_same_sequence(self, batch):
        labels = batch["labels"]
        half_bs = labels.shape[0] // 2
        for i in range(0, half_bs):
            s1 = labels[i]
            s2 = labels[i+half_bs]
            idx_1 = s1 != self.label_pad_token_id
            idx_2 = s2 != self.label_pad_token_id
            slist1 = s1[idx_1].tolist()
            slist2 = s2[idx_2].tolist()
            diff = difflib.ndiff(list(map(str, slist1)), list(map(str, slist2)))

            OUTPUT1 = []
            OUTPUT2 = []
            for word in diff:
                if word.startswith('- '):
                    OUTPUT1.append(int(word[2:]))
                elif word.startswith('+ '):
                    OUTPUT2.append(int(word[2:]))
                elif word.startswith('  '):
                    OUTPUT1.append(self.label_pad_token_id)
                    OUTPUT2.append(self.label_pad_token_id)
            labels[i][idx_1] = torch.tensor(OUTPUT1, device=s1.device, dtype=s1.dtype)
            labels[i+half_bs][idx_2] = torch.tensor(OUTPUT2, device=s2.device, dtype=s2.dtype)
        return batch

    def __call__(self, features: Sequence[Dict[str, Any]], skip_mm_inputs=False) -> Dict[str, "torch.Tensor"]:
        r"""
        Pads batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        concatenated_features = []
        key = "chosen"
        for feature in features:
            target_feature = {
                "input_ids": feature["{}_input_ids".format(key)],
                "attention_mask": feature["{}_attention_mask".format(key)],
                "labels": feature["{}_labels".format(key)],
                "images": feature["images"][0:1],
                "videos": feature["videos"],
            }
            concatenated_features.append(target_feature)

        key = "rejected"
        for feature in features:
            target_feature = {
                "input_ids": feature["{}_input_ids".format(key)],
                "attention_mask": feature["{}_attention_mask".format(key)],
                "labels": feature["{}_labels".format(key)],
                "images": feature["images"][0:1],
                "videos": feature["videos"],
            }
            concatenated_features.append(target_feature)

        batch = super().__call__(concatenated_features, skip_mm_inputs)
        if self.mask_same:
            return self.mask_same_sequence(batch)
        return batch
    

@dataclass
class MDPODifCollator(MetaDifCollator):
    r"""
    Data collator for pairwise data.
    """
    def __init__(self, mask_same=True, use_negative_data=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_same = mask_same
        self.use_nega_data = use_negative_data

    def mask_same_sequence(self, batch):
        labels = batch["labels"]
        if self.use_nega_data:
            half_bs = labels.shape[0] // 2
            for i in range(0, half_bs, 2):
                s1 = labels[i]
                s2 = labels[i+1]
                idx_1 = s1 != self.label_pad_token_id
                idx_2 = s2 != self.label_pad_token_id
                slist1 = s1[idx_1].tolist()
                slist2 = s2[idx_2].tolist()
                diff = difflib.ndiff(list(map(str, slist1)), list(map(str, slist2)))

                OUTPUT1 = []
                OUTPUT2 = []
                for word in diff:
                    if word.startswith('- '):
                        OUTPUT1.append(int(word[2:]))
                    elif word.startswith('+ '):
                        OUTPUT2.append(int(word[2:]))
                    elif word.startswith('  '):
                        OUTPUT1.append(self.label_pad_token_id)
                        OUTPUT2.append(self.label_pad_token_id)
                labels[i][idx_1] = torch.tensor(OUTPUT1, device=s1.device, dtype=s1.dtype)
                labels[i+1][idx_2] = torch.tensor(OUTPUT2, device=s2.device, dtype=s2.dtype)
                labels[i+half_bs][idx_2] = torch.tensor(OUTPUT2, device=s2.device, dtype=s2.dtype)
                labels[i+half_bs+1][idx_1] = torch.tensor(OUTPUT1, device=s1.device, dtype=s1.dtype)
        else:
            bs = labels.shape[0] // 3
            for i in range(bs):
                s1 = labels[i]
                s2 = labels[bs+2*i]
                idx_1 = s1 != self.label_pad_token_id
                idx_2 = s2 != self.label_pad_token_id
                slist1 = s1[idx_1].tolist()
                slist2 = s2[idx_2].tolist()
                diff = difflib.ndiff(list(map(str, slist1)), list(map(str, slist2)))

                OUTPUT1 = []
                OUTPUT2 = []
                for word in diff:
                    if word.startswith('- '):
                        OUTPUT1.append(int(word[2:]))
                    elif word.startswith('+ '):
                        OUTPUT2.append(int(word[2:]))
                    elif word.startswith('  '):
                        OUTPUT1.append(self.label_pad_token_id)
                        OUTPUT2.append(self.label_pad_token_id)
                labels[i][idx_1] = torch.tensor(OUTPUT1, device=s1.device, dtype=s1.dtype)
                labels[bs+2*i][idx_2] = torch.tensor(OUTPUT2, device=s2.device, dtype=s2.dtype)
                labels[bs+2*i+1][idx_1] = torch.tensor(OUTPUT1, device=s1.device, dtype=s1.dtype)
        return batch

    def __call__(self, features: Sequence[Dict[str, Any]], skip_mm_inputs=False) -> Dict[str, "torch.Tensor"]:
        r"""
        Pads batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        concatenated_features = []
        for feature in features:
            # print(feature["images"])
            target_feature = {
                "input_ids": feature["chosen_input_ids"],
                "attention_mask": feature["chosen_attention_mask"],
                "labels": feature["chosen_labels"],
                "images": feature["images"][0:1],
                "videos": feature["videos"],
            }
            concatenated_features.append(target_feature)
            if self.use_nega_data:
                target_feature = {
                    "input_ids": feature["rejected_input_ids"],
                    "attention_mask": feature["rejected_attention_mask"],
                    "labels": feature["rejected_labels"],
                    "images": feature["images"][1:2],
                    "videos": feature["videos"],
                }
                concatenated_features.append(target_feature)
        for feature in features:
            target_feature = {
                "input_ids": feature["rejected_input_ids"],
                "attention_mask": feature["rejected_attention_mask"],
                "labels": feature["rejected_labels"],
                "images": feature["images"][0:1],
                "videos": feature["videos"],
            }
            concatenated_features.append(target_feature)
            target_feature = {
                "input_ids": feature["chosen_input_ids"],
                "attention_mask": feature["chosen_attention_mask"],
                "labels": feature["chosen_labels"],
                "images": feature["images"][1:2],
                "videos": feature["videos"],
            }
            concatenated_features.append(target_feature)
        batch = super().__call__(concatenated_features, skip_mm_inputs)
        if self.mask_same:
            return self.mask_same_sequence(batch)
        return batch


@dataclass
class oriMDPODifCollator(MetaDifCollator):
    r"""
    Data collator for pairwise data.
    """
    def __init__(self, mask_same=True, use_negative_data=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_same = mask_same
        self.use_nega_data = use_negative_data

    def mask_same_sequence(self, batch):
        labels = batch["labels"]
        if self.use_nega_data:
            half_bs = labels.shape[0] // 2
            for i in range(0, half_bs, 2):
                s1 = labels[i]
                s2 = labels[i+1]
                idx_1 = s1 != self.label_pad_token_id
                idx_2 = s2 != self.label_pad_token_id
                slist1 = s1[idx_1].tolist()
                slist2 = s2[idx_2].tolist()
                diff = difflib.ndiff(list(map(str, slist1)), list(map(str, slist2)))

                OUTPUT1 = []
                OUTPUT2 = []
                for word in diff:
                    if word.startswith('- '):
                        OUTPUT1.append(int(word[2:]))
                    elif word.startswith('+ '):
                        OUTPUT2.append(int(word[2:]))
                    elif word.startswith('  '):
                        OUTPUT1.append(self.label_pad_token_id)
                        OUTPUT2.append(self.label_pad_token_id)
                labels[i][idx_1] = torch.tensor(OUTPUT1, device=s1.device, dtype=s1.dtype)
                labels[i+1][idx_2] = torch.tensor(OUTPUT2, device=s2.device, dtype=s2.dtype)
                labels[i+half_bs][idx_2] = torch.tensor(OUTPUT2, device=s2.device, dtype=s2.dtype)
                labels[i+half_bs+1][idx_1] = torch.tensor(OUTPUT1, device=s1.device, dtype=s1.dtype)
        else:
            bs = labels.shape[0] // 3
            for i in range(bs):
                s1 = labels[i]
                s2 = labels[bs+2*i]
                idx_1 = s1 != self.label_pad_token_id
                idx_2 = s2 != self.label_pad_token_id
                slist1 = s1[idx_1].tolist()
                slist2 = s2[idx_2].tolist()
                diff = difflib.ndiff(list(map(str, slist1)), list(map(str, slist2)))

                OUTPUT1 = []
                OUTPUT2 = []
                for word in diff:
                    if word.startswith('- '):
                        OUTPUT1.append(int(word[2:]))
                    elif word.startswith('+ '):
                        OUTPUT2.append(int(word[2:]))
                    elif word.startswith('  '):
                        OUTPUT1.append(self.label_pad_token_id)
                        OUTPUT2.append(self.label_pad_token_id)
                labels[i][idx_1] = torch.tensor(OUTPUT1, device=s1.device, dtype=s1.dtype)
                labels[bs+2*i][idx_2] = torch.tensor(OUTPUT2, device=s2.device, dtype=s2.dtype)
                labels[bs+2*i+1][idx_1] = torch.tensor(OUTPUT1, device=s1.device, dtype=s1.dtype)
        return batch

    def __call__(self, features: Sequence[Dict[str, Any]], skip_mm_inputs=False) -> Dict[str, "torch.Tensor"]:
        r"""
        Pads batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        concatenated_features = []
        for feature in features:
            # print(feature["images"])
            target_feature = {
                "input_ids": feature["chosen_input_ids"],
                "attention_mask": feature["chosen_attention_mask"],
                "labels": feature["chosen_labels"],
                "images": feature["images"][0:1],
                "videos": feature["videos"],
            }
            concatenated_features.append(target_feature)
            if self.use_nega_data:
                target_feature = {
                    "input_ids": feature["rejected_input_ids"],
                    "attention_mask": feature["rejected_attention_mask"],
                    "labels": feature["rejected_labels"],
                    "images": feature["images"][1:2],
                    "videos": feature["videos"],
                }
                concatenated_features.append(target_feature)
        for feature in features:
            target_feature = {
                "input_ids": feature["rejected_input_ids"],
                "attention_mask": feature["rejected_attention_mask"],
                "labels": feature["rejected_labels"],
                "images": feature["images"][0:1],
                "videos": feature["videos"],
            }
            concatenated_features.append(target_feature)
            target_feature = {
                "input_ids": feature["chosen_input_ids"],
                "attention_mask": feature["chosen_attention_mask"],
                "labels": feature["chosen_labels"],
                "images": feature["images"][0:1],
                "videos": feature["videos"],
            }
            concatenated_features.append(target_feature)
        batch = super().__call__(concatenated_features, skip_mm_inputs)
        if self.mask_same:
            return self.mask_same_sequence(batch)
        return batch
    

@dataclass
class NoRefSVCODifCollator(MetaDifCollator):
    r"""
    Data collator for pairwise data.
    """
    def __init__(self, mask_same=True, use_negative_data=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_same = mask_same
        self.use_nega_data = use_negative_data

    def mask_same_sequence(self, batch):
        raise NotImplementedError

    def __call__(self, features: Sequence[Dict[str, Any]], skip_mm_inputs=False) -> Dict[str, "torch.Tensor"]:
        r"""
        Pads batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        concatenated_features = []
        for feature in features:
            # print(feature["images"])
            concatenated_features.append({
                "input_ids": feature["chosen_input_ids"],
                "attention_mask": feature["chosen_attention_mask"],
                "labels": feature["chosen_labels"],
                "images": feature["images"][0:1],
                "videos": feature["videos"],
            })
            if self.use_nega_data:
                target_feature = {
                    "input_ids": feature["chosen_input_ids"],
                    "attention_mask": feature["chosen_attention_mask"],
                    "labels": feature["chosen_labels"],
                    "images": feature["images"][1:2],
                    "videos": feature["videos"],
                }
                concatenated_features.append(target_feature)
            
        for feature in features:
            if self.use_nega_data:
                target_feature = {
                    "input_ids": feature["rejected_input_ids"],
                    "attention_mask": feature["rejected_attention_mask"],
                    "labels": feature["rejected_labels"],
                    "images": feature["images"][0:1],
                    "videos": feature["videos"],
                }
                concatenated_features.append(target_feature)
            concatenated_features.append({
                "input_ids": feature["rejected_input_ids"],
                "attention_mask": feature["rejected_attention_mask"],
                "labels": feature["rejected_labels"],
                "images": feature["images"][1:2],
                "videos": feature["videos"],
            })
        batch = super().__call__(concatenated_features, skip_mm_inputs)
        if self.mask_same:
            return self.mask_same_sequence(batch)
        return batch


@dataclass
class SVCODifCollator(MetaDifCollator):
    r"""
    Data collator for pairwise data.
    """
    def __init__(self, mask_same=True, use_negative_data=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_same = mask_same
        self.use_nega_data = use_negative_data

    def mask_same_sequence(self, batch):
        raise NotImplementedError

    def __call__(self, features: Sequence[Dict[str, Any]], skip_mm_inputs=False) -> Dict[str, "torch.Tensor"]:
        r"""
        Pads batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        concatenated_features = []
        for feature in features:
            # print(feature["images"])
            concatenated_features.append({
                "input_ids": feature["chosen_input_ids"],
                "attention_mask": feature["chosen_attention_mask"],
                "labels": feature["chosen_labels"],
                "images": feature["images"][0:1],
                "videos": feature["videos"],
            })
            if self.use_nega_data:
                concatenated_features.append({
                    "input_ids": feature["rejected_input_ids"],
                    "attention_mask": feature["rejected_attention_mask"],
                    "labels": feature["rejected_labels"],
                    "images": feature["images"][1:2],
                    "videos": feature["videos"],
                })
        for feature in features:
            target_feature = {
                "input_ids": feature["chosen_input_ids"],
                "attention_mask": feature["chosen_attention_mask"],
                "labels": feature["chosen_labels"],
                "images": feature["images"][1:2],
                "videos": feature["videos"],
            }
            concatenated_features.append(target_feature)
            if self.use_nega_data:
                target_feature = {
                    "input_ids": feature["rejected_input_ids"],
                    "attention_mask": feature["rejected_attention_mask"],
                    "labels": feature["rejected_labels"],
                    "images": feature["images"][0:1],
                    "videos": feature["videos"],
                }
                concatenated_features.append(target_feature)

        for feature in features:
            concatenated_features.append({
                "input_ids": feature["chosen_input_ids"],
                "attention_mask": feature["chosen_attention_mask"],
                "labels": feature["chosen_labels"],
                "images": [Image.new("RGB", (64, 64), (255, 255, 255))],
                "videos": feature["videos"]
            })
            if self.use_nega_data:
                concatenated_features.append({
                    "input_ids": feature["rejected_input_ids"],
                    "attention_mask": feature["rejected_attention_mask"],
                    "labels": feature["rejected_labels"],
                    "images": [Image.new("RGB", (64, 64), (255, 255, 255))],
                    "videos": feature["videos"]
                })
        batch = super().__call__(concatenated_features, skip_mm_inputs)
        if self.mask_same:
            return self.mask_same_sequence(batch)
        return batch
    

@dataclass
class SVCOTextRefDifCollator(MetaDifCollator):
    r"""
    Data collator for pairwise data.
    """
    def __init__(self, mask_same=True, use_negative_data=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_same = mask_same
        self.use_nega_data = use_negative_data

    def mask_same_sequence(self, batch):
        raise NotImplementedError

    def __call__(self, features: Sequence[Dict[str, Any]], skip_mm_inputs=False) -> Dict[str, "torch.Tensor"]:
        r"""
        Pads batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        concatenated_features = []
        for feature in features:
            # print(feature["images"])
            concatenated_features.append({
                "input_ids": feature["chosen_input_ids"],
                "attention_mask": feature["chosen_attention_mask"],
                "labels": feature["chosen_labels"],
                "images": feature["images"][0:1],
                "videos": feature["videos"],
            })
            if self.use_nega_data:
                concatenated_features.append({
                    "input_ids": feature["rejected_input_ids"],
                    "attention_mask": feature["rejected_attention_mask"],
                    "labels": feature["rejected_labels"],
                    "images": feature["images"][1:2],
                    "videos": feature["videos"],
                })
        for feature in features:
            target_feature = {
                "input_ids": feature["chosen_input_ids"],
                "attention_mask": feature["chosen_attention_mask"],
                "labels": feature["chosen_labels"],
                "images": feature["images"][1:2],
                "videos": feature["videos"],
            }
            concatenated_features.append(target_feature)
            if self.use_nega_data:
                target_feature = {
                    "input_ids": feature["rejected_input_ids"],
                    "attention_mask": feature["rejected_attention_mask"],
                    "labels": feature["rejected_labels"],
                    "images": feature["images"][0:1],
                    "videos": feature["videos"],
                }
                concatenated_features.append(target_feature)
        for feature in features:
            loc = feature["chosen_input_ids"].index(self.IMAGE_TOKEN_ID)
            num = feature["chosen_input_ids"].count(self.IMAGE_TOKEN_ID)
            concatenated_features.append({
                "input_ids": feature["chosen_input_ids"][:loc] + feature["chosen_input_ids"][loc+num:],
                "attention_mask": feature["chosen_attention_mask"][:loc] + feature["chosen_attention_mask"][loc+num:],
                "labels": feature["chosen_labels"][:loc] + feature["chosen_labels"][loc+num:],
                "images": [],
                "videos": feature["videos"]
            })
            if self.use_nega_data:
                loc = feature["rejected_input_ids"].index(self.IMAGE_TOKEN_ID)
                num = feature["rejected_input_ids"].count(self.IMAGE_TOKEN_ID)
                concatenated_features.append({
                    "input_ids": feature["rejected_input_ids"][:loc] + feature["rejected_input_ids"][loc+num:],
                    "attention_mask": feature["rejected_attention_mask"][:loc] + feature["rejected_attention_mask"][loc+num:],
                    "labels": feature["rejected_labels"][:loc] + feature["rejected_labels"][loc+num:],
                    "images": [],
                    "videos": feature["videos"]
                })
        batch = super().__call__(concatenated_features, skip_mm_inputs)
        if self.mask_same:
            return self.mask_same_sequence(batch)
        return batch
    

@dataclass
class ORMDifCollator(MetaDifCollator):
    r"""
    Data collator for pairwise data.
    """
    def __init__(self, mask_same=True, use_negative_data=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_same = mask_same
        self.use_nega_data = use_negative_data

    def mask_same_sequence(self, batch):
        raise NotImplementedError

    def __call__(self, features: Sequence[Dict[str, Any]], skip_mm_inputs=False) -> Dict[str, "torch.Tensor"]:
        r"""
        Pads batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        concatenated_features = []
        for feature in features:
            # print(feature["images"])
            concatenated_features.append({
                "input_ids": feature["chosen_input_ids"],
                "attention_mask": feature["chosen_attention_mask"],
                "labels": feature["chosen_labels"],
                "images": feature["images"][0:1],
                "videos": feature["videos"],
            })
            if self.use_nega_data:
                concatenated_features.append({
                    "input_ids": feature["rejected_input_ids"],
                    "attention_mask": feature["rejected_attention_mask"],
                    "labels": feature["rejected_labels"],
                    "images": feature["images"][1:2],
                    "videos": feature["videos"],
                })
        for feature in features:
            target_feature = {
                "input_ids": feature["chosen_input_ids"],
                "attention_mask": feature["chosen_attention_mask"],
                "labels": feature["chosen_labels"],
                "images": feature["images"][1:2],
                "videos": feature["videos"],
            }
            concatenated_features.append(target_feature)
            if self.use_nega_data:
                target_feature = {
                    "input_ids": feature["rejected_input_ids"],
                    "attention_mask": feature["rejected_attention_mask"],
                    "labels": feature["rejected_labels"],
                    "images": feature["images"][0:1],
                    "videos": feature["videos"],
                }
                concatenated_features.append(target_feature)
        for feature in features:
            target_feature = {
                "input_ids": feature["rejected_input_ids"],
                "attention_mask": feature["rejected_attention_mask"],
                "labels": feature["rejected_labels"],
                "images": feature["images"][0:1],
                "videos": feature["videos"],
            }
            concatenated_features.append(target_feature)
            if self.use_nega_data:
                target_feature = {
                    "input_ids": feature["chosen_input_ids"],
                    "attention_mask": feature["chosen_attention_mask"],
                    "labels": feature["chosen_labels"],
                    "images": feature["images"][1:2],
                    "videos": feature["videos"],
                }
                concatenated_features.append(target_feature)
        for feature in features:
            loc = feature["chosen_input_ids"].index(self.IMAGE_TOKEN_ID)
            num = feature["chosen_input_ids"].count(self.IMAGE_TOKEN_ID)
            concatenated_features.append({
                "input_ids": feature["chosen_input_ids"][:loc] + feature["chosen_input_ids"][loc+num:],
                "attention_mask": feature["chosen_attention_mask"][:loc] + feature["chosen_attention_mask"][loc+num:],
                "labels": feature["chosen_labels"][:loc] + feature["chosen_labels"][loc+num:],
                "images": [],
                "videos": feature["videos"]
            })
            if self.use_nega_data:
                loc = feature["rejected_input_ids"].index(self.IMAGE_TOKEN_ID)
                num = feature["rejected_input_ids"].count(self.IMAGE_TOKEN_ID)
                concatenated_features.append({
                    "input_ids": feature["rejected_input_ids"][:loc] + feature["rejected_input_ids"][loc+num:],
                    "attention_mask": feature["rejected_attention_mask"][:loc] + feature["rejected_attention_mask"][loc+num:],
                    "labels": feature["rejected_labels"][:loc] + feature["rejected_labels"][loc+num:],
                    "images": [],
                    "videos": feature["videos"]
                })
        batch = super().__call__(concatenated_features, skip_mm_inputs)
        if self.mask_same:
            return self.mask_same_sequence(batch)
        return batch