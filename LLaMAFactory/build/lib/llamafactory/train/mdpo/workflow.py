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

"""
cd ~/LLaMA-Factory-new; conda activate llamafactorynew
pip install .
llamafactory-cli train --model_name_or_path "/root/model/Qwen2-VL-7B-Instruct" --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset POVID --dataset_dir "/root/LLaMA-Factory/data" --template qwen2_vl_new --cutoff_len 2048 --max_samples 1000 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/qwen2_vl-7b/lora/dpo" --logging_steps 1 --save_steps 500 --plot_loss --overwrite_output_dir --per_device_train_batch_size 1 --gradient_accumulation_steps 2 --learning_rate 5e-6 --num_train_epochs 3 --lr_scheduler_type cosine --warmup_ratio 0.1 --bf16 --ddp_timeout 180000000 --val_size 0.1 --per_device_eval_batch_size 1 --eval_strategy steps --eval_steps 500 --test
CUDA_VISIBLE_DEVICES=0 python -m src.llamafactory.train.mdpo.workflow --model_name_or_path "/root/model/Qwen2-VL-7B-Instruct" --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset POVID --dataset_dir "/root/LLaMA-Factory/data" --template qwen2_vl_new --cutoff_len 2048 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/qwen2_vl-7b/lora/dpo" --logging_steps 10 --save_steps 0.05 --plot_loss --overwrite_output_dir --per_device_train_batch_size 2 --gradient_accumulation_steps 16 --learning_rate 2e-5 --num_train_epochs 3 --eval_on_start --lr_scheduler_type cosine --warmup_ratio 0.1 --bf16 --ddp_timeout 180000000 --val_size 0.1 --per_device_eval_batch_size 4 --eval_strategy steps --eval_steps 0.05 --mask_same_sequence --lora_rank 256 --vtcl_mode same --test

SFT
CUDA_VISIBLE_DEVICES=0 python -m src.llamafactory.train.mdpo.workflow --model_name_or_path "/root/model/Qwen2-VL-7B-Instruct" --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset POVID --dataset_dir "/root/LLaMA-Factory/data" --template qwen2_vl_new --cutoff_len 2048 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/qwen2_vl-7b/lora/dpo" --logging_steps 10 --save_steps 0.05 --plot_loss --overwrite_output_dir --per_device_train_batch_size 2 --gradient_accumulation_steps 16 --learning_rate 2e-5 --num_train_epochs 3 --eval_on_start --lr_scheduler_type cosine --warmup_ratio 0.1 --bf16 --ddp_timeout 180000000 --val_size 0.1 --per_device_eval_batch_size 4 --eval_strategy steps --eval_steps 0.05 --mask_same_sequence --lora_rank 256 --vtcl_mode single
CUDA_VISIBLE_DEVICES=1 python -m src.llamafactory.train.mdpo.workflow --model_name_or_path "/root/model/Qwen2-VL-7B-Instruct" --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset POVID --dataset_dir "/root/LLaMA-Factory/data" --template qwen2_vl_new --cutoff_len 2048 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/qwen2_vl-7b/lora/dpo" --logging_steps 10 --save_steps 0.05 --plot_loss --overwrite_output_dir --per_device_train_batch_size 2 --gradient_accumulation_steps 16 --learning_rate 1e-5 --num_train_epochs 3 --eval_on_start --lr_scheduler_type cosine --warmup_ratio 0.1 --bf16 --ddp_timeout 180000000 --val_size 0.1 --per_device_eval_batch_size 4 --eval_strategy steps --eval_steps 0.05 --mask_same_sequence --lora_rank 256 --vtcl_mode single
CUDA_VISIBLE_DEVICES=2 python -m src.llamafactory.train.mdpo.workflow --model_name_or_path "/root/model/Qwen2-VL-7B-Instruct" --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset POVID --dataset_dir "/root/LLaMA-Factory/data" --template qwen2_vl_new --cutoff_len 2048 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/qwen2_vl-7b/lora/dpo" --logging_steps 10 --save_steps 0.05 --plot_loss --overwrite_output_dir --per_device_train_batch_size 2 --gradient_accumulation_steps 16 --learning_rate 3e-5 --num_train_epochs 3 --eval_on_start --lr_scheduler_type cosine --warmup_ratio 0.1 --bf16 --ddp_timeout 180000000 --val_size 0.1 --per_device_eval_batch_size 4 --eval_strategy steps --eval_steps 0.05 --mask_same_sequence --lora_rank 256 --vtcl_mode single
CUDA_VISIBLE_DEVICES=3 python -m src.llamafactory.train.mdpo.workflow --model_name_or_path "/root/model/Qwen2-VL-7B-Instruct" --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset POVID --dataset_dir "/root/LLaMA-Factory/data" --template qwen2_vl_new --cutoff_len 2048 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/qwen2_vl-7b/lora/dpo" --logging_steps 10 --save_steps 0.05 --plot_loss --overwrite_output_dir --per_device_train_batch_size 2 --gradient_accumulation_steps 16 --learning_rate 4e-5 --num_train_epochs 3 --eval_on_start --lr_scheduler_type cosine --warmup_ratio 0.1 --bf16 --ddp_timeout 180000000 --val_size 0.1 --per_device_eval_batch_size 4 --eval_strategy steps --eval_steps 0.05 --mask_same_sequence --lora_rank 256 --vtcl_mode single
CUDA_VISIBLE_DEVICES=4 python -m src.llamafactory.train.mdpo.workflow --model_name_or_path "/root/model/Qwen2-VL-7B-Instruct" --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset POVID --dataset_dir "/root/LLaMA-Factory/data" --template qwen2_vl_new --cutoff_len 2048 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/qwen2_vl-7b/lora/dpo" --logging_steps 10 --save_steps 0.05 --plot_loss --overwrite_output_dir --per_device_train_batch_size 2 --gradient_accumulation_steps 16 --learning_rate 2e-5 --num_train_epochs 3 --eval_on_start --lr_scheduler_type cosine --warmup_ratio 0.1 --bf16 --ddp_timeout 180000000 --val_size 0.1 --per_device_eval_batch_size 4 --eval_strategy steps --eval_steps 0.05 --lora_rank 256 --vtcl_mode single
CUDA_VISIBLE_DEVICES=5 python -m src.llamafactory.train.mdpo.workflow --model_name_or_path "/root/model/Qwen2-VL-7B-Instruct" --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset POVID --dataset_dir "/root/LLaMA-Factory/data" --template qwen2_vl_new --cutoff_len 2048 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/qwen2_vl-7b/lora/dpo" --logging_steps 10 --save_steps 0.05 --plot_loss --overwrite_output_dir --per_device_train_batch_size 2 --gradient_accumulation_steps 16 --learning_rate 1e-5 --num_train_epochs 3 --eval_on_start --lr_scheduler_type cosine --warmup_ratio 0.1 --bf16 --ddp_timeout 180000000 --val_size 0.1 --per_device_eval_batch_size 4 --eval_strategy steps --eval_steps 0.05 --lora_rank 256 --vtcl_mode single
CUDA_VISIBLE_DEVICES=6 python -m src.llamafactory.train.mdpo.workflow --model_name_or_path "/root/model/Qwen2-VL-7B-Instruct" --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset POVID --dataset_dir "/root/LLaMA-Factory/data" --template qwen2_vl_new --cutoff_len 2048 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/qwen2_vl-7b/lora/dpo" --logging_steps 10 --save_steps 0.05 --plot_loss --overwrite_output_dir --per_device_train_batch_size 2 --gradient_accumulation_steps 16 --learning_rate 3e-5 --num_train_epochs 3 --eval_on_start --lr_scheduler_type cosine --warmup_ratio 0.1 --bf16 --ddp_timeout 180000000 --val_size 0.1 --per_device_eval_batch_size 4 --eval_strategy steps --eval_steps 0.05 --lora_rank 256 --vtcl_mode single
CUDA_VISIBLE_DEVICES=7 python -m src.llamafactory.train.mdpo.workflow --model_name_or_path "/root/model/Qwen2-VL-7B-Instruct" --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset POVID --dataset_dir "/root/LLaMA-Factory/data" --template qwen2_vl_new --cutoff_len 2048 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/qwen2_vl-7b/lora/dpo" --logging_steps 10 --save_steps 0.05 --plot_loss --overwrite_output_dir --per_device_train_batch_size 2 --gradient_accumulation_steps 16 --learning_rate 4e-5 --num_train_epochs 3 --eval_on_start --lr_scheduler_type cosine --warmup_ratio 0.1 --bf16 --ddp_timeout 180000000 --val_size 0.1 --per_device_eval_batch_size 4 --eval_strategy steps --eval_steps 0.05 --lora_rank 256 --vtcl_mode single

test
CUDA_VISIBLE_DEVICES=0 python -m src.llamafactory.train.mdpo.workflow --model_name_or_path "/root/model/Qwen2-VL-7B-Instruct" --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset POVID --dataset_dir "/root/LLaMA-Factory/data" --template qwen2_vl_new --cutoff_len 2048 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/qwen2_vl-7b/lora/dpo" --logging_steps 10 --save_steps 0.05 --plot_loss --overwrite_output_dir --per_device_train_batch_size 1 --gradient_accumulation_steps 32 --learning_rate 4e-5 --num_train_epochs 3 --eval_on_start --lr_scheduler_type cosine --warmup_ratio 0.1 --bf16 --ddp_timeout 180000000 --val_size 0.1 --per_device_eval_batch_size 2 --eval_strategy steps --eval_steps 0.05 --lora_rank 256 --vtcl_mode single --strategy dpo

DPO and DPO-single
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train --model_name_or_path "/root/model/Qwen2-VL-7B-Instruct" --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset POVID --dataset_dir "/root/LLaMA-Factory/data" --template qwen2_vl_new --cutoff_len 2048 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/qwen2_vl-7b/lora/dpo" --logging_steps 10 --save_steps 0.05 --plot_loss --overwrite_output_dir --per_device_train_batch_size 1 --gradient_accumulation_steps 32 --learning_rate 4e-5 --num_train_epochs 3 --eval_on_start --lr_scheduler_type cosine --warmup_ratio 0.1 --bf16 --ddp_timeout 180000000 --val_size 0.1 --per_device_eval_batch_size 2 --eval_strategy steps --eval_steps 0.05 --lora_rank 256 --vtcl_mode single --strategy dpo
CUDA_VISIBLE_DEVICES=2,3 llamafactory-cli train --model_name_or_path "/root/model/Qwen2-VL-7B-Instruct" --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset POVID --dataset_dir "/root/LLaMA-Factory/data" --template qwen2_vl_new --cutoff_len 2048 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/qwen2_vl-7b/lora/dpo" --logging_steps 10 --save_steps 0.05 --plot_loss --overwrite_output_dir --per_device_train_batch_size 1 --gradient_accumulation_steps 32 --learning_rate 2e-5 --num_train_epochs 3 --eval_on_start --lr_scheduler_type cosine --warmup_ratio 0.1 --bf16 --ddp_timeout 180000000 --val_size 0.1 --per_device_eval_batch_size 2 --eval_strategy steps --eval_steps 0.05 --lora_rank 256 --vtcl_mode single --strategy dpo
CUDA_VISIBLE_DEVICES=4,5 llamafactory-cli train --model_name_or_path "/root/model/Qwen2-VL-7B-Instruct" --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset POVID_single --dataset_dir "/root/LLaMA-Factory/data" --template qwen2_vl_new --cutoff_len 2048 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/qwen2_vl-7b/lora/dpo" --logging_steps 10 --save_steps 0.05 --plot_loss --overwrite_output_dir --per_device_train_batch_size 1 --gradient_accumulation_steps 32 --learning_rate 4e-5 --num_train_epochs 3 --eval_on_start --lr_scheduler_type cosine --warmup_ratio 0.1 --bf16 --ddp_timeout 180000000 --val_size 0.1 --per_device_eval_batch_size 2 --eval_strategy steps --eval_steps 0.05 --lora_rank 256 --vtcl_mode difBatch --strategy dpo
CUDA_VISIBLE_DEVICES=6,7 llamafactory-cli train --model_name_or_path "/root/model/Qwen2-VL-7B-Instruct" --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset POVID_single --dataset_dir "/root/LLaMA-Factory/data" --template qwen2_vl_new --cutoff_len 2048 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/qwen2_vl-7b/lora/dpo" --logging_steps 10 --save_steps 0.05 --plot_loss --overwrite_output_dir --per_device_train_batch_size 1 --gradient_accumulation_steps 32 --learning_rate 2e-5 --num_train_epochs 3 --eval_on_start --lr_scheduler_type cosine --warmup_ratio 0.1 --bf16 --ddp_timeout 180000000 --val_size 0.1 --per_device_eval_batch_size 2 --eval_strategy steps --eval_steps 0.05 --lora_rank 256 --vtcl_mode difBatch --strategy dpo

SFT-difBatch (mask and not mask)
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train --model_name_or_path "/root/model/Qwen2-VL-7B-Instruct" --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset POVID_single --dataset_dir "/root/LLaMA-Factory/data" --template qwen2_vl_new --cutoff_len 2048 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/qwen2_vl-7b/lora/dpo" --logging_steps 10 --save_steps 0.05 --plot_loss --overwrite_output_dir --per_device_train_batch_size 2 --gradient_accumulation_steps 16 --learning_rate 4e-5 --num_train_epochs 3 --eval_on_start --lr_scheduler_type cosine --warmup_ratio 0.1 --bf16 --ddp_timeout 180000000 --val_size 0.1 --per_device_eval_batch_size 4 --eval_strategy steps --eval_steps 0.05 --lora_rank 256 --vtcl_mode difBatch
CUDA_VISIBLE_DEVICES=2,3 llamafactory-cli train --model_name_or_path "/root/model/Qwen2-VL-7B-Instruct" --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset POVID_single --dataset_dir "/root/LLaMA-Factory/data" --template qwen2_vl_new --cutoff_len 2048 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/qwen2_vl-7b/lora/dpo" --logging_steps 10 --save_steps 0.05 --plot_loss --overwrite_output_dir --per_device_train_batch_size 2 --gradient_accumulation_steps 16 --learning_rate 2e-5 --num_train_epochs 3 --eval_on_start --lr_scheduler_type cosine --warmup_ratio 0.1 --bf16 --ddp_timeout 180000000 --val_size 0.1 --per_device_eval_batch_size 4 --eval_strategy steps --eval_steps 0.05 --lora_rank 256 --vtcl_mode difBatch
# WRONG learning rate!!!! CUDA_VISIBLE_DEVICES=4,5 llamafactory-cli train --model_name_or_path "/root/model/Qwen2-VL-7B-Instruct" --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset POVID_single --dataset_dir "/root/LLaMA-Factory/data" --template qwen2_vl_new --cutoff_len 2048 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/qwen2_vl-7b/lora/dpo" --logging_steps 10 --save_steps 0.05 --plot_loss --overwrite_output_dir --per_device_train_batch_size 2 --gradient_accumulation_steps 16 --learning_rate 2e-5 --num_train_epochs 3 --eval_on_start --lr_scheduler_type cosine --warmup_ratio 0.1 --bf16 --ddp_timeout 180000000 --val_size 0.1 --per_device_eval_batch_size 4 --eval_strategy steps --eval_steps 0.05 --mask_same_sequence --lora_rank 256 --vtcl_mode difBatch
CUDA_VISIBLE_DEVICES=6,7 llamafactory-cli train --model_name_or_path "/root/model/Qwen2-VL-7B-Instruct" --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset POVID_single --dataset_dir "/root/LLaMA-Factory/data" --template qwen2_vl_new --cutoff_len 2048 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/qwen2_vl-7b/lora/dpo" --logging_steps 10 --save_steps 0.05 --plot_loss --overwrite_output_dir --per_device_train_batch_size 2 --gradient_accumulation_steps 16 --learning_rate 2e-5 --num_train_epochs 3 --eval_on_start --lr_scheduler_type cosine --warmup_ratio 0.1 --bf16 --ddp_timeout 180000000 --val_size 0.1 --per_device_eval_batch_size 4 --eval_strategy steps --eval_steps 0.05 --mask_same_sequence --lora_rank 256 --vtcl_mode difBatch
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train --model_name_or_path "/root/model/Qwen2-VL-7B-Instruct" --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset POVID_single --dataset_dir "/root/LLaMA-Factory/data" --template qwen2_vl_new --cutoff_len 2048 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/qwen2_vl-7b/lora/dpo" --logging_steps 10 --save_steps 0.05 --plot_loss --overwrite_output_dir --per_device_train_batch_size 2 --gradient_accumulation_steps 16 --learning_rate 1e-5 --num_train_epochs 3 --eval_on_start --lr_scheduler_type cosine --warmup_ratio 0.1 --bf16 --ddp_timeout 180000000 --val_size 0.1 --per_device_eval_batch_size 4 --eval_strategy steps --eval_steps 0.05 --lora_rank 256 --vtcl_mode difBatch
CUDA_VISIBLE_DEVICES=2,3 llamafactory-cli train --model_name_or_path "/root/model/Qwen2-VL-7B-Instruct" --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset POVID_single --dataset_dir "/root/LLaMA-Factory/data" --template qwen2_vl_new --cutoff_len 2048 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/qwen2_vl-7b/lora/dpo" --logging_steps 10 --save_steps 0.05 --plot_loss --overwrite_output_dir --per_device_train_batch_size 2 --gradient_accumulation_steps 16 --learning_rate 3e-5 --num_train_epochs 3 --eval_on_start --lr_scheduler_type cosine --warmup_ratio 0.1 --bf16 --ddp_timeout 180000000 --val_size 0.1 --per_device_eval_batch_size 4 --eval_strategy steps --eval_steps 0.05 --lora_rank 256 --vtcl_mode difBatch
CUDA_VISIBLE_DEVICES=4,5 llamafactory-cli train --model_name_or_path "/root/model/Qwen2-VL-7B-Instruct" --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset POVID_single --dataset_dir "/root/LLaMA-Factory/data" --template qwen2_vl_new --cutoff_len 2048 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/qwen2_vl-7b/lora/dpo" --logging_steps 10 --save_steps 0.05 --plot_loss --overwrite_output_dir --per_device_train_batch_size 2 --gradient_accumulation_steps 16 --learning_rate 1e-5 --num_train_epochs 3 --eval_on_start --lr_scheduler_type cosine --warmup_ratio 0.1 --bf16 --ddp_timeout 180000000 --val_size 0.1 --per_device_eval_batch_size 4 --eval_strategy steps --eval_steps 0.05 --mask_same_sequence --lora_rank 256 --vtcl_mode difBatch
CUDA_VISIBLE_DEVICES=6,7 llamafactory-cli train --model_name_or_path "/root/model/Qwen2-VL-7B-Instruct" --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset POVID_single --dataset_dir "/root/LLaMA-Factory/data" --template qwen2_vl_new --cutoff_len 2048 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/qwen2_vl-7b/lora/dpo" --logging_steps 10 --save_steps 0.05 --plot_loss --overwrite_output_dir --per_device_train_batch_size 2 --gradient_accumulation_steps 16 --learning_rate 3e-5 --num_train_epochs 3 --eval_on_start --lr_scheduler_type cosine --warmup_ratio 0.1 --bf16 --ddp_timeout 180000000 --val_size 0.1 --per_device_eval_batch_size 4 --eval_strategy steps --eval_steps 0.05 --mask_same_sequence --lora_rank 256 --vtcl_mode difBatch

VTCL-sameBatch
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train --model_name_or_path "/root/model/Qwen2-VL-7B-Instruct" --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset POVID --dataset_dir "/root/LLaMA-Factory/data" --template qwen2_vl_new --cutoff_len 2048 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/qwen2_vl-7b/lora/dpo" --logging_steps 10 --save_steps 0.05 --plot_loss --overwrite_output_dir --per_device_train_batch_size 1 --gradient_accumulation_steps 16 --learning_rate 4e-5 --num_train_epochs 3 --eval_on_start --lr_scheduler_type cosine --warmup_ratio 0.1 --bf16 --ddp_timeout 180000000 --val_size 0.1 --per_device_eval_batch_size 1 --eval_strategy steps --eval_steps 0.05 --lora_rank 256 --vtcl_mode sameBatch --strategy vtcl
CUDA_VISIBLE_DEVICES=2,3 llamafactory-cli train --model_name_or_path "/root/model/Qwen2-VL-7B-Instruct" --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset POVID --dataset_dir "/root/LLaMA-Factory/data" --template qwen2_vl_new --cutoff_len 2048 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/qwen2_vl-7b/lora/dpo" --logging_steps 10 --save_steps 0.05 --plot_loss --overwrite_output_dir --per_device_train_batch_size 1 --gradient_accumulation_steps 16 --learning_rate 3e-5 --num_train_epochs 3 --eval_on_start --lr_scheduler_type cosine --warmup_ratio 0.1 --bf16 --ddp_timeout 180000000 --val_size 0.1 --per_device_eval_batch_size 1 --eval_strategy steps --eval_steps 0.05 --lora_rank 256 --vtcl_mode sameBatch --strategy vtcl
CUDA_VISIBLE_DEVICES=4,5 llamafactory-cli train --model_name_or_path "/root/model/Qwen2-VL-7B-Instruct" --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset POVID --dataset_dir "/root/LLaMA-Factory/data" --template qwen2_vl_new --cutoff_len 2048 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/qwen2_vl-7b/lora/dpo" --logging_steps 10 --save_steps 0.05 --plot_loss --overwrite_output_dir --per_device_train_batch_size 1 --gradient_accumulation_steps 16 --learning_rate 2e-5 --num_train_epochs 3 --eval_on_start --lr_scheduler_type cosine --warmup_ratio 0.1 --bf16 --ddp_timeout 180000000 --val_size 0.1 --per_device_eval_batch_size 1 --eval_strategy steps --eval_steps 0.05 --lora_rank 256 --vtcl_mode sameBatch --strategy vtcl
CUDA_VISIBLE_DEVICES=6,7 llamafactory-cli train --model_name_or_path "/root/model/Qwen2-VL-7B-Instruct" --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset POVID --dataset_dir "/root/LLaMA-Factory/data" --template qwen2_vl_new --cutoff_len 2048 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/qwen2_vl-7b/lora/dpo" --logging_steps 10 --save_steps 0.05 --plot_loss --overwrite_output_dir --per_device_train_batch_size 1 --gradient_accumulation_steps 16 --learning_rate 1e-5 --num_train_epochs 3 --eval_on_start --lr_scheduler_type cosine --warmup_ratio 0.1 --bf16 --ddp_timeout 180000000 --val_size 0.1 --per_device_eval_batch_size 1 --eval_strategy steps --eval_steps 0.05 --lora_rank 256 --vtcl_mode sameBatch --strategy vtcl

VTCL-sameBatch (tokenwise)
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train --model_name_or_path "/root/model/Qwen2-VL-7B-Instruct" --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset POVID --dataset_dir "/root/LLaMA-Factory/data" --template qwen2_vl_new --cutoff_len 2048 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/qwen2_vl-7b/lora/dpo" --logging_steps 10 --save_steps 0.05 --plot_loss --overwrite_output_dir --per_device_train_batch_size 1 --gradient_accumulation_steps 16 --learning_rate 4e-5 --num_train_epochs 3 --eval_on_start --lr_scheduler_type cosine --warmup_ratio 0.1 --bf16 --ddp_timeout 180000000 --val_size 0.1 --per_device_eval_batch_size 1 --eval_strategy steps --eval_steps 0.05 --lora_rank 256 --vtcl_mode sameBatch --strategy vtcl --mask_same_sequence
CUDA_VISIBLE_DEVICES=2,3 llamafactory-cli train --model_name_or_path "/root/model/Qwen2-VL-7B-Instruct" --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset POVID --dataset_dir "/root/LLaMA-Factory/data" --template qwen2_vl_new --cutoff_len 2048 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/qwen2_vl-7b/lora/dpo" --logging_steps 10 --save_steps 0.05 --plot_loss --overwrite_output_dir --per_device_train_batch_size 1 --gradient_accumulation_steps 16 --learning_rate 3e-5 --num_train_epochs 3 --eval_on_start --lr_scheduler_type cosine --warmup_ratio 0.1 --bf16 --ddp_timeout 180000000 --val_size 0.1 --per_device_eval_batch_size 1 --eval_strategy steps --eval_steps 0.05 --lora_rank 256 --vtcl_mode sameBatch --strategy vtcl --mask_same_sequence
CUDA_VISIBLE_DEVICES=4,5 llamafactory-cli train --model_name_or_path "/root/model/Qwen2-VL-7B-Instruct" --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset POVID --dataset_dir "/root/LLaMA-Factory/data" --template qwen2_vl_new --cutoff_len 2048 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/qwen2_vl-7b/lora/dpo" --logging_steps 10 --save_steps 0.05 --plot_loss --overwrite_output_dir --per_device_train_batch_size 1 --gradient_accumulation_steps 16 --learning_rate 2e-5 --num_train_epochs 3 --eval_on_start --lr_scheduler_type cosine --warmup_ratio 0.1 --bf16 --ddp_timeout 180000000 --val_size 0.1 --per_device_eval_batch_size 1 --eval_strategy steps --eval_steps 0.05 --lora_rank 256 --vtcl_mode sameBatch --strategy vtcl --mask_same_sequence
CUDA_VISIBLE_DEVICES=6,7 llamafactory-cli train --model_name_or_path "/root/model/Qwen2-VL-7B-Instruct" --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset POVID --dataset_dir "/root/LLaMA-Factory/data" --template qwen2_vl_new --cutoff_len 2048 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/qwen2_vl-7b/lora/dpo" --logging_steps 10 --save_steps 0.05 --plot_loss --overwrite_output_dir --per_device_train_batch_size 1 --gradient_accumulation_steps 16 --learning_rate 1e-5 --num_train_epochs 3 --eval_on_start --lr_scheduler_type cosine --warmup_ratio 0.1 --bf16 --ddp_timeout 180000000 --val_size 0.1 --per_device_eval_batch_size 1 --eval_strategy steps --eval_steps 0.05 --lora_rank 256 --vtcl_mode sameBatch --strategy vtcl --mask_same_sequence

VTCL-difBatch (concatenate bug)
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train --model_name_or_path "/root/model/Qwen2-VL-7B-Instruct" --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset POVID_single --dataset_dir "/root/LLaMA-Factory/data" --template qwen2_vl_new --cutoff_len 2048 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/qwen2_vl-7b/lora/dpo" --logging_steps 10 --save_steps 0.05 --plot_loss --overwrite_output_dir --per_device_train_batch_size 2 --gradient_accumulation_steps 16 --learning_rate 4e-5 --num_train_epochs 3 --eval_on_start --lr_scheduler_type cosine --warmup_ratio 0.1 --bf16 --ddp_timeout 180000000 --val_size 0.1 --per_device_eval_batch_size 2 --eval_strategy steps --eval_steps 0.05 --lora_rank 256 --vtcl_mode difBatch --strategy vtcl
CUDA_VISIBLE_DEVICES=2,3 llamafactory-cli train --model_name_or_path "/root/model/Qwen2-VL-7B-Instruct" --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset POVID_single --dataset_dir "/root/LLaMA-Factory/data" --template qwen2_vl_new --cutoff_len 2048 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/qwen2_vl-7b/lora/dpo" --logging_steps 10 --save_steps 0.05 --plot_loss --overwrite_output_dir --per_device_train_batch_size 2 --gradient_accumulation_steps 16 --learning_rate 3e-5 --num_train_epochs 3 --eval_on_start --lr_scheduler_type cosine --warmup_ratio 0.1 --bf16 --ddp_timeout 180000000 --val_size 0.1 --per_device_eval_batch_size 2 --eval_strategy steps --eval_steps 0.05 --lora_rank 256 --vtcl_mode difBatch --strategy vtcl
CUDA_VISIBLE_DEVICES=4,5 llamafactory-cli train --model_name_or_path "/root/model/Qwen2-VL-7B-Instruct" --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset POVID_single --dataset_dir "/root/LLaMA-Factory/data" --template qwen2_vl_new --cutoff_len 2048 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/qwen2_vl-7b/lora/dpo" --logging_steps 10 --save_steps 0.05 --plot_loss --overwrite_output_dir --per_device_train_batch_size 2 --gradient_accumulation_steps 16 --learning_rate 2e-5 --num_train_epochs 3 --eval_on_start --lr_scheduler_type cosine --warmup_ratio 0.1 --bf16 --ddp_timeout 180000000 --val_size 0.1 --per_device_eval_batch_size 2 --eval_strategy steps --eval_steps 0.05 --lora_rank 256 --vtcl_mode difBatch --strategy vtcl
CUDA_VISIBLE_DEVICES=6,7 llamafactory-cli train --model_name_or_path "/root/model/Qwen2-VL-7B-Instruct" --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset POVID_single --dataset_dir "/root/LLaMA-Factory/data" --template qwen2_vl_new --cutoff_len 2048 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/qwen2_vl-7b/lora/dpo" --logging_steps 10 --save_steps 0.05 --plot_loss --overwrite_output_dir --per_device_train_batch_size 2 --gradient_accumulation_steps 16 --learning_rate 1e-5 --num_train_epochs 3 --eval_on_start --lr_scheduler_type cosine --warmup_ratio 0.1 --bf16 --ddp_timeout 180000000 --val_size 0.1 --per_device_eval_batch_size 2 --eval_strategy steps --eval_steps 0.05 --lora_rank 256 --vtcl_mode difBatch --strategy vtcl


CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train --model_name_or_path "/root/model/Qwen2-VL-7B-Instruct" --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset POVID --dataset_dir "/root/LLaMA-Factory/data" --template qwen2_vl_new --cutoff_len 2048 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/qwen2_vl-7b/lora/dpo" --logging_steps 10 --save_steps 0.05 --plot_loss --overwrite_output_dir --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --learning_rate 4e-5 --num_train_epochs 3 --eval_on_start --lr_scheduler_type cosine --warmup_ratio 0.1 --bf16 --ddp_timeout 180000000 --val_size 0.1 --per_device_eval_batch_size 4 --eval_strategy steps --eval_steps 0.05 --lora_rank 256 --vtcl_mode sameBatch --strategy vtcl
CUDA_VISIBLE_DEVICES=4,5,6,7 llamafactory-cli train --model_name_or_path "/root/model/Qwen2-VL-7B-Instruct" --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset POVID --dataset_dir "/root/LLaMA-Factory/data" --template qwen2_vl_new --cutoff_len 2048 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/qwen2_vl-7b/lora/dpo" --logging_steps 10 --save_steps 0.05 --plot_loss --overwrite_output_dir --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --learning_rate 1e-4 --num_train_epochs 3 --eval_on_start --lr_scheduler_type cosine --warmup_ratio 0.1 --bf16 --ddp_timeout 180000000 --val_size 0.1 --per_device_eval_batch_size 4 --eval_strategy steps --eval_steps 0.05 --lora_rank 256 --vtcl_mode sameBatch --strategy vtcl
CUDA_VISIBLE_DEVICES=0 python -m src.llamafactory.train.mdpo.workflow --model_name_or_path "/root/model/Qwen2-VL-7B-Instruct" --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset POVID --dataset_dir "/root/LLaMA-Factory/data" --template qwen2_vl_new --cutoff_len 2048 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/qwen2_vl-7b/lora/dpo" --logging_steps 10 --save_steps 0.05 --plot_loss --overwrite_output_dir --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --learning_rate 4e-5 --num_train_epochs 3 --eval_on_start --lr_scheduler_type cosine --warmup_ratio 0.1 --bf16 --ddp_timeout 180000000 --val_size 0.1 --per_device_eval_batch_size 4 --eval_strategy steps --eval_steps 0.05 --lora_rank 256 --vtcl_mode sameBatch --strategy vtcl

llamafactory-cli train --model_name_or_path "/root/model/llava-1.5-7b-hf" --stage mdpo --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --logging_steps 0.002 --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --gradient_accumulation_steps 8 --eval_strategy steps --eval_steps 0.05 --metric_for_best_model rewards/accuracies --learning_rate 1e-5 --num_train_epochs 3.0 --lr_scheduler_type cosine --warmup_ratio 0.05 --ddp_timeout 1800000000 --dataset vtcl_coco2014 --template llava --cutoff_len 1024 --preprocessing_num_workers 16 --output_dir 1 --val_size 0.1 --do_train --do_eval --overwrite_output_dir --load_best_model_at_end --bf16 --plot_loss --overwrite_cache --save_steps 0.05 --lora_rank 256 --strategy vtcl --faster_loader
llamafactory-cli train --model_name_or_path "/root/model/llava-1.5-7b-hf" --stage mdpo --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --logging_steps 0.002 --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --gradient_accumulation_steps 8 --eval_strategy steps --eval_steps 0.05 --learning_rate 1e-5 --num_train_epochs 1.25 --lr_scheduler_type cosine --warmup_ratio 0.05 --ddp_timeout 1800000000 --dataset POVID --template llava --cutoff_len 1024 --preprocessing_num_workers 16 --output_dir 1 --val_size 0.1 --do_train --do_eval --overwrite_output_dir --bf16 --plot_loss --overwrite_cache --save_steps 0.2 --lora_rank 256 --strategy vtcl --faster_loader
# 1 item = 2 samples, num_gpu * per_gpu_bs * bs = 32

llamafactory-cli train --model_name_or_path "/root/model/llava-1.5-7b-hf" --stage mdpo --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --logging_steps 0.002 --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --gradient_accumulation_steps 16 --eval_strategy steps --eval_steps 0.05 --learning_rate 1e-5 --num_train_epochs 1.25 --lr_scheduler_type cosine --warmup_ratio 0.05 --ddp_timeout 1800000000 --dataset POVID_single --template llava --cutoff_len 1024 --preprocessing_num_workers 16 --output_dir 1 --val_size 0.1 --do_train --do_eval --overwrite_output_dir --bf16 --plot_loss --overwrite_cache --save_steps 0.2 --lora_rank 256 --strategy dpo
llamafactory-cli train --model_name_or_path "/root/model/llava-1.5-7b-hf" --stage mdpo --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --logging_steps 0.002 --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --gradient_accumulation_steps 16 --eval_strategy steps --eval_steps 0.05 --learning_rate 1e-5 --num_train_epochs 3.0 --lr_scheduler_type cosine --warmup_ratio 0.05 --ddp_timeout 1800000000 --dataset vtcl_coco2014_single --template llava --cutoff_len 1024 --preprocessing_num_workers 16 --output_dir 1 --val_size 0.1 --do_train --do_eval --overwrite_output_dir --load_best_model_at_end --bf16 --plot_loss --overwrite_cache --save_steps 0.05 --lora_rank 256 --strategy mdpo

llamafactory-cli train --model_name_or_path /root/model/llama3-llava-next-8b-hf --template llava_next_llama3 --stage mdpo --finetuning_type lora --lora_target all --save_only_model --pref_beta 0.1 --pref_loss sigmoid --logging_steps 0.002 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 16 --eval_strategy steps --eval_steps 0.05 --metric_for_best_model rewards/accuracies --learning_rate 1e-5 --num_train_epochs 1.5 --lr_scheduler_type cosine --warmup_ratio 0.05 --ddp_timeout 1800000000 --dataset POVID --cutoff_len 2048 --preprocessing_num_workers 16 --output_dir 1 --val_size 0.1 --do_train --do_eval --overwrite_output_dir --bf16 --plot_loss --overwrite_cache --save_steps 0.05 --lora_rank 256 --strategy vtcl --faster_loader

llamafactory-cli train --model_name_or_path "/root/model/llava-1.5-7b-hf" --stage mdpo --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --logging_steps 0.002 --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --gradient_accumulation_steps 8 --eval_strategy steps --eval_steps 0.05 --metric_for_best_model rewards/accuracies --learning_rate 1e-5 --num_train_epochs 1.25 --lr_scheduler_type cosine --warmup_ratio 0.05 --ddp_timeout 1800000000 --dataset POVID,vtcl_coco2014 --template llava --cutoff_len 1024 --preprocessing_num_workers 16 --output_dir 1 --val_size 0.1 --do_train --do_eval --overwrite_output_dir --bf16 --plot_loss --overwrite_cache --save_steps 0.2 --lora_rank 256 --strategy vtcl --faster_loader

llamafactory-cli train --model_name_or_path "/root/model/llama3-llava-next-8b-hf" --stage mdpo --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --logging_steps 0.002 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 64 --eval_strategy steps --eval_steps 0.05 --metric_for_best_model loss --learning_rate 2e-5 --num_train_epochs 1.25 --lr_scheduler_type cosine --warmup_ratio 0.05 --ddp_timeout 1800000000 --dataset POVID_single --template llava_next_llama3 --cutoff_len 1024 --preprocessing_num_workers 16 --output_dir 1 --val_size 0.1 --do_train --do_eval --overwrite_output_dir --bf16 --plot_loss --overwrite_cache --save_steps 0.2 --lora_rank 64 --strategy sft
llamafactory-cli train --model_name_or_path "/root/model/llama3-llava-next-8b-hf" --stage mdpo --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --logging_steps 0.002 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 64 --eval_strategy steps --eval_steps 0.05 --metric_for_best_model loss --learning_rate 2e-5 --num_train_epochs 2.5 --lr_scheduler_type cosine --warmup_ratio 0.05 --ddp_timeout 1800000000 --dataset POVID --template llava_next_llama3 --cutoff_len 1024 --preprocessing_num_workers 16 --output_dir 1 --val_size 0.1 --do_train --do_eval --overwrite_output_dir --bf16 --plot_loss --overwrite_cache --save_steps 0.2 --lora_rank 64 --strategy sft

llamafactory-cli train --model_name_or_path "/root/model/llama3-llava-next-8b-hf" --stage mdpo --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --logging_steps 0.002 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 64 --eval_strategy steps --eval_steps 0.05 --metric_for_best_model rewards/accuracies --learning_rate 2e-5 --num_train_epochs 1.25 --lr_scheduler_type cosine --warmup_ratio 0.05 --ddp_timeout 1800000000 --dataset POVID_original_single --template llava_next_llama3 --cutoff_len 1024 --preprocessing_num_workers 16 --output_dir 1 --val_size 0.1 --do_train --do_eval --overwrite_output_dir --bf16 --plot_loss --overwrite_cache --save_steps 0.05 --lora_rank 64 --strategy dpo
llamafactory-cli train --model_name_or_path "/root/model/llama3-llava-next-8b-hf" --stage mdpo --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --logging_steps 0.002 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 64 --eval_strategy steps --eval_steps 0.05  --learning_rate 5e-6 --num_train_epochs 1.25 --lr_scheduler_type cosine --warmup_ratio 0.05 --ddp_timeout 1800000000 --dataset POVID_special --template llava_next_llama3 --cutoff_len 1024 --preprocessing_num_workers 16 --output_dir 1 --val_size 0.1 --do_train --do_eval --overwrite_output_dir --bf16 --plot_loss --overwrite_cache --save_steps 0.05 --lora_rank 64 --strategy vtcl --faster_loader --special_vtcl
CUDA_VISIBLE_DEVICES=6 llamafactory-cli train --model_name_or_path "/root/model/llama3-llava-next-8b-hf" --stage mdpo --finetuning_type lora --lora_target all --pref_beta 0.1 --save_only_model --pref_loss sigmoid --logging_steps 0.002 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 16 --eval_strategy steps --eval_steps 0.05  --learning_rate 2e-5 --num_train_epochs 1.25 --lr_scheduler_type cosine --warmup_ratio 0.05 --ddp_timeout 1800000000 --dataset POVID --template llava_next_llama3 --cutoff_len 1024 --preprocessing_num_workers 16 --output_dir 1 --val_size 0.1 --do_train --do_eval --overwrite_output_dir --bf16 --plot_loss --overwrite_cache --save_steps 0.05 --lora_rank 64 --strategy vtcl --iter_training
CUDA_VISIBLE_DEVICES=7 llamafactory-cli train --model_name_or_path "/root/model/llama3-llava-next-8b-hf" --stage mdpo --finetuning_type lora --lora_target all --pref_beta 0.1 --save_only_model --pref_loss sigmoid --logging_steps 0.002 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 16 --eval_strategy steps --eval_steps 0.05  --learning_rate 5e-6 --num_train_epochs 1.25 --lr_scheduler_type cosine --warmup_ratio 0.05 --ddp_timeout 1800000000 --dataset POVID --template llava_next_llama3 --cutoff_len 1024 --preprocessing_num_workers 16 --output_dir 1 --val_size 0.1 --do_train --do_eval --overwrite_output_dir --bf16 --plot_loss --overwrite_cache --save_steps 0.05 --lora_rank 64 --strategy vtcl --iter_training
"""

import dataclasses
import json
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import List, Optional

import torch

from ...data import get_dataset, get_template_and_fix_tokenizer
from ...data.formatter import EmptyFormatter, FunctionFormatter, StringFormatter, ToolFormatter
from ...data.mm_plugin import get_mm_plugin, Qwen2vlPlugin
from ...data.template import _register_template
from ...extras.constants import IGNORE_INDEX, IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER
from ...extras.ploting import plot_loss
from ...hparams import ModelArguments
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push, create_ref_model
from .trainer import CustomDPOTrainer
from .dataloader import SFTDifCollator, MDPODifCollator, DPODifCollator


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
    "vtcl": MDPODifCollator,
    "mdpo": MDPODifCollator,
    "dpo": DPODifCollator,
}


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


def extra_args_examination(data_args, vtcl_args: VTCLArguments):
    if vtcl_args.load_double_dataset:
        if type(data_args.dataset) is str:
            assert data_args.dataset.endswith("single"), f"Training mode: difBatch, need customed dataset (now '{data_args.dataset}')"
    else:
        if type(data_args.dataset) is str:
            assert not data_args.dataset.endswith("single"), f"Training mode: {vtcl_args.vtcl_mode}, customed dataset not supported."
    assert vtcl_args.strategy != "mdpo" or not vtcl_args.use_negative_data, "No pairwise MDPO"


def process_messages(
    messages: Sequence[Dict[str, str]],
    images: Sequence["ImageInput"],
    videos: Sequence["VideoInput"],
    processor: Optional["ProcessorMixin"],
    self,
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


def run_mdpo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    vtcl_args: "VTCLArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    extra_args_examination(data_args, vtcl_args)

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="rm", **tokenizer_module)
    # print(dataset_module["train_dataset"][0])
    # breakpoint()
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    if vtcl_args.test:  # for dubugging
        training_args.logging_dir = "/root/LLaMA-Factory-new/saves/test"
        training_args.output_dir = "/root/LLaMA-Factory-new/saves/test"
    elif training_args.do_train:
        if training_args.do_predict:
            raise KeyError("Not supported: Please run another program to test the trained model on test dataset")
        model_name = model_args.model_name_or_path.split('/')[-1]
        training_args.logging_dir = f"/root/LLaMA-Factory-new/tensorboard/mdpo_{vtcl_args.strategy}/{model_name}"
        training_args.output_dir = f"/root/LLaMA-Factory-new/saves/mdpo_{vtcl_args.strategy}/{model_name}"
        if training_args.world_size == 1 or is_process_zero(training_args):
            suffix = get_suffix(training_args.output_dir)
            training_args.logging_dir += suffix
            training_args.output_dir += suffix
            print("logging dir:", training_args.logging_dir)
            Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
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

    # Initialize our Trainer
    trainer = CustomDPOTrainer(
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
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
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
