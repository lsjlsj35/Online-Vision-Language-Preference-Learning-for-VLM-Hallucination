# Preparation

## Environment

- Train

The training codes are based on code from the [LLaMAFactory](https://github.com/hiyouga/LLaMA-Factory) project.

Due to the old version we used and the significant modifications made, we provide the modified source code directly in this repository instead of applying a `git clone` + `patch`.
```bash
apt install python3.11-venv -y  # sudo apt install python3.11-venv -y
python -m venv ovip
source ovip/bin/activate
cd LLaMAFactory
pip install -e .
pip install torch==2.4.1 transformers==4.45.0 accelerate==0.34.2 deepspeed==0.15.4 httpx==0.23.3 torchvision==0.19.1
pip install spacy==3.7.5
sudo pip uninstall torchaudio lmdeploy ms-swift -y
pip install fastapi diffusers
cd ../
```

- Evaluate

In avoiding env conflicts, we create another env for evaluation, you can skip this step if you only want to run the training process.
```bash
python -m venv amber
source amber/bin/activate
pip install -U spacy
pip install nltk
# pip install ./spacy/en_core_web_lg-3.8.0-py3-none-any.whl  # install en_core_web_lg
# pip install ./spacy/en_core_web_trf-3.8.0-py3-none-any.whl  # install en_core_web_trf
pip install openai
pip install jsonlines
pip install httpx==0.23.3
deactivate

apt install jq -y
apt install yq -y
```


## Models
By default, we use **llava-1.5-7b-hf** for training (note that it's different with **llava-1.5-7b** which is based on the `llava` package). **Qwen2.5-7b-instruct** and **FLUX.1-dev** are for score prediction and negative image generation.
```bash
huggingface-cli download llava-hf/llava-1.5-7b-hf --local-dir model/llava-1.5-7b-hf
huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir model/FLUX.1-dev
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir model/qwen2.5-7b-instruct
```

## Data
We provide a few samples for training. 

```bash
tar -zxvf OVIP_data.tar.gz
```

# Codes
## Training
We implement OViP and other algorithms based on *llamafactory*.
### What's New?
- We define some new arguments in `LLaMAFactory/src/llamafactory/hparams/vtcl_args.py`
- We add a new stage `mdpo` in `LLaMAFactory/src/llamafactory/train`. All the dataloaders, trainers and workflow are in this directory.

In `LLaMAFactory/src/llamafactory/train/mdpo/trainer.py`:
- Offline learning uses `CustomDPOTrainer`.
- Online learning (except for GRPO) uses `OnlineCustomDPOTrainerWithBuffer`.
- GRPO uses `GRPOTrainerWithBuffer`.

## Evaluation
We implement the evaluation framework based on *s-vco*'s repo. The inference code in their repo has a bug, which can lead to significant performance drops. We fix the bug and provide a clearer code for data loading and inference.
### What's New?
- We provide a clear and easy-to-use inference framework in `svco-eval/eval_vlm/llamafactory_eval`. 
  - We separate the inference codes and the data loading codes. The data flow during inference is controlled by callbacks of different templates defined in `svco-eval/eval_vlm/llamafactory_eval/template.py`.
  - Inference engine for Huggingface-styled llava model (we use this model for training) is defined in `svco-eval/eval_vlm/llamafactory_eval/model.py`.
  - Inference engine for liuhaotian/llava-v1.5-7b with lora adapters is defined in `svco-eval/eval_vlm/llamafactory_eval/model_llava.py`.


# Training
- First, start LLM and diffusion model services:
```bash
tmux new -s 4; cd LLaMAFactory/annotations; CUDA_VISIBLE_DEVICES=4 MODEL=generate uvicorn annotate:app --port 8001
tmux new -s 5; cd LLaMAFactory/annotations; CUDA_VISIBLE_DEVICES=5 MODEL=generate uvicorn annotate:app --port 8003
tmux new -s 6; cd LLaMAFactory/annotations; CUDA_VISIBLE_DEVICES=6 MODEL=desc uvicorn annotate:app --port 8002
```

- If you run the above codes to get models and data, then you can start training using this script:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train --model_name_or_path "model/llava-1.5-7b-hf" --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset dummy --eval_dataset ovip_eval --dataset_dir ../data --template llava_new --cutoff_len 2048 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/llava-1.5/lora/dpo" --logging_steps 10 --save_strategy steps --save_steps 4 --plot_loss --overwrite_output_dir --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --learning_rate 1e-6 --num_train_epochs 2 --lr_scheduler_type cosine --ddp_timeout 18000000 --eval_strategy steps --eval_steps 4 --lora_rank 256 --save_total_limit 100 --vtcl_mode single --strategy online-orm --not_half_dpo_loss --anchor_ratio 0.0 --ds_path ../data/ovip_sample.json --experience_buffer --exp_per_batch 1
```

- Or you can replace "**{{xxx}}**" based on your own project structure:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train --model_name_or_path {{where you store the model parameter}} --adapter_name_or_path {{if exists}} --trust_remote_code --stage mdpo --do_train --finetuning_type lora --lora_target all --pref_beta 0.1 --pref_loss sigmoid --save_only_model --dataset dummy --eval_dataset {{EVALUATION dataset}} --dataset_dir {{Training data DPO}} --template llava_new --cutoff_len 2048 --overwrite_cache --preprocessing_num_workers 16 --output_dir "saves/test/llava-1.5/lora/dpo" --logging_steps 10 --save_strategy steps --save_steps 4 --plot_loss --overwrite_output_dir --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --learning_rate 1e-6 --num_train_epochs 2 --lr_scheduler_type cosine --ddp_timeout 18000000 --eval_strategy steps --eval_steps 4 --lora_rank 256 --save_total_limit 100 --vtcl_mode single --strategy online-orm --not_half_dpo_loss --anchor_ratio 0.0 --ds_path {{Your OVIP data}} --experience_buffer --exp_per_batch 1
```

- The generated online data are saved at `LLaMAFactory/online_data/*/llava-1.5-7b-hf_Exp*`.
  - The time for each stage (e.g. sampling, scoring, image generation) is logged in `LLaMAFactory/online_data/*/llava-1.5-7b-hf_Exp*/time_log.jsonl`.
  - `LLaMAFactory/online_data/*/llava-1.5-7b-hf_Exp*/online_sample_data.jsonl` stores the original sampled responses and their scores.
  - `LLaMAFactory/online_data/*/llava-1.5-7b-hf_Exp*/data.jsonl` is the training data after filtering, which also contains the path to the generated images.

# Evaluation
To evaluate the trained lora checkpoint, run:
```bash
cd svco-eval; bash generate_all_eval_rule.sh --ckpt_path="none" --device=0
```
Here, "none" indicates that no additional checkpoint is loaded. It can be replaced with the path to a LoRA checkpoint. We include 1 samples from the AMBER benchmark and 1 from RealworldQA to ensure the script runs successfully. 

Our evaluation framework separates the testing code for each benchmark to enable easier customization. The datasets for TextVQA, RQA, CVBench, and Llava-Bench are downloaded using the same procedures as in svco. To download MMHal, ObjectHal and MMStar, please follow the following instructions:
```bash
# MMHal
huggingface-cli download Shengcao1006/MMHal-Bench --local-dir benchmark/mmhal --repo-type dataset

# ObjectHal
cd benchmark/objectHal; wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip; unzip annotations_trainval2014.zip; cd ../..

# MMStar
huggingface-cli download Lin-Chen/MMStar --local-dir benchmark/MMStar --repo-type dataset
```

For AMBER benchmark, download the [images](https://drive.google.com/file/d/1MaCHgtupcZUjf007anNl4_MV0o4DjXvl/view?usp=sharing) to `benchmark/image`, and replace `svco-eval/eval_vlm/playground/data/eval/amber_gen/data/annotations.json` with `annotations.json` from https://github.com/junyangwang0410/AMBER/tree/master/data. Additionally, convert `query_discriminative.json` and `query_generative.json` from https://github.com/junyangwang0410/AMBER/tree/master/data/query into jsonl format, and use them to replace:
- `svco-eval/eval_vlm/playground/data/eval/amber_gen/amber_gen.jsonl`
- `svco-eval/eval_vlm/playground/data/eval/amber_dis/amber_dis.jsonl`
