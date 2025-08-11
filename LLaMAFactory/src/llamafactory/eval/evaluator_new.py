"""
llamafactory-cli eval --model_name_or_path /root/model/llava-1.5-7b-hf --template llava --task POPE_new --max_new_tokens 128 --stage vtcl --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory/saves/mdpo/llava-1.5-7b-hf_Exp17/checkpoint-258/
llamafactory-cli eval --model_name_or_path /root/model/llava-1.5-7b-hf --template llava --task POPE_new --max_new_tokens 128 --stage vtcl --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory/saves/mdpo/llava-1.5-7b-hf_Exp22/checkpoint-702

CUDA_VISIBLE_DEVICES=4 llamafactory-cli eval --model_name_or_path /root/model/llava-1.5-7b-hf --template llava --task COCOEval --max_new_tokens 256 --stage vtcl --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory/saves/mdpo/llava-1.5-7b-hf_Exp24/checkpoint-636; 
CUDA_VISIBLE_DEVICES=4 llamafactory-cli eval --model_name_or_path /root/model/llava-1.5-7b-hf --template llava --task MMHalBench --max_new_tokens 128 --stage vtcl --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory/saves/mdpo/llava-1.5-7b-hf_Exp24/checkpoint-636; 
CUDA_VISIBLE_DEVICES=4 llamafactory-cli eval --model_name_or_path /root/model/llava-1.5-7b-hf --template llava --task MMVetV1 --max_new_tokens 384 --stage vtcl --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory/saves/mdpo/llava-1.5-7b-hf_Exp24/checkpoint-636; 
CUDA_VISIBLE_DEVICES=4 llamafactory-cli eval --model_name_or_path /root/model/llava-1.5-7b-hf --template llava --task AMBER_discriminative --max_new_tokens 10 --stage vtcl --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory/saves/mdpo/llava-1.5-7b-hf_Exp24/checkpoint-636; 
CUDA_VISIBLE_DEVICES=5 llamafactory-cli eval --model_name_or_path /root/model/llava-1.5-7b-hf --template llava --task HalluBench --max_new_tokens 10 --stage vtcl --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory/saves/mdpo/llava-1.5-7b-hf_Exp24/checkpoint-636; 
CUDA_VISIBLE_DEVICES=5 llamafactory-cli eval --model_name_or_path /root/model/llava-1.5-7b-hf --template llava --task RBench --max_new_tokens 10 --stage vtcl --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory/saves/mdpo/llava-1.5-7b-hf_Exp24/checkpoint-636; 
CUDA_VISIBLE_DEVICES=5 llamafactory-cli eval --model_name_or_path /root/model/llava-1.5-7b-hf --template llava --task AMBER_generative --max_new_tokens 384 --stage vtcl --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory/saves/mdpo/llava-1.5-7b-hf_Exp24/checkpoint-636; 
CUDA_VISIBLE_DEVICES=5 llamafactory-cli eval --model_name_or_path /root/model/llava-1.5-7b-hf --template llava --task POPE --max_new_tokens 64 --stage vtcl --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory/saves/mdpo/llava-1.5-7b-hf_Exp24/checkpoint-636

CUDA_VISIBLE_DEVICES=5 llamafactory-cli eval --model_name_or_path /root/model/llama3-llava-next-8b-hf --template llava_next_llama3 --task MMHalBench --max_new_tokens 128 --stage vtcl --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory/saves/mdpo/llama3-llava-next-8b-hf_Exp11/checkpoint-443; 
CUDA_VISIBLE_DEVICES=5 llamafactory-cli eval --model_name_or_path /root/model/llama3-llava-next-8b-hf --template llava_next_llama3 --task MMVetV1 --max_new_tokens 384 --stage vtcl --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory/saves/mdpo/llama3-llava-next-8b-hf_Exp11/checkpoint-443; 
CUDA_VISIBLE_DEVICES=2 llamafactory-cli eval --model_name_or_path /root/model/llama3-llava-next-8b-hf --template llava_next_llama3 --task HalluBench --max_new_tokens 10 --stage vtcl --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory/saves/mdpo/llama3-llava-next-8b-hf_Exp6/checkpoint-222; 
CUDA_VISIBLE_DEVICES=2 llamafactory-cli eval --model_name_or_path /root/model/llama3-llava-next-8b-hf --template llava_next_llama3 --task RBench --max_new_tokens 10 --stage vtcl --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory/saves/mdpo/llama3-llava-next-8b-hf_Exp6/checkpoint-222; 
CUDA_VISIBLE_DEVICES=2 llamafactory-cli eval --model_name_or_path /root/model/llama3-llava-next-8b-hf --template llava_next_llama3 --task COCOEval --max_new_tokens 256 --stage vtcl --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory/saves/mdpo/llama3-llava-next-8b-hf_Exp6/checkpoint-222; 
CUDA_VISIBLE_DEVICES=5 llamafactory-cli eval --model_name_or_path /root/model/llama3-llava-next-8b-hf --template llava_next_llama3 --task AMBER_discriminative --max_new_tokens 10 --stage vtcl --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory/saves/mdpo/llama3-llava-next-8b-hf_Exp6/checkpoint-222; 
CUDA_VISIBLE_DEVICES=5 llamafactory-cli eval --model_name_or_path /root/model/llama3-llava-next-8b-hf --template llava_next_llama3 --task POPE --max_new_tokens 64 --stage vtcl --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory/saves/mdpo/llama3-llava-next-8b-hf_Exp6/checkpoint-222

CUDA_VISIBLE_DEVICES=6 llamafactory-cli eval --model_name_or_path /root/model/llava-1.5-7b-hf --template llava --task MMHalBench --max_new_tokens 128 --stage vtcl --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory/saves/mdpo/llava-1.5-7b-hf_Exp28/checkpoint-222;
CUDA_VISIBLE_DEVICES=6 llamafactory-cli eval --model_name_or_path /root/model/llava-1.5-7b-hf --template llava --task MMVetV1 --max_new_tokens 384 --stage vtcl --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory/saves/mdpo/llava-1.5-7b-hf_Exp28/checkpoint-222; 

CUDA_VISIBLE_DEVICES=3 llamafactory-cli eval --model_name_or_path /root/model/llama3-llava-next-8b-hf --template llava_next_llama3 --task AMBER_generative --max_new_tokens 384 --stage vtcl --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory/saves/mdpo/llama3-llava-next-8b-hf_Exp4/checkpoint-222; 

llamafactory-cli eval --model_name_or_path /root/model/llama3-llava-next-8b-hf --template llava_next_llama3 --task AMBER_generative --max_new_tokens 384 --stage vtcl --finetuning_type lora
CUDA_VISIBLE_DEVICES=0 llamafactory-cli eval --model_name_or_path /root/model/llama3-llava-next-8b-hf --template llava_next_llama3 --task HalluBench --max_new_tokens 10 --stage vtcl --finetuning_type lora
CUDA_VISIBLE_DEVICES=1 llamafactory-cli eval --model_name_or_path /root/model/llama3-llava-next-8b-hf --template llava_next_llama3 --task RBench --max_new_tokens 10 --stage vtcl --finetuning_type lora
CUDA_VISIBLE_DEVICES=2 llamafactory-cli eval --model_name_or_path /root/model/llama3-llava-next-8b-hf --template llava_next_llama3 --task COCOEval --max_new_tokens 256 --stage vtcl --finetuning_type lora
CUDA_VISIBLE_DEVICES=3 llamafactory-cli eval --model_name_or_path /root/model/llama3-llava-next-8b-hf --template llava_next_llama3 --task MMHalBench --max_new_tokens 128 --stage vtcl --finetuning_type lora
CUDA_VISIBLE_DEVICES=4 llamafactory-cli eval --model_name_or_path /root/model/llama3-llava-next-8b-hf --template llava_next_llama3 --task MMHalBench_new --max_new_tokens 128 --stage vtcl --finetuning_type lora
CUDA_VISIBLE_DEVICES=5 llamafactory-cli eval --model_name_or_path /root/model/llama3-llava-next-8b-hf --template llava_next_llama3 --task MMVetV1 --max_new_tokens 384 --stage vtcl --finetuning_type lora
CUDA_VISIBLE_DEVICES=6 llamafactory-cli eval --model_name_or_path /root/model/llama3-llava-next-8b-hf --template llava_next_llama3 --task POPE_new --max_new_tokens 128 --stage vtcl --finetuning_type lora

CUDA_VISIBLE_DEVICES=5 llamafactory-cli eval --model_name_or_path /root/model/llama3-llava-next-8b-hf --template llava_next_llama3 --task MMHalBench --max_new_tokens 128 --stage vtcl --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory/saves/mdpo/llama3-llava-next-8b-hf_Exp18/checkpoint-885; 
CUDA_VISIBLE_DEVICES=5 llamafactory-cli eval --model_name_or_path /root/model/llama3-llava-next-8b-hf --template llava_next_llama3 --task MMVetV1 --max_new_tokens 384 --stage vtcl --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory/saves/mdpo/llama3-llava-next-8b-hf_Exp18/checkpoint-885;
CUDA_VISIBLE_DEVICES=5 llamafactory-cli eval --model_name_or_path /root/model/llama3-llava-next-8b-hf --template llava_next_llama3 --task MMVetV2 --max_new_tokens 384 --stage vtcl --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory/saves/mdpo/llama3-llava-next-8b-hf_Exp18/checkpoint-885;
CUDA_VISIBLE_DEVICES=2 llamafactory-cli eval --model_name_or_path /root/model/llama3-llava-next-8b-hf --template llava_next_llama3 --task MMHalBench --max_new_tokens 128 --stage vtcl --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory/saves/mdpo/llama3-llava-next-8b-hf_Exp18/checkpoint-810; 
CUDA_VISIBLE_DEVICES=2 llamafactory-cli eval --model_name_or_path /root/model/llama3-llava-next-8b-hf --template llava_next_llama3 --task MMVetV1 --max_new_tokens 384 --stage vtcl --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory/saves/mdpo/llama3-llava-next-8b-hf_Exp18/checkpoint-810;
CUDA_VISIBLE_DEVICES=3 llamafactory-cli eval --model_name_or_path /root/model/llama3-llava-next-8b-hf --template llava_next_llama3 --task MMVetV2 --max_new_tokens 384 --stage vtcl --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory/saves/mdpo/llama3-llava-next-8b-hf_Exp18/checkpoint-810;

CUDA_VISIBLE_DEVICES=6 llamafactory-cli eval --model_name_or_path /root/model/llama3-llava-next-8b-hf --template llava_next_llama3 --task MMVetV2 --max_new_tokens 384 --stage vtcl --finetuning_type lora;
CUDA_VISIBLE_DEVICES=6 llamafactory-cli eval --model_name_or_path /root/model/llama3-llava-next-8b-hf --template llava_next_llama3 --task MMVetV2 --max_new_tokens 384 --stage vtcl --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory/saves/mdpo/llama3-llava-next-8b-hf_Exp4/checkpoint-222;
CUDA_VISIBLE_DEVICES=0 llamafactory-cli eval --model_name_or_path /root/model/llama3-llava-next-8b-hf --template llava_next_llama3 --task MMVetV2 --max_new_tokens 384 --stage vtcl --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory/saves/mdpo/llama3-llava-next-8b-hf_Exp6/checkpoint-222;
CUDA_VISIBLE_DEVICES=0 llamafactory-cli eval --model_name_or_path /root/model/llama3-llava-next-8b-hf --template llava_next_llama3 --task MMVetV2 --max_new_tokens 384 --stage vtcl --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory/saves/mdpo/llama3-llava-next-8b-hf_Exp9/checkpoint-222;
CUDA_VISIBLE_DEVICES=1 llamafactory-cli eval --model_name_or_path /root/model/llama3-llava-next-8b-hf --template llava_next_llama3 --task MMVetV2 --max_new_tokens 384 --stage vtcl --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory/saves/mdpo/llama3-llava-next-8b-hf_Exp11/checkpoint-443;
CUDA_VISIBLE_DEVICES=1 llamafactory-cli eval --model_name_or_path /root/model/llama3-llava-next-8b-hf --template llava_next_llama3 --task MMVetV2 --max_new_tokens 384 --stage vtcl --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory/saves/mdpo/llama3-llava-next-8b-hf_Exp12/checkpoint-443;
CUDA_VISIBLE_DEVICES=7 llamafactory-cli eval --model_name_or_path /root/model/llama3-llava-next-8b-hf --template llava_next_llama3 --task MMVetV2 --max_new_tokens 384 --stage vtcl --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory/saves/mdpo/llama3-llava-next-8b-hf_Exp14/checkpoint-443;
CUDA_VISIBLE_DEVICES=7 llamafactory-cli eval --model_name_or_path /root/model/llama3-llava-next-8b-hf --template llava_next_llama3 --task MMVetV2 --max_new_tokens 384 --stage vtcl --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory/saves/mdpo/llama3-llava-next-8b-hf_Exp15/checkpoint-443;

CUDA_VISIBLE_DEVICES=0 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task MMHalBench --max_new_tokens 128 --stage mdpo --finetuning_type lora

VTCL
for i in {1..3}; do CUDA_VISIBLE_DEVICES=0 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task MMHalBench --max_new_tokens 128 --stage mdpo --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory-new/saves/mdpo_vtcl/Qwen2-VL-7B-Instruct_Exp1; done
for i in {1..3}; do CUDA_VISIBLE_DEVICES=1 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task MMHalBench --max_new_tokens 128 --stage mdpo --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory-new/saves/mdpo_vtcl/Qwen2-VL-7B-Instruct_Exp2; done
for i in {1..3}; do CUDA_VISIBLE_DEVICES=2 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task MMHalBench --max_new_tokens 128 --stage mdpo --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory-new/saves/mdpo_vtcl/Qwen2-VL-7B-Instruct_Exp3; done
for i in {1..3}; do CUDA_VISIBLE_DEVICES=3 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task MMHalBench --max_new_tokens 128 --stage mdpo --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory-new/saves/mdpo_vtcl/Qwen2-VL-7B-Instruct_Exp4; done
for i in {1..3}; do CUDA_VISIBLE_DEVICES=4 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task MMHalBench --max_new_tokens 128 --stage mdpo --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory-new/saves/mdpo_vtcl/Qwen2-VL-7B-Instruct_Exp5; done
for i in {1..3}; do CUDA_VISIBLE_DEVICES=5 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task MMHalBench --max_new_tokens 128 --stage mdpo --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory-new/saves/mdpo_vtcl/Qwen2-VL-7B-Instruct_Exp6; done
for i in {1..3}; do CUDA_VISIBLE_DEVICES=6 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task MMHalBench --max_new_tokens 128 --stage mdpo --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory-new/saves/mdpo_vtcl/Qwen2-VL-7B-Instruct_Exp7; done
for i in {1..3}; do CUDA_VISIBLE_DEVICES=7 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task MMHalBench --max_new_tokens 128 --stage mdpo --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory-new/saves/mdpo_vtcl/Qwen2-VL-7B-Instruct_Exp8; done

mDPO (MMHalBench+AMBER_GEN)
CUDA_VISIBLE_DEVICES=3 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task MMHalBench --max_new_tokens 128 --stage mdpo --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory-new/saves/mdpo_mdpo/Qwen2-VL-7B-Instruct_Exp1
CUDA_VISIBLE_DEVICES=4 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task MMHalBench --max_new_tokens 128 --stage mdpo --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory-new/saves/mdpo_mdpo/Qwen2-VL-7B-Instruct_Exp2
CUDA_VISIBLE_DEVICES=5 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task MMHalBench --max_new_tokens 128 --stage mdpo --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory-new/saves/mdpo_mdpo/Qwen2-VL-7B-Instruct_Exp3
CUDA_VISIBLE_DEVICES=6 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task MMHalBench --max_new_tokens 128 --stage mdpo --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory-new/saves/mdpo_mdpo/Qwen2-VL-7B-Instruct_Exp4
CUDA_VISIBLE_DEVICES=7 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task MMHalBench --max_new_tokens 128 --stage mdpo --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory-new/saves/mdpo_mdpo/Qwen2-VL-7B-Instruct_Exp5

DPO
CUDA_VISIBLE_DEVICES=0 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task MMHalBench --max_new_tokens 128 --stage mdpo --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory-new/saves/mdpo_dpo/Qwen2-VL-7B-Instruct_Exp1
CUDA_VISIBLE_DEVICES=1 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task MMHalBench --max_new_tokens 128 --stage mdpo --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory-new/saves/mdpo_dpo/Qwen2-VL-7B-Instruct_Exp2
CUDA_VISIBLE_DEVICES=2 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task MMHalBench --max_new_tokens 128 --stage mdpo --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory-new/saves/mdpo_dpo/Qwen2-VL-7B-Instruct_Exp3
CUDA_VISIBLE_DEVICES=3 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task MMHalBench --max_new_tokens 128 --stage mdpo --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory-new/saves/mdpo_dpo/Qwen2-VL-7B-Instruct_Exp4

SFT-dif batch-sequence
CUDA_VISIBLE_DEVICES=4 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task MMHalBench --max_new_tokens 128 --stage mdpo --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory-new/saves/mdpo_sft/Qwen2-VL-7B-Instruct_Exp20
CUDA_VISIBLE_DEVICES=5 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task MMHalBench --max_new_tokens 128 --stage mdpo --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory-new/saves/mdpo_sft/Qwen2-VL-7B-Instruct_Exp21
CUDA_VISIBLE_DEVICES=6 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task MMHalBench --max_new_tokens 128 --stage mdpo --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory-new/saves/mdpo_sft/Qwen2-VL-7B-Instruct_Exp22
CUDA_VISIBLE_DEVICES=7 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task MMHalBench --max_new_tokens 128 --stage mdpo --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory-new/saves/mdpo_sft/Qwen2-VL-7B-Instruct_Exp23

MMVetV1
CUDA_VISIBLE_DEVICES=0 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task MMVetV1 --max_new_tokens 384 --stage mdpo --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory-new/saves/mdpo_vtcl/Qwen2-VL-7B-Instruct_Exp1
CUDA_VISIBLE_DEVICES=1 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task MMVetV1 --max_new_tokens 384 --stage mdpo --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory-new/saves/mdpo_vtcl/Qwen2-VL-7B-Instruct_Exp4
CUDA_VISIBLE_DEVICES=2 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task MMVetV1 --max_new_tokens 384 --stage mdpo --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory-new/saves/mdpo_vtcl/Qwen2-VL-7B-Instruct_Exp11
CUDA_VISIBLE_DEVICES=3 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task MMVetV1 --max_new_tokens 384 --stage mdpo --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory-new/saves/mdpo_vtcl/Qwen2-VL-7B-Instruct_Exp12
CUDA_VISIBLE_DEVICES=4 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task MMVetV1 --max_new_tokens 384 --stage mdpo --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory-new/saves/mdpo_sft/Qwen2-VL-7B-Instruct_Exp20
CUDA_VISIBLE_DEVICES=5 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task MMVetV1 --max_new_tokens 384 --stage mdpo --finetuning_type lora

HalluBench
CUDA_VISIBLE_DEVICES=0 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task HalluBench --max_new_tokens 10 --stage mdpo --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory-new/saves/mdpo_vtcl/Qwen2-VL-7B-Instruct_Exp1
CUDA_VISIBLE_DEVICES=1 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task HalluBench --max_new_tokens 10 --stage mdpo --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory-new/saves/mdpo_vtcl/Qwen2-VL-7B-Instruct_Exp4
CUDA_VISIBLE_DEVICES=2 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task HalluBench --max_new_tokens 10 --stage mdpo --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory-new/saves/mdpo_vtcl/Qwen2-VL-7B-Instruct_Exp11
CUDA_VISIBLE_DEVICES=3 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task HalluBench --max_new_tokens 10 --stage mdpo --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory-new/saves/mdpo_vtcl/Qwen2-VL-7B-Instruct_Exp12
CUDA_VISIBLE_DEVICES=4 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task HalluBench --max_new_tokens 10 --stage mdpo --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory-new/saves/mdpo_sft/Qwen2-VL-7B-Instruct_Exp20
CUDA_VISIBLE_DEVICES=5 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task HalluBench --max_new_tokens 10 --stage mdpo --finetuning_type lora

RBench & POPE & AMBER_dis & AMBER_gen
CUDA_VISIBLE_DEVICES=0 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task RBench --max_new_tokens 10 --stage mdpo --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory-new/saves/mdpo_vtcl/Qwen2-VL-7B-Instruct_Exp1; 

llavaBenchintheWild
CUDA_VISIBLE_DEVICES=2 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task llavaBenchintheWild --max_new_tokens 256 --stage mdpo --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory-new/saves/mdpo_vtcl/Qwen2-VL-7B-Instruct_Exp11

GQA
CUDA_VISIBLE_DEVICES=2 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task GQA --max_new_tokens 128 --stage mdpo --finetuning_type lora
CUDA_VISIBLE_DEVICES=3 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task GQA --max_new_tokens 128 --stage mdpo --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory-new/saves/mdpo_vtcl/Qwen2-VL-7B-Instruct_Exp4
CUDA_VISIBLE_DEVICES=4 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task GQA --max_new_tokens 128 --stage mdpo --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory-new/saves/mdpo_vtcl/Qwen2-VL-7B-Instruct_Exp12
CUDA_VISIBLE_DEVICES=5 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task GQA --max_new_tokens 128 --stage mdpo --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory-new/saves/mdpo_vtcl/Qwen2-VL-7B-Instruct_Exp1
CUDA_VISIBLE_DEVICES=6 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task GQA --max_new_tokens 128 --stage mdpo --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory-new/saves/mdpo_sft/Qwen2-VL-7B-Instruct_Exp20
CUDA_VISIBLE_DEVICES=7 python -m src.llamafactory.eval.evaluator_new --model_name_or_path /root/model/Qwen2-VL-7B-Instruct --template qwen2_vl --task GQA --max_new_tokens 128 --stage mdpo --finetuning_type lora --adapter_name_or_path /root/LLaMA-Factory-new/saves/mdpo_mdpo/Qwen2-VL-7B-Instruct_Exp1

cd ~/LLaMA-Factory-new ; conda activate llamafactorynew
"""

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, Optional

import PIL
import torch
import transformers
from tqdm import tqdm
# from qwen_vl_utils import process_vision_info

from ..data import get_template_and_fix_tokenizer
from ..hparams import get_eval_args
from ..model import load_model, load_tokenizer
from .template_new import get_eval_template


IMAGE_PLACEHOLDER = "<image>"


def get_suffix(p):
    idx = 0
    while Path(p + f"_Exp{idx}").exists():
        idx += 1
    return f"_Exp{idx}"


def set_seed(seed=0):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


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


class Evaluator:
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        self.model_args, self.data_args, self.eval_args, self.finetuning_args, generation_args, self.vtcl_args = get_eval_args(args)
        set_seed(self.eval_args.seed)
        pro_tok = load_tokenizer(self.model_args)
        self.processor = pro_tok["processor"]
        self.tokenizer = pro_tok["tokenizer"]
        if self.tokenizer.pad_token:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # self.tokenizer.padding_side = "right"  # avoid overflow issue in batched inference for llama2
        self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)
        self.model = load_model(self.tokenizer, self.model_args, self.finetuning_args).eval().to("cuda")
        self.eval_template = get_eval_template(self.eval_args.task)
        self.generation_config = generation_args.to_dict()
        
        adapter_path = self.model_args.adapter_name_or_path[0] if self.model_args.adapter_name_or_path else self.model_args.adapter_name_or_path
        if self.model_args.adapter_name_or_path:
            trained_info = adapter_path[adapter_path.find("saves"):].split('/', 1)[1]
            self.save_path = f"/root/LLaMA-Factory-new/evaluation/{self.eval_args.task}/{trained_info.replace('/', '_')}"
        else:
            self.save_path = f"/root/LLaMA-Factory-new/evaluation/{self.eval_args.task}/{self.model_args.model_name_or_path.rsplit('/', 1)[-1]}"
        print("logging dir:", self.save_path)
        if not Path(self.save_path).exists():
            Path(self.save_path).mkdir(parents=True)
            total_args, filtered_args = as_dict(self.model_args, self.data_args, self.eval_args, self.finetuning_args, generation_args, self.vtcl_args)
            torch.save(total_args, Path(self.save_path) / "total_args.pt")
            with open(Path(self.save_path) / "config.txt", "w") as f:
                json.dump(filtered_args, f, indent=4)

    def encode(self, messages):
        encoded_messages = self.template._encode(self.tokenizer, messages, None, None)
        prompt_ids = []
        for encoded_ids in encoded_messages:
            prompt_ids += encoded_ids
        return prompt_ids

    def eval(self):
        dataset = self.eval_template.load_data()
        i = 1
        while (Path(self.save_path) / f"trial_{i}.jsonl").exists():
            i += 1
        with open(Path(self.save_path) / f"trial_{i}.jsonl", 'a') as f:
            pbar = tqdm(dataset, total=self.eval_template.length)
            for item in pbar:
                if self.eval_template.customed_inference:
                    response, other_results_dict = self.eval_template.inference(item, self.encode, self.template, self.model, self.tokenizer, self.processor, self.generation_config)
                    item.pop("IMAGE", None)
                    json.dump({**item, "result": response, **other_results_dict}, f)
                    f.write('\n')
                else:
                    messages, images = self.eval_template.format_example(item)
                    if images and images.size[0] * images.size[1] > 1800000:
                        scale = (images.size[0] * images.size[1] / 1800000.) ** 0.5
                        images = images.resize((int(images.size[0] / scale), int(images.size[1] / scale)))
                    # print("shape", images.size)
                    content = messages[0]["content"]
                    if content.startswith("<image>"):
                        content = content[7:].strip()
                    if images:
                        messages[0]["content"] = [
                            {"type": "image"}, {"type": "text", "text": content}
                        ]
                    if self.processor:
                        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                    else:
                        text_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                    if images:
                        inputs = self.processor(
                            text=[text_prompt], images=[images] if images else None, padding=True, return_tensors="pt"
                        )
                    else:
                        inputs = self.tokenizer([text_prompt], padding=True, return_tensors="pt")
                    inputs = inputs.to("cuda")
                    with torch.no_grad():
                        # output = self.model.generate(**inputs, **self.generation_config)[0][len(input_ids):-1]  # <eos>
                        output = self.model.generate(**inputs, pad_token_id=self.tokenizer.eos_token_id, **self.generation_config)[0][inputs["input_ids"].shape[-1]:-1]
                    response = self.tokenizer.decode(output)
                    item.pop("IMAGE", None)
                    json.dump({**item, "result": response}, f)
                    f.write('\n')
                    torch.cuda.empty_cache()
        self.eval_template.post_process(Path(self.save_path), Path(self.save_path) / f"trial_{i}.jsonl")


def run_eval() -> None:
    Evaluator().eval()


if __name__ == "__main__":
    run_eval()