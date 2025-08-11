while [ $# -gt 0 ]; do
  case "$1" in
    --ckpt_path=*)
      CKPT_PATH="${1#*=}" 
      ;;
    --template=*)
      TEMPLATE="${1#*=}"
      ;;
    --device=*)
      DEVICE="${1#*=}"
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
  shift
done

if [ -z "$TEMPLATE" ]; then
  TEMPLATE="llava"
fi

OPENAI_API_KEY=$(yq -r '.api_key' config.yaml)

declare -A T2M

while IFS="=" read -r key value; do
  T2M["$key"]="$value"
done < <(cat ./template2model.json | jq -r 'to_entries | .[] | "\(.key)=\(.value)"')

MODEL_BASE=${T2M["${TEMPLATE}"]}

RUN_NAME=$(echo "$CKPT_PATH" | sed 's|.*saves/||' | sed 's/mdpo_//g' | sed 's/\//_/g')
DELETE_TMP_MODEL=false

export MODEL_BASE
export RUN_NAME
export CKPT_PATH
export OPENAI_API_KEY
export DEVICE
export OUTPUT_DIR="LOG_EVAL/${RUN_NAME}"

echo "MODEL_BASE = ${MODEL_BASE}"  # ["llava-hf/llava-interleave-qwen-7b-hf", "llava-hf/llava-1.5-7b-hf"]
echo "RUN_NAME = ${RUN_NAME}"  # experiment name --> used for eval result logging
echo "CKPT_PATH = ${CKPT_PATH}" # ABSOLUTE path of checkpoints
echo "OPENAI_API_KEY = ${OPENAI_API_KEY}"  # needed for mmhal
echo "DEVICE = CUDA:${DEVICE}"  # cuda device id
echo "OUTPUT_DIR = ${OUTPUT_DIR}"  # cuda device id

mkdir -p ${OUTPUT_DIR}

source ../ovip/bin/activate
echo "===== Generating ObjectHalBench"
set -x
(
DST_FILE="eval_results/objecthal/answers/${RUN_NAME}.jsonl"
if [ ! -f "$DST_FILE" ]; then
    CUDA_VISIBLE_DEVICES=${DEVICE} python eval_vlm/main.py --answers-file ${DST_FILE} \
    --template objecthal --question-file ../benchmark/objectHal/obj_halbench_300_with_image.jsonl \
    --model-base ${MODEL_BASE} --model-path ${CKPT_PATH} --max_new_tokens 384
else
    echo "File ${DST_FILE} already exists. Skipping these steps."
fi
source ../amber/bin/activate
python './eval_vlm/llava/eval/objectHal_gpt_eval.py' --cap_file ${DST_FILE} --openai_key ${OPENAI_API_KEY}
python './eval_vlm/llava/eval/summarize_gpt_obj_halbench_review.py' --cap_file "eval_results/objecthal/answers/hall_${RUN_NAME}_-1.json"
deactivate
) > "${OUTPUT_DIR}/ObjectHal.txt" 2>&1 &
wait

echo "===== Generating MMHal"
(
DST_FILE="eval_results/mmhal/answers/${RUN_NAME}.jsonl"
if [ ! -f "$DST_FILE" ]; then
    CUDA_VISIBLE_DEVICES=${DEVICE} python eval_vlm/main.py --answers-file ${DST_FILE} \
    --template mmhal --image-folder ../benchmark/mmhal/images --question-file ../benchmark/mmhal/response_template.json \
    --model-base ${MODEL_BASE} --model-path ${CKPT_PATH} --max_new_tokens 384
else
    echo "File ${DST_FILE} already exists. Skipping these steps."
fi
python './eval_vlm/llava/eval/eval_gpt_mmhal_vis.py' --response ${DST_FILE} --api-key ${OPENAI_API_KEY}
) > "${OUTPUT_DIR}/MMHal.txt" 2>&1 &
wait

echo "===== Eval LLavaBench"
(
DST_FILE="eval_results/llavabench/answers/${RUN_NAME}.jsonl"
if [ ! -f "$DST_FILE" ]; then
    CUDA_VISIBLE_DEVICES=${DEVICE} python eval_vlm/main.py --answers-file ${DST_FILE} \
    --template llavabench --image-folder ./eval_vlm/playground/data/eval/llava-bench-in-the-wild/images --question-file ./eval_vlm/playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --model-base ${MODEL_BASE} --model-path ${CKPT_PATH} --max_new_tokens 1024
else
    echo "File ${DST_FILE} already exists. Skipping these steps."
fi
python ./eval_vlm/llava/eval/llavabench_gpt_review.py --question ./eval_vlm/playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --context ./eval_vlm/playground/data/eval/llava-bench-in-the-wild/context.jsonl --rule ./eval_vlm/llava/eval/llavabench_gpt_review_rule.json \
    --answer-list ./eval_vlm/playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl ${DST_FILE} \
    --output eval_results/llavabench/reviews/${RUN_NAME}.jsonl \
    --api-key ${OPENAI_API_KEY}
python ./eval_vlm/llava/eval/llavabench_gpt_review_summarize.py -f eval_results/llavabench/reviews/${RUN_NAME}.jsonl
) > "${OUTPUT_DIR}/llavabench.txt" 2>&1 &
wait

echo "===== Evaluating CVBench"
(
DST_FILE="eval_results/cvbench/answers/${RUN_NAME}.jsonl"
if [ ! -f "$DST_FILE" ]; then
    CUDA_VISIBLE_DEVICES=${DEVICE} python eval_vlm/main.py --answers-file ${DST_FILE} \
    --template cvbench --question-file ./eval_vlm/playground/data/eval/cvbench/cvbench_test.jsonl \
    --image-folder ../benchmark/cvbench/img --model-base ${MODEL_BASE} --model-path ${CKPT_PATH}
else
    echo "File ${DST_FILE} already exists. Skipping these steps."
fi
python eval_vlm/llava/eval/cvbench_rule_grader.py --answer_file "${DST_FILE}"
) > "${OUTPUT_DIR}/cvbench.txt" 2>&1 &
wait

echo "===== Evaluating AMBER generative"
(
DST_FILE="eval_results/amber_gen/answers/${RUN_NAME}.jsonl"
if [ ! -f "$DST_FILE" ]; then
    CUDA_VISIBLE_DEVICES=${DEVICE} python eval_vlm/main.py --answers-file ${DST_FILE} \
    --template amber --question-file ./eval_vlm/playground/data/eval/amber_gen/amber_gen.jsonl \
    --image-folder ../benchmark/image --model-base ${MODEL_BASE} --model-path ${CKPT_PATH} --max_new_tokens 384
else
    echo "File ${DST_FILE} already exists. Skipping these steps."
fi
python ./eval_vlm/playground/data/eval/amber_gen/transform_output2amberForm.py --target_file "${RUN_NAME}.jsonl"
FINAL_DST_FILE="eval_results/amber_gen/results/${RUN_NAME}.json"
source ../amber/bin/activate
python ./eval_vlm/playground/data/eval/amber_gen/inference.py --evaluation_type g --inference_data "${FINAL_DST_FILE}"
deactivate
) > "${OUTPUT_DIR}/amber_gen.txt" 2>&1 &
wait

source ../ovip/bin/activate
echo "===== Evaluating AMBER discriminative"
(
DST_FILE="eval_results/amber_dis/answers/${RUN_NAME}.jsonl"
if [ ! -f "$DST_FILE" ]; then
    CUDA_VISIBLE_DEVICES=${DEVICE} python eval_vlm/main.py --answers-file ${DST_FILE} \
    --template amber --question-file ./eval_vlm/playground/data/eval/amber_dis/amber_dis.jsonl \
    --image-folder ../benchmark/image --model-base ${MODEL_BASE} --model-path ${CKPT_PATH} --max_new_tokens 10
else
    echo "File ${DST_FILE} already exists. Skipping these steps."
fi
python ./eval_vlm/playground/data/eval/amber_dis/transform_output2amberForm.py --target_file "${RUN_NAME}.jsonl"
FINAL_DST_FILE="eval_results/amber_dis/results/${RUN_NAME}.json"
source ../amber/bin/activate
python ./eval_vlm/playground/data/eval/amber_gen/inference.py --evaluation_type d --inference_data "${FINAL_DST_FILE}"
deactivate
) > "${OUTPUT_DIR}/amber_dis.txt" 2>&1 &
wait

source ../ovip/bin/activate
echo "===== Evaluating RealWorldQA"
(
DST_FILE="eval_results/realworldqa/answers/${RUN_NAME}.jsonl"
if [ ! -f "$DST_FILE" ]; then
    CUDA_VISIBLE_DEVICES=${DEVICE} python eval_vlm/main.py --answers-file ${DST_FILE} \
    --template rqa --question-file ./eval_vlm/playground/data/eval/realworldqa/llava_test_realworldqa.jsonl \
    --image-folder ./eval_vlm/playground/data/eval/realworldqa/images --model-base ${MODEL_BASE} --model-path ${CKPT_PATH}
else
    echo "File ${DST_FILE} already exists. Skipping these steps."
fi
python eval_vlm/llava/eval/realworldqa_rule_grader.py --answer_file "${DST_FILE}"
) > "${OUTPUT_DIR}/realworldqa.txt" 2>&1 &
wait

echo "===== Evaluating TextVQA"
(
DST_FILE="eval_results/textvqa/answers/${RUN_NAME}.jsonl"
if [ ! -f "$DST_FILE" ]; then
    CUDA_VISIBLE_DEVICES=${DEVICE} python eval_vlm/main.py --answers-file ${DST_FILE} \
    --template tqa --question-file ./eval_vlm/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./eval_vlm/playground/data/eval/textvqa/train_images --model-base ${MODEL_BASE} --model-path ${CKPT_PATH}
else
    echo "File ${DST_FILE} already exists. Skipping these steps."
fi
cd eval_vlm
python -m llava.eval.textvqa_grader \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file "${DST_FILE}"
cd ../
) > "${OUTPUT_DIR}/textvqa.txt" 2>&1 &
wait

echo "===== Evaluating MMStar"
(
DST_FILE="eval_results/mmstar/answers/${RUN_NAME}.jsonl"
if [ ! -f "$DST_FILE" ]; then
    CUDA_VISIBLE_DEVICES=${DEVICE} python eval_vlm/main.py --answers-file ${DST_FILE} \
    --template mmstar --question-file ../benchmark/MMStar  \
    --model-base ${MODEL_BASE} --model-path ${CKPT_PATH}
else
    echo "File ${DST_FILE} already exists. Skipping these steps."
fi
cd eval_vlm
python -m llava.eval.mmstar_rule_grader \
    --answer_file "${DST_FILE}"
cd ../
) > "${OUTPUT_DIR}/mmstar.txt" 2>&1 &
wait