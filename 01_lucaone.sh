#!/usr/bin/env bash
set -euo pipefail

# Default values (can be overridden by CLI args)
INPUT_FILE="./UniVH-model-main/test/PV241319.1_nt.fasta"
SAVE_PATH="./UniVH-model-main/test/embedding"
GPU_ID="0"

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input_file) INPUT_FILE="$2"; shift 2 ;;
    --save_path)  SAVE_PATH="$2"; shift 2 ;;
    --gpu_id)     GPU_ID="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

cd ./UniVH-model-main/LucaOneApp/LucaOneApp-master/algorithms

python inference_embedding_lucaone.py \
    --llm_dir ./UniVH-model-main/LucaOneApp/models \
    --llm_type lucaone_gplm \
    --llm_version v2.0 \
    --llm_task_level token_level,span_level,seq_level,structure_level \
    --llm_time_str 20231125113045 \
    --llm_step 5600000 \
    --truncation_seq_length 100000 \
    --trunc_type right \
    --seq_type gene \
    --input_file "${INPUT_FILE}" \
    --save_path "${SAVE_PATH}" \
    --embedding_type vector \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --embedding_fixed_len_a_time 1000 \
    --gpu_id "${GPU_ID}"
