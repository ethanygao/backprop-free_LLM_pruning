#!/bin/bash

# Get all parameters passed to the script
GPUS="$@"

# Check if GPU parameters are passed
if [ -z "$GPUS" ]; then
  echo "Usage: $0 <gpu1> [gpu2 gpu3 ...]"
  exit 1
fi

# Splices GPU parameters into a string, separated by commas','
GPU_STRING=$(IFS=,; echo "$GPUS")

# Define an array with the rates you want to use
rates=("0.70")

# Loop over the rates
for rate in "${rates[@]}"
do
  CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES="$GPU_STRING" python prune_channel/main.py \
  --model-name meta-llama/Meta-Llama-3-8B \
  --init-data-name c4 \
  --train-data-name c4 \
  --test-data-name wikitext2 \
  --dataset-size 120000 \
  --train-batch-size 8 \
  --test-batch-size 8 \
  --max-seq-length 128 \
  --init-seqlen 1024 \
  --nsamples 128 \
  --n-workers 1 \
  --init-type wanda-sp \
  --score-from-metric sigmap \
  --prune-rate-start $rate \
  --prune-rate-target $rate \
  --prune-start-iter-percentage 0.0 \
  --prune-end-iter-percentage 0.1 \
  --init-rate $rate \
  --attn-score-lr 0.006 \
  --mlp-score-lr 0.006 \
  --score-init-constant 1.0 \
  --ma-window-size 5 \
  --eval-per-steps 1000 \
  --exp-name "wanda-sp-sigmap/llama3-8b<ds=12e4,lr=6e-3,initseqlen=1024>" \
  --save-folder exp \
  --not-save-model \
  --penalty-lamda-init -1 \
  --penalty-lamda-final 0.0 \
  --K 2
done