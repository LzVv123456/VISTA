#!/bin/bash


# run mmhal eval
CUDA_VISIBLE_DEVICES=0 \
python chair_eval.py \
--model "llava-1.5" \
--vsv \
--vsv-lambda 0.17 \
--logits-aug \
--logits-alpha 0.3 \

# # read the result file
# python chair_ans.py \
# --response 'path to result file' \