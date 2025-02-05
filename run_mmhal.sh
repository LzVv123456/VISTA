#!/bin/bash

# run chair eval
CUDA_VISIBLE_DEVICES=0 \
python mmhal_eval.py \
--model "llava-1.5" \
--vsv \
--vsv-lambda 0.1 \
--logits-aug \
--logits-alpha 0.3 \

# # read the result file
# python mmhal_ans.py \
# --response 'path to result file' \