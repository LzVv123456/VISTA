#!/bin/bash

# run pope eval
CUDA_VISIBLE_DEVICES=0 \
python pope_eval.py \
--model "llava-1.5" \
--data-path 'YOUR_PATH_TO_COCO' \
--pope-type 'random' \
--vsv \
--vsv-lambda 0.01 \
--logits-aug \
--logits-alpha 0.3 \


# # read the result file
# python pope_ans.py \
# --ans_file 'path to result file' \