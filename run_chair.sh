#!/bin/bash

# run chair eval
CUDA_VISIBLE_DEVICES=0 \
python chair_eval.py \
--model "llava-1.5" \
--data-path 'YOUR_PATH_TO_COCO' \
--vsv \
--vsv-lambda 0.1 \
--logits-aug \
--logits-alpha 0.3 \


# # read the result file
# python chair_ans.py \
# --cap_file 'path to result file' \