#!/bin/bash

# run pope eval
CUDA_VISIBLE_DEVICES=0 \
python pope_eval.py \
--model "llava-1.5" \
--data-path 'path to coco val2014' \
--pope-type 'random' \
--vsv \
--lamda 0.01 \
--logits_aug \
--logits_alpha 0.3 \


# # read the result file
# python pope_ans.py \
# --ans_file 'path to result file' \