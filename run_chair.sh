#!/bin/bash

# run chair eval
CUDA_VISIBLE_DEVICES=0 \
python chair_eval.py \
--model "llava-1.5" \
--data-path 'PATH TO COCO_2014' \
--vsv \
--lamda 0.1 \
--logits_aug \
--logits_alpha 0.3 \


# # read the result file
# python chair_ans.py \
# --cap_file 'path to result file' \