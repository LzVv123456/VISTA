#!/bin/bash

# # run chair eval
# CUDA_VISIBLE_DEVICES=0 \
# python chair_eval.py \
# --model "llava-1.5" \
# --data-path '../download_datasets/COCO_2014/val2014' \
# --vsv \
# --vsv-lambda 0.17 \
# --logits-aug \
# --logits-alpha 0.3 \


# read the result file
python chair_ans.py \
--cap_file 'exp_results/chair_eval/llava-1.5/seed1994_vsv_lambda_0.17_logaug_loglayer_25,30_logalpha_0.3_greedy_max_new_tokens_512.jsonl' \