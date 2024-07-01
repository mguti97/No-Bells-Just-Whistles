#!/bin/bash

# Set parameters
# ROOT_DIR="datasets/WC-2014"
ROOT_DIR="datasets/TS-WorldCup/"
SPLIT="test"
CFG="config/hrnetv2_w48.yaml"
CFG_L="config/hrnetv2_w48_l.yaml"
WEIGHTS_KP="weights/SV_FT_TSWC_kp"
WEIGHTS_L="weights/SV_FT_TSWC_lines"
SAVE_DIR="inference/inference_2D/inference_tswc/"
DEVICE="cuda:0"
PRED_FILE="${SAVE_DIR}${SPLIT}_pred.zip"


# Run inference script
python scripts/inference_tswc.py --cfg $CFG --cfg_l $CFG_L --weights_kp $WEIGHTS_KP --weights_line $WEIGHTS_L --root_dir $ROOT_DIR --split $SPLIT --save_dir $SAVE_DIR --cuda $DEVICE

# Run evaluation script
python scripts/eval_tswc.py --root_dir $ROOT_DIR --split $SPLIT --pred_file $PRED_FILE
