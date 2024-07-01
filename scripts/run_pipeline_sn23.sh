#!/bin/bash

# Set parameters
# ROOT_DIR="datasets/WC-2014"
ROOT_DIR="datasets/calibration-2023/"
SPLIT="test"
CFG="config/hrnetv2_w48.yaml"
CFG_L="config/hrnetv2_w48_l.yaml"
WEIGHTS_KP="weights/MV_kp"
WEIGHTS_L="weights/MV_lines"
SAVE_DIR="inference/inference_3D/inference_sn23/"
DEVICE="cuda:0"
GT_FILE="${SAVE_DIR}${SPLIT}.zip"
PRED_FILE="${SAVE_DIR}${SPLIT}_pred.zip"


# Run inference script
python scripts/inference_sn.py --cfg $CFG --cfg_l $CFG_L --weights_kp $WEIGHTS_KP --weights_line $WEIGHTS_L --root_dir $ROOT_DIR --split $SPLIT --save_dir $SAVE_DIR --cuda $DEVICE

# Run evaluation script
python sn_calibration/src/evalai_camera.py -s $GT_FILE  -p $PRED_FILE 
