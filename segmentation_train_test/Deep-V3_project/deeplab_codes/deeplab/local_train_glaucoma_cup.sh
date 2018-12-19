#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script is used to run local test on PASCAL VOC 2012. Users could also
# modify from this script for their use case.
#
# Usage:
#   # From the tensorflow/models/research/deeplab directory.
#   sh ./local_test.sh
#
#

# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd -


# Set up the working environment.
WORK_DIR=/root/mount_out/work2018/private_glaucoma_recognition/Deep-V3_project/deeplab_codes/deeplab

DATASET_DIR="datasets"

# Set up the working directories.
EXP_FOLDER="training-cup"
#INIT_FOLDER="${WORK_DIR}/init_models/xception"
#INIT_FOLDER="${WORK_DIR}/cup_resume_models/xception"
INIT_FOLDER="${WORK_DIR}/cup_resume_models/resume"
TRAIN_LOGDIR="${WORK_DIR}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${EXP_FOLDER}/export"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

GLAUCOMA_DATASET=/root/mount_out/data/glaucoma_cup_seg_train_val_data_20180926/tfrecords


## Train 10 iterations.
##NUM_ITERATIONS=260900 # disc: number of iterations for 200 epoches and batch_size=8
#NUM_ITERATIONS=763200 # cup: number of iterations for 200 epoches and batch_size=8
###NUM_ITERATIONS=600000 #244300
#python train.py \
#  --logtostderr \
#  --train_split="train" \
#  --train_batch_size=8 \
#  --base_learning_rate=.001 \
#  --learning_rate_decay_step=500000 \
#  --learning_rate_decay_factor=0.1 \
#  --model_variant="xception_65" \
#  --atrous_rates=6 \
#  --atrous_rates=12 \
#  --atrous_rates=18 \
#  --output_stride=16 \
#  --decoder_output_stride=4 \
#  --train_crop_size=513 \
#  --train_crop_size=513 \
#  --train_batch_size=2 \
#  --training_number_of_steps="${NUM_ITERATIONS}" \
#  --fine_tune_batch_norm=true \
#  --tf_initial_checkpoint="${INIT_FOLDER}/model.ckpt" \
#  --train_logdir="${TRAIN_LOGDIR}" \
#  --dataset_dir="${GLAUCOMA_DATASET}"

# #--tf_initial_checkpoint="${INIT_FOLDER}/xception/model.ckpt" \
# Run evaluation. This performs eval over the full val split (1449 images) and
# will take a while.
# Using the provided checkpoint, one should expect mIOU=82.20%.
#NUM_ITERATIONS=572902
NUM_ITERATIONS=438706
python eval.py \
  --logtostderr \
  --eval_split="val" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --eval_crop_size=513 \
  --eval_crop_size=513 \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --eval_logdir="${EVAL_LOGDIR}" \
  --dataset_dir="${GLAUCOMA_DATASET}" \
  --max_number_of_evaluations=1

# Visualize the results.
python vis.py \
  --logtostderr \
  --vis_split="val" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --vis_crop_size=513 \
  --vis_crop_size=513 \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --vis_logdir="${VIS_LOGDIR}" \
  --dataset_dir="${GLAUCOMA_DATASET}" \
  --max_number_of_iterations=1

# Export the trained checkpoint.
CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
#CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-1031438"
EXPORT_PATH="${EXPORT_DIR}/frozen_cup_seg_inference_graph.pb"

python export_model.py \
  --logtostderr \
  --checkpoint_path="${CKPT_PATH}" \
  --export_path="${EXPORT_PATH}" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --num_classes=2 \
  --crop_size=513 \
  --crop_size=513 \
  --inference_scales=1.0

# Run inference with the exported checkpoint.
# Please refer to the provided deeplab_demo.ipynb for an example.
