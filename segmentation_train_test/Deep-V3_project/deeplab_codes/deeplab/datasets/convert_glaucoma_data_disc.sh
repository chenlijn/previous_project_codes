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
# Script to download and preprocess the PASCAL VOC 2012 dataset.
#
# Usage:
#   bash ./download_and_convert_voc2012.sh
#
# The folder structure is assumed to be:
#  + datasets
#     - build_data.py
#     - build_voc2012_data.py
#     - download_and_convert_voc2012.sh
#     - remove_gt_colormap.py
#     + pascal_voc_seg
#       + VOCdevkit
#         + VOC2012
#           + JPEGImages
#           + SegmentationClass
#

# Exit immediately if a command exits with a non-zero status.
set -e

# Build TFRecords of the dataset.
# First, create output directory for storing TFRecords.
WORK_DIR=/root/mount_out/work2018/private_glaucoma_recognition/Deep-V3_project/deeplab_codes/deeplab
OUTPUT_DIR=/root/mount_out/data/glaucoma_disc_segmentation_sorted_data_full_20180917/tfrecords
mkdir -p "${OUTPUT_DIR}"

# convert all images
IMAGE_FOLDER=/root/mount_out/data/glaucoma_disc_segmentation_sorted_data_full_20180917/trainingset_final/images

SEMANTIC_SEG_FOLDER=/root/mount_out/data/glaucoma_disc_segmentation_sorted_data_full_20180917/trainingset_final/cleaned_masks


LIST_FOLDER=/root/mount_out/data/glaucoma_disc_segmentation_sorted_data_full_20180917/lists
mkdir -p "${LIST_FOLDER}"

echo "Converting glaucoma dataset..."
python ./build_voc2012_data.py \
  --image_folder="${IMAGE_FOLDER}" \
  --semantic_segmentation_folder="${SEMANTIC_SEG_FOLDER}" \
  --list_folder="${LIST_FOLDER}" \
  --image_format="jpg" \
  --label_format="png" \
  --output_dir="${OUTPUT_DIR}"
