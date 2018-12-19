#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
set -e

EXAMPLE=/home/gaia/share/cataractData/codes/clearity/lmdb
#EXAMPLE=/mnt/lijian/mount_out/work/cataract_project/lmdb
DATA=/mnt/lijian/mount_out/zhongshan_ophthalmology_data20180206/imgs_for_cls_txt/combined_to_train
#DATA=/mnt/lijian/mount_out/codes/cataract/combined_illness_train_val

TOOLS=/home/gaia/caffe/build/tools

TRAIN_DATA_ROOT="" #/home/gaia/share/cataractData/cat_tv/
VAL_DATA_ROOT="" #/home/gaia/share/cataractData/cat_tv/


# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=224
  RESIZE_WIDTH=224
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

#if [ ! -d "$TRAIN_DATA_ROOT" ]; then
#  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
#  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
#       "where the ImageNet training data is stored."
#  exit 1
#fi
#
#if [ ! -d "$VAL_DATA_ROOT" ]; then
#  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
#  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
#       "where the ImageNet validation data is stored."
#  exit 1
#fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    "" \
    $DATA/clearity_train_val.txt \
    $EXAMPLE/clearity_train_val_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    "" \
    $DATA/clearity_val.txt \
    $EXAMPLE/clearity_val_lmdb

echo "Done."
