#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

#EXAMPLE=/home/gaia/share/cataractData/codes/lmdb_imaging_types
#DATA=/home/gaia/share/cataractData/codes/lmdb_imaging_types
#EXAMPLE=/home/gaia/share/cataractData/codes/lmdb_severity
#DATA=/home/gaia/share/cataractData/codes/lmdb_severity
EXAMPLE=/home/gaia/share/cataractData/codes/lmdb_new_dec
DATA=/home/gaia/share/cataractData/codes/lmdb_new_dec
TOOLS=/home/gaia/caffe/build/tools

$TOOLS/compute_image_mean $EXAMPLE/illness3_train_val_lmdb_slit\
  $DATA/illness3_mean_slit.binaryproto

echo "Done."
