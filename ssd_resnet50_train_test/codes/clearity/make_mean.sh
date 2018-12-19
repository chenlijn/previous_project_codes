#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

#EXAMPLE=/home/gaia/share/cataractData/codes/lmdb_imaging_types
#DATA=/home/gaia/share/cataractData/codes/lmdb_imaging_types
#EXAMPLE=/home/gaia/share/cataractData/codes/lmdb_severity
#DATA=/home/gaia/share/cataractData/codes/lmdb_severity
#EXAMPLE=/home/gaia/share/cataractData/codes/lmdb_new_dec
EXAMPLE=/home/gaia/share/cataractData/codes/clearity/lmdb
DATA=/home/gaia/share/cataractData/codes/clearity/lmdb
#EXAMPLE=/home/gaia/share/cataractData/codes/lmdb_961
#DATA=/home/gaia/share/cataractData/codes/lmdb_961
TOOLS=/home/gaia/caffe/build/tools

$TOOLS/compute_image_mean $EXAMPLE/clearity_train_val_lmdb\
  $DATA/clearity_mean.binaryproto

echo "Done."
