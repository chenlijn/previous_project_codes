#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

#EXAMPLE=/home/gaia/share/cataractData/codes/lmdb_imaging_types
#DATA=/home/gaia/share/cataractData/codes/lmdb_imaging_types
EXAMPLE=/home/gaia/share/cataractData/codes/lmdb_severity
DATA=/home/gaia/share/cataractData/codes/lmdb_severity
TOOLS=/home/gaia/caffe/build/tools

$TOOLS/compute_image_mean $EXAMPLE/severity_slit_train_val_lmdb\
  $DATA/severity_slit_mean.binaryproto

echo "Done."
