#!/bin/bash

work_root=/root/caffe 
$work_root/python/draw_net.py  ../../resnet-protofiles-master-new/ResNet50_illness3_train_val_split_diff.prototxt  ./visualization.png
