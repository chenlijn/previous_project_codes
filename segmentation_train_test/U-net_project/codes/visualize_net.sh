#!/bin/bash
caffe_root=/home/gaia/caffe 
#./python/draw_net.py  /home/gaia/share/work/u-net/dense_unet_train2.prototxt  /home/gaia/share/work/u-net/visualization.png
#./python/draw_net.py  /home/gaia/share/work/u-net/disc_cup_seg_train.prototxt  /home/gaia/share/work/u-net/visualization.png
#./python/draw_net.py  /home/gaia/share/work/u-net/phseg_v5-train-scratch.prototxt  /home/gaia/share/work/u-net/visualization.png
$caffe_root/python/draw_net.py ../models/dense_unet_train2.prototxt visualization.png
#disc_cup_seg_train.prototxt
