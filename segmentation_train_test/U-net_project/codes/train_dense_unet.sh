caffe_root=/home/gaia/caffe
$caffe_root/build/tools/caffe train \
	-solver  "../models/solver_dense_unet.prototxt" \
	-gpu 1 



#	-weights  "/home/gaia/share/work/u-net/phseg_v5.caffemodel" \
	#-weights  "/home/gaia/share/work/u-net/snapshots/unet_scratch_iter_8000.caffemodel" \

	#-weights  "/home/gaia/share/work/u-net/snapshots/unet_scratch_iter_21000.caffemodel" \
