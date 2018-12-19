cd /home/gaia/caffe
./build/tools/caffe train \
	-solver  "/home/gaia/share/cataractData/codes/resnet-protofiles-master/solver_imaging_types.prototxt" \
	-snapshot  "/home/gaia/share/cataractData/codes/snapshots_dec/resnet50_imaging_types_iter_50000.solverstate" \
	-gpu 1 



	#-weights "/home/gaia/share/cataractData/codes/OneDrive-2017-08-22/ResNet-50-model.caffemodel" \
	#-weights  "/home/gaia/share/cataractData/codes/trained_models/resnet_iter_imaging_40000.caffemodel" \
