
#cd /home/gaia/caffe
work_root=/root/caffe
$work_root/build/tools/caffe train \
	-solver  "../../resnet-protofiles-master-new/solver_illness3_split_diff.prototxt" \
	-weights "../../OneDrive-2017-08-22/ResNet-50-model.caffemodel" \
	-gpu 0 #\
        #2>&1 | tee /home/gaia/share/cataractData/codes/illness3_train_logs.log   	
         



	#-weights "/home/gaia/share/cataractData/codes/OneDrive-2017-08-22/ResNet-50-model.caffemodel" \
	#-snapshot "/home/gaia/share/cataractData/codes/snapshots_dec/resnet_illness3_diffuse_iter_150000.solverstate" \
	#-snapshot "/home/gaia/share/cataractData/codes/snapshots/resnet_illness_iter_50000.solverstate" \
	#-weights "/home/gaia/share/cataractData/codes/OneDrive-2017-08-22/ResNet-50-model.caffemodel" \
	#-weights  "/home/gaia/share/cataractData/codes/trained_models/resnet_iter_diag_40000.caffemodel" \
