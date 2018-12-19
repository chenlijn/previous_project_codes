
work_root=/root/caffe
$work_root/build/tools/caffe train \
	-solver  "../../resnet-protofiles-master-new/solver_illness3_split_slit.prototxt" \
	-weights "../../OneDrive-2017-08-22/ResNet-50-model.caffemodel" \
	-gpu 1 #\
        #2>&1 | tee /home/gaia/share/cataractData/codes/illness3_train_logs.log   	
         



	#-weights "/home/gaia/share/cataractData/codes/OneDrive-2017-08-22/ResNet-50-model.caffemodel" \
	#-weights "/home/gaia/share/cataractData/codes/OneDrive-2017-08-22/ResNet-50-model.caffemodel" \
	#-weights  "/home/gaia/share/cataractData/codes/trained_models/resnet_iter_diag_40000.caffemodel" \
