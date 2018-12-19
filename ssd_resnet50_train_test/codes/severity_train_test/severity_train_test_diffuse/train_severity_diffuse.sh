
cd /home/gaia/caffe
./build/tools/caffe train \
	-solver  "/home/gaia/share/cataractData/codes/resnet-protofiles-master/solver_severity_diffuse.prototxt" \
	-weights "/home/gaia/share/cataractData/codes/OneDrive-2017-08-22/ResNet-50-model.caffemodel"
        -gpu 0 
        #2>&1 | tee /home/gaia/share/cataractData/codes/severity_train_logs.log   	
         



	#-snapshot "/home/gaia/share/cataractData/codes/snapshots/resnet50_severity_iter_14000.solverstate" \
	#-weights  "/home/gaia/share/cataractData/codes/trained_models/resnet_iter_diag_40000.caffemodel" \
