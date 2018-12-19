this is the repository of cataract recognition.

Execute the download\_models.sh to get all trained models.  

# 1. environment
python2.7  
caffe-ssd  

# 2. data 
The data is sorted into txt files, and the txts are stored under this folder, but the absolute path of each image should be modified to suit your need.  

cataract\_data\_txts/cataract:  
    - combined\_illness\_train\_val: combined train and validation set for training illness classifier   
    - combined\_illness\_test: test set for illness classifier    
    - sorted\_original\_data\_txts: sorted original data  
    - train\_val\_split: the split txts for trainning, validation and test set.  
    - imaging\_types: sorted data for imaging types (capture mode) classifier  


cataract\_data\_txts/cataract\_severity: sorted data for training and testing severity classifier  

occlusion\_axis\_clearity\_data\_txts: sorted data for clearity, optic\_axis, occlusion classifiers' training and testing   
    - supplement\_data\_for\_illness3\_train\_test: the sorted clearity, optic\_axis, occlusion data for illness classifier training  

# 3. codes 
This folder contains codes for every task. The codes for each task are under a folder with the task name.  
   - illness3\_train\_test:  codes for training and testing illness(healthy, ill, after-surgery)  
   - imaging\_types\_train\_test: codes for training and testing imaging types.  
   - severity\_train\_test: codes for severity classifier  
   - optic\_axis: codes for optic\_axis involvement classifier  
   - clearity: clearity  
   - occlusion: occlusion  
   
   - paper\_result\_generate\_codes: codes for generating paper materials such as: ROC curve, excel files.  
   - resnet-protofiles-master-new: prototxts and solver for different tasks  

## 3.1 example
under illness3\_train\_test/illness\_diffuse\_train\_test:
    - visualize\_net.sh: visualize the prototxt    
    - create\_lmdb\_data.sh: to create lmdb data for training  
    - make\_owndata\_mean\_diffuse.sh: compute the data mean from lmdb for training  
    - convert\_binaryproto\_to\_npyfile\_diffuse\_ill.py: convert binaryproto to npy for testing  
    - train\_illness3\_diffuse.sh: train illness classifier  

## 3.2 codes in detection folder
please see the instructions on wiki. 
 
