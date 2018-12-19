
This is the glaucoma project including U-net, Deep-V3, M-Net. 

Execute download\_models.sh to get trained models.  
  
# 1. data\_sorting\_codes  
It contains codes for sorting and preparing data for training and testing.   
There codes for optic disc detection, enhancement, data augmentation, disc-splitting, drawing masks and other utilities.   
  
Use ***disc\_detection.py*** to detect optic disc and crop it out from original images via yolov3.   
Use ***sort\_cup\_data/split\_discs\_by\_mask.py*** to split double-disc images into one-disc images.  
Use ***sort\_cup\_data/keras\_data\_aug.py*** to do data augmentation.  
Use ***sort\_cup\_data/change\_img\_names.py*** to change image names to md5 names.  
Use ***sort\_cup\_data/draw\_img\_names.py*** to change image names to md5 names.  
Use codes in ***sort\_cup\_data/sort\_REFUGE\_data/*** to prepare the REFUGE dataset for training and testing.  
  
  
# 2. U-net\_project  
codes for training and testing U-Net with caffe.   

# 3. M-Net\_project  
It contains codes preparing data, training and testing the M-Net. It comes from the interns in Airdoc Shanghai Office.   


# 4. DeepLab-v3\_project 
This project depends on codes in ***deeplab\_codes/slim***. Thus, the slim folder should be included. For more information of DeepLab, see: https://github.com/tensorflow/models/tree/master/research/deeplab.   
  
## 4.1 How to prepare training and validation data for DeepLabV3?
Data preparation codes are under ***deeplab\_codes/deeplab/datasets***. The data will be prepared as tfrecord format. After generating the tfrecord format data, you can use ***deeplab\_codes/deeplab/test\_tfrecord.py*** to check the data.  

Use ***convert\_glaucoma\_val\_data\_cup.sh*** to prepare cup segmentation training and validation data.  
Use ***convert\_glaucoma\_val\_data\_disc.sh*** to prepare disc segmentation training and validation data.  
  
Specify the number of classes of the ***pascal\_voc part in segmentation.py***, because we reuse the pascal\_voc codes here    
Modify the ***local\_train\_glaucoma.sh*** to train your own data    
More parameters of learning policies can be found in ***train.py***    
More modifications can be made in ***build\_voc2012\_data.py*** and ***build\_data.py***, such as add image preprocessing codes.  
  
  
## 4.2 procedure to train  

Under ***deeplab\_codes/deeplab/***  
    
Use ***local\_train\_glaucoma\_disc.sh*** to train the disc segmentation model.  
Use ***local\_train\_glaucoma\_cup.sh*** to train the disc segmentation model.  
  
However, only one model can be trained at one time due to the codes and model are the same.  
  

## 4.3 Deploy
After finishing training and frozeeing the models, you can deploy the frozen models with the codes under ***deeplab\_codes/deeplab/deploy/***.  

Use ***deploy\_deeplab\_frozen\_pb\_with\_gt.py*** to test images with grouth truth.  
Use ***deploy\_deeplab\_frozen\_pb\_with\_gt\_REFUGE.py*** to REFUGE images with grouth truth.  
  
Use ***deploy\_to\_test.py*** to test images without grouth truth.  
Use ***deploy\_to\_test\_REFUGE.py*** to test REFUGE images without grouth truth.  
  

# 5. generate\_pku\_paper\_materials

## 5.1 procedure to test images and generate the materials
**step 1:** Deploy yolov3 to detect and crop disc area from original images as well as masks. Store somewhere.   
**step 2:** Deploy trained DeepLabV3 to segment out disc and cup. Store somewhere.     
**step 3:** Deploy ***eye-side classifier*** to classify each disc into right-eye or left-eye, and store the results as txt file. See: http://172.16.0.2/chenlijian/eye-side-classification/tree/master    
**step 4:** Make three folders: good\_case, bad\_case, csv.  
**step 5:** Use ***compute\_cup\_disc\_ratio\_with\_eyeside.py*** to compute cup-to-disc ratios and generate CSV files.  

To test REFUGE dataset, use ***compute\_cup\_disc\_ratio\_with\_eyesidei\_REFUGE.py***.  

You need to execute eye-side classification if you want to compute ISNT. If not, you can modify ***compute\_cup\_disc\_ratio\_with\_eyeside.py*** and ***compute\_cup\_disc\_ratio\_with\_eyeside\_REFUGE.py*** to ignore ISNT computing.   

If you want to re-train yolov3 detection, see: http://172.16.0.2/chenlijian/yolov3\_disc\_detection/tree/master.  

