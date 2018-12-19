#==========================================
#to test the SSD detection
#==========================================
import numpy as np 
import matplotlib.pyplot as plt 
import time 

#set display defaults
plt.rcParams["figure.figsize"]=(10,10) 
plt.rcParams["image.interpolation"]="nearest" 
plt.rcParams["image.cmap"]="gray"  

#load caffe 
caffe_root = "/home/gaia/caffe/"
import caffe


#set up GPU
caffe.set_device(2) #use the first GPU
caffe.set_mode_gpu() 

#load the net
model_root = "/home/gaia/data/eyeground/SSD/SSD_500x500/"
model_def = model_root + "disc_filter_deploy.prototxt"  
model_weights = model_root + "VGG_eye_SSD_500x500_iter_15000.caffemodel"

net = caffe.Net(model_def, model_weights, caffe.TEST) #use test mode


#input preprocessing: 'data' is the name of the input blob == net.input[0]
#load the mean ImageNet image
mu = np.load(caffe_root + "python/caffe/imagenet/ilsvrc_2012_mean.npy")
mu = mu.mean(1).mean(1) #average over pixels to obtain the mean pixel value, BGR respectively
                        #it's B: 104.01, G: 116.67, R: 122.68  

transformer = caffe.io.Transformer({'data': net.blobs["data"].data.shape})  
transformer.set_transpose('data', (2,0,1)) #move image channels to outermost dimension
transformer.set_mean('data', mu) #subtract the dataset-mean value in each channel 
transformer.set_raw_scale('data', 255) #rescale from [0,1] to [0,255] 
transformer.set_channel_swap('data',(2,1,0)) #swap trannels from RGB to BGR 

#set the size of input here if you want 
batch_size = 1
image_resize = 500
channels = 3
net.blobs['data'].reshape(batch_size, channels, image_resize, image_resize)  


#measure the time for processing an image 
imgfile = "/mnt/data/eyeground/test.txt"
#imgNum = 1
#imgDir = "/mnt/lijian/mount_out/tmp_test/1000_left.jpeg" 

with open(imgfile, 'r') as f: 
   data = f.read().splitlines() 
imgNum = len(data) 

time_cost_total = 0
i = 0
correct_num = 0

#process the images one by one 
for imgDir in data:

#tart = time.clock() 
#start = time.time() #it is the wall-time  
    #load an image 
    imgDir = '/home/gaia/data/eyeGroundCommonTest/JPEGImages_3class/000016.jpeg' #test one image
    start = time.time() #it is the wall-time  
    image = caffe.io.load_image(imgDir)  
#    start = time.time() #it is the wall-time  
    transformed_image = transformer.preprocess('data', image)  
    #plt.imshow(image)  

    #copy the image into the memory allocated for the net 
    net.blobs['data'].data[0] = transformed_image 


    #start = time.time() #it is the wall-time  
    #perform detection, output format: [image_ID, label, score, xmin, ymin, xmax, ymax ] 
    output = net.forward()
    #end = time.clock() 
    #end = time.time() #wall-time
    #time_cost = end - start
    #time_cost_total = time_cost_total + time_cost  

    #-----test-----
    #--------------

    #check the output
    detection_result = output['detection_out'][0,0,:,:]  #output format: [image_ID, label, score, xmin, ymin, xmax, ymax]
    sorted_result = detection_result[detection_result[:,2].argsort()[::-1]] #sorted w.r.t. the confidence 
    end = time.time() #wall-time
    time_cost = end - start
    time_cost_total = time_cost_total + time_cost  

    boxNum = sorted_result.shape[0] 
    if boxNum > 0: 
       confidence_of_bestBox = sorted_result[0,2]
       if confidence_of_bestBox > 0.5:
          correct_num += 1           
       else:
          print '\n', "Failed:", imgDir, '\n'

    #index
    i = i+1
    print i


print '\n', "detection rate: ", np.float32(correct_num) / imgNum, '(', correct_num, '/', imgNum, ')', '\n' 
print "total time:", time_cost_total, '\n', "time cost per image: %d", time_cost_total / imgNum, '\n' 

#calculate the IOU 
#detection_result = output['detection_out'][0,0,:,:]  #output format: [image_ID, label, score, xmin, ymin, xmax, ymax]


#print detection_result.shape
#print "\n"
#print detection_result
#print '\n'



#print '\n' 
#sorted_result = detection_result[detection_result[:,2].argsort()[::-1]]  
#print sorted_result 

#print detection_result[0,0,:,0]
#print '\n'
#print detection_result[0,0,:,2], '\n' 

#print'\n' 
#print output.keys(), '\n', output.values()   
 
