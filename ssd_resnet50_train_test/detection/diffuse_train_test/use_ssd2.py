#==========================================
#to test the SSD detection
#==========================================
import numpy as np 
import matplotlib.pyplot as plt 
import cv2
import time 
import os


#load caffe 
caffe_root = "/home/gaia/caffe/"
import caffe
import math 


#set up GPU
caffe.set_device(1) #use the first GPU
caffe.set_mode_gpu() 

#load the net
model_root = "/home/gaia/data/eyeground/SSD/SSD_300x300/"
model_def = model_root + "disc_filter_deploy.prototxt" #"deploy.prototxt"  
model_weights = model_root + "VGG_eye_SSD_300x300_iter_20000.caffemodel"
#model_weights = "/home/gaia/data/eyeground/trainedModels/" + "VGG_eye_SSD_300x300_iter_10000.caffemodel"

net = caffe.Net(model_def, model_weights, caffe.TEST) #use test mode


#input preprocessing: 'data' is the name of the input blob == net.input[0]

#load the mean ImageNet image
mu = np.load(caffe_root + "python/caffe/imagenet/ilsvrc_2012_mean.npy")
mu = mu.mean(1).mean(1) #average over pixels to obtain the mean pixel value, BGR respectively
                        #it's B: 104.01, G: 116.67, R: 122.68  

#preprocessing
transformer = caffe.io.Transformer({'data': net.blobs["data"].data.shape})  
transformer.set_transpose('data', (2,0,1)) #move image channels to outermost dimension
transformer.set_mean('data', mu) #subtract the dataset-mean value in each channel 
transformer.set_raw_scale('data', 255) #rescale from [0,1] to [0,255] 
transformer.set_channel_swap('data',(2,1,0)) #swap trannels from RGB to BGR 

#set the size of input here if you want 
batch_size = 1
image_resize = 300
channels = 3
net.blobs['data'].reshape(batch_size, channels, image_resize, image_resize)  


#measure the time for processing an image 
#imgfile = "/mnt/data/eyeground/test.txt"
imgPath = '/home/gaia/share/glaucomaData_filtered_kan20170825_final/cls/healthy/'
#imgPath = '/home/gaia/share/glaucomaData_filtered_kan20170825_final/cropped_cls/healthy/'
#imgNum = 1
#imgDir = "/mnt/lijian/mount_out/tmp_test/1000_left.jpeg" 

#with open(imgfile, 'r') as f: 
#   data = f.read().splitlines() 
imgnames = os.listdir(imgPath)
imgNum = len(imgnames)   

time_cost_total = 0
i = 0
correct_num = 0

#process the images one by one 
for name in imgnames[0:2]:

#tart = time.clock() 
#start = time.time() #it is the wall-time  
    start = time.time() #it is the wall-time  
    #load an image 
    #imgDir = '/home/gaia/data/eyeGroundCommonTest/JPEGImages_3class/000016.jpeg'  #test specific image
    imgDir = imgPath + name 
    #savefile = '/home/gaia/share/work/u-net/inter_results/' + name 
    savefile = '/home/gaia/share/glaucomaData_filtered_kan20170825_final/ssd_crop_resized/healthy/' + name 
    image = caffe.io.load_image(imgDir)  
    #print "image size: ", image.shape, 'data type: ', image.dtype, "\n"
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

    #check the output
    tempout = output['detection_out']
    print 'all dim: ', tempout.shape
    detection_result = output['detection_out'][0,0,:,:]  #output format: [image_ID, label, score, xmin, ymin, xmax, ymax]
    print 'dim: ', detection_result.shape
    #print "blob size: ", detection_result.shape, '\n' 
    sorted_result = detection_result[detection_result[:,2].argsort()[::-1]] #sorted w.r.t. the confidence 
    #print "the sorted scores: ", sorted_result[:,2], "\n"
    discBox = sorted_result[0,3:7]
    discBox[0] *= image.shape[1] #columns <-> x
    discBox[1] *= image.shape[0] #rows <-> y
    discBox[2] *= image.shape[1]
    discBox[3] *= image.shape[0]

    #enlarge the area by 40%  
    #discBox[0] *= 0.7 
    #discBox[1] *= 0.7 
    #discBox[2] *= 1.3 
    #discBox[3] *= 1.3

    
    srcimg = cv2.imread(imgDir, cv2.IMREAD_UNCHANGED)
    print imgDir
   


    #print "before: ", discBox, '\n'
    discBox = np.int32(discBox)
    #print "after: ", discBox, '\n'  
    #print 'the box: ', discBox, '\n'

    #construct to a mask 
    #maskImg = np.zeros([image.shape[0], image.shape[1]],np.uint8)
    #maskImg[discBox[1]:discBox[3], discBox[0]:discBox[2]] = 255
    #cv2.namedWindow("mask", cv2.WINDOW_NORMAL) 
    #cv2.imshow("mask", maskImg) 
    #srcimg = cv2.imread(imgDir, cv2.IMREAD_UNCHANGED)
    print savefile
    cv2.imwrite(savefile, srcimg[discBox[1]:discBox[3], discBox[0]:discBox[2], :])
    #cv2.imwrite(savefile, cut_image)   
    

    #imgshow = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #imgshow = cv2.rectangle(imgshow, (discBox[0],discBox[1]), (discBox[2],discBox[3]), (0,255,0), 3)
    #cv2.namedWindow("show", cv2.WINDOW_NORMAL)
    #cv2.imshow("show", imgshow)   
    #cv2.waitKey(0)   
    ####boxNum = sorted_result.shape[0] 
    #end = time.time()
    #time_cost = end - start
    #time_cost_total = time_cost_total + time_cost


    #boxNum = sorted_result.shape[0] 
    #if boxNum > 0: 
    #   confidence_of_bestBox = sorted_result[0,2]
    #   if confidence_of_bestBox > 0.5:
    #      correct_num += 1           
    #   else:
    #      print '\n', "Failed:", imgDir, '\n'

    #index
    i = i+1
    print i
#    break 


#print '\n', "detection rate: ", np.float32(correct_num) / imgNum, '(', correct_num, '/', imgNum, ')', '\n' 
#print "total time:", time_cost_total, '\n', "time cost per image: %d", time_cost_total / imgNum, '\n' 

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
 
