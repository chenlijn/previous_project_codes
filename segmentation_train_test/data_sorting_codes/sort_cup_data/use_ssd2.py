#==========================================
#to test the SSD detection
#==========================================
import numpy as np 
import matplotlib.pyplot as plt 
import time 
import os
import math 

import cv2
import caffe

class SSDDetection(object):

    def __init__(self):
        #load the net
        self._model_root = "/home/gaia/data/eyeground/SSD/SSD_300x300/"
        self._model_def = self._model_root + "disc_filter_deploy.prototxt" #"deploy.prototxt"  
        self._model_weights = self._model_root + "VGG_eye_SSD_300x300_iter_20000.caffemodel"

        #set up GPU
        caffe.set_device(1) #use the first GPU
        caffe.set_mode_gpu() 
        self._net = caffe.Net(self._model_def, self._model_weights, caffe.TEST) #use test mode

        #load the mean ImageNet image
        self._caffe_root = "/home/gaia/caffe/"
        mu = np.load(self._caffe_root + "python/caffe/imagenet/ilsvrc_2012_mean.npy")
        mu = mu.mean(1).mean(1) #average over pixels to obtain the mean pixel value, BGR respectively

        #preprocessing
        self._transformer = caffe.io.Transformer({'data': self._net.blobs["data"].data.shape})  
        self._transformer.set_transpose('data', (2,0,1)) #move image channels to outermost dimension
        self._transformer.set_mean('data', mu) #subtract the dataset-mean value in each channel 
        self._transformer.set_raw_scale('data', 255) #rescale from [0,1] to [0,255] 
        self._transformer.set_channel_swap('data',(2,1,0)) #swap trannels from RGB to BGR 

        #set the size of input here if you want 
        self._target_crop_size = 961
        self._box_enlarge_ratio = 0.5
        self._batch_size = 1
        self._image_resize = 300
        self._channels = 3
        self._net.blobs['data'].reshape(self._batch_size, self._channels, self._image_resize, self._image_resize)  

        return

    def disc_detection(self, img_path, dst_path, mask_path=None, mask_dst=None):
        imgnames = os.listdir(mask_path)
        imgNum = len(imgnames)   
        
        time_cost_total = 0
        i = 0
        correct_num = 0
        
        #process the images one by one 
        for name in imgnames:
            if name.endswith('.png'):
                print name 
                img_file = img_path + name.split('.')[0] + '.jpg' 
                image = caffe.io.load_image(img_file)  
                transformed_image = self._transformer.preprocess('data', image)  
                self._net.blobs['data'].data[0] = transformed_image 
                output = self._net.forward()
        
                #check the output
                tempout = output['detection_out']
                detection_result = output['detection_out'][0,0,:,:]  #output format: [image_ID, label, score, xmin, ymin, xmax, ymax]
                sorted_result = detection_result[detection_result[:,2].argsort()[::-1]] #sorted w.r.t. the confidence 
                discBox = sorted_result[0,3:7]
                discBox[0] *= image.shape[1] #columns <-> x
                discBox[1] *= image.shape[0] #rows <-> y
                discBox[2] *= image.shape[1]
                discBox[3] *= image.shape[0]

                # enlarge the disc region
                bw = discBox[2] - discBox[0]
                bh = discBox[3] - discBox[1]
                discBox[0] -= bw * self._box_enlarge_ratio
                discBox[1] -= bh * self._box_enlarge_ratio
                discBox[2] += bw * self._box_enlarge_ratio
                discBox[3] += bh * self._box_enlarge_ratio

                #if bw < self._target_crop_size: 
                #    extra_w = (self._target_crop_size - bw)/2 
                #    discBox[0] -= extra_w
                #    discBox[2] += extra_w 
                #if bh < self._target_crop_size:
                #    extra_h = (self._target_crop_size - bh)/2
                #    discBox[1] -= extra_h
                #    discBox[3] += extra_h 
                
                
                discBox[0] = max(0, discBox[0])
                discBox[1] = max(0, discBox[1])
                discBox[2] = min(image.shape[1], discBox[2])
                discBox[3] = min(image.shape[0], discBox[3])
                
                discBox = np.int32(discBox)
                srcimg = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
                savefile = dst_path + name 
                cv2.imwrite(savefile, srcimg[discBox[1]:discBox[3], discBox[0]:discBox[2], :])

                #save mask image
                if mask_path != None:
                    mask_file = mask_path + name.split('.')[0] + '.png' 
                    mask_img = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
                    savefile = mask_dst + name
                    cv2.imwrite(savefile, mask_img[discBox[1]:discBox[3], discBox[0]:discBox[2]])
        return

        
if __name__=='__main__':

    detector = SSDDetection()
    version = 'v2'
    img_type = 'abnormal'
    #img_path = '/mnt/lijian/mount_out/data/online_glaucoma_imgs_labeled_20180307/201802_data2label_v1/glaucoma_data2label_v1-bing/' + img_type + '/'
    img_path = '/mnt/lijian/mount_out/data/online_glaucoma_imgs_labeled_20180307/201802_data2label_v2/glaucoma_data2label_v2/' + img_type + '/'
    dst_path = '/mnt/lijian/mount_out/data/online_glaucoma_imgs_labeled_20180307/sorted_data/' + version + '_resized/images_' + img_type + '/' 
    mask_path = '/mnt/lijian/mount_out/data/online_glaucoma_imgs_labeled_20180307/sorted_data/mask_from_xml/' + version + '/' + img_type + '/' 
    mask_dst = '/mnt/lijian/mount_out/data/online_glaucoma_imgs_labeled_20180307/sorted_data/' + version + '_resized/' + img_type + '/' 

    detector.disc_detection(img_path, dst_path, mask_path, mask_dst) 
    
