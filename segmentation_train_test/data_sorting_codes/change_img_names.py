# coding: utf-8

import os
import sys
import numpy
import cv2
import base64
from hashlib import md5

#input_file='val_img_list.txt' #image list without label 
#input_file='/mnt/lijian/mount_out/data/glaucoma_segmentation_imgs/sorted_disc/img_list.txt'
input_file = '/root/mount_out/data/original_glaucoma_data/all_data_sorted/glaucoma_cls_test_set/serious_docker.txt'
#input_file='test.txt'
#mask_dir = '/mnt/lijian/mount_out/data/glaucoma_segmentation_imgs/sorted_disc/masks/'

with open(input_file,'r') as f:
    img_list = f.readlines()
    #print img_list

    for imgfile in img_list:
        with open(imgfile.strip('\n'), 'rb') as imgf:
            imgstr = base64.b64encode(imgf.read())

            #print imgfile
    #        print "hello world"
            path = os.path.dirname(imgfile)
            #print path
            imgname = imgfile.split('/')[-1]
            #img = cv2.imread(imgfile)
            md5_num = md5(imgstr).hexdigest()
            dst_name = path + '/' + md5_num + '.jpg'
            print(imgfile) 
            #print dst_name
            
            os.rename(imgfile.strip('\n'), dst_name)
            #os.rename(mask_dir+imgname.split('.')[0]+'.png', mask_dir + md5_num + '.png')
            #print 'path: {}\n imgname: {}\n md5: {}\n'.format(path, imgname, md5_num)


