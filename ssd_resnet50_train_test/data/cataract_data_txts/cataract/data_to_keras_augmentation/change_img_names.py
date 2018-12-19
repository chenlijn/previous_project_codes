# coding: utf-8

import os
import sys
import numpy
import cv2
import base64
from hashlib import md5

input_file='val_img_list.txt' #image list without label 
#input_file='test.txt'

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
            img = cv2.imread(imgfile)
            md5_num = md5(imgstr).hexdigest()
            dst_name = path + '/' + md5_num + '.jpg'
            print imgfile 
            #print dst_name
            
            os.rename(imgfile.strip('\n'), dst_name)
            #print 'path: {}\n imgname: {}\n md5: {}\n'.format(path, imgname, md5_num)


