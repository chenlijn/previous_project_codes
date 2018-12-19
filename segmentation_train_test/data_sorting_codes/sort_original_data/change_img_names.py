# coding: utf-8

import os
import sys
import numpy
import cv2
import base64
from hashlib import md5

#input_file='val_img_list.txt' #image list without label 
#input_file='test.txt'

#work_root = '/mnt/lijian/mount_out/data/original_glaucoma_data/all_data_sorted/disc_2000/disc-800/20170808-kang-disk/'
work_root = '/root/mount_out/data/original_glaucoma_data/all_data_sorted/healthy/1/'
img_dir = work_root + 'scan-normal_xing/images/'
xmls_dir = work_root + 'scan-normal_xing/xmls/'
img_list = os.listdir(img_dir)


#input_file='/mnt/lijian/mount_out/data/glaucoma_segmentation_imgs/sorted_disc/img_list.txt'
#mask_dir = '/mnt/lijian/mount_out/data/glaucoma_segmentation_imgs/sorted_disc/masks/'
#with open(input_file,'r') as f:
#    img_list = f.readlines()
#    #print img_list


for img_name in img_list:
    if not img_name.endswith('.jpg'):
        continue

    imgfile = img_dir + img_name

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
        try:
            os.rename(xmls_dir+imgname.split('.')[0]+'.xml', xmls_dir + md5_num + '.xml')
        except:
            print('no corresponding xml file !')
            continue
        #print 'path: {}\n imgname: {}\n md5: {}\n'.format(path, imgname, md5_num)


