from tensorflow.core.framework import graph_pb2
import tensorflow as tf
from PIL import Image
import os

import numpy as np
import cv2

import deploy_frozen_yolov3_model as yolo #mport DsicDetection


if __name__=='__main__':
    
    # set up the network
    detector = yolo.DiscDetection()
    detector.load_network()

    # deploy

    # glaucoma dsic 2018
    #image_type = 'abnormal'
    #data_root = '/root/mount_out/data/glaucoma_disc_segmentation_sorted_data_full_20180810/glaucoma_disc_20180720/'
    #list_file = data_root + image_type + '_imgs.txt'
    #model_input_size = (513, 513)
    #img_path = data_root + image_type + '/images/'
    #dst_img_path = data_root + image_type + '/detected_resized_images/'

    #mask_path = data_root + image_type + '/masks/'
    #dst_mask_path = data_root + image_type + '/detected_resized_masks/'


    # -------------glaucoma dsic 2017-----------------------
    #data_root = '/root/mount_out/data/glaucoma_disc_segmentation_sorted_data_full_20180810/disc_2017/'
    #src_root = '/root/mount_out/data/glaucomaData_filtered_kan20170825_final/seg/'
    src_root = "/root/mount_out/data/online_glaucoma_imgs_labeled_20180307/online_glaucoma_labeled_data_sorted_20180307/"
    dst_root = '/root/mount_out/data/glaucoma_cup_seg_train_val_data_20180823/'
    list_file = dst_root + 'n_train.txt'
    model_input_size = (513, 513)
    img_path = src_root + 'images_normal/' 
    dst_img_path = dst_root + 'detected_resized_images_train/'

    mask_path = src_root + 'mask_normal/'
    dst_mask_path = dst_root + 'detected_resized_masks_train/'
    #------------------------------------------------------ 

    show = False

    if not os.path.exists(dst_img_path):
        os.mkdir(dst_img_path)
    if not os.path.exists(dst_mask_path):
        os.mkdir(dst_mask_path)

    with open(list_file, 'r') as lf:
        img_files = lf.readlines()


    for imgfile in img_files:
        print(imgfile)
        imgfile = imgfile.strip('\n')
        #if imgfile.endswith('.jpg'):
        if imgfile.endswith('.png'):
            imgname = imgfile.split('.')[0]
            mask_name = imgname + '.png'
            image_file = img_path + imgfile
            mask_file = mask_path + mask_name
            print image_file

            image = cv2.imread(image_file)
            print image.shape
            mask = cv2.imread(mask_file)
            image_box = detector.remove_black_edge(image)
            xmin, ymin, xmax, ymax = image_box
            image_remove = image[ymin:ymax, xmin:xmax, :]
            mask_remove = mask[ymin:ymax, xmin:xmax, :]

            box = detector.predict(Image.fromarray(image_remove[:,:,::-1]))
            if box is not None:
                top, left, bottom, right = box
                detected_img = image_remove[top:bottom, left:right, :]
                detected_mask = mask_remove[top:bottom, left:right, :]

                # resize
                resized_img = cv2.resize(detected_img, model_input_size) 
                resized_mask = cv2.resize(detected_mask, model_input_size)
                if show:
                    resized_mask = np.where(resized_mask > 0, 255, 0)
                else:
                    resized_mask = np.where(resized_mask > 0, 1, 0)
                
                cv2.imwrite(dst_img_path + imgfile.split('.')[0] + '.jpg', resized_img)
                cv2.imwrite(dst_mask_path + mask_name, resized_mask)
            else:
                cv2.imwrite('bad_case/' + imgfile, image)




