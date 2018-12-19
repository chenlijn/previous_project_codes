from tensorflow.core.framework import graph_pb2
import tensorflow as tf
from PIL import Image
import os

import numpy as np
import cv2

import deploy_frozen_yolov3_model as yolo #mport DsicDetection
from lj_utils import draw_seg_mask_on_img


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


    ## -------------glaucoma dsic-----------------------
    #data_root = '/root/mount_out/data/glaucoma_disc_segmentation_sorted_data_full_20180917/disc_2000/split/'
    ##list_file = data_root + 'glaucoma_imgs.txt'
    #model_input_size = (513, 513)
    #img_path = data_root + 'images/'
    #dst_img_path = data_root + 'detected_resized_images/'

    #mask_path = data_root + 'masks/'
    #dst_mask_path = data_root + 'detected_resized_masks/'

    #show_path = data_root + 'show/'
    ##------------------------------------------------------ 


    # -------------glaucoma cup-----------------------
    #data_root = '/root/mount_out/data/glaucoma_disc_segmentation_sorted_data_full_20180917/disc_2000/split/'
    data_root = '/root/mount_out/data/glaucoma_cup_seg_train_val_data_20180926/train_val_data/val/'
    #list_file = data_root + 'glaucoma_imgs.txt'
    model_input_size = (513, 513)
    img_path = data_root + 'images/'
    dst_img_path = data_root + 'detected_resized_images/'

    mask_path = data_root + 'masks/'
    dst_mask_path = data_root + 'detected_resized_masks/'

    show_path = data_root + 'show/'
    #------------------------------------------------------ 


    ## ----------------------------------------------------------------------------------------
    #data_root = '/root/mount_out/work2018/private_glaucoma_recognition/data/'
    #data_root = '/mnt/lijian_mount_out/data/glaucoma_disc_segmentation_sorted_data_full_20180917/disc_2000/split'
    #model_input_size = (513, 513)
    #img_type = 'healthy'
    #img_path = data_root + "test/" + img_type + "/images/"
    #dst_img_path = data_root + 'results/' + img_type + "/det/"
    ## ----------------------------------------------------------------------------------------


    show = False #True

    if not os.path.exists(dst_img_path):
        os.mkdir(dst_img_path)
    if not os.path.exists(dst_mask_path):
        os.mkdir(dst_mask_path)
    if not os.path.exists(show_path):
        os.mkdir(show_path)

    #with open(list_file, 'r') as lf:
    #    img_files = lf.readlines()

    img_files = os.listdir(mask_path)


    for imgfile in img_files:
        print(imgfile)
        imgfile = imgfile.strip('\n')
        if imgfile.endswith('.png'):
            imgname = imgfile.split('.')[0]
            mask_name = imgname + '.png'
            image_file = img_path + imgname + '.jpg'
            mask_file = mask_path + mask_name

            image = cv2.imread(image_file)
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
                
                cv2.imwrite(dst_img_path + image_file, resized_img)
                cv2.imwrite(dst_mask_path + mask_name, resized_mask)
                cv2.imwrite(show_path + mask_name, draw_seg_mask_on_img(resized_img, resized_mask))
            #else:
            #    cv2.imwrite('bad_case/' + imgfile, image)




