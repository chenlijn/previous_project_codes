
import os
import numpy as np
import cv2

def clean_mask(mask_img):
    rows, cols = np.where(mask_img>0)
    mask_img[rows,cols] = np.uint8(1)
    return mask_img


if __name__=="__main__":

    #src_dir= "/root/mount_out/data/glaucoma_disc_segmentation_sorted_data_full_20180810/augmented_data/masks/" #"./augmented_data_online/new_mask_abnormal/"
    #dst_dir= "/root/mount_out/data/glaucoma_disc_segmentation_sorted_data_full_20180810/augmented_data/clean_masks/" #"./augmented_data_online/new_mask_abnormal/"
    #src_dir = "/mnt/lijian/mount_out/data/glaucoma_disc_segmentation_sorted_data_full_20180810/disc_2017/detected_resized_masks/"
    #dst_dir = "/mnt/lijian/mount_out/data/glaucoma_disc_segmentation_sorted_data_full_20180810/disc_2017/new_detected_resized_masks/"

    src_dir = "/root/mount_out/data/glaucoma_cup_seg_train_val_data_20180823/detected_resized_masks_train/"
    dst_dir = "/root/mount_out/data/glaucoma_cup_seg_train_val_data_20180823/detected_resized_masks_train-1c/"

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    img_names = os.listdir(src_dir)

    for img_name in img_names:
        if img_name.endswith('.png'):
            img_file = src_dir + img_name 
            mask = cv2.imread(img_file, 0)
            # print "max:{}".format(img.max())
            #mask = img[:,:]
            new_mask = clean_mask(mask)
            save_file = dst_dir + img_name 
            cv2.imwrite(save_file, new_mask)



    
