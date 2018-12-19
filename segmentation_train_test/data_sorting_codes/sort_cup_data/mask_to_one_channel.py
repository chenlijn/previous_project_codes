
import os
import numpy as np
import cv2

def get_three_channel_mask(mask_file):
    mask_img = cv2.imread(mask_file)
    return mask_img

def get_one_channel_mask(mask_file):
    mask_img = cv2.imread(mask_file, 0)
    return mask_img


if __name__ == "__main__":

    #src_dir = "./train_val/resized_data/mask_abnormal/1/"
    #dst_dir = "./train_val/resized_data/3_channel_mask_abnormal/"
    #img_type = "val"
    #src_dir = "/root/mount_out/data/glaucoma_disc_segmentation_sorted_data_full_20180810/train_val/" + img_type + "/masks/"
    #dst_dir = "/root/mount_out/data/glaucoma_disc_segmentation_sorted_data_full_20180810/train_val/" + img_type + "/one_channel_masks/"

    src_dir = "/root/mount_out/data/glaucoma_cup_seg_train_val_data_20180823/detected_resized_masks_train/" 
    dst_dir = "/root/mount_out/data/glaucoma_cup_seg_train_val_data_20180823/detected_resized_masks_train-1c/" 

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)


    img_names = os.listdir(src_dir)
    for img_name in img_names:
        print img_name 
        if img_name.endswith(".png"):
            img_file = src_dir + img_name 
            mask_img = get_one_channel_mask(img_file)
            print mask_img.shape
            save_file = dst_dir + img_name 
            cv2.imwrite(save_file, mask_img)


