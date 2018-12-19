
import os
import numpy as np
import cv2



def draw_seg_mask_on_img(img_folder, mask_folder, dst_folder, img_suffix='.jpg'):
    img_names = os.listdir(img_folder)
    for img_name in img_names:
        if img_name.endswith(img_suffix):
            img = cv2.imread(img_folder+img_name)
            mask = cv2.imread(mask_folder+img_name.split('.')[0] + '.png', 0)

            # get the edge of mask area
            kernel = np.ones((5,5), np.uint8)
            temp_mask = np.copy(mask)
            eroded_mask = cv2.erode(temp_mask, kernel, iterations=1)
            edge = mask - eroded_mask

            # draw
            rows, cols = np.where(edge > 0)
            img[rows, cols, 0] = 0
            img[rows, cols, 1] = 255
            img[rows, cols, 2] = 0
            cv2.imwrite(dst_folder+img_name, img)
    return



def show_masks(mask_folder, dst_folder, img_suffix='.png'):
    img_names = os.listdir(mask_folder)
    for img_name in img_names:
        if img_name.endswith(img_suffix):
            mask = cv2.imread(mask_folder+img_name, 0)
            cv2.imwrite(dst_folder + img_name, np.where(mask>0, 255, 0))





if __name__=="__main__":
    #img_folder = '/root/mount_out/data/glaucoma_disc_segmentation_sorted_data_full_20180810/augmented_data/images/' 
    #mask_folder = '/root/mount_out/data/glaucoma_disc_segmentation_sorted_data_full_20180810/augmented_data/clean_masks/' 
    #dst_folder = '/root/mount_out/data/glaucoma_disc_segmentation_sorted_data_full_20180810/augmented_data/show/' 

    img_folder = '/root/mount_out/data/glaucoma_cup_seg_train_val_data_20180926/augmented_data/images/'
    mask_folder = '/root/mount_out/data/glaucoma_cup_seg_train_val_data_20180926/augmented_data/masks/'
    dst_folder = '/root/mount_out/data/glaucoma_cup_seg_train_val_data_20180926/augmented_data/show_masks/'
    if not os.path.exists(dst_folder):
        os.mkdir(dst_folder)

    draw_seg_mask_on_img(img_folder, mask_folder, dst_folder, img_suffix='.jpg')
    #show_masks(mask_folder, dst_folder, img_suffix='.png')


