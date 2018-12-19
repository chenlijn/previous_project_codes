
import numpy as np
import cv2
import os


def extract_disc_cup_masks(mix_mask):
    h, w = mix_mask.shape
    disc = np.zeros((h,w), np.uint8)
    cup = np.zeros((h,w), np.uint8)
    rows, cols = np.where(mix_mask<=128)  # disc
    disc[rows, cols] = 255
    rows, cols = np.where(mix_mask==0)  # cup
    cup[rows, cols] = 255
    return disc, cup
 


if __name__ == '__main__':

    file_path = '/root/mount_out/data/public_datasets/disc_cup_segmentation_dataset/Annotation-Training400/Disc_Cup_Masks/Glaucoma/'
    #file_path = '/root/mount_out/data/public_datasets/disc_cup_segmentation_dataset/Annotation-Training400/Disc_Cup_Masks/Non-Glaucoma/'
    #file_path = '/root/mount_out/data/public_datasets/disc_cup_segmentation_dataset/REFUGE-Validation400-GT/Disc_Cup_Masks/'
    dst_path = '/root/mount_out/data/public_datasets/disc_cup_segmentation_dataset/Annotation-Training400/all_masks/'
    #dst_path = '/root/mount_out/data/public_datasets/disc_cup_segmentation_dataset/REFUGE-Validation400-GT/all_masks/'
    all_files = os.listdir(file_path)

    for file_name in all_files:
        print(file_name)
        if file_name.endswith('bmp'):
            mask = cv2.imread(file_path+file_name, 0)
            disc, cup = extract_disc_cup_masks(mask)
            cv2.imwrite(dst_path+'disc/'+file_name, disc)
            cv2.imwrite(dst_path+'cup/'+file_name, cup)
       
 
    #print(all_files[:3])
    #print(mask.shape)


