import numpy as np
import os
import cv2

from lj_utils import split_disc_by_mask


def split_discs(src_path, dst_path):

    img_path = src_path + 'images/'
    mask_path = src_path + 'masks/'
    dst_img_path = dst_path + 'split/images/'
    dst_mask_path = dst_path + 'split/masks/'

    file_names = os.listdir(mask_path)

    for name in file_names:
    #for name in file_names[388:389]:
        if name.endswith('.png'):
            print(name)
            raw_name = name.split('.')[0]
            img = cv2.imread(img_path + raw_name+'.jpg')
            mask = cv2.imread(mask_path + raw_name + '.png', 0)
            print(img.shape, mask.shape)

            split_img, split_mask = split_disc_by_mask(img, mask)
            print(split_img.shape, split_mask.shape)
            print(dst_img_path + name)
            cv2.imwrite(dst_img_path + raw_name + '.jpg', split_img)
            cv2.imwrite(dst_mask_path + raw_name + '.png', split_mask)




if __name__ == "__main__":
    #src_path = '/root/mount_out/data/glaucoma_disc_segmentation_sorted_data_full_20180917/disc_2000/'
    #dst_path = '/root/mount_out/data/glaucoma_disc_segmentation_sorted_data_full_20180917/disc_2000/'
    src_path = '/root/mount_out/data/original_glaucoma_data/all_data_sorted/healthy/'
    dst_path = '/root/mount_out/data/original_glaucoma_data/all_data_sorted/healthy/'
    split_discs(src_path, dst_path)
