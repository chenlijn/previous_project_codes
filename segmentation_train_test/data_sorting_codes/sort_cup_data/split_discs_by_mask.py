import numpy as np
import os
import cv2

from lj_utils import split_disc_by_mask


def split_discs(src_path, dst_path):

    img_path = src_path + 'images/'
    mask_path = src_path + 'mask/'
    dst_img_path = dst_path + 'disc_split_image/'
    dst_mask_path = dst_path + 'disc_split_mask/'

    if not os.path.exists(dst_img_path):
        os.mkdir(dst_img_path)
    if not os.path.exists(dst_mask_path):
        os.mkdir(dst_mask_path)

    file_names = os.listdir(img_path)

    for name in file_names:
    #for name in file_names[388:389]:
        if name.endswith('jpg'):
            print name
            raw_name = name.split('.')[0]
            img = cv2.imread(img_path + name)
            mask = cv2.imread(mask_path + raw_name + '.png', 0)
            print img.shape, mask.shape

            split_img, split_mask = split_disc_by_mask(img, mask, 1)
            print split_img.shape, split_mask.shape
            print dst_img_path + name
            cv2.imwrite(dst_img_path + name, split_img)
            cv2.imwrite(dst_mask_path + raw_name + '.png', split_mask)




if __name__ == "__main__":
    src_path = '/root/mount_out/data/glaucomaData_filtered_kan20170825_final/seg/'
    dst_path = '/root/mount_out/data/glaucomaData_filtered_kan20170825_final/seg/'
    split_discs(src_path, dst_path)
