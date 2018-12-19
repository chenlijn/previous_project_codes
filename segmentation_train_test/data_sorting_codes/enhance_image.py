#generate HDF5 data files

import os
import numpy as np 
import cv2

from lj_utils import enhance_image_by_clahe

img_file = '/mnt/lijian/mount_out/work2018/private_glaucoma_recognition/generate_pku_paper_materials/disc_cup_seg_yelin/result_data/healthy/det/a707725fc32ab89349003f01bd68cf6b.png' 
image = cv2.imread(img_file)
res_img = enhance_image_by_clahe(image)
cv2.imwrite('img_clahe.png', res_img)

        


