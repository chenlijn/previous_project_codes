import os
import shutil


img_path = '/root/mount_out/data/glaucoma_disc_segmentation_sorted_data_full_20180810/combined_train_data/images/'
dst_path = '/root/mount_out/data/glaucoma_disc_segmentation_sorted_data_full_20180810/combined_train_data/images_jpg/'

if not os.path.exists(dst_path):
    os.mkdir(dst_path)

image_files = os.listdir(img_path)

for idx, imgname in enumerate(image_files):

    imgfile = img_path + imgname
    savefile = dst_path + imgname.split('.')[0] + '.jpg'

    shutil.copy(imgfile, savefile)

    print idx 


