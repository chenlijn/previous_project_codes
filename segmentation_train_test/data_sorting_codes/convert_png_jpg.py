import os
import shutil


#img_path = '/root/mount_out/data/glaucoma_disc_segmentation_sorted_data_full_20180917/augmented_data/images/'
#dst_path = '/root/mount_out/data/glaucoma_disc_segmentation_sorted_data_full_20180917/augmented_data/images/'

img_path = '/root/mount_out/data/glaucoma_cup_seg_train_val_data_20180926/augmented_data/images/'
dst_path = '/root/mount_out/data/glaucoma_cup_seg_train_val_data_20180926/augmented_data/images/'

if not os.path.exists(dst_path):
    os.mkdir(dst_path)

image_files = os.listdir(img_path)

for idx, imgname in enumerate(image_files):

    imgfile = img_path + imgname
    savefile = dst_path + imgname.split('.')[0] + '.jpg'

    #shutil.copy(imgfile, savefile)
    os.rename(imgfile, savefile)

    print(idx)


