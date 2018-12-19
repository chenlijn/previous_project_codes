
import os
import numpy as np
from keras import models 
from keras.preprocessing.image import ImageDataGenerator


#src_dir_parent = "/root/mount_out/data/glaucoma_disc_segmentation_sorted_data_full_20180917/train_val_data/train/"
src_dir_parent = '/root/mount_out/data/glaucoma_cup_seg_train_val_data_20180926/train_val_data/train/'
dst_dir_parent = '/root/mount_out/data/glaucoma_cup_seg_train_val_data_20180926/augmented_data/'
#dst_dir_parent = "/root/mount_out/data/glaucoma_disc_segmentation_sorted_data_full_20180917/augmented_data/"

# folders=["healthy", "ill", "after_surgery"]
# img_types=["mydriasis_diffuse_light", "mydriasis_slit_light",
#            "small_pupil_diffuse_light", "small_pupil_slit_light"]

folders=[""]
img_types=[""]


# target_aug_num=1380*4
#target_aug_num = 8000 # disc 
target_aug_num = 25500 # cup

tsize= 513
batchsize=64 
iter_num = target_aug_num // batchsize 

saveprefix="aug"

data_gen_args = dict(
                     featurewise_center=False,
                     featurewise_std_normalization=False,
                     #zca_epsilon=,
                     zca_whitening=False,
                     rotation_range=20,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=np.pi/32,
                     zoom_range=0.2,
                     channel_shift_range=0,
                     fill_mode="nearest",
                     horizontal_flip=True,
                     vertical_flip=True,
                     rescale=0,
                    )

#mask_data_gen_args = dict(
#                     featurewise_center=False,
#                     featurewise_std_normalization=False,
#                     #zca_epsilon=,
#                     zca_whitening=False,
#                     rotation_range=15,
#                     width_shift_range=0.05,
#                     height_shift_range=0.05,
#                     shear_range=0, #np.pi/32,
#                     zoom_range=0.1,
#                     channel_shift_range=0,
#                     fill_mode="nearest",
#                     horizontal_flip=True,
#                     vertical_flip=True,
#                     rescale=0,
#                    )

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# image_generator = image_datagen.flow_from_directory(src_dir, target_size=(tsize, tsize), batch_size=batchsize, 
#                                                     save_to_dir=dst_dir, save_prefix=saveprefix, seed=1)
# 
# mask_generator = mask_datagen.flow_from_directory(src_dir, target_size=(tsize, tsize), batch_size=batchsize, 
#                                                     save_to_dir=dst_dir, save_prefix=saveprefix, seed=1)


for folder in folders:
    for img_t in img_types:
        img_src_dir = src_dir_parent + "detected_resized_images"  
        mask_src_dir = src_dir_parent + "detected_resized_masks"

        img_dst_dir = dst_dir_parent + "images"
        mask_dst_dir = dst_dir_parent + "masks"

        imgs_list = os.listdir(img_src_dir + '/1')
        masks_list = os.listdir(mask_src_dir + '/1')
        imgs_num = len(imgs_list) 
        
        print(folder, img_t)
        print("imgs_num:{}, iternum: {}".format(imgs_num, iter_num))
        seed = 1
        for _ in range(iter_num):
            saveprefix="aug_" + str(_)
            image_generator = image_datagen.flow_from_directory(img_src_dir, target_size=(tsize, tsize), batch_size=batchsize, 
                                                                save_to_dir=img_dst_dir, save_prefix=saveprefix, seed=seed)
            
            mask_generator = mask_datagen.flow_from_directory(mask_src_dir, target_size=(tsize, tsize), batch_size=batchsize, 
                                                                save_to_dir=mask_dst_dir, save_prefix=saveprefix, seed=seed)

            #train_generator = zip(image_generator, mask_generator) 
            #model.fit_generator(train_generator, steps_per_epoch=1, epochs=1)

            image_generator.next()
            mask_generator.next()
            # del train_generator 
            del image_generator
            del mask_generator 
            seed += 1
            print(_)

#models.Sequential.fit_generator(train_data_generator, steps_per_epoch=10, epochs=1)

