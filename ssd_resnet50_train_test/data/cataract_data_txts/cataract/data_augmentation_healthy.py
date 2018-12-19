#=========================
# image data augmentation
#=========================
import os
import numpy as np
import cv2
import random
import shutil 


class image_data_augmentation(object):

    def __init__(self):
        self.imgNum = 0
        return

    def pixel_pertubation(self, image, dst, aug_num): 
        '''change the pixel intensities randomly in a small range of [0, 10]'''
        height, width, depth = image.shape
        inten_range = 10

        for i in range(aug_num):
            #add
            pertur_mat = np.random.random(image.shape) * 10
            new_add_image = (image + pertur_mat).astype(int)
            savefile = dst + '/per_add_' + str(self.imgNum) + '_' + str(i) + '.jpg' 
            cv2.imwrite(savefile, new_add_image)  

            #minus
            pertur_mat = np.random.random(image.shape) * 10
            new_minu_image = (image - pertur_mat).astype(int)
            savefile = dst + '/per_minu_' + str(self.imgNum) + '_' + str(i) + '.jpg' 
            cv2.imwrite(savefile, new_minu_image)  


    #one times the num
    def random_crop(self, image, crop_size, dst, aug_num):
        height, width, depth = image.shape
        ratio = 0.9
        #self.imgNum += 1
        new_size = int(crop_size/ratio)
        resize_img = cv2.resize(image, (new_size, new_size))

        #crop
        crop_img_num = aug_num
        for i in range(crop_img_num):
            up_left_x = random.randint(0,new_size-crop_size)
            up_left_y = random.randint(0,new_size-crop_size)
            bot_right_x = up_left_x + crop_size
            bot_right_y = up_left_y + crop_size 
            crop_img = resize_img[up_left_y:bot_right_y, up_left_x:bot_right_x, :]
            savefile = dst + '/crop_' + str(self.imgNum) + '_' + str(i) + '.jpg' 
            cv2.imwrite(savefile, crop_img)  
        return

    #one times the num
    def random_translation(self, image, dst, aug_num):
        height, width, depth = image.shape
        ratio = 0.1

        #translation, only in the x direction 
        trans_num = aug_num
        for i in range(trans_num):
            x_random = random.randint(0, int(ratio*width))
            M = np.float32([[1,0,x_random],[0,1,0]])
            b,g,r = cv2.split(image)
            trans_b = cv2.warpAffine(b, M, (width, height))
            trans_g = cv2.warpAffine(g, M, (width, height))
            trans_r = cv2.warpAffine(r, M, (width, height))
            trans_img = cv2.merge([trans_b, trans_g, trans_r])  
            savefile = dst + '/trans_' + str(self.imgNum) + '_' + str(i) + '.jpg' 
            cv2.imwrite(savefile, trans_img)  
        return

    #one times the num 
    def random_rotation(self, image, dst, aug_num):
        height, width, depth = image.shape
        angle_range = 20
        
        #rotate
        rotate_num = aug_num  
        for i in range(rotate_num):
            rotate_angle = random.randint(0,angle_range) 
            M = cv2.getRotationMatrix2D((width/2,height/2),rotate_angle,1)
            b,g,r = cv2.split(image)
            rotate_r = cv2.warpAffine(r,M,(width, height))
            rotate_g = cv2.warpAffine(g,M,(width, height))
            rotate_b = cv2.warpAffine(b,M,(width, height))
            rotate_img = cv2.merge([rotate_b, rotate_g, rotate_r])  
            savefile = dst + '/rotate_' + str(self.imgNum) + '_' + str(i) + '.jpg' 
            cv2.imwrite(savefile, rotate_img)  
        return

    #two times the num 
    def flip_img(self, image, dst):  
        height, width, depth = image.shape

        #flip  
        flip_img = cv2.flip(image, 1)
        savefile = dst + '/vflip_' + str(self.imgNum) + '.jpg' 
        cv2.imwrite(savefile, flip_img)
        flip_img = cv2.flip(image, 0)
        savefile = dst + '/hflip_' + str(self.imgNum) + '.jpg' 
        cv2.imwrite(savefile, flip_img)
        return



if __name__ == '__main__':

    workroot = os.getcwd() 
    data_type = 'mydriasis_diffuse_light'
    #data_type = 'mydriasis_diffuse_light'
    ori_src_dir = '/mnt/lijian/mount_out/zhongshan_ophthalmology_cataract_images_20170906/healthy/' + data_type
    dst_dir = '/mnt/lijian/mount_out/cataract_augmented_data_dec2/healthy/' + data_type

   
    augmenter = image_data_augmentation()
    aug_num = 1

    train_file = os.getcwd() + '/train_val_split/cleaned_healthy_' + data_type + '_train.txt'
    with open(train_file, 'r') as trf:
        train_names = trf.readlines() 
    #print train_names
    
    #imgnames = []
    for name in train_names[:40]:
        label = name.strip('\n').split(' ')[-1]
        print label
        imgfile = name.strip('\n').strip(label).strip()
        print imgfile

        #imgnames.append(imgfile)
        image = cv2.imread(imgfile, cv2.IMREAD_UNCHANGED)
#        print image

        if image == None:
            print 'empty image !'
            continue
        #shutil.copy(imgfile, dst_dir)
        augmenter.imgNum += 1
        target_size = 512
        #augmenter.random_rotation(image, dst_dir, aug_num)
        augmenter.pixel_pertubation(image, dst_dir, 5) 
        #augmenter.flip_img(image, dst_dir)
        #augmenter.random_translation(image, dst_dir, aug_num)
        #augmenter.random_crop(image, target_size, dst_dir, aug_num)

    #print 'scr img num: {}'.format(augmenter.imgNum)  




