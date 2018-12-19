#==========================================================
# these files contain general utilities for preprocessing
#==========================================================
import os
import sys
import numpy as np
import cv2
import shutil


class utils:
    def __init__(self):
        pass

    def resizeMask4Seg(self, maskImg, targetSize):
        '''
        resize a mask containing only 2 classes which are background and foreground
        and targetSize must be a tuple
        '''
        resImg = cv2.resize(maskImg, targetSize) #using bilinear interpolation by default
        rows, colns = np.where(resImg >= 0.5) 
        resImg[:,:] = 0
        resImg[rows, colns] = 1
        return resImg.astype(np.uint8) 

    def resizeImg4Seg(self, img_path, mask_path, dst_path, show_path, targetSize, suffix=""):
        #all_images = os.listdir(img_path)
        all_masks = os.listdir(mask_path)

        image_dst_path = dst_path + '/images_' + suffix + '/'
        mask_dst_path = dst_path + '/mask_' + suffix + '/' 

        if os.path.exists(image_dst_path):
            shutil.rmtree(image_dst_path)
        if os.path.exists(mask_dst_path):
            shutil.rmtree(mask_dst_path) 

        os.makedirs(image_dst_path)
        os.makedirs(mask_dst_path) 

        for mask_name in all_masks:
            if mask_name.endswith('.png'):
                name = mask_name.split('.')[0]
                img_file = img_path + name + '.png'
                save_img_file = image_dst_path + name + '.png'
                mask_file = mask_path + mask_name 
                save_mask_file = mask_dst_path + name + '.png'
                show_file = show_path + name + '.png'

                # read and resize 
                print img_file 
                img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
                print img.shape 
                mask = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
                resized_img = cv2.resize(img, targetSize)  
                resized_mask = self.resizeMask4Seg(mask, targetSize)
                cv2.imwrite(save_img_file, resized_img)
                cv2.imwrite(save_mask_file, resized_mask)
                
                # draw masks on images
                rows, cols = np.where(resized_mask>0)
                resized_img[rows, cols, 0] = 0
                resized_img[rows, cols, 1] = 255
                resized_img[rows, cols, 2] = 0
                cv2.imwrite(show_file, resized_img) 

        return 



    # def resizeImg4Seg(self, src_path, dst_path, show_path, targetSize):
    #     image_path = dst_path + '/images'
    #     mask_path = dst_path + '/mask' 

    #     if os.path.exists(image_path):
    #         shutil.rmtree(image_path)
    #     if os.path.exists(mask_path):
    #         shutil.rmtree(mask_path) 

    #     os.makedirs(image_path)
    #     os.makedirs(mask_path) 

    #     for path, subdirs, files in os.walk(src_path):
    #         for name in files:
    #             wholeName = os.path.join(path, name)
    #             if 'JPEGImages' in wholeName:
    #                 srcimg = cv2.imread(wholeName, cv2.IMREAD_UNCHANGED) 
    #                 resSRC = cv2.resize(srcimg, targetSize)  
    #                 savefile_src = image_path + '/' + name 
    #                 cv2.imwrite(savefile_src, resSRC)   
    #             if 'Segmentations' in wholeName:
    #                 maskimg = cv2.imread(wholeName, cv2.IMREAD_UNCHANGED) 
    #                 if len(maskimg.shape) == 3:
    #                     resMask = self.resizeMask4Seg(maskimg[:,:,1], targetSize)
    #                 else:
    #                     resMask = self.resizeMask4Seg(maskimg, targetSize)

    #                 savefile_mask = mask_path + '/' + name 
    #                 cv2.imwrite(savefile_mask, resMask) 
    #                 cv2.imwrite(show_path + '/' + name, resMask * 255) 

    #             print wholeName 


    def draw_mask_on_imgs(self, img_path, mask_path, dst_path):
        #imgnames = os.listdir(img_path)
        imgnames = os.listdir(mask_path)
        for name in imgnames:
            if name.endswith('.png'):
                #imgfile = img_path + name.split('.')[0] + '.jpg'
                imgfile = img_path + name.split('.')[0] + '.png'
                img = cv2.imread(imgfile, cv2.IMREAD_UNCHANGED)
                temp_img = np.copy(img) 
                #maskfile = mask_path + '/' + name.split('.')[0] + '_mask.png' 
                maskfile = mask_path + name 
                print maskfile
              
                mask = cv2.imread(maskfile, cv2.IMREAD_UNCHANGED)
                print mask.shape 
                rows, colns = np.where(mask == 1) 
                temp_img[rows,colns,0] = 0
                temp_img[rows,colns,1] = 255
                temp_img[rows,colns,2] = 0
                savefile = dst_path+'/' + name.split('.')[0] + '.png' 
                concat_img = np.concatenate((img,temp_img),axis=1)  
                cv2.imwrite(savefile, concat_img)
                #print imgfile
   

if __name__ == "__main__":

    from kan_utils import kan_data_proc
    from parse_xml import get_segmentation_polygon

    data_root = '/mnt/lijian/mount_out/data/online_glaucoma_imgs_labeled_20180307/'
    
    #version = 'v1'
    #img_type = 'abnormal'
    #img_path = data_root + '201802_data2label_v1/glaucoma_data2label_v1-bing/' + img_type + '/' # v1
    #img_path = data_root + '201802_data2label_v2/glaucoma_data2label_v2/' + img_type + '/' # v2
    #img_path = data_root + '201802_data2label_v2/glaucoma_data2label_v2-bing/normal/'
    #new_mask_path = '/mnt/lijian/mount_out/data/online_glaucoma_imgs_labeled_20180307/sorted_data/mask_from_xml/' + version + '/' + img_type + '/' 
    #dst_path = data_root + 'sorted_data/'
    test_path = data_root + '/test/'

    ## -------------resize image----------------- 
    #img_type = "normal"
    #data_root = '/mnt/lijian/mount_out/data/online_glaucoma_imgs_labeled_20180307/online_glaucoma_labeled_data_sorted_20180307/'
    #img_path = data_root + "images_" + img_type + '/' 
    #mask_path = data_root + "mask_" + img_type + '/'
 
    #targetSize = (512, 512)

    util = utils()

    ## resize images and masks to target size
    #dst_path = '/mnt/lijian/mount_out/data/online_glaucoma_imgs_labeled_20180307/train_val/resized_data/'
    #show_path = '/mnt/lijian/mount_out/data/online_glaucoma_imgs_labeled_20180307/test/'
    #util.resizeImg4Seg(img_path, mask_path, dst_path, show_path, targetSize, img_type)
    ## ------------------------------------------


    ##generate the masks
    ##mask_path = img_path + img_type + '-annotation-bing/' # v1
    ##mask_path = img_path + 'normal-xing/' # v2
    #mask_path = img_path + 'abnormal-kang/' # v2
    #annot_files = os.listdir(mask_path) 
    #for annot_file in annot_files:
    #    if annot_file.endswith('.xml'):
    #        name = os.path.splitext(annot_file)[0]
    #        imgfile = img_path + name + '.jpg'
    #        img = cv2.imread(imgfile)
    #        h,w,_ = img.shape 
    #        polygons = get_segmentation_polygon(mask_path + annot_file) 
    #        gen_mask = np.zeros((h,w), np.uint8)
    #        cv2.fillPoly(gen_mask, polygons, 1)
    #        save_name = new_mask_path + name + '.png' 
    #        cv2.imwrite(save_name, gen_mask)   
    #        print 'max val of mask: {}'.format(gen_mask.max())

            

    ##filtered the mask 
    #mask_path = img_path + img_type + '-annotation-bing/'
    #file_names = os.listdir(mask_path)
    #for file_name in file_names:
    #    if file_name.endswith('.jpg'):
    #        mask_img = cv2.imread(mask_path+file_name, cv2.IMREAD_UNCHANGED)
    #        filtered_mask = kan_data_proc.filter_mask_img(mask_img)
    #        save_name = dst_path + version + '/' + 'mask_' + img_type + '/' + os.path.splitext(os.path.basename(file_name))[0] + '.png'
    #        #cv2.imwrite(save_name, filtered_mask)    
    #        cv2.imwrite(save_name, mask_img*255)    
    #        print file_name 

    #        ##resize mask 
    #        #resized_mask = resizeMask4Seg(filtered_mask, targetSize)
    #        #mask_savename = dst_path + version + '_resized/' + img_type + '/' + os.path.splitext(os.path.basename(file_name))[0] + '.png'
    #        #cv2.imwrite(mask_savename, resized_mask)

    ##util.resizeImg4Seg(file_path, new_path, show_path, targetSize)
    #util.draw_mask_on_imgs(img_path, new_mask_path, test_path)


    # draw masks on images
    img_src_dir = '/mnt/lijian/mount_out/docker_share/glaucoma/augmented_data/images_normal/'
    mask_src_dir = '/mnt/lijian/mount_out/docker_share/glaucoma/augmented_data/mask_normal_clean/'
    util.draw_mask_on_imgs(img_src_dir, mask_src_dir, test_path)
    

