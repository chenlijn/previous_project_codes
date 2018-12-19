import numpy as np
import sys
import os
import caffe
import cv2
import params_config as cfg
from disc_detector_api import disc_detector
from disc_cup_seg_api import disc_cup_seg
from unet_surgery import UnetCupSegmentation

#sys.path.append('/home/gaia/caffe/python')


def draw_mask(image,result_disc = None,result_cup = None):

	result_disc = np.asarray(result_disc,dtype=np.uint8)
	result_disc[result_disc>0]=1
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
	result_disc_ = cv2.erode(result_disc, kernel)
	result_cup_ = cv2.erode(result_cup, kernel)

	result_disc_b = result_disc-result_disc_
	result_cup_b = result_cup-result_cup_

	image = np.array(image)

	rd_rows,rd_cols = np.where(result_disc_b==1)
	rc_rows,rc_cols = np.where(result_cup_b==1)

	image[rd_rows,rd_cols,0] = 0
	image[rd_rows,rd_cols,1] = 0
	image[rd_rows,rd_cols,2] = 255

	image[rc_rows,rc_cols,0] = 0
	image[rc_rows,rc_cols,1] = 255
	image[rc_rows,rc_cols,2] = 0

	cv2.imwrite(os.path.join(show_path,image_name),image)

def main(image_name, image_path, save_path, ddtr, disc_seg):
#def main(image_name, image_path, save_path, ddtr, unet_cup_seg):
#def main(image_name, image_path, save_path, ddtr, cup_seg):

	image = cv2.imread(os.path.join(image_path,image_name))
        print os.path.join(image_path,image_name)
        print image.shape
	img_disc = ddtr.forward_net(image)
        #cv2.imwrite(os.path.join(save_path+'/det', image_name.split('.')[0]+'.png'), img_disc)
	#img_disc = cv2.imread(os.path.join(save_path,image_name))
	result_disc = disc_seg.forward_net(img_disc)
	#result_cup = cup_seg.forward_net(img_disc)
	#result_cup = unet_cup_seg.net_forward(img_disc)


	cv2.imwrite(os.path.join(save_path+'/disc',image_name.split('.')[0]+'.png'), result_disc*255)
	#cv2.imwrite(os.path.join(save_path+'/cup',image_name.split('.')[0]+'.png'), result_cup*255)
	#cv2.imwrite(os.path.join(save_path+'/cup_yelin',image_name.split('.')[0]+'.png'), result_cup*255)

	#draw_mask(img_disc,result_disc,result_cup)

if __name__ == '__main__':
        #image_path_root = '/mnt/lijian/mount_out/data/glaucomaData_filtered_kan20170825_final/disc_split_cls/'
        image_path_root = '/mnt/lijian/mount_out/data/pku_glaucoma_201712_label_consistency_check/source_images/'
        #image_name = '00000.jpg'
        #image_name = '00000_disc_detected.png'
        
        #save_path_root = '../result_data/'
        save_path_root = '/mnt/lijian/mount_out/data/pku_glaucoma_201712_label_consistency_check/algo_result/'
        #show_path = '../result_show/'

        #img_types = ['healthy', 'light', 'mid', 'serious']
        img_types = ['light', 'mid']
        #img_types = ['light', 'mid', 'serious']

	ddtr = disc_detector(cfg.params.disc_det_proto(), cfg.params.disc_det_weight(), cfg.params.get_gpu_id_0(), cfg.params.disc_det_mean())
	disc_seg = disc_cup_seg(cfg.params.disc_seg_proto(), cfg.params.disc_seg_weight(), cfg.params.get_gpu_id_0())
	#cup_seg = disc_cup_seg(cfg.params.cup_seg_proto(), cfg.params.cup_seg_weight(),cfg.params.get_gpu_id_0())
        # my cup segmetation
        model_file = '../models/dense_unet_deploy.prototxt'
        #weights = '../models/dense_unet_iter_44000.caffemodel'
        weights = '../models/dense_unet_iter_50000.caffemodel'
        meanfile = '../models/mean.npy'
        #unet_cup_seger = UnetCupSegmentation(model_file, weights, meanfile, 0) 
        #net, transformer = unet_surgery.load_net(model_file, weights, meanfile)

        for img_type in img_types:
            image_path = image_path_root + img_type
            save_path = save_path_root + img_type
            imgnames = os.listdir(image_path)
            for image_name in imgnames:
                #if image_name.endswith('.png'):
                if image_name.endswith('.jpg'):
                    main(image_name, image_path, save_path, ddtr, disc_seg)
                    #main(image_name, image_path, save_path, ddtr, unet_cup_seger)
                    #main(image_name, image_path, save_path, ddtr, cup_seg)




