
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from google.protobuf import text_format
import cv2


class DeepLabModel(object):
    """class to load deeplab model and run inference"""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, graph_file, binary_or_not):
        """creates and loads pretrained model"""

        graph_def = graph_pb2.GraphDef()
        with open(graph_file, 'rb') as f:
            if binary_or_not:
                graph_def.ParseFromString(f.read())
            else:
                text_format.Merge(f.read(), graph_def)
        
        tf.import_graph_def(graph_def, name='')
        self.sess = tf.Session()


    def run(self, imagefile):
        image = Image.open(imagefile)
        width, height = image.size
        #resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        #target_size = (int(resize_ratio * width), int(resize_ratio * height))
        target_size = (self.INPUT_SIZE, self.INPUT_SIZE)
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)

        # preprocessing
        print type(resized_image)
        cv_image = np.array(resized_image)
        cv_image = cv_image.copy()
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        print 'cv: ', cv_image.dtype
        cv_image[:,:,0] = clahe.apply(cv_image[:,:,0])
        cv_image[:,:,1] = clahe.apply(cv_image[:,:,1])
        cv_image[:,:,2] = clahe.apply(cv_image[:,:,2])
        #cv2.imwrite('res_img_c.png', cv_image)
        #resized_image.save('res_img.png')
        batch_seg_map = self.sess.run(self.OUTPUT_TENSOR_NAME, 
                                      feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(cv_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map



if __name__ == "__main__":

    #img_dir = '/root/mount_out/work2018/disc_cup_seg_yelin/result_data/'
    img_dir = '/root/mount_out/data/original_glaucoma_data/sorted_data/glaucoma_cls_test_set/splits/'
    work_root = '/root/mount_out/work2018/private_glaucoma_recognition/Deep-V3_project/deeplab_codes/deeplab/'

    # disc
    graph_file = work_root + 'training-disc/export/frozen_disc_seg_inference_graph.pb'

    # cup
    #graph_file = 'frozen_cup_seg_inference_graph.pb'

    test_model = DeepLabModel(graph_file, 1)
    #img_type = 'healthy'
    #img_type = 'light'
    #img_type = 'mid'
    #img_type = 'serious'
    #img_type = 'online_abnormal_test'

    img_types = ['healthy', 'light', 'mid', 'serious', 'online_abnormal_test', 'online_normal_test']

    for img_type in img_types:

        #dst_dir = '/root/mount_out/work2018/disc_cup_seg_yelin/result_data/' + img_type + '/cup_deeplabv3/' 
        #dst_dir = '/root/mount_out/work2018/private_glaucoma_recognition/generate_pku_paper_materials/disc_cup_seg_yelin/result_data/' + img_type + '/cup_deeplabv3/' 
        dst_dir = img_dir + 'seg/disc/' + img_type + '/' 
        #dst_dir = img_dir + 'seg/cup/' + img_type + '/' 
        img_folder = img_dir + 'det/' + img_type + '/' 
        #mask_folder = img_dir + 'train_val/val_1/one_channel_masks/'
        img_list_file = img_dir + 'det/' + img_type + '.txt'

        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)

        with open(img_list_file) as f:
            imgnames = f.readlines()

        #IOUs = []
        for imgname in imgnames:
            img = imgname.strip('\n')
            imgfile = img_folder + img
            #maskfile = mask_folder + img.split('.')[0] + '.png'
            res_img, predict = test_model.run(imgfile)

            ## concatenate and save
            #concat = np.concatenate((cv_image, cv2.cvtColor(predict*255, cv2.COLOR_GRAY2RGB)), axis=1)
            savename = dst_dir + img.split('.')[0] + '.png'
            cv2.imwrite(savename, np.where(predict>0, 255, 0))

    #print "mean IOU: {}".format(np.mean(IOUs))



