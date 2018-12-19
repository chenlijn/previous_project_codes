
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
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)

        # preprocessing
        print type(resized_image)
        cv_image = np.array(resized_image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cv_image = cv_image.copy()
        print 'cv: ', cv_image.dtype
        cv_image[:,:,0] = clahe.apply(cv_image[:,:,0])
        cv_image[:,:,1] = clahe.apply(cv_image[:,:,1])
        cv_image[:,:,2] = clahe.apply(cv_image[:,:,2])
        #cv2.imwrite('res_img_c.png', cv_image)
        #resized_image.save('res_img.png')
        batch_seg_map = self.sess.run(self.OUTPUT_TENSOR_NAME, 
                                      feed_dict={self.INPUT_TENSOR_NAME: [cv_image]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map



if __name__ == "__main__":

    #img_dir = '/root/mount_out/data/glaucoma_disc_segmentation_sorted_data_full_20180810/'
    work_root = '/root/mount_out/work2018/private_glaucoma_recognition/Deep-V3_project/deeplab_codes/deeplab/'

    #img_dir = '/root/mount_out/data/glaucoma_cup_seg_train_val_data_20180926/train_val_data/val/'
    img_dir = '/root/mount_out/data/public_datasets/disc_cup_segmentation_dataset/det/validation/'

    # disc
    #graph_file = 'frozen_disc_seg_inference_graph.pb'

    # cup
    graph_file = 'frozen_cup_seg_inference_graph.pb'

    test_model = DeepLabModel(graph_file, 1)

    dst_dir = img_dir + 'prediction_show_cup/'
    seg_dst_dir = img_dir + 'seg/cup/'
    if not os.path.exists(seg_dst_dir):
        os.mkdir(seg_dst_dir)

    img_folder = img_dir + 'images/'
    mask_folder = img_dir + 'cup/'
    imgnames = os.listdir(img_folder)

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    IOUs = []
    for imgname in imgnames:
        img = imgname.strip('\n')
        if not img.endswith('.jpg'):
            continue
        imgfile = img_folder + img
        maskfile = mask_folder + img.split('.')[0] + '.png'
        res_img, predict = test_model.run(imgfile)
        savename = seg_dst_dir + img.split('.')[0] + '.png' 
        r, c = np.where(predict > 0)
        predict[r,c] = 1
        predict = predict.astype(np.uint8)
        cv2.imwrite(savename, predict*255)
        height, width = res_img.size
        mask = cv2.imread(maskfile, 0)
        res_mask = cv2.resize(mask, (height, width))



        # draw mask and prediction on image
        kernel = np.ones((5,5), np.uint8)
        erosion = cv2.erode(res_mask, kernel, iterations=1)
        edge = res_mask - erosion
        rows, cols = np.where(edge > 0)
        
        cv_image = np.array(res_img)
        cv_image = cv_image[:,:,::-1].copy()

        # draw mask, blue, ground truth
        cv_image[rows, cols, 0] = 255
        cv_image[rows, cols, 1] = 0
        cv_image[rows, cols, 2] = 0

        # prediction drawing 
        kernel = np.ones((5,5), np.uint8)
        erosion = cv2.erode(predict, kernel, iterations=1)
        edge = predict - erosion
        rows, cols = np.where(edge > 0)
        cv_image[rows, cols, 0] = 0
        cv_image[rows, cols, 1] = 255
        cv_image[rows, cols, 2] = 0

        # compute IOUs
        rs, cs = np.where(res_mask>0)
        res_mask[rs, cs] = 1
        union = res_mask + predict
        intersect = np.zeros_like(union)
        rows, cols = np.where(union==2)
        intersect[rows, cols] = 1
        intersect_area = intersect.sum()
        union_area = union.sum() - intersect_area
        iou = intersect_area / float(union_area)
        IOUs.append(iou)

        # concatenate and save
        concat = np.concatenate((cv_image, cv2.cvtColor(predict*255, cv2.COLOR_GRAY2RGB)), axis=1)
        savename = dst_dir + img.split('.')[0] + '.png'
        cv2.imwrite(savename, concat)

    print("mean IOU: {}".format(np.mean(IOUs)))



