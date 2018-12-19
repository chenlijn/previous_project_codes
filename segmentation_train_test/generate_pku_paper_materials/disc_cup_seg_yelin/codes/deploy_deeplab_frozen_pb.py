
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
        cv_image = cv_image[:,:,::-1].copy()
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        print 'cv: ', cv_image.dtype
        cv_image[:,:,0] = clahe.apply(cv_image[:,:,0])
        cv_image[:,:,1] = clahe.apply(cv_image[:,:,1])
        cv_image[:,:,2] = clahe.apply(cv_image[:,:,2])
        #cv2.imwrite('res_img_c.png', cv_image)
        #resized_image.save('res_img.png')
        batch_seg_map = self.sess.run(self.OUTPUT_TENSOR_NAME, 
                                      feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map



if __name__ == "__main__":

    work_root = '/root/host_share/glaucoma/'
    graph_file = work_root + 'deeplab/training/export/frozen_inference_graph.pb'
    test_model = DeepLabModel(graph_file, 1)

    dst_dir = work_root + 'deeplab/prediction_show_my/'
    img_folder = work_root + 'deeplab/all_images/'
    mask_folder = work_root + 'deeplab/all_masks/'
    img_list_file = work_root + 'deeplab/lists/val.txt'
    with open(img_list_file) as f:
        imgnames = f.readlines()

    IOUs = []
    for imgname in imgnames:
        img = imgname.strip('\n')
        imgfile = img_folder + img
        maskfile = mask_folder + img.split('.')[0] + '.png'
        res_img, predict = test_model.run(imgfile)
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
        r, c = np.where(predict > 0)
        predict = predict.astype(np.uint8)
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

    print "mean IOU: {}".format(np.mean(IOUs))



