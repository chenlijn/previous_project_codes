import os
import numpy as np
import cv2
import tensorflow as tf

from tensorflow.core.framework import graph_pb2
from lj_utils import confusion_matrix2sens_spec

class DiscClassifier(object):

    def __init__(self):
        self._input_size = (961, 961)
        self._class_num = 5
        self._model_file = 'frozen_model/inception-v3_frozen.pb'

    def letterbox_image(self, image):
        '''resize image with unchanged aspect ratio using padding'''
        ih, iw, _ = image.shape
        h, w = self._input_size
        scale = min(1.0*w/iw, 1.0*h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        temp = cv2.resize(image, (nw, nh))
        new_image = np.ones((h,w,3), np.float32)*128
        start_h = (h-nh)/2
        end_h = start_h + nh
        start_w = (w-nw)/2
        end_w = start_w + nw
        new_image[start_h:end_h, start_w:end_w, :] = temp[:,:,:]
        return new_image

    def load_network(self):
        self._graph_def = graph_pb2.GraphDef()
        with open(self._model_file, 'rb') as f:
            self._graph_def.ParseFromString(f.read())
        tf.import_graph_def(self._graph_def, name="")
        self._sess = tf.Session()

    def predict(self, image):
        resized_image = self.letterbox_image(image) 
        in_image = np.expand_dims(resized_image[:,:,::-1], axis=0).astype(np.float32)
        in_image -= 127.5
        in_image /= 127.5
        y_ = self._sess.run('dense_2/Softmax:0', feed_dict={'input_1:0': in_image})
        return y_


if __name__=='__main__':

    disc_cls = DiscClassifier()
    disc_cls.load_network()

    data_root = 'data/test_apollo_disc_v2/disc/'
    test_file = data_root + 'test.txt'
    with open(test_file, 'r') as tf:
        imgfiles = tf.readlines()

    conf_mat = np.zeros((disc_cls._class_num, disc_cls._class_num))


    for line in imgfiles[:5]:
        img_path = data_root + line.split()[0]
        label = line.split()[1].strip('\n')
        image = cv2.imread(img_path)
        #resized_image = disc_cls.letterbox_image(image) 
        #in_image = np.expand_dims(resized_image[:,:,::-1], axis=0).astype(np.float32)
        predictions = disc_cls.predict(image)
        conf_mat[predictions.argmax(), int(label)] += 1
        print('predict: {}, ground truth: {}'.format(predictions.argmax(), label))
    np.save('result/conf_mat.npy', conf_mat)
    print('sensitivity and specificity: \n {}'.format(confusion_matrix2sens_spec(conf_mat)))



