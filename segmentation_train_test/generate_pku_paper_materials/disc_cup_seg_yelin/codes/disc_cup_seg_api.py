import numpy as np
import sys
import os
import caffe
import cv2
import math
import copy

class disc_cup_seg(object):
	'''
	Detector api class
	'''
	def __init__(self, prototx_addr, caffemodel_addr, gpu_id):
		self._prototxt = prototx_addr
		self._caffemodel = caffemodel_addr
		self._gup_id = gpu_id

		self.prob_layer = 'prob'
		self.data_layer = 'data'
		self.class_num = 2
		self.base_size = 512

		caffe.set_mode_gpu()
		caffe.set_device(gpu_id)

		self._net = caffe.Net(self._prototxt, self._caffemodel, caffe.TEST)

	def forward_net(self, image):
		in_image = cv2.resize(image,(self.base_size,self.base_size))

		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
		cl0 = clahe.apply(in_image[:,:,0])
		cl1 = clahe.apply(in_image[:,:,1])
		cl2 = clahe.apply(in_image[:,:,2])
		in_image[:,:,0] = cl0
		in_image[:,:,1] = cl1
		in_image[:,:,2] = cl2

		grey = in_image[:,:,1]
		x = cv2.Sobel(grey,cv2.CV_16S,1,0)
		y = cv2.Sobel(grey,cv2.CV_16S,0,1)
		absX = cv2.convertScaleAbs(x)   # coverted to uint8
		absY = cv2.convertScaleAbs(y)
		sobel = cv2.addWeighted(absX,0.5,absY,0.5,0)

		h, w, d = in_image.shape

		_sobel = cv2.resize(sobel,(self.base_size,self.base_size))

		score_map = cv2.resize(self.scale_process(in_image, _sobel), (w, h))

		cup_result = score_map.argmax(2)

		return cup_result

	def scale_process(self,_image,_sobel):
		_image = np.asarray(_image, dtype=np.float32)
		_sobel = np.asarray(_sobel, dtype=np.float32)
		score = self.caffe_process(_image, _sobel)

		return score

	def caffe_process(self,_image,_sobel):
		h, w, d = _image.shape
		_score = np.zeros((h, w, self.class_num), dtype=np.float32)
		_input = np.zeros((h, w, 4), dtype=np.float32)
		_input[:,:,0:3] = _image
		_input[:,:,3] = _sobel
		_input = _input.transpose(2, 0, 1)
		_input = _input.reshape((1,) + _input.shape)
		self._net.blobs['data'].reshape(*_input.shape)
		self._net.blobs['data'].data[...] = _input

		self._net.forward()
		_score += self._net.blobs[self.prob_layer].data[0].transpose(1, 2, 0)

		return _score





