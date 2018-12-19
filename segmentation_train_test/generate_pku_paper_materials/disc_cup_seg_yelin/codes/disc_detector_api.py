import numpy as np
import sys
import os
import caffe
import cv2
import math
import copy

class disc_detector(object):
	'''
	Detector api class
	'''
	def __init__(self, prototx_addr, caffemodel_addr, gpu_id, mean_val):
		self._prototxt = prototx_addr
		self._caffemodel = caffemodel_addr
		self._gup_id = gpu_id
		self._mean_val = mean_val

		self.img_size = 300

		caffe.set_mode_gpu()
		caffe.set_device(gpu_id)

		self._net = caffe.Net(self._prototxt, self._caffemodel, caffe.TEST)

	def forward_net(self, image, mask):
		src_img = copy.deepcopy(image)
		image_ = cv2.resize(image,(self.img_size, self.img_size))
		in_ = np.array(image_, dtype=np.float32)
		in_ -= np.array(self._mean_val)
		in_ = in_.transpose((2, 0, 1))

		self._net.blobs['data'].reshape(1, 3, self.img_size, self.img_size)
		self._net.blobs['data'].data[...] = in_

		detection_result = self._net.forward()['detection_out'][0,0,:,:]
		sorted_result = detection_result[detection_result[:,2].argsort()[::-1]] #sorted w.r.t. the confidence
		discBox = sorted_result[0,3:7]

		#test
		print("discBox = "+str(discBox))

		discBox[0] *= image.shape[1]
		discBox[1] *= image.shape[0]
		discBox[2] *= image.shape[1]
		discBox[3] *= image.shape[0]

		cent_x = (discBox[0] + discBox[2])/2.0
		cent_y = (discBox[1] + discBox[3])/2.0

		box_w = discBox[2] - discBox[0]
		box_h = discBox[3] - discBox[1]

		cut_w = box_w * 3
		cut_h = box_h * 3

		base_size = 512
		img_height = image.shape[0]
		img_width = image.shape[1]

		if cut_w < base_size:
			cut_w = base_size
		if cut_h < base_size:
			cut_h = base_size

		half_width = math.ceil(cut_w/2)
		cut_w = half_width*2
		half_height = math.ceil(cut_h/2)
		cut_h = half_height*2


		if img_height < cut_h or img_width < cut_w:
                    cut_img = cv2.resize(src_img, (base_size, base_size))
                    cut_mask = cv2.resize(mask, (base_size, base_size))
                    cut_mask = np.where(cut_mask > 0, 255, 0)
                    return cut_img, cut_mask
			#raise Exception("the shape of image is too small")

		if cent_x+half_width > img_width:
			cent_x = img_width - half_width
		if cent_y+half_height > img_height:
			cent_y = img_height - half_height
	
		if cent_x-half_width < 0:
			cent_x = half_width
		if cent_y-half_height < 0:
			cent_y = half_height

                cut_h = np.int(cut_h)
                cut_w = np.int(cut_w)
                cent_x = np.int(cent_x)
                cent_y = np.int(cent_y)
                half_height = np.int(half_height)
                half_width = np.int(half_width)


		cut_img = np.zeros((cut_h, cut_w,3), dtype=np.uint8)
		cut_mask = np.zeros((cut_h, cut_w,3), dtype=np.uint8)
		cut_img[0:cut_h,0:cut_w,0:3] = src_img[cent_y-half_height:cent_y+half_height,cent_x-half_width:cent_x+half_width,0:3]
		cut_mask[0:cut_h,0:cut_w,0:3] = mask[cent_y-half_height:cent_y+half_height,cent_x-half_width:cent_x+half_width,0:3]
		cut_img = cv2.resize(cut_img,(base_size,base_size))
		cut_mask = cv2.resize(cut_mask,(base_size,base_size))
                cut_mask = np.where(cut_mask > 0, 255, 0)

		return cut_img, cut_mask







