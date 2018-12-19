# coding: utf-8

"""
Kan's module for data processing
"""

import os
import sys
import numpy as np
import cv2
import shutil
import random
import base64
from hashlib import md5


class KanDataProc(object):
    """data processing functions"""
    
    def __init__(self):
        # private data
        self.__cwd = os.getcwd()
        self.__suffix = ".txt"
        self.__split_perc = [0.7, 0.2, 0.1]
        return


    def split_train_val_test(self, data_list_file, dst_dir):
        in_name = os.path.splitext(data_list_file)[0]
        train_name = dst_dir + in_name + "_train" + self.__suffix
        val_name = dst_dir + in_name + "_val" + self.__suffix
        test_name = dst_dir + in_name + "_test" + self.__suffix

        # read files and split
        with open(data_list_file, 'r') as df:
            data_list = df.readlines()
            random.shuffle(data_list)
            data_num = len(data_list)
            train_limit = int(data_num * self.__split_perc[0])
            val_limit = int(data_num * sum(self.__split_perc[:2]))
            with open(train_name, "w+") as trf:
                with open(val_name, "w+") as vf:
                    with open(test_name, "w+") as tf:
                        for idx, data in enumerate(data_list):
                            if idx <= train_limit:
                                trf.write(data)
                            elif idx <= val_limit:
                                vf.write(data)
                            else:
                                tf.write(data)
        return

    
    def filter_mask_img(self, maskimg):
        hist_val, bin_edges = np.histogram(maskimg, bins = range(256))
        #print hist_val
        mask_val = bin_edges[hist_val.argsort()[-2]] # the value with the second largest occurrence #
                                                     # the most frequent value is zero.
        #rows, cols = np.where(maskimg == mask_val) 
        rows, cols = np.where(maskimg > 0) 
        temp_mask = np.zeros_like(maskimg)
        temp_mask[rows, cols] = np.uint8(1)

        kernel = np.ones((3,3), np.uint8) 
        erosion = cv2.erode(temp_mask,kernel,iterations = 1)
        dilation = cv2.dilate(erosion, kernel, iterations = 1) 

        return dilation
        #return temp_mask # return the filtered mask 
            

kan_data_proc = KanDataProc()

#if __name__ == "__main__":

