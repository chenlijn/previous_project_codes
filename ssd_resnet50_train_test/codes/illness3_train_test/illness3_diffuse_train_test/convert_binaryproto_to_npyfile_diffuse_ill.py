import caffe 
import sys
import numpy as np 
import os

binaryprotofile = '../../../data/lmdbs/illness3_mean_diffuse.binaryproto'
savefile = 'illness3_mean_diffuse.npy'

blob = caffe.proto.caffe_pb2.BlobProto()
data = open(binaryprotofile, 'rb').read()
blob.ParseFromString(data)  
arr = np.array(caffe.io.blobproto_to_array(blob)) 
a2 = arr[0]
np.save(savefile, a2)
print a2.shape
