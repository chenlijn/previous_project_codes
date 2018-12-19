import caffe 
import sys
import numpy as np 
import os

#binaryprotofile = '/home/gaia/share/cataractData/codes/lmdb_new_dec/illness3_mean_diffuse.binaryproto'
binaryprotofile = '/home/gaia/share/cataractData/codes/optic_axis/lmdb/optic_axis_mean.binaryproto'
#savefile = os.getcwd() + '/lmdb_new_dec/illness3_mean_diffuse.npy'
savefile = os.getcwd() + '/lmdb/optic_axis_mean.npy'
#savefile = os.getcwd() + '/lmdb_961/illness3_mean_diffuse.npy'
#savefile = os.getcwd() + '/lmdb_severity/severity_slit_mean.npy'
#savefile = os.getcwd() + '/lmdb_imaging_types/imaging_types_mean.npy'
#savefile = os.getcwd() + '/lmdb_imaging_types/imaging_types_mean.npy'
#savefile = os.getcwd() + '/lmdb_severity/severity_slit_mean.npy'
blob = caffe.proto.caffe_pb2.BlobProto()
data = open(binaryprotofile, 'rb').read()
blob.ParseFromString(data)  
arr = np.array(caffe.io.blobproto_to_array(blob)) 
a2 = arr[0]
np.save(savefile, a2)
print a2.shape
