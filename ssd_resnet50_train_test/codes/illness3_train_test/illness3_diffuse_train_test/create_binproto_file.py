#create a binary protofile

import numpy as np
import caffe

classNum = 3 
w0 = 1 #5#3/5.0
w1 = 1#1/5.0
w2 = 1#1/5.0
H = np.zeros((classNum,classNum), dtype='float32')

H[0,0] = w0
H[1,1] = w1
#H[1,2] = w1
H[2,2] = w2

blob = caffe.io.array_to_blobproto(H.reshape((1,1,classNum,classNum))) 
print blob.shape
with open('./illness3_loss_weight.binaryproto', 'wb') as f:
    f.write(blob.SerializeToString())  



