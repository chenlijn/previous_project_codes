import os
import numpy as np
import cv2
import h5py


workroot = '/mnt/lijian/mount_out/docker_share/glaucoma/combined_data_set'
h5folder = workroot + '/hdf5/'
mean_img = np.load('/mnt/lijian/mount_out/docker_share/glaucoma/combined_data_set/mean.npy')

allfiles = os.listdir(h5folder) 
hfile = h5folder + allfiles[2500]

f = h5py.File(hfile, 'r')

data = f['data'][:]
print "h5 image data: ", data.shape
tempimg = data[0,:,:,:]

img = np.transpose(tempimg, (1,2,0))#tempimg.reshape(tempimg.shape[1], tempimg.shape[2], tempimg.shape[0])   
img += mean_img 
#tempimg2 = img[:,:,0]
savefile = 'check_data.png'
cv2.imwrite(savefile, img)

#print "b image shape: ", b

labeldata = f['label'][:]
print 'h5mask shape: ', labeldata.shape
maskdata = labeldata[0,0,:,:]
savefile = 'mask.png'
cv2.imwrite(savefile, maskdata*255)

# draw mask on image
temp_mask = np.copy(maskdata)
kernel = np.ones((5,5), np.uint8)
erosion = cv2.erode(temp_mask, kernel, iterations=1)
edge = maskdata - erosion
rows, cols = np.where(edge > 0)
img[rows,cols,0] = 0
img[rows,cols,1] = 255
img[rows,cols,2] = 0
cv2.imwrite("check_match.png", img)


