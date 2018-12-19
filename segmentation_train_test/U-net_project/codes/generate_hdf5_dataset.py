#generate HDF5 data files

import os
import numpy as np 
import h5py as h5
import cv2

workroot = '/mnt/lijian/mount_out/docker_share/glaucoma'
file_path = workroot + '/combined_data_set'
img_path = file_path + '/images'
aug_img_path = file_path + '/enhanced_images'
sobel_img_path = file_path + '/sobelImgs'
h5_folder = file_path + '/hdf5'
mask_path = file_path + '/masks' 
tf = open('train_h5.txt', 'w+')
vf = open('val.txt', 'w+')

if not os.path.exists(aug_img_path):
    os.makedirs(aug_img_path)

if not os.path.exists(sobel_img_path):
    os.makedirs(sobel_img_path)

if not os.path.exists(h5_folder):
    os.makedirs(h5_folder)

imgnames = os.listdir(img_path)
imgNum = len(imgnames)
#os.listdir(mask_path)  
print imgNum 
##enhance the images
#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#i = 0
#for name in imgnames:
#    imgfile = img_path + '/' + name
#    img = cv2.imread(imgfile, cv2.IMREAD_UNCHANGED)
#
#    img[:,:,0] = clahe.apply(img[:,:,0])
#    img[:,:,1] = clahe.apply(img[:,:,1])
#    img[:,:,2] = clahe.apply(img[:,:,2])
#    savefile = aug_img_path + '/' + name
#    cv2.imwrite(savefile, img) 
#    #print savefile 
#    print i
#    i += 1
#
#    #sobel image
#    gray = cv2.imread(imgfile, 0)
#    hedges = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = 3)
#    vedges = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = 3)
#    hvedges = (np.abs(hedges) + np.abs(vedges))
#    hvedges /= hvedges.max()
#    hvedges *= 255
#    hvedges.astype(np.uint8) 
#    savefile = sobel_img_path + '/' + name
#    cv2.imwrite(savefile, hvedges)  
   

##compute the mean image
#imgsize = 512
#fimgNum = np.float(imgNum)
#meanimgB = np.zeros([imgsize, imgsize], np.float)
#meanimgG = np.zeros([imgsize, imgsize], np.float)
#meanimgR = np.zeros([imgsize, imgsize], np.float)
#
#for name in imgnames:
#    imgfile = aug_img_path + '/' + name
#    img = cv2.imread(imgfile, cv2.IMREAD_UNCHANGED)
#    meanimgB += img[:,:,0] / fimgNum
#    meanimgG += img[:,:,1] / fimgNum
#    meanimgR += img[:,:,2] / fimgNum
#
#meanimg = cv2.merge([meanimgB, meanimgG, meanimgR])
#np.save(workroot + '/combined_data_set/mean.npy', meanimg)
#print 'mean computed !!'

meanimg = np.load(workroot + '/combined_data_set/mean.npy')
print "mean image shape: ", meanimg.shape 

#generate hd files
i = 0
trainNum = np.int(imgNum)
h5ftrain = os.getcwd() + '/combined_data_set/train.h5'
for name in imgnames:
    print name
    if i > trainNum:
        imgfile = aug_img_path + '/' + name 
        vf.write(imgfile + '\n')
        i += 1
    else:
        imgfile = aug_img_path + '/' + name 
        img = cv2.imread(imgfile) 
        
        ##histogram equalization 
        #img[:,:,0] = cv2.equalizeHist(img[:, :, 0])
        #img[:,:,1] = cv2.equalizeHist(img[:, :, 1])
        #img[:,:,2] = cv2.equalizeHist(img[:, :, 2])

        ##CLAHE equalization 
        #img[:,:,0] = clahe.apply(img[:,:,0])
        #img[:,:,1] = clahe.apply(img[:,:,1])
        #img[:,:,2] = clahe.apply(img[:,:,2])

        tempimg = (img - meanimg)
        
        np.set_printoptions(threshold=np.nan)

        temps = name.split('.')[-1]
        maskfile = mask_path + '/' + name.strip(temps) + 'png' 
        maskimg = cv2.imread(maskfile, cv2.IMREAD_UNCHANGED)

        pureName = name.split('.')[0].replace(" ", "")    

        h5file = h5_folder + '/' + pureName + '.h5'
        f = h5.File(h5file, 'w')  #create an hdf5 file 
      
       
        print 'max: ', np.max(tempimg), ' min: ', np.min(tempimg), ' bottom: ', np.max(tempimg)-np.min(tempimg)   
         
        #tempimg = tempimg / 255.0
        #tempimg = (tempimg - np.min(tempimg)) / (np.max(tempimg) - np.min(tempimg)) #simple scale after mean subtracting 
        if np.max(tempimg)==float('Inf') or np.min(tempimg)==float('NaN'):
            print "Inf or Nan"
            break 
        #newtemp = np.rollaxis(tempimg, axis=2, start=0)

        transImg = np.transpose(tempimg,(2,0,1))


        f['data'] = transImg.reshape(1,tempimg.shape[2], tempimg.shape[0], tempimg.shape[1])  #scale, and transpos  
        print 'transImg shape: ', transImg.shape
        newmask = maskimg.reshape(1,1, maskimg.shape[0], maskimg.shape[1])   
        f['label'] = newmask 
        f.close()
        tf.write(h5file + '\n') 
        i += 1
        #break
tf.close()
vf.close()

        









##write into HDF5 files
#imgData = np.ones([10,10], np.uint8)  
#imgData1 = np.zeros([10,10], np.uint8)  
#f = h5.File(os.getcwd() + '/test.h5', 'w')
#f['data'][0] = imgData
#f['data'][1] = imgData1
#f['labels'] = range(100)  
#f.close()
#
#
#
##read HDF5 files
#f = h5.File(os.getcwd() + '/test.h5', 'r')
#print 'keys: ', f.keys()
#print 'data: ', f['data'][:] 
#f.close()




