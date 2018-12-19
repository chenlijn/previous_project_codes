#===================================================================
# this file is for deploying the segmentation model 
#===================================================================
import sys
import os
import numpy as np
import cv2
import caffe
import h5py
import time

class UnetCupSegmentation(object):

    def __init__(self, model, weights, meanfile, gpu_index=1):

    #def load_net(model, weights, meanfile):
    
        #set up the GPU and load the net
        caffe.set_device(gpu_index)
        caffe.set_mode_gpu()
        self.net = caffe.Net(model, weights, caffe.TEST) #test mode does not perform dropouts
    
        #set up the preprocessing of images
        inputsize = 512
        inputsize2 = 512
        #mu = np.array([128])
        mu = np.load(meanfile) 
        meanimg = np.transpose(mu, (2,0,1))
        #meanimg = meanimg.reshape(1,meanimg.shape[0], meanimg.shape[1], meanimg.shape[2])
        #mu = np.array( [mu[:,:,0].mean() , mu[:,:,1].mean(), mu[:,:,2].mean()] )
        
    #    print 'mean values: ', zip('BGR', mu)
        #print 'mean: ', meanimg.shape
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})   
        #print 'data shape: ', net.blobs['data'].data.shape
        self.transformer.set_transpose('data', (2, 0, 1)) 
        self.transformer.set_mean('data', meanimg) 
        #transformer.set_raw_scale('data', 255)
        #transformer.set_channel_swap('data', (2,1,0)) 
    
        self.net.blobs['data'].reshape(1, 3, inputsize, inputsize2)  
    
    
    
    #def net_forward(self, imgfile):
    def net_forward(self, img):
        #enhance the images
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        #img = cv2.imread(imgfile)
        
        img[:,:,0] = clahe.apply(img[:,:,0])
        img[:,:,1] = clahe.apply(img[:,:,1])
        img[:,:,2] = clahe.apply(img[:,:,2])
    
        ##image = caffe.io.load_image(imgfile,False)  #read gray image
        #image = caffe.io.load_image(imgfile)  #read color image  
        cv2.imwrite('enhanced_img.png', img)
        transformed_image = self.transformer.preprocess('data', img)
        self.net.blobs['data'].data[...] = transformed_image
    
        output = self.net.forward()
        temp = output['prob'] 
        #temp = output['sigmoid_score'] 
        temp = self.net.blobs['prob'].data  
        #print 'prob size: ', temp.shape
        
        #output_prob = output['prob'][0]
        #print 'prob: ', output_prob
        pred = temp[0,1,:,:]
        pred = np.where(pred >= 0.5, 1, 0)
    
        return pred
        #return temp[0,0,:,:] , temp[0,1,:,:]
        #return temp[0,0,:,:] #, temp[0,1,:,:]




if __name__ == '__main__':
    
    #caffe_root = '/home/gaia/caffe'
    #sys.path.insert()
    workroot = os.getcwd() + '/'


    #model for semantic segmentation   
    model_file = '../models/dense_unet_deploy.prototxt'
    weights = '/home/gaia/share/work/u-net/glaucoma2018/snapshots/dense_unet_iter_44000.caffemodel' 
    meanfile = '/mnt/lijian/mount_out/docker_share/glaucoma/combined_data_set/mean.npy'
    
    valfiles = workroot + '/test.txt' 
    f = open(valfiles,"r") 
    valImgNames = f.read().splitlines()  

    net, transformer = load_net(model_file, weights, meanfile)

    #print the weights
    #encoder
    layername1 = 'conv_d0a-b-3'
    layername2 = 'conv_d0b-c'
    layername = 'conv_d1a-b'
    layername4 = 'conv_d1b-c'
    layername5 = 'conv_d4b-c'

    #decoder
    layername1 = 'upconv_d4c_u3a' 
    layername2 = 'upconv_u3d_u2a' 
    layername3 = 'upconv_u2d_u1a' 
    layername4 = 'upconv_u1d_u0a' 
    layername18 = 'conv_u0c-d' 
    layername19 = 'conv_u0d-score-new' 
    
    w = net.params[layername][0].data[...] 
    b = net.params[layername][1].data[...] 

    #print 'weights shape: ', w.shape
    #np.set_printoptions(threshold=np.nan)
    pf = open('print.txt', 'w+')
    print >>pf, "weights:\n", w
    print >>pf, "bias:\n", b
    pf.close()

    
    

    #imgfile = workroot + '/u-net-release/PhC-C2DH-U373/01/t000.tif'
    img_path = '/mnt/lijian/mount_out/docker_share/glaucoma/test_set/combined_test/images/' 
    mask_path = '/mnt/lijian/mount_out/docker_share/glaucoma/test_set/combined_test/masks/' 
    iou_sum = 0.0 
    test_num = len(valImgNames)       
    for j in range(len(valImgNames)):
        imgfile = valImgNames[j]
        #imgfile = drimgnames[0]
        print imgfile
        maskfile = mask_path + imgfile.split('/')[-1].split('.')[0]+'.png'
        masktemp = cv2.imread(maskfile)
        maskimg = masktemp[:,:,1]

        
        savefile1= '../test/showResults/' + imgfile.split('/')[-1].split('.')[0]+'.png'
        savefile2 = '../test/res1.png'
        savefile3 = '../test/res2.png'
        #savefile4 = workroot + '/mask.png'
        res1, res2 = net_forward(net, transformer, imgfile)
   
   
        diffimg = res2 - maskimg
        img = cv2.imread(imgfile)
        #combimg = np.concatenate((img, res2*255),axis = 1)
        #cv2.imwrite(savefile1, combimg)
        cv2.imwrite(savefile2, res1*255)
        cv2.imwrite(savefile3, res2*255)
        #src_res2 = np.copy(res2)   
        res2_r = np.copy(res2)
        res2_b = np.copy(res2)
        mergeres2 = cv2.merge([res2_b, res2, res2_r])


        #compute the IOU
        rows, colns = np.where(res2 > 0.5)
        res2[:,:] = 0
        res2[rows, colns] = 1
        summask = res2 + maskimg
        intersection = np.count_nonzero(summask == 2)
        union = np.count_nonzero(summask)
        iou = intersection / np.float32(union)
        iou_sum += iou



        #draw the contour on the source image
        rows, colns = np.where(res2 > 0.5)
        res2[:,:] = 0
        res2[rows, colns] = 1
        kernel_size = 3
        kernel = np.ones((kernel_size,kernel_size), np.uint8)
        erosion = cv2.erode(res2, kernel, 1)  
        contour = res2 - erosion
        rows, colns = np.where(contour==1)  
        img[rows, colns, 0] = 0 
        img[rows, colns, 1] = 255 
        img[rows, colns, 2] = 0 

        #draw the ground truth on the source image
        gterosion = cv2.erode(maskimg, kernel, 1)
        gtcont = maskimg - gterosion
        rows, colns = np.where(gtcont == 1)
        img[rows, colns, 0] = 0
        img[rows, colns, 1] = 0 
        img[rows, colns, 2] = 255 

        print "img shape: {}, merger shape: {}".format(img.shape, mergeres2.shape)
        combimg = np.concatenate((img, mergeres2*255),axis = 1)
        cv2.imwrite(savefile1, combimg)


        savefile4 = '../test/summat.png' 
   #     cv2.imwrite(savefile4, summat*255)  

        #difference image  
        #diffimg = res2 - maskimg
        rows, colns = np.where(diffimg != 0)  
        diffimg[:,:] = 0
        diffimg[rows, colns]= 255
        savefile = '../test/prediction_error.png'
        cv2.imwrite(savefile, diffimg)   
    
    print "number of test images: ",test_num,  ", average IOU: ", iou_sum/test_num

    #cv2.imwrite(savefile4, maskdata*255)

    #np.set_printoptions(threshold = np.nan)
    #print res1

    #cv2.imshow('src img', img)

    #cv2.imshow('res1', res1)
    #cv2.imshow('res2', res2)
    
    #cv2.waitKey(0)  
    
    #valf = open(valfiles)
    #filenames = valf.readlines()
    #valf.close()

    #nmap = open(classesmap)
    #classes = nmap.readlines()  
    #nmap.close()    

    #errf = open(recordfile, 'w+')
    #errors = 0
    #total_time = 0 
    #for line in filenames:
    #    imgfile = line.split(' ')[0]  
    #    gt = int(line.split(' ')[1])  
    #    begin = time.time()
    #    prediction = net_forward(net, transformer, imgfile)
    #    total_time = total_time +  time.time() - begin
    #    savefile = misclsImgFolder + '/'+ imgfile.split('/')[-1] 
    #    
    #   
    #    if prediction != gt:
    #        errors += 1 
    #        errf.write(line + 'prediction: ' + str(prediction) + ', ' + classes[prediction].split(' ')[0] +  '\n')
    #        putText = classes[prediction].split(' ')[0]  
    #        img = cv2.imread(imgfile)  
    #        cv2.putText(img,putText, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,255,0), 6)   
    #   
    #  
    # 
    #        cv2.imwrite(savefile, img)  
    #    print line 
    #    

    #errf.close()  
    #valnum = float(len(filenames))
    #print 'mis-classified num: ', errors, 'accuracy: ', (valnum-errors) * 100 / valnum  
    #print 'average time cost per image: ', total_time/valnum, '  seconds' 

    ##imgfile = filenames[0].split(' ')[0]
    ##gt = filenames[0].split(' ')[1]
    #
    ##prediction = net_forward(net, transformer, imgfile)
    ##print 'groud truth: ', gt, '  prediction: ', prediction 



