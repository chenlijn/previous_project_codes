#=================================================================
#this file is for deploying the catarat imaging method classifier
#=================================================================
import sys
import os
import numpy as np
import cv2
import caffe
import time
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True 


def load_net(model, weights, meanfile):

    #set up the GPU and load the net
    caffe.set_device(1)
    caffe.set_mode_gpu()
    net = caffe.Net(model, weights, caffe.TEST) #test mode does not perform dropouts

    #set up the preprocessing of images
    inputsize = 224
    mu = np.load(meanfile)
    mu = mu.mean(1).mean(1) 
    #print 'mean values: ', zip('BGR', mu)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})   
    transformer.set_transpose('data', (2, 0, 1)) 
    transformer.set_mean('data', mu) 
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2,1,0)) 

    net.blobs['data'].reshape(1, 3, inputsize, inputsize)  

    return net, transformer


def net_forward(net, transformer, imgfile):
    image = caffe.io.load_image(imgfile)
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image

    output = net.forward()
    temp = output['prob'] 
##    print 'probs shape: ', temp.shape
    
    output_prob = output['prob'][0]
    #print 'prob: ', output_prob

    # return output_prob.argmax()
    return output_prob 




if __name__ == '__main__':
    
    #model for determining the illness
    model_file = '../../resnet-protofiles-master-new/ResNet50_illness3_deploy_slit.prototxt'
    weights = 'snapshots/resnet_illness3_slit_iter_100000.caffemodel'
    meanfile = 'illness3_mean_slit.npy'

    recordfile = 'misclassification_illness3.txt'
    misclsImgFolder = 'bad_case'
    #valfiles = '/mnt/lijian/mount_out/codes/cataract/combined_illness_train_val/val_diffuse.txt'
    #valfiles = '/mnt/lijian/mount_out/codes/cataract/combined_illness_test/test_slit.txt'
    #valfiles = '../../../data/cataract_data_txts/cataract/combined_illness_test/clearity_slit_test.txt'
    valfiles = '../../../data/cataract_data_txts/cataract/combined_illness_test/optic_axis_slit_test.txt'
    #valfiles = '/mnt/lijian/mount_out/codes/cataract/combined_illness_test/mydriasis_slit_light_test.txt'
    
    #outside validation set 
    #valfiles = '/mnt/lijian/mount_out/zhongshan_ophthalmology_data20180206/sorted_txt/slit_illness_val.txt'


    classesmap = '../../../data/cataract_data_txts/cataract/cataract_illness_name_mapping.txt'
    correctResults = 'good_case'


    net, transformer = load_net(model_file, weights, meanfile)
    
    
    valf = open(valfiles)
    filenames = valf.readlines()
    valf.close()

    nmap = open(classesmap)
    classes = nmap.readlines()  
    nmap.close()    


    #the confusion matrix
    cls_num = len(classes)
    confusion_mat = np.zeros([cls_num, cls_num], np.int)

    #the label-prob matrix 
    label_prob = np.zeros([len(filenames), 4], np.float)
   

    errf = open(recordfile, 'w+')
    errors = 0
    valnum = float(len(filenames))
    total_time=0
    for idx, line in enumerate(filenames):
        gt_s = line.split(' ')[-1]
        imgfile = line.strip(gt_s).strip()
        checkimg = cv2.imread(imgfile,cv2.IMREAD_UNCHANGED)
        if checkimg is None:
            print 'skip empty image'
            valnum -= 1
            continue
        gt = int(gt_s)
        begin = time.time()
        prediction = net_forward(net, transformer, imgfile)
        total_time = total_time +  time.time() - begin
        savefile = misclsImgFolder + '/'+ imgfile.split('/')[-1] 
        saveCorrectFile = correctResults + '/'+ imgfile.split('/')[-1]
        
        #record the confusion matrix
        confusion_mat[prediction.argmax(), gt] += 1

        #record the lable prob matrix
        label_prob[idx,0:3] = prediction[:]
        label_prob[idx,3] = gt 

        ##save the correctly predict images
        #putText1 = 'gtrue: ' + classes[gt].split(' ')[0]
        #putText2 = 'predict: ' + classes[prediction.argmax()].split(' ')[0]
        #img = cv2.imread(imgfile)
        #cv2.putText(img,putText1, (100,200), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,255,0), 6)
        #cv2.putText(img,putText2, (100,400), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,255,0), 6)
        ##cv2.imwrite(saveCorrectFile, img)

        if prediction.argmax() != gt:
            errors += 1 
            errf.write(line + 'prediction: ' + str(prediction.argmax()) + ', ' + classes[prediction.argmax()].split(' ')[0] +  '\n')
            #cv2.imwrite(savefile, img)  
        #else:
            #cv2.imwrite(saveCorrectFile, img)

        print "ground truth:{}, prediction:{}".format(gt, prediction.argmax()) 
        

    errf.close()  

    #save the confusion matrix
    savefd = "./"
    con_mat_savename = savefd + valfiles.split('/')[-1].split('.')[0] + '_cm.npy'
    np.save(con_mat_savename, confusion_mat)
    print 'sens and specificity: ', confusion_mat

    lb_prob_name = savefd + valfiles.split('/')[-1].split('.')[0] + '_label_prob.npy' 
    np.save(lb_prob_name, label_prob)  
    #print label_prob[0:5,:]

    print 'data distribution: ', classes[0]+': ', confusion_mat[:,0].sum(), classes[1]+': ', confusion_mat[:,1].sum(), classes[2]+': ', confusion_mat[:,2].sum(), 'total # of images: ', confusion_mat.sum()
    print 'average time cost per image: ', total_time/valnum, '  seconds' 


