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
    #inputsize = 961
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
    #print 'probs shape: ', temp.shape
    
    output_prob = output['prob'][0]
    print 'prob: ', output_prob

    return output_prob.argmax()




if __name__ == '__main__':
    
    #caffe_root = '/home/gaia/caffe'
    #sys.path.insert()
    workroot = '/home/gaia/share/cataractData'

    ##googlenet deployment 
    #model_file = '/home/gaia/share/cataractData/codes/bvlc_googlenet/googlenet_deploy.prototxt'
    #weights = '/home/gaia/share/cataractData/codes/bvlc_googlenet/snapshots/googlenet_quick_iter_420000.caffemodel' 
    #meanfile = '/home/gaia/share/cataractData/codes/lmdb_new_Feb/illness3_mean_diffuse.npy'


    ##inception-v3 deployment 
    #model_file = '/home/gaia/share/cataractData/codes/inception_v3/incept_v3_deploy.prototxt'
    #weights = '/home/gaia/share/cataractData/codes/inception_v3/snapshots/inception_v3_iter_220000.caffemodel' 
    #meanfile = '/home/gaia/share/cataractData/codes/lmdb_961/illness3_mean_diffuse.npy'


    ##alexnet deployment 
    #model_file = '/home/gaia/share/cataractData/codes/bvlc_alexnet/deploy.prototxt'
    #weights = '/home/gaia/share/cataractData/codes/bvlc_alexnet/snapshots/caffe_alexnet_train_iter_220000.caffemodel' 
    #meanfile = '/home/gaia/share/cataractData/codes/lmdb_march/illness3_mean_diffuse.npy'


    ##model for determining the illness
    model_file = '/home/gaia/share/cataractData/codes/split_trainning/small_pupil_diffuse/ResNet50_illness3_deploy_diff.prototxt'
    #weights = '/home/gaia/share/cataractData/codes/snapshots_dec/resnet_illness3_diffuse_iter_180000.caffemodel'
    #meanfile = '/home/gaia/share/cataractData/codes/lmdb_march/illness3_mean_diffuse.npy'
    weights = '/home/gaia/share/cataractData/codes/split_trainning/small_pupil_diffuse/snapshots/resnet_sm_pu_diffuse_iter_250000.caffemodel'
    meanfile = '/home/gaia/share/cataractData/codes/split_trainning/small_pupil_diffuse/lmdb/small_pupil_diffuse_mean.npy'
    recordfile = workroot + '/codes/misclassification_illness3.txt'
    misclsImgFolder = workroot + '/codes/miscls_diffuse'
    #valfiles = '/mnt/lijian/mount_out/codes/cataract/combined_illness_train_val/val_diffuse.txt'
    #valfiles = '/mnt/lijian/mount_out/codes/cataract/combined_illness_test/test_diffuse.txt'

    #valfiles = '/mnt/lijian/mount_out/codes/cataract/combined_illness_test/small_pupil_diffuse_light_test.txt'

    #outside validation set
    valfiles = '/mnt/lijian/mount_out/zhongshan_ophthalmology_data20180206/sorted_txt/small_pupil_diffuse_val.txt'
    #valfiles="./outside_test_img/mis_label.txt"
    #valfiles="./outside_test_img/src_label.txt"
    #valfiles = '/mnt/lijian/mount_out/codes/cataract/combined_illness_test/mydriasis_diffuse_light_test.txt'

    classesmap = '/mnt/lijian/mount_out/codes/cataract/cataract_illness_name_mapping.txt'
    correctResults = workroot + '/codes/correctResults_diffuse'

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

    errf = open(recordfile, 'w+')
    errors = 0
    total_time = 0 
    valnum = float(len(filenames))

    for line in filenames:
        gt_s = line.split(' ')[-1]
        imgfile = line.strip(gt_s).strip()
        checkimg = cv2.imread(imgfile,cv2.IMREAD_UNCHANGED)
        if checkimg == None:
            print 'skip empty image'
            valnum -= 1
            continue
        #imgfile = line.split(' ')[0]  
        #gt = int(line.split(' ')[1])  
        gt = int(gt_s)
        begin = time.time()
        prediction = net_forward(net, transformer, imgfile)
        total_time = total_time +  time.time() - begin
        savefile = misclsImgFolder + '/'+ imgfile.split('/')[-1] 
        saveCorrectFile = correctResults + '/'+ imgfile.split('/')[-1]

        #record the confusion matrix
        confusion_mat[prediction, gt] += 1       
       


        #save the correctly predict images
        putText1 = 'gtrue: ' + classes[gt].split(' ')[0]
        putText2 = 'predict: ' + classes[prediction].split(' ')[0]
        img = cv2.imread(imgfile)
        cv2.putText(img,putText1, (100,200), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,255,0), 6)
        cv2.putText(img,putText2, (100,400), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,255,0), 6)
        #cv2.imwrite(saveCorrectFile, img)

        if prediction != gt:
            errors += 1 
            errf.write(line + 'prediction: ' + str(prediction) + ', ' + classes[prediction].split(' ')[0] +  '\n')
        #    cv2.imwrite(savefile, img)  
        #else:
        #    cv2.imwrite(saveCorrectFile, img)

        print 'ground truth: {}, prediction: {}'.format(gt, prediction)

        

    errf.close()  

    #save the confusion matrix
    con_mat_savename = valfiles.split('/')[-1].split('.')[0] + '_cm.npy'
    np.save(con_mat_savename, confusion_mat)
    print "sensi and specificity:\n ", confusion_mat 


    #calculate the overall accuracy 
    overall_acc = float(sum(confusion_mat.diagonal())) / confusion_mat.sum()

    print 'the overall accuracy: acc: {}'.format(overall_acc)      


    print 'data distribution: ', classes[0]+': ', confusion_mat[:,0].sum(), classes[1]+': ', confusion_mat[:,1].sum(), classes[2]+': ', confusion_mat[:,2].sum(), 'total # of images: ', confusion_mat.sum()
    print 'average time cost per image: ', total_time/valnum, '  seconds' 





