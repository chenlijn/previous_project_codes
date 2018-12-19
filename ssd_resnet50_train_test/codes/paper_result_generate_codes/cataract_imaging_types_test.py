#=================================================================
#this file is for deploying the catarat imaging method classifier
#=================================================================
import sys
import os
import numpy as np
import cv2
import caffe
import time
from random import shuffle
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True 

def load_net(model, weights, meanfile):

    #set up the GPU and load the net
    caffe.set_device(0)
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
    print 'probs shape: ', temp.shape
    
    output_prob = output['prob'][0]
    #print 'prob: ', output_prob

    # return output_prob.argmax()
    return output_prob




if __name__ == '__main__':
    
    #caffe_root = '/home/gaia/caffe'
    #sys.path.insert()
    workroot = '/home/gaia/share/cataractData'

    ##model for classifing the imaging types
    #model_file = '/home/gaia/share/cataractData/codes/resnet-protofiles-master/ResNet_50_deploy.prototxt'
    #weights = '/home/gaia/share/cataractData/codes/trained_models/resnet_iter_imaging_40000.caffemodel'
    #meanfile = '/home/gaia/share/cataractData/codes/cataract_mean.npy'
    #recordfile = workroot + '/codes/misclassification.txt'
    #misclsImgFolder = workroot + '/codes/misclassifications'
    #valfiles = workroot + '/testimgs/val.txt'  
    #classesmap = workroot + '/testimgs/name_mapping.txt'    

    ##model for determining the illness
    #model_file = '/home/gaia/share/cataractData/codes/resnet-protofiles-master/ResNet50_diag_deploy.prototxt'
    #weights = '/home/gaia/share/cataractData/codes/trained_models/resnet_iter_diag_40000.caffemodel'
    #meanfile = '/home/gaia/share/cataractData/codes/lmdb/cataract_diag_mean.npy'
    #recordfile = workroot + '/codes/misclassification_diag.txt'
    #misclsImgFolder = workroot + '/codes/misclassifications_diag'
    #valfiles = workroot + '/cat_tv/val.txt'  
    #classesmap = workroot + '/cat_tv/name_mapping.txt'    


    #model for classifing the imaging types
    model_file = '/home/gaia/share/cataractData/codes/resnet-protofiles-master/ResNet50_imaging_types_deploy.prototxt'
    #weights = '/home/gaia/share/cataractData/codes/snapshots/resnet50_imaging_types_iter_250000.caffemodel'
    weights = '/home/gaia/share/cataractData/codes/snapshots_dec_v0/resnet50_imaging_types_iter_175000.caffemodel'
    meanfile = '/home/gaia/share/cataractData/codes/lmdb_imaging_types/imaging_types_mean.npy'
    recordfile = workroot + '/codes/misclassification.txt'
    misclsImgFolder = workroot + '/codes/miscls_img_types'
    correctResults = workroot + '/codes/correctResults_img_types'
    #valfiles = workroot + '/allData/imaging_types_val.txt'  
    valfiles = '/mnt/lijian/mount_out/codes/cataract/imaging_types/combined_test/imaging_types_test.txt'  

    #outside validation set
    # valfiles = '/mnt/lijian/mount_out/zhongshan_ophthalmology_data20180206/sorted_txt/imaging_types_val.txt'
    #classesmap = workroot + '/allData/imaging_type_name_mapping.txt'    
    classesmap = '/mnt/lijian/mount_out/codes/cataract/imaging_types/imaging_types_name_mapping.txt' 

    net, transformer = load_net(model_file, weights, meanfile)
    
    
    valf = open(valfiles)
    filenames = valf.readlines()
    #shuffle(filenames)  
    valf.close()

    nmap = open(classesmap)
    classes = nmap.readlines()  
    nmap.close()    

    #the confusion matrix
    cls_num = len(classes)
    confusion_mat = np.zeros([cls_num, cls_num], np.int)

    #the label-prob matrix
    label_prob = np.zeros([len(filenames), 5], np.float)


    errf = open(recordfile, 'w+')
    errors = 0
    total_time = 0 
    type0_total_num = 0
    type0_correct_num = 0
    type1_total_num = 0
    type1_correct_num = 0
    type2_total_num = 0
    type2_correct_num = 0
    type3_total_num = 0
    type3_correct_num = 0
    for idx, line in enumerate(filenames):
        # print line 
        #linesplit = line.split(' ')[0]  
        gt_s = (line.split(' ')[-1]) 
        imgfile = line.strip(gt_s).strip() 
        #imgfile = '/mnt/lijian/mount_out/zhongshan_ophthalmology_cataract_images_20170906/ill/mydriasis_diffuse_light/90169.jpg'
        gt = int(gt_s)
        begin = time.time()
        #print "here !"
        prediction = net_forward(net, transformer, imgfile)
        total_time = total_time +  time.time() - begin
        savefile = misclsImgFolder + '/'+ imgfile.split('/')[-1] 
        saveCorrectFile = correctResults + '/'+ imgfile.split('/')[-1] 
        print 'label: ',gt,  'prediction: ', prediction.argmax() 
        
        #record the confusion matrix
        confusion_mat[prediction.argmax(), gt] += 1

        # record label-prob
        label_prob[idx,0:4] = prediction[:]
        label_prob[idx,4] = gt 

       
        #compute the accuracy with respect to every classes
        if gt == 0:
            type0_total_num += 1
            if prediction.argmax() == gt:
                type0_correct_num += 1
        elif gt == 1:
            type1_total_num += 1
            if prediction.argmax() == gt:
                type1_correct_num += 1
        elif gt == 2:
            type2_total_num += 1
            if prediction.argmax() == gt:
                type2_correct_num += 1
        elif gt == 3:
            type3_total_num += 1
            if prediction.argmax() == gt:
                type3_correct_num += 1

        #save the correctly predict images
        putText1 = 'gtrue: ' + classes[gt].split(' ')[0]
        putText2 = 'predict: ' + classes[prediction.argmax()].split(' ')[0]   
        img = cv2.imread(imgfile)  
        cv2.putText(img,putText1, (100,200), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,255,0), 6)   
        cv2.putText(img,putText2, (100,400), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,255,0), 6)   
        #cv2.imwrite(saveCorrectFile, img)  

        if prediction.argmax() != gt:
            errors += 1 
            errf.write(line + 'prediction: ' + str(prediction.argmax()) + ', ' + classes[prediction.argmax()].split(' ')[0] +  '\n')
            #putText = classes[prediction].split(' ')[0]  
            #img = cv2.imread(imgfile)  
            #cv2.putText(img,putText, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,255,0), 6)   
            cv2.imwrite(savefile, img)  
        else:
            cv2.imwrite(saveCorrectFile, img)  



        #print line 
        

    errf.close()  
    #save the confusion matrix
    savefd = "./paper_material/"
    con_mat_savename = savefd + valfiles.split('/')[-1].split('.')[0] + '_cm.npy'
    np.save(con_mat_savename, confusion_mat)

    lb_prob_savename = savefd + valfiles.split('/')[-1].split('.')[0] + '_label_prob.npy' 
    np.save(lb_prob_savename, label_prob)   
    # print label_prob


    valnum = float(len(filenames))
    acc0 = type0_correct_num/max(1.0, float(type0_total_num))
    acc1 = type1_correct_num/max(1.0, float(type1_total_num))
    acc2 = type2_correct_num/max(1.0, float(type2_total_num))
    acc3 = type3_correct_num/max(1.0, float(type3_total_num)) 
    aver_acc = (acc0 + acc1 + acc2 + acc3) / 4.0 


    #compute the specificity: true negative / all negatives
    spec0 = float(type1_correct_num + type2_correct_num + type3_correct_num) / max(1.0, (type1_total_num + type2_total_num + type3_total_num))  
    spec1 = float(type0_correct_num + type2_correct_num + type3_correct_num) / max(1.0, (type0_total_num + type2_total_num + type3_total_num))  
    spec2 = float(type1_correct_num + type0_correct_num + type3_correct_num) / max(1.0, (type1_total_num + type0_total_num + type3_total_num))  
    spec3 = float(type1_correct_num + type2_correct_num + type0_correct_num) / max(1.0, (type1_total_num + type2_total_num + type0_total_num))  

    #calculate the overall accuracy
    #overall_acc0 = float(type0_correct_num + type1_correct_num + type2_correct_num) / max(1.0, (type0_total_num + type1_total_num + type2_total_num))  
    overall_acc1 = float(sum(confusion_mat.diagonal())) / confusion_mat.sum()
    
    print 'the overall accuracy: acc1: {}'.format(overall_acc1)



    #aver_acc = (acc0 + acc1) / 2.0 
    total_data_num = type0_total_num + type1_total_num + type2_total_num + type3_total_num
    print 'data distribution: type0: {0}, tyep1: {1}, type2: {2}, type3: {3}, total data: {4}'.format(type0_total_num, type1_total_num, type2_total_num, type3_total_num, total_data_num) 
    print 'sensitivity: type 0 accurarcy: {0}, type 1 accurarcy: {1}, type 2 accuracy: {2}, type 3 accuracy: {3}, average accuracy: {4}'.format(acc0, acc1, acc2, acc3, aver_acc)   
    print 'specificity:  type 0 spec: {0}, type 1 spec: {1}, type 2 spec: {2}, type 3 spec: {3}'.format(spec0, spec1, spec2, spec3)   
    print 'total mis-classified num: ', errors, 'total accuracy: ', (valnum-errors) * 100 / valnum  
    print 'average time cost per image: ', total_time/valnum, '  seconds' 

    with open('outputs.txt', 'w+') as f:
        f.write(str(spec0))
        f.write(str(spec1))
        f.write(str(spec2))
        f.write(str(spec3))

    #imgfile = filenames[0].split(' ')[0]
    #gt = filenames[0].split(' ')[1]
    
    #prediction = net_forward(net, transformer, imgfile)
    #print 'groud truth: ', gt, '  prediction: ', prediction 



