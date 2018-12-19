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
    #print 'probs shape: ', temp.shape
    
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


    ##model for classifing the imaging types
    #model_file = '/home/gaia/share/cataractData/codes/resnet-protofiles-master/ResNet50_imaging_types_deploy.prototxt'
    #weights = '/home/gaia/share/cataractData/codes/snapshots/resnet50_imaging_types_iter_90000.caffemodel'
    #meanfile = '/home/gaia/share/cataractData/codes/lmdb/imaging_types_mean.npy'
    #recordfile = workroot + '/codes/misclassification.txt'
    #misclsImgFolder = workroot + '/codes/misclassifications'
    #valfiles = workroot + '/allData/imaging_types_val.txt'  
    #classesmap = workroot + '/allData/imaging_type_name_mapping.txt'    

    #model for determining the illness
    model_file = '/home/gaia/share/cataractData/codes/resnet-protofiles-master/ResNet50_severity_slit_deploy.prototxt'
    #weights = '/home/gaia/share/cataractData/codes/snapshots/resnet_illness3_iter_90000.caffemodel'
    #weights = '/home/gaia/share/cataractData/codes/snapshots/resnet_illness3_pure_iter_100000.caffemodel'
    weights = '/home/gaia/share/cataractData/codes/trained_models_dec_2017/resnet_severity_slit_iter_40000.caffemodel'
    #meanfile = '/home/gaia/share/cataractData/codes/lmdb/imaging_types_mean.npy'
    meanfile = '/home/gaia/share/cataractData/codes/trained_models_dec_2017/severity_slit_mean.npy'
    #meanfile = '/home/gaia/share/cataractData/codes/cataract_project_output/cataract_illness_classifier_v0_20171107/illness3_mean.npy'
    recordfile = workroot + '/codes/misclassification_illness3.txt'
    misclsImgFolder = workroot + '/codes/misclassifications_diag'
    #valfiles = workroot + '/allData/illness3_val.txt'  
    #valfiles = workroot + '/augmented_data/illness3_val.txt'  
    #valfiles = '/mnt/lijian/mount_out/codes/cataract/combined_illness_train_val/val_diffuse.txt'
    #valfiles = '/mnt/lijian/mount_out/codes/cataract/combined_illness_test/test_diffuse.txt'
    #valfiles = '/mnt/lijian/mount_out/codes/cataract/combined_illness_test/small_pupil_diffuse_light_test.txt'
    #valfiles = '/mnt/lijian/mount_out/codes/cataract_severity/slit/combined_test/severity_slit_test.txt'
    valfiles = '/mnt/lijian/mount_out/codes/cataract_severity/slit/combined_test/mydriasis_slit_light_test.txt'
    #valfiles = '/mnt/lijian/mount_out/codes/cataract_severity/slit/combined_test/small_pupil_slit_light_test.txt'
    #valfiles = workroot + '/codes/cataract_project_output/cataract_illness_classifier_v0_20171107/illness3_val.txt'  
    #valfiles = workroot + '/codes/cataract_project_output/txtfiles/illness3_val_diffuse_light_mydriasis.txt'  
    #valfiles = workroot + '/codes/cataract_project_output/txtfiles/illness3_val_slit_light_small.txt'  
    #valfiles = workroot + '/codes/cataract_project_output/txtfiles/illness3_val_slit_light_small.txt'  
    #valfiles = workroot + '/augmented_data/split_data/val_ori_mydriasis_diffuse_light.txt'
    #valfiles = workroot + '/allData/ill_type2_unseen.txt'  
    #classesmap = workroot + '/allData/illness3_name_mapping.txt'    
    classesmap = '/mnt/lijian/mount_out/codes/cataract_severity/slit/cataract_severity_name_mapping.txt'
    correctResults = workroot + '/codes/allResults'


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
    label_prob = np.zeros([len(filenames), 3], np.float)

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
    h_true_negative = 0
    i_true_negative = 0
    s_true_negative = 0 
    true_neg0 = 0
    true_neg1 = 0
    true_neg2 = 0
    valnum = float(len(filenames))
    for idx, line in enumerate(filenames):
        #print 'line', line
        gt_s = line.split(' ')[-1]
        imgfile = line.strip(gt_s).strip()
        checkimg = cv2.imread(imgfile,cv2.IMREAD_UNCHANGED)
        if checkimg == None:
            print 'skip empty image'
            valnum -= 1
            continue
        #print 'img: ', imgfile
        #imgfile = line.split(' ')[0]  
        #gt = int(line.split(' ')[1])  
        gt = int(gt_s)
        begin = time.time()
        prediction = net_forward(net, transformer, imgfile)
        total_time = total_time +  time.time() - begin
        savefile = misclsImgFolder + '/'+ imgfile.split('/')[-1] 
        saveCorrectFile = correctResults + '/'+ imgfile.split('/')[-1]
        
        #record the confusion matrix
        confusion_mat[prediction.argmax(), gt] += 1

        #record the label_prob
        label_prob[idx, 0:2] = prediction[:]
        label_prob[idx, 2] = gt 
       
        #compute the accuracy with respect to every classes
        if gt == 0:
            type0_total_num += 1
            if prediction.argmax() == gt:
                type0_correct_num += 1
        else:
            if prediction.argmax() != 0:
                h_true_negative += 1

        if gt == 1:
            type1_total_num += 1
            if prediction.argmax() == gt:
                type1_correct_num += 1
        else:
            if prediction.argmax() != 1:
                i_true_negative += 1
            
        if gt == 2:
            type2_total_num += 1
            if prediction.argmax() == gt:
                type2_correct_num += 1
        else:
            if prediction.argmax() != 2:
                s_true_negative += 1

        # print 'h,i,s true neg:', h_true_negative, i_true_negative, s_true_negative
        print "label: {}, prediction: {}".format(gt, prediction.argmax())   

        #elif gt == 3:
        #    type3_total_num += 1
        #    if prediction == gt:
        #        type3_correct_num += 1

        # #save the correctly predict images
        # putText1 = 'gtrue: ' + classes[gt].split(' ')[0]
        # putText2 = 'predict: ' + classes[prediction.argmax()].split(' ')[0]
        # img = cv2.imread(imgfile)
        # cv2.putText(img,putText1, (100,200), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,255,0), 6)
        # cv2.putText(img,putText2, (100,400), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,255,0), 6)
        # #cv2.imwrite(saveCorrectFile, img)

        ##count the classified positive and negative numbers
        #if (prediction != 0) and (gt = 0):
        #    posit0 += 1
        #elif prediction == 1 and gt != 1:
        #    posit1 += 1
        #elif prediction == 2:
        #    posit2 += 1
        

        if prediction.argmax() != gt:
            errors += 1 
            errf.write(line + 'prediction: ' + str(prediction.argmax()) + ', ' + classes[prediction.argmax()].split(' ')[0] +  '\n')
            #cv2.imwrite(savefile, img)  

        # print line 
        

    errf.close()  
    #save the confusion matrix
    savefd = "./paper_material/severity_"
    con_mat_savename = savefd + valfiles.split('/')[-1].split('.')[0] + '_cm.npy'
    np.save(con_mat_savename, confusion_mat)
    print confusion_mat 

    # save the label-prob matrix
    lb_prob_savename = savefd + valfiles.split('/')[-1].split('.')[0] + '_label_prob.npy'
    np.save(lb_prob_savename, label_prob)   
    print label_prob


    type0_total_num = max(1.0, type0_total_num)
    type1_total_num = max(1.0, type1_total_num)
    #type2_total_num = max(1.0, type2_total_num)

    acc0 = type0_correct_num/float(type0_total_num)
    acc1 = type1_correct_num/float(type1_total_num)
    #acc2 = type2_correct_num/float(type2_total_num)
    aver_acc = (acc0 + acc1) / 2.0 

    total_num = type0_total_num + type1_total_num#total number of the images

    #specificity, the portion of true negative (not the right classes)
    spec0 = float(h_true_negative)/(type1_total_num)
    spec1 = float(i_true_negative)/(type0_total_num)

    #calculate the overall accuracy
    overall_acc0 = float(type0_correct_num + type1_correct_num + type2_correct_num) / (type0_total_num + type1_total_num + type2_total_num)
    overall_acc1 = float(sum(confusion_mat.diagonal())) / confusion_mat.sum()
    
    print 'the overall accuracy: acc0: {}, acc1: {}'.format(overall_acc0, overall_acc1)


    print 'data distribution: ', classes[0]+': ', type0_total_num, classes[1]+': ', type1_total_num,  'total # of images: ', total_num  
    print 'Specificity: slight spec0: {}, severe spec1: {}'.format(spec0, spec1)  
    print 'Sensitivity: slight accurarcy: {0}, severe accurarcy: {1}, average accuracy: {2}'.format(acc0, acc1,aver_acc)   
    print 'total mis-classified num: ', errors, 'total accuracy: ', (valnum-errors)  / valnum  
    print 'average time cost per image: ', total_time/valnum, '  seconds' 


    #imgfile = filenames[0].split(' ')[0]
    #gt = filenames[0].split(' ')[1]
    
    #prediction = net_forward(net, transformer, imgfile)
    #print 'groud truth: ', gt, '  prediction: ', prediction 



