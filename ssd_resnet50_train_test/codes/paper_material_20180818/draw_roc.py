
'''draw ROC curve and compute the Area under curve'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc
from statsmodels.stats.proportion import proportion_confint  #.stats #as stats
from pandas import DataFrame

from utils_lijian import utils_lijian as utl 


# namemap_file = '/mnt/lijian/mount_out/codes/cataract/imaging_types/imaging_types_name_mapping.txt'
# with open(namemap_file) as f:
#     names

## imaging types
#label_prob = np.load("imaging_types_test_label_prob.npy")
#conf_mat = np.load("imaging_types_test_cm.npy")
#prefix = "imgtype_"
#label_names = ["mydriasis_diffuse_light", "mydriasis_slit_light", "small_pupil_diffuse_light", "small_pupil_slit_light"]
#record_filename = "imaging_types_intervals.txt"

## illness types, mydriasis diffuse
#label_prob = np.load("diffuse_illness_test_data_label_prob.npy")
#conf_mat = np.load("diffuse_illness_test_data_cm.npy")
#prefix = "diffuse_illness_"
#label_names = ["healthy", "ill"]   
#record_filename = "diffuse_illness_intervals.txt"

## illness types, mydriasis diffuse
#label_prob = np.load("illness_test_data_label_prob.npy")
#conf_mat = np.load("slit_illness_test_data_cm.npy")
#prefix = "slit_illness_"
#label_names = ["healthy", "ill"]   
#record_filename = "slit_illness_intervals.txt"

## illness types, mydriasis slit
#label_prob = np.load("illness_label_prob.npy")
#conf_mat = np.load("illness_cm.npy")
#prefix = "slit_illness_"
#label_names = ["healthy", "ill"]   
#record_filename = "slit_illness_intervals.txt"


## illness types, small pupil diffuse
#label_prob = np.load("small_pupil_diffuse_light_test_label_prob.npy")
#conf_mat = np.load("small_pupil_diffuse_light_test_cm.npy")
#prefix = "illness_smpu_dif_"
#label_names = ["healthy", "ill", "after_surgery"]   
#record_filename = "illness_smpu_dif_intervals.txt"

## illness types, small pupil slit 
#label_prob = np.load("small_pupil_slit_light_test_label_prob.npy")
#conf_mat = np.load("small_pupil_slit_light_test_cm.npy")
#prefix = "illness_smpu_slit_"
#label_names = ["healthy", "ill", "after_surgery"]   
#record_filename = "illness_smpu_slit_intervals.txt"

## illness types, slit
#label_prob = np.load("test_slit_label_prob.npy")
#conf_mat = np.load("test_slit_cm.npy")
#prefix = "slit_"
#label_names = ["healthy", "ill", "after_surgery"]   
#record_filename = "test_slit_intervals.txt"

## severity, mydriasis diffuse
#label_prob = np.load("severity_mydriasis_diffuse_light_test_label_prob.npy")
#conf_mat = np.load("severity_mydriasis_diffuse_light_test_cm.npy")
#prefix = "severity_mydr_dif_"
#label_names = ["slight", "severe"]   
#record_filename ="severity_mydr_dif_intervals.txt"

## severity, small pupil diffuse
#label_prob = np.load("severity_small_pupil_diffuse_light_test_label_prob.npy")
#conf_mat = np.load("severity_small_pupil_diffuse_light_test_cm.npy")
#prefix = "severity_smpu_dif_"
#label_names = ["slight", "severe"]   
#record_filename ="severity_smpu_dif_intervals.txt"

# severity, small pupil slit
label_prob = np.load("severity_severity_val_label_prob.npy")
conf_mat = np.load("severity_severity_val_cm.npy")
prefix = "severity_smpu_slit_"
label_names = ["slight", "severe"]   
record_filename ="severity_smpu_slit_intervals.txt"

## severity, mydriasis, slit
#label_prob = np.load("severity_mydriasis_slit_light_test_label_prob.npy")
#conf_mat = np.load("severity_mydriasis_slit_light_test_cm.npy")
#prefix = "severity_mydr_slit_"
#label_names = ["slight", "severe"]   
#record_filename ="severity_mydr_slit_intervals.txt"


##optic axis
#label_prob = np.load("optic_axis_test_label_prob.npy")
#conf_mat = np.load("optic_axis_test_cm.npy")
#prefix = "optic_axis_"
#label_names = ["optic_axis_not_involved", "optic_axis_involved"]
#record_filename = "optic_axis_intervals.txt"

##occlusion
#label_prob = np.load("occlusion_test_label_prob.npy")
#conf_mat = np.load("occlusion_test_cm.npy")
#prefix = "occlusion_"
#label_names = ["partial_occlusion", "complete_occlusion"]
#record_filename = "occlusion_intervals.txt"

##clearity 
#label_prob = np.load("clearity_test_label_prob.npy")
#conf_mat = np.load("clearity_test_cm.npy") 
##print label_prob 
#prefix = "clearity_"
#label_names = ["clear", "opaque"]
#record_filename = "clearity_intervals.txt"


#----------for 3 x 3 matrix ----------------------------------------
## combine 3x3 matrix into 2x2 matrix
#new_conf_mat = conf_mat[:2, :2]
#new_conf_mat[0,0] += conf_mat[2,2]
#new_conf_mat[0,0] += conf_mat[2,0] + conf_mat[2,0] + conf_mat[0,2]
#new_conf_mat[0,1] += conf_mat[2,1]
#new_conf_mat[1,0] += conf_mat[1,2]
##print conf_mat
#print new_conf_mat
#
### compute the sensitivity and specificity
#sens_spec = utl.confusion_matrix2sens_spec(new_conf_mat)
#overall_precision = utl.overall_precision(new_conf_mat)
#-------------------------------------------------------------------

sens_spec = utl.confusion_matrix2sens_spec(conf_mat)
overall_precision = utl.overall_precision(conf_mat)
print "sens_spec: {}\n".format(sens_spec)  
print "overall accuracy: {}\n".format(overall_precision)  

success_num = conf_mat.diagonal().sum()
acc_interval = proportion_confint(success_num, conf_mat.sum(), method='beta')


#compute the confidence intervals
_, cls_num = sens_spec.shape
with open(record_filename,'w+') as f:
    f.write("{}\n".format(sens_spec))
    f.write("accuracy: {}\n".format(overall_precision))
    f.write("acc interval: {}\n".format(acc_interval))   
    for cls in range(cls_num):
        f.write("\n {}: \n".format(label_names[cls]))

        # sensitivity
        tp_num = conf_mat[cls,cls]
        p_num = conf_mat[:,cls].sum()
        sens_interval = proportion_confint(tp_num, p_num, method='beta')
        f.write("sensitivity: {}\n".format(float(tp_num)/max(1,p_num)))
        f.write("sensitivity intervals: {}\n".format(sens_interval))
        
        # specificity
        n_num = conf_mat.sum() - conf_mat[:,cls].sum()
        tn_num = n_num - conf_mat[cls,:].sum() + conf_mat[cls, cls]
        spec_interval = proportion_confint(tn_num, n_num, method='beta')
        f.write("specificity: {}\n".format(float(tn_num)/n_num))
        f.write("specificity intervals: {}\n".format(spec_interval))
        



# draw the ROC curve and compute the confidence interval 
for label, name in enumerate(label_names):

    roc_savename = prefix + name + "_roc.png" 
    roc_data_savename = prefix + name + "_roc.xls" 
    # plt_title = "mydriasis diffuse "

    # compute the confidence interval for ROC 
    #true_positive = conf_mat[label, label]
    #negative_sum = conf_mat.sum() - conf_mat[:,label].sum()
    #true_negative = negative_sum - conf_mat[label,:].sum()  
    #success_num = true_positive + true_negative 
    #total_trial_num = conf_mat.sum()   # conf_mat[:,label].sum() 
    
    scores = label_prob[:,label] 
    #y = label_prob[:,len(label_names)]  
    y = label_prob[:, -1]  
    
    #compute the roc curve
    fpr, tpr, thresholds = roc_curve(y, scores, pos_label=label) 
    
    #compute area under curve
    binary_y = np.zeros_like(y)
    binary_y[np.where(y==label)] = 1
    #weights = np.ones_like(y)
    #weights[np.where(y!=label)] = 0.5 #conf_mat[:,label].sum()/float(conf_mat.sum())
    #weights[np.where(y==label)] = 0.5 #(conf_mat.sum() - conf_mat[:,label].sum())/float(conf_mat.sum())  
    #area = roc_auc_score(binary_y,scores, average='weighted', sample_weight = weights)
    area = roc_auc_score(binary_y,scores)

    # compute 95% CI for AUC
    q1 = area/(2-area)
    q2 = 2*area**2/(1+area)
    N1 = conf_mat[:,label].sum() # positive num
    negative_sum = conf_mat.sum() - N1
    N2 = negative_sum
    numerator = area*(1-area) + (N1-1)*(q1-area**2) + (N2 - 1)*(q2 - area**2)
    SE = np.sqrt(numerator/(N1*N2))
    z = 1.96 # 95% confidence  
    interval = z * SE
    #print "auc: {}, auc interal: ({}, {})".format(area, area-interval, area+interval)  

    #print "test auc: {}, test auc ci: {}".format(float(true_positive)/conf_mat[label,:].sum(),proportion_confint(true_positive, conf_mat[label,:].sum(), method='beta'))  

    
    # compute the AUC confidence interval 
    #auc_ci_low, auc_ci_up = proportion_confint(count=success_num, nobs=total_trial_num, method='beta') 

    #print "auc: {}, confidence interval: ({}, {})".format(area, auc_ci_low, auc_ci_up)  

    # print fpr
    # print tpr
    # print thresholds

    #with open(roc_data_savename, 'w+') as rdf:
    #    rdf.write('false positive rate: ' + str(fpr*100) + '\n')
    #    rdf.write('true positive rate: ' + str(tpr*100) + '\n')
    df = DataFrame({'false positive rate':(fpr*100), 'true positive rate':(tpr*100)})  
    df.to_excel(roc_data_savename, sheet_name='sheet1', index=False)  
    
    plt.figure()
    plt.plot(fpr*100, tpr*100, color='blue',label='ROC curve', linewidth=2)
    plt.plot([0, 1], [0, 1], color='blue',  linestyle='--')
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 100])
    plt.ylim([0.0, 100])
    plt.xlabel('False Positive Rate', fontsize=10)
    plt.ylabel('True Positive Rate', fontsize=10)
    plt.title('AUC, {:4.2f}%;  95% CI, ({:4.2f}%, {:4.2f}%)'.format(area*100, (area-interval)*100, min(100.00, (area+interval)*100)))
    plt.legend(loc="lower right")
    #plt.xlim([0.0, 30])
    #plt.ylim([70.0, 100])
    plt.savefig(roc_savename)
    #plt.show()
    plt.close()
    
