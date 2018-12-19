#this file stores the general utility functions
import numpy as np

class utils(object):

    def __init__(self):
        return

    def confusion_matrix2sens_spec(self, confusion_matrix):
        rows, cols = confusion_matrix.shape
        
        #define the sensitivity and specificity matrix
        sens_spec_matrix = np.zeros([2,rows], np.float32)

        #compute sensitivity and specificity
        for i in range(rows):
            #sens
            sens_spec_matrix[0,i] = np.float32(confusion_matrix[i,i]) / max(1, (confusion_matrix[:,i]).sum())

            #spec, one versas all
            negative_sum = (confusion_matrix).sum() - (confusion_matrix[:,i]).sum()
            true_negative = negative_sum - ((confusion_matrix[i,:]).sum()-confusion_matrix[i,i])  
            sens_spec_matrix[1,i] = np.float32(true_negative) / max(negative_sum, 1)

        return sens_spec_matrix

    def overall_precision(self, confusion_matrix):
        return np.float(np.trace(confusion_matrix)) / max(1, confusion_matrix.sum())




utils_lijian = utils()

if __name__ == '__main__':

    # conf_file = 'imaging_types_val_cm.npy'
    #conf_file = '_val_cm.npy'
    #conf_file = 'diffuse_illness_val_cm.npy'
    #conf_file = 'slit_illness_cm.npy'
    conf_file = 'severity_severity_val_cm.npy'
    conf_mat = np.load(conf_file)
    print "confusion matrix: \n", conf_mat

    sens_spec = utils_lijian.confusion_matrix2sens_spec(conf_mat)
    print "sens_spec: ", sens_spec

    print "total num: ", conf_mat.sum(), "overall precision: ", utils_lijian.overall_precision(conf_mat)

