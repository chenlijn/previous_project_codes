#filter the glaucoma images and split those with two discs

import os
import numpy as np
import cv2


def enhance_image_by_clahe(img):

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    if len(img.shape)==3:
        img[:,:,0] = clahe.apply(img[:,:,0])
        img[:,:,1] = clahe.apply(img[:,:,1])
        img[:,:,2] = clahe.apply(img[:,:,2])
        return img
    else:
        return  clahe.apply(img)
        

def draw_seg_mask_on_img(img, in_mask):

    # get the edge of mask area
    print(img.shape)
    print(in_mask.shape)
    print(in_mask.dtype)
    if len(in_mask.shape)>2:
        mask = in_mask[:,:,0]
    else:
        mask = in_mask

    kernel = np.ones((5,5), np.uint8)
    temp_mask = np.copy(mask)
    eroded_mask = cv2.erode(temp_mask*1.0, kernel, iterations=1)
    edge = mask - eroded_mask

    # draw
    rows, cols = np.where(edge > 0)
    img[rows, cols, 0] = 0
    img[rows, cols, 1] = 255
    img[rows, cols, 2] = 0

    return img


def split_disc_by_mask(image, mask):
    """split the double discs by mask"""

    mask = np.where(mask>0, 1, 0)
    rows, cols = np.where(mask==1)
    print(rows, cols)
    mean_x = int(np.mean(cols))
    mean_y = int(np.mean(rows))

    height, width, _ = image.shape
    center_x = width // 2

    wid = 10
    #center_area = mask[:, center_x - wid : center_x + wid]
    #center_img_area = image[:, center_x - wid : center_x + wid, 1]

    # mid area
    mid_area = image[mean_y - wid : mean_y + wid, width//4 : 3*width//4, 1] # green channel
    mid_rows, mid_cols = np.where(mid_area < 30)
    mid_area = np.where(mid_area<30, 255, 0)
    cv2.imwrite('mid_area.png', mid_area)
    print('rows len: ', len(mid_rows))
    if len(mid_rows) >= 2*wid * wid:
        split_x = int(np.mean(mid_cols)) + width//4
        #print 'split_x: {}'.format(split_x)

        if mean_x < center_x:
             return image[:, :split_x, :], mask[:, :split_x]
        else:
             return image[:, split_x:, :], mask[:, split_x:]
    else:
        return image, mask
  




def split_two_disc(v,image):
    """split the double discs in images
    
    args:
        v - v of hsv 
        image - original image"""

    #the input image should be gray image
    w = image.shape[1]
    h = image.shape[0]

    centx = w//2

    area_width = 30
    value_thresh = 40

    
    mid_area1 = v[:, centx-5-area_width:centx-5]
    mid_area2 = v[:, w//3:2*w//3]  
    mid_thresh1 = mid_area1 < value_thresh
    mid_thresh2 = mid_area2 < value_thresh
    rate1 = np.sum(mid_thresh1) / float(area_width * h) 
    rate2 = np.sum(mid_thresh2) / float((2*w/3) * h) 
    print('rate: ', rate1, rate2)
    if rate1 > 0.75:
        saveimg = image[:, 0:centx-5, :]
        return saveimg
    elif rate2 > 0.1:
        saveimg = image[:, 0:centx-5, :]
        return saveimg 
    else:
        return image


def confusion_matrix2sens_spec(confusion_matrix):
    """the columns of confusion matrix are groun d truth, rows are predictions"""
    rows, cols = confusion_matrix.shape

    #define the sensitivity and specificity matrix
    sens_spec_matrix = np.zeros([2,rows], np.float32)

    #compute sensitivity and specificity
    for i in range(rows):
        #sens
        sens_spec_matrix[0,i] = np.float32(confusion_matrix[i,i]) / (confusion_matrix[:,i]).sum()

        #spec, one versas all
        negative_sum = (confusion_matrix).sum() - (confusion_matrix[:,i]).sum()
        true_negative = negative_sum - ((confusion_matrix[i,:]).sum()-confusion_matrix[i,i])
        sens_spec_matrix[1,i] = np.float32(true_negative) / negative_sum

    return sens_spec_matrix


def overall_precision(confusion_matrix):
    """the columns of confusion matrix are groun d truth, rows are predictions"""

    return np.float(np.trace(confusion_matrix)) / confusion_matrix.sum()



if __name__ == '__main__':

    src_dir = '/home/gaia/share/glaucomaData_filtered_kan20170825_final/cls/light/'
    dst_dir = '/home/gaia/share/glaucomaData_filtered_kan20170825_final/cropped_cls/light/'
    imgfiles = os.listdir(src_dir)
    num = len(imgfiles)

    i = 0
    for img in imgfiles:
           i += 1
           #img = '1_276.jpg'
        #if 'jpg' in img:
           imgfile = src_dir + img
           image = cv2.imread(imgfile)
           hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
           h,s,v = cv2.split(hsv)
           saveimg = split_two_disc(v,image)  
           savefile = dst_dir + img 
           cv2.imwrite(savefile, saveimg)
           print(i)


