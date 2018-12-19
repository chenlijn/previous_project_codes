
import os
import shutil
import numpy as np
import cv2

from pandas import DataFrame


def filter_masks(img):
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # sort contours
    point_num_list = []
    for i in range(len(contours)):
        point_num_list.append(contours[i].shape[0])
    if len(point_num_list)==0:
        return img, None, None
    contours = [contours[np.argmax(point_num_list)]]
    print contours[0].shape
    
    #temp_img = np.zeros_like(img)
    new_img = np.zeros_like(img)
    #cv2.fillPoly(new_img, contours, 255)
    hull = cv2.convexHull(contours[0])
    cv2.fillPoly(new_img, [hull], 255)


    # get the boundary
    kernel = np.ones((3,3), np.uint8)
    erosion = cv2.erode(new_img, kernel, iterations=1)
    boundary = new_img - erosion
    #cv2.imwrite('boundary.png', boundary)
    rows, cols = np.where(boundary==255)
    points = np.zeros((len(rows), 2), np.int)
    points[:,0] = cols
    points[:,1] = rows

    num = points.shape[0]
    cent_x = points[:, 0].sum() / num
    cent_y = points[:, 1].sum() / num
    centroid = (cent_x, cent_y) 
 
    return new_img, points, centroid


def compute_diameter(points, centroid):
    #print 'points', points
    #print 'centroid', centroid
    if points is None:
        return None, None, None, None
    #print points.shape
    #print centroid
    rows1 = np.where(points[:,0]==centroid[0]) # vertical points
    vertical_pnts = points[rows1]
    rows2 = np.where(points[:,1]==centroid[1]) # horizontal points
    horizontal_pnts = points[rows2]

    if len(vertical_pnts) == 0 or len(horizontal_pnts) == 0:
        return None, None, None, None
    #idxs3, rows3 = np.where(points[:,1]==centroid[1]+2) # horizontal points
    #print "vertical points: ", vertical_pnts
    #print "horizontal points: ", horizontal_pnts

    vert_pnts = []
    vert_pnts.append(vertical_pnts[0,:])
    vert_pnts.append(vertical_pnts[-1,:])
    hori_pnts = []
    hori_pnts.append(horizontal_pnts[0,:])
    hori_pnts.append(horizontal_pnts[-1,:])

    #print "horizontal near points: ", points[idxs3, rows3, :]

    # vertical diameter
    vert_d1 = centroid[1] - vert_pnts[0][1]
    vert_d2 = vert_pnts[1][1] - centroid[1]

    hori_d1 = centroid[0] - hori_pnts[0][0]
    hori_d2 = hori_pnts[1][0] - centroid[0]
    #vert_d = vert_pnts[1][1] - vert_pnts[0][1]
    #hori_d = hori_pnts[1][0] - hori_pnts[0][0]

    return vert_d1, vert_d2, hori_d1, hori_d2


def compute_cup2disc_ratio(disc_mask, cup_mask):
    rows, cols = np.where(disc_mask==255)
    if len(rows) == 0:
        return 0
    d_height = np.max(rows) - np.min(rows)
    
    c_rows, c_cols = np.where(cup_mask==255)
    if len(c_rows) == 0:
        return 0
    c_height = np.max(c_rows) - np.min(c_rows)
    
    print "disc_area: {}, cup_area: {}, ratio: {}".format(d_height, c_height, float(c_height)/d_height)
    
    return float(c_height)/d_height


def glaucoma_classifier(disc_mask, cup_mask):
    new_disc, disc_points, disc_centroid = filter_masks(disc_mask)
    new_cup, cup_points, cup_centroid = filter_masks(cup_mask)
    cup2disc_ratio = compute_cup2disc_ratio(new_disc, new_cup)
    if cup_points is None:
        return 0, cup2disc_ratio
    disc_v_d1, disc_v_d2, disc_h_d1, disc_h_d2 = compute_diameter(disc_points, cup_centroid)
    if disc_v_d1 is None:
        return 1, cup2disc_ratio
    cup_v_d1, cup_v_d2, cup_h_d1, cup_h_d2 = compute_diameter(cup_points, cup_centroid)

    # glaucoma recognition
    is_glaucoma = 0
    distance1 = 1.2
    distance2 = 0.8
    if (cup_v_d1 + cup_v_d2)/(cup_h_d1 + cup_h_d2) >= distance1:
        print "A < B"
        is_glaucoma = 1
    elif (disc_v_d2 - cup_v_d2)/(disc_h_d1 - cup_h_d1) < distance2:
        print "a < c"
        is_glaucoma = 1
    elif (disc_v_d1 - cup_v_d1)/(disc_h_d1 - cup_h_d1) < distance2:
        print "b < c"
        is_glaucoma = 1

    return is_glaucoma, cup2disc_ratio




if __name__ == '__main__':

    #img_types = ['healthy', 'light', 'mid', 'serious']
    #img_types = ['light', 'mid', 'serious']
    img_types = ['healthy']
    image_path_root = '/root/mount_out/work2018/private_glaucoma_recognition/data/results/'
    src_img_path_root = '/root/mount_out/work2018/private_glaucoma_recognition/data/test/'
    bad_case_path = '../bad_case/'

    for img_type in img_types:
        record_file = "csv/" + img_type + '_cup_to_disc_ratio.csv'

        src_img_path = src_img_path_root + img_type
        det_path = image_path_root + img_type + '/det/'
        disc_path = image_path_root + img_type + '/disc/'
        cup_path = image_path_root + img_type + '/cup/'
        #cup_path = image_path_root + img_type + '/cup_yelin/'

        imgnames = os.listdir(disc_path)
        ratio_list = []
        decisions = []
        threshold = 0.65

        for imgname in imgnames:
            if imgname.endswith('.png'):
                disc = cv2.imread(os.path.join(disc_path, imgname), 0)
                #disc = cv2.resize(in_disc, (512, 512))
                #rs, cs = np.where(disc>0)
                #disc[rs,cs]=255
                cup = cv2.imread(os.path.join(cup_path, imgname), 0)

                show_disc, _, _ = filter_masks(np.copy(disc))
                show_cup, _, _ = filter_masks(np.copy(cup))
                kernel = np.ones((3,3), np.uint8)
                disc_erosion = cv2.erode(show_disc, kernel, iterations=1)
                disc_edge = show_disc - disc_erosion
                cup_erosion = cv2.erode(show_cup, kernel, iterations=1)
                cup_edge = show_cup - cup_erosion
                det_img = cv2.imread(os.path.join(det_path, imgname.split('.')[0]+'.jpg'))
                det_img = cv2.resize(det_img, (513,513))
                rows, cols= np.where(disc_edge==255)
                det_img[rows, cols, 0] = 0
                det_img[rows, cols, 1] = 255
                det_img[rows, cols, 2] = 0
                rows, cols= np.where(cup_edge==255)
                det_img[rows, cols, 0] = 255
                det_img[rows, cols, 1] = 0 
                det_img[rows, cols, 2] = 0
                #cv2.imwrite(image_path_root+img_type + '/show_yelin/' + imgname, det_img)
                #cv2.imwrite(image_path_root+img_type + '/show_lijian/' + imgname, det_img)

                disc_dilation = cv2.dilate(disc_erosion, kernel, iterations=1)
                cup_dilation = cv2.dilate(cup_erosion, kernel, iterations=1)

                #cup2disc_ratio = compute_cup2disc_ratio(disc_dilation, cup_dilation)
                is_glaucoma, cup2disc_ratio = glaucoma_classifier(disc, cup)
                ratio_list.append(cup2disc_ratio)

                #if is_glaucoma or cup2disc_ratio > threshold:
                if cup2disc_ratio > threshold:
                    if img_type == 'healthy':
                        #srcimg = cv2.imread(os.path.join(src_img_path, imgname))
                        #srcimg = cv2.imread(os.path.join(image_path_root + img_type + '/show_lijian/', imgname))
                        cv2.imwrite(bad_case_path + img_type + '/' + imgname, det_img)
                    decisions.append(1)
                else:
                    decisions.append(0)
                    if img_type != 'healthy':
                        #srcimg = cv2.imread(os.path.join(src_img_path, imgname))
                        #srcimg = cv2.imread(os.path.join(image_path_root + img_type + '/show_lijian/', imgname))
                        cv2.imwrite(bad_case_path + img_type + '/' + imgname, det_img)
                        #shutil.copy(os.path.join(src_img_path, imgname), bad_case_path + img_type + '/')

               
        print "length imgnames: {}, ratio_list: {}, decision: {}".format(len(imgnames), len(ratio_list), len(decisions))

        df = DataFrame({"image name": imgnames, "cup to disc ratio": ratio_list, "glaucoma": decisions})
        df = df[["image name", "cup to disc ratio", "glaucoma"]]
        img_num = len(ratio_list)
        df.loc[img_num] = ['mean ratio', np.mean(ratio_list), '']
        df.loc[img_num+1] = ['', '', '']
        df.loc[img_num+2] = ['threshold', threshold, '']
        df.loc[img_num+3] = ['total num', 'healthy num', 'glaucoma num']
        df.loc[img_num+4] = [img_num, img_num - np.sum(decisions), np.sum(decisions)]
        df.loc[img_num+5] = ['rate', 1 - np.sum(decisions)/float(img_num), np.sum(decisions)/float(img_num)]
        df.to_csv(record_file, index=False)


