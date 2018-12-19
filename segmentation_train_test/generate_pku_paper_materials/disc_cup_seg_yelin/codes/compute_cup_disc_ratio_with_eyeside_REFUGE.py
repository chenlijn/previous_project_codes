
import os
import shutil
import numpy as np
import cv2

from pandas import DataFrame


def filter_mask(img):
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours
    point_num_list = []
    for i in range(len(contours)):
        point_num_list.append(contours[i].shape[0])
    if len(point_num_list)==0:
        return img, None, (0,0)
    contours = [contours[np.argmax(point_num_list)]]
    print contours[0].shape
    
    new_img = np.zeros_like(img)
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


def draw_seg_and_centroid(img, in_disc, in_cup):
    """draw the edges of disc and cup masks as well as their centroids"""
    disc, _, disc_centroid = filter_mask(in_disc)
    cup, _, cup_centroid = filter_mask(in_cup)

    kernel = np.ones((3,3), np.uint8)
    disc_erosion = cv2.erode(disc, kernel, iterations=1)
    disc_edge = disc - disc_erosion

    cup_erosion = cv2.erode(cup, kernel, iterations=1)
    cup_edge = cup - cup_erosion
 
    # draw green
    rows, cols= np.where(disc_edge==255)
    img[rows, cols, 0] = 0
    img[rows, cols, 1] = 255
    img[rows, cols, 2] = 0
    cv2.circle(img, disc_centroid, 1, (0,255,0), 3)

    rows, cols= np.where(cup_edge==255)
    img[rows, cols, 0] = 255
    img[rows, cols, 1] = 0 
    img[rows, cols, 2] = 0
    cv2.circle(img, cup_centroid, 1, (255,0,0), 3)

    # compute the mean centroid and draw it
    #if (disc_centroid is not None) and (cup_centroid is not None):
    mean_cent = tuple((np.array(disc_centroid)+np.array(cup_centroid))/2)
    h, w = disc.shape
    cv2.line(img, (mean_cent[0],0), (mean_cent[0], h), (0, 255,0), 1)
    cv2.line(img, (0,mean_cent[1]), (w,mean_cent[1]), (0, 255,0), 1)
    return img


def get_edge_width(disc_edge, cup_edge, mean_centroid, img=None):
    x = mean_centroid[0]
    y = mean_centroid[1]
    hori_line = disc_edge[y, :]
    col_idx = np.where(hori_line==255)[0]
    if len(col_idx)==0:
        disc_hori_edge_pnts = None
    else:
        disc_hori_edge_pnts = [(col_idx[0], y), (col_idx[-1],y)]

    vert_line = disc_edge[:,x]
    rs_idx = np.where(vert_line==255)[0]
    if len(rs_idx)==0:
        disc_vert_edge_pnts = None
    else:
        disc_vert_edge_pnts = [(x, rs_idx[0]), (x, rs_idx[-1])]

    hori_line = cup_edge[y, :]
    col_idx = np.where(hori_line==255)[0]
    if len(col_idx)==0:
        cup_hori_edge_pnts = None
    else:
        cup_hori_edge_pnts = [(col_idx[0], y), (col_idx[-1],y)]

    vert_line = cup_edge[:,x]
    rs_idx = np.where(vert_line==255)[0]
    if len(rs_idx)==0:
        cup_vert_edge_pnts = None
    else:
        cup_vert_edge_pnts = [(x, rs_idx[0]), (x, rs_idx[-1])]

    # draw for test
    if img is not None:
        if disc_hori_edge_pnts and disc_vert_edge_pnts:
            cv2.circle(img, disc_hori_edge_pnts[0], 1, (0, 0, 255), 3)
            cv2.circle(img, disc_hori_edge_pnts[1], 1, (0, 0, 255), 3)
            cv2.circle(img, disc_vert_edge_pnts[0], 1, (0, 0, 255), 3)
            cv2.circle(img, disc_vert_edge_pnts[1], 1, (0, 0, 255), 3)

        if cup_hori_edge_pnts and cup_vert_edge_pnts:
            cv2.circle(img, cup_hori_edge_pnts[0], 1, (0, 0, 255), 3)
            cv2.circle(img, cup_hori_edge_pnts[1], 1, (0, 0, 255), 3)
            cv2.circle(img, cup_vert_edge_pnts[0], 1, (0, 0, 255), 3)
            cv2.circle(img, cup_vert_edge_pnts[1], 1, (0, 0, 255), 3)
        return img

    # compute edge widths
    if disc_hori_edge_pnts and cup_hori_edge_pnts:
        hdl = cup_hori_edge_pnts[0][0] - disc_hori_edge_pnts[0][0]
        hdr = disc_hori_edge_pnts[1][0] - cup_hori_edge_pnts[1][0]
        hori_ratio = float(cup_hori_edge_pnts[1][0]-cup_hori_edge_pnts[0][0])/(disc_hori_edge_pnts[1][0] - disc_hori_edge_pnts[0][0])
    else:
        hdl = 0
        hdr = 0
        hori_ratio = 0

    if disc_vert_edge_pnts and cup_vert_edge_pnts:
        vdu = cup_vert_edge_pnts[0][1] - disc_vert_edge_pnts[0][1]
        vdd = disc_vert_edge_pnts[1][1] - cup_vert_edge_pnts[1][1]
        vert_ratio = float(cup_vert_edge_pnts[1][1]-cup_vert_edge_pnts[0][1])/(disc_vert_edge_pnts[1][1] - disc_vert_edge_pnts[0][1])
    else:
        vdu = 0
        vdd = 0
        vert_ratio = 0
    return hdl, hdr, vdu, vdd, hori_ratio, vert_ratio


def get_edge_wid_from_range(disc, cup, mean_cent, eyeside, img=None):
    hdl = compute_edge_wid_by_eyeside(disc, cup, mean_cent, 2, eyeside, img=None)
    hdr = compute_edge_wid_by_eyeside(disc, cup, mean_cent, 3, eyeside, img=None)
    vdu = compute_edge_wid_by_eyeside(disc, cup, mean_cent, 0, eyeside, img=None)
    vdd = compute_edge_wid_by_eyeside(disc, cup, mean_cent, 1, eyeside, img=None)
    vert_cup_d = compute_cup_diameter_from_sector(cup, mean_cent, (7*np.pi/16, 9*np.pi/16), img=None)
    hori_cup_d = compute_cup_diameter_from_sector(cup, mean_cent, (15*np.pi/16, 17*np.pi/16), img=None)

    vert_dd = compute_cup_diameter_from_sector(disc, mean_cent, (7*np.pi/16, 9*np.pi/16), img=None)
    hori_dd = compute_cup_diameter_from_sector(disc, mean_cent, (15*np.pi/16, 17*np.pi/16), img=None)
    return hdl, hdr, vdu, vdd, hori_cup_d, vert_cup_d, hori_dd, vert_dd


def draw_pnts(pnts, img):
    #print(pnts)
    h, w, _ = img.shape
    for pnt in pnts:
        x, y = pnt
        if y > 0 and y < h:
            cv2.circle(img, (x,y), 1, (0, 0, 255), 3)



def get_isnt(in_disc,in_cup, eyeside=0, img=None):
    disc, _, disc_centroid = filter_mask(in_disc)
    cup, _, cup_centroid = filter_mask(in_cup)

    kernel = np.ones((3,3), np.uint8)
    disc_erosion = cv2.erode(disc, kernel, iterations=1)
    disc_edge = disc - disc_erosion

    cup_erosion = cv2.erode(cup, kernel, iterations=1)
    cup_edge = cup - cup_erosion
 
    # compute the mean centroid and draw it
    mean_cent = tuple((np.array(disc_centroid)+np.array(cup_centroid))/2)
    #pnts, values = extract_line_at_specific_angle(cup_edge, mean_cent, -2*np.pi/6)

    #d = compute_edge_wid_by_eyeside(disc_edge, cup_edge, mean_cent, 1, eyeside, img)
    #print('minimum d: {}'.format(d))
    # only for showing the pnts
    if img is not None:
        img = get_edge_width(disc_edge, cup_edge, mean_cent, img=img)
        #draw_pnts(pnts, img)
        return img

    #hdl, hdr, vdu, vdd, hori_ratio, vert_ratio = get_edge_width(disc_edge, cup_edge, mean_cent)
    hdl, hdr, vdu, vdd, hori_cd, vert_cd, hori_dd, vert_dd = get_edge_wid_from_range(disc_edge, cup_edge, mean_cent, eyeside, img=None)

    if eyeside: # right eye
        interior = vdd
        superior = vdu
        nasal = hdr
        temporal = hdl
    else: # left eye
        interior = vdd
        superior = vdu
        nasal = hdl
        temporal = hdr

    #return interior, superior, nasal, temporal, hori_ratio, vert_ratio
    return interior, superior, nasal, temporal, hori_cd, vert_cd, hori_dd, vert_dd


def extract_line_at_specific_angle(edge_mask, mean_cent, angle):
    """extract the pixels and their coordinates along a line in image
    
    args:
    disc_cup_comb: binary image containing 0 or 1
    """

    pnts = []
    values = []
    h, w = edge_mask.shape
    cx, cy = mean_cent
    if (angle >= -np.pi/4  and angle <= np.pi/4) or \
       (angle >= 7*np.pi/4 and angle <= 2*np.pi) or \
       (angle >= 3*np.pi/4 and angle <= 5*np.pi/4):
        k = np.tan(angle)
        for x in range(w):
            y = int(k * (cx - x) + cy)
            pnts.append((x, y))
            if y>=0 and y<h:
                values.append(edge_mask[y, x])
            else:
                values.append(-1)
    else:
        angle += np.pi/2
        k = np.tan(angle)
        for y in range(h):
            x = int(k * (y - cy) + cx)
            pnts.append((x,y))
            if x >=0 and x < w:
                values.append(edge_mask[y, x])
            else:
                values.append(-1)
    #print('values {}'.format(values))
    return pnts, values


def get_intersect_pnts_from_line(mean_cent, pnts, values, direction):
    """
    args:
    direction: 0 - top, 1 - bottom, 2 - left, 3 - right, 4 - return both side
    """

    cx, cy = mean_cent

    if direction == 0:
       idx = np.where(np.array(values)==1)
       temp_p = np.array(pnts)[idx]
       n = 0
       x, y = 0, 0
       for pnt in temp_p:
           if pnt[1] <= cy:
               n += 1
               x += pnt[0]
               y += pnt[1]
       return (int(x/max(n,1)), int(y/max(1,n)))
    elif direction == 1:
       idx = np.where(np.array(values)==1)
       temp_p = np.array(pnts)[idx]
       n = 0
       x, y = 0, 0
       for pnt in temp_p:
           if pnt[1] > cy:
               n += 1
               x += pnt[0]
               y += pnt[1]
       return (int(x/max(n,1)), int(y/max(1,n)))
    elif direction == 2:
       idx = np.where(np.array(values)==1)
       temp_p = np.array(pnts)[idx]
       n = 0
       x, y = 0, 0
       for pnt in temp_p:
           if pnt[0] <= cx:
               n += 1
               x += pnt[0]
               y += pnt[1]
       return (int(x/max(n,1)), int(y/max(1,n)))
    elif direction == 3:
       idx = np.where(np.array(values)==1)
       temp_p = np.array(pnts)[idx]
       n = 0
       x, y = 0, 0
       for pnt in temp_p:
           if pnt[0] > cx:
               n += 1
               x += pnt[0]
               y += pnt[1]
       return (int(x/max(n,1)), int(y/max(1,n)))
    elif direction == 4:
       idx = np.where(np.array(values)==1)
       temp_p = np.array(pnts)[idx]
       if len(temp_p) <= 1:
           return [(0,0),(0,0)]
       else:
           return [tuple(temp_p[0]), tuple(temp_p[1])]


def pnt_distance(pnt1, pnt2):
    x1, y1 = pnt1
    x2, y2 = pnt2
    return np.sqrt(np.square(x1-x2)+np.square(y1-y2))


def compute_edge_width_from_sector(disc, cup, mean_cent, sector, direction, img=None):
    """
    argx:
    sector: (start_angle, finsih_angle)
    """

    cx, cy = mean_cent
    start_angle, finish_angle = sector
    sample_num = 20
    step = 1.0*(finish_angle-start_angle) / sample_num
    widths = []
    for i in range(sample_num):
        angle = start_angle + step * i
        pnts, values = extract_line_at_specific_angle(disc, mean_cent, angle)
        intersect_pnt1 = get_intersect_pnts_from_line(mean_cent, pnts, values, direction)
        pnts, values = extract_line_at_specific_angle(cup, mean_cent, angle)
        intersect_pnt2 = get_intersect_pnts_from_line(mean_cent, pnts, values, direction)
        d = pnt_distance(intersect_pnt1, intersect_pnt2)
        if d > 0:
            widths.append(d)
        if img is not None:
            cv2.line(img, intersect_pnt1, intersect_pnt2, (0, 0, 255), 1)
    if len(widths)==0:
        return 0.0
    else:
        return np.sort(widths)[0]


def compute_cup_diameter_from_sector(cup, mean_cent, sector, img=None):
    """
    argx:
    sector: (start_angle, finsih_angle)
    """

    cup = np.where(cup>0, 1, 0)
    cx, cy = mean_cent
    start_angle, finish_angle = sector
    sample_num = 20
    step = 1.0*(finish_angle-start_angle) / sample_num
    widths = []
    for i in range(sample_num):
        angle = start_angle + step * i
        pnts, values = extract_line_at_specific_angle(cup, mean_cent, angle)
        pnt1, pnt2 = get_intersect_pnts_from_line(mean_cent, pnts, values, 4)
        #print('cup edge: {}, {}'.format(pnt1, pnt2))
        d = pnt_distance(pnt1, pnt2)
        if d > 0:
            widths.append(d)
        if img is not None:
            cv2.line(img, pnt1, pnt2, (0, 0, 255), 1)
    if len(widths)==0:
        return 0.0
    else:
        return np.sort(widths)[-1]



def compute_edge_wid_by_eyeside(disc, cup, mean_cent, direction, eyeside, img=None):
    disc = np.where(disc>0, 1, 0)
    cup = np.where(cup>0, 1, 0)
    step = np.pi / 6
    if eyeside: # right
        if direction==0: # up
            sector = (2*step, 5*step)
        elif direction==1: # down
            sector = (7*step, 10*step)
        elif direction==2: # left
            sector = (5*step, 7*step)
        elif direction==3: # right
            sector = (-2*step, 2*step)
    else: # left
        if direction==0: # up
            sector = (step, 4*step)
        elif direction==1: # down
            sector = (8*step, 11*step)
        elif direction==2: # left
            sector = (4*step, 8*step)
        elif direction==3: # right
            sector = (-1*step, step)
    return compute_edge_width_from_sector(disc, cup, mean_cent, sector, direction, img=img)



def isnt_classifier(interior, superior, nasal, temporal, hori_cd, vert_cd, hori_dd, vert_dd, threshold=0.65):
    rate = 1
    threshold = threshold
    #if (interior and nasal) or (superior and nasal):
    #    if interior < nasal*rate or superior < nasal*rate:
    #        return 1
    #    else:
    #        return 0
    #    
    #if vert_cd < 1.0 * hori_cd:
    #    return 0
    ratio = vert_cd * 1.0 / max(0.1, vert_dd)
    print(vert_cd, vert_dd, ratio)
    if ratio < threshold:
        return 0, ratio
    else:
        return 1, ratio



def compute_diameter(points, centroid):
    if points is None:
        return None, None, None, None
    rows1 = np.where(points[:,0]==centroid[0]) # vertical points
    vertical_pnts = points[rows1]
    rows2 = np.where(points[:,1]==centroid[1]) # horizontal points
    horizontal_pnts = points[rows2]

    if len(vertical_pnts) == 0 or len(horizontal_pnts) == 0:
        return None, None, None, None
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


def get_eyeside_dict(eyeside_txt):
    with open(eyeside_txt, 'r') as f:
        lines = f.readlines()
    imgnames = []
    eyeside = []
    for line in lines:
        side = line.strip('\n').split()[-1]
        name = line[:-3]
        imgnames.append(name.split('.')[0])
        eyeside.append(int(side))
    return dict(zip(imgnames, eyeside))



if __name__ == '__main__':

    #image_path_root = '/root/mount_out/data/original_glaucoma_data/all_data_sorted/glaucoma_cls_test_set/splits/'
    image_path_root = '/root/mount_out/data/public_datasets/disc_cup_segmentation_dataset/det/validation/'
    #mask_path = '/root/mount_out/data/original_glaucoma_data/all_data_sorted/healthy/masks/'
    #mask_path = '/root/mount_out/data/glaucoma_cup_seg_train_val_data_20180926/train_val_data/val/cleaned_masks/'
    #image_path_root = '/root/mount_out/data/public_datasets/disc_cup_segmentation_dataset/det/validation/'
    img_types = ['glaucoma', 'non_glaucoma']
    #img_types = ['online_abnormal_test']

    for img_type in img_types:
        show_path = 'bad_case/{}'.format(img_type)
        eyeside_txt = image_path_root + img_type + '_with_path_eye_side.txt'
        eyeside_dict = get_eyeside_dict(eyeside_txt)

        record_file = "csv/" + img_type + '_cup_to_disc_ratio.csv'

        det_path = image_path_root + 'images/' + img_type 
        disc_path = image_path_root + 'seg/disc/' + img_type
        cup_path = image_path_root + 'seg/cup/' + img_type

        imgnames = os.listdir(disc_path)
        ratio_list = []
        decisions = []
        threshold = 0.65
        sides = ['left', 'right']

        #for idx, imgname in enumerate(imgnames[:3]):
        for idx, imgname in enumerate(imgnames):
            if imgname.endswith('.png'):
                disc = cv2.imread(os.path.join(disc_path, imgname), 0)
                cup = cv2.imread(os.path.join(cup_path, imgname), 0)
                #cup = cv2.imread(os.path.join(mask_path, imgname), 0)

                ## to draw ans store the intermediate results
                #eyeside = eyeside_dict[imgname.split('.')[0]]
                #img = cv2.imread(os.path.join(det_path, imgname.split('.')[0]+'.jpg'))
                #show_img = draw_seg_and_centroid(np.copy(img), disc, cup)
                #show_img = get_isnt(disc, cup, eyeside=eyeside, img=show_img)
                #cv2.imwrite(os.path.join(show_path, imgname), show_img)
                
                eyeside = eyeside_dict[imgname.split('.')[0]]

                i,s,n,t, hori_cd, vert_cd, hori_dd, vert_dd = get_isnt(disc, cup, eyeside=eyeside)
                glaucoma, ratio = isnt_classifier(i,s,n,t, hori_cd, vert_cd, hori_dd, vert_dd, threshold=threshold)
                if glaucoma:
                    if img_type == 'healthy' or img_type == 'online_normal_test' or img_type == 'non_glaucoma':
                        # to draw ans store the intermediate results
                        img = cv2.imread(os.path.join(det_path, imgname.split('.')[0]+'.jpg'))
                        show_img = draw_seg_and_centroid(np.copy(img), disc, cup)
                        show_img = get_isnt(disc, cup, eyeside=eyeside, img=show_img)
                        put_text= 'side: {}, i {}, s {}, n {}, t {}'.format(sides[eyeside], int(i), int(s), int(n), int(t)) 
                        cv2.putText(show_img, put_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)
                        put_text2 = 'cup_to_disc_ratio: {:6.4f}'.format(ratio)
                        cv2.putText(show_img, put_text2, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)
                        cv2.imwrite(os.path.join(show_path, imgname), show_img)
                    decisions.append(1)
                    ratio_list.append(ratio)
                    #ratio_list.append('-')
                else:
                    decisions.append(0)
                    ratio_list.append(ratio)
                    #ratio_list.append('-')
                    if img_type != 'healthy' and img_type != 'online_normal_test' and img_type != 'non_glaucoma':
                        # to draw ans store the intermediate results
                        img = cv2.imread(os.path.join(det_path, imgname.split('.')[0]+'.jpg'))
                        show_img = draw_seg_and_centroid(np.copy(img), disc, cup)
                        show_img = get_isnt(disc, cup, eyeside=eyeside, img=show_img)
                        #put_text= 'side: {}'.format(eyeside) 
                        #put_text= 'side: {}, i {}, s {}, n {}, t {}'.format(eyeside, i, s, n, t) 
                        put_text= 'side: {}, i {}, s {}, n {}, t {}'.format(sides[eyeside], int(i), int(s), int(n), int(t)) 
                        cv2.putText(show_img, put_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)
                        put_text2 = 'cup_to_disc_ratio: {:6.4f}'.format(ratio)
                        cv2.putText(show_img, put_text2, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)
                        cv2.imwrite(os.path.join(show_path, imgname), show_img)

                
        print "length imgnames: {}, ratio_list: {}, decision: {}".format(len(imgnames), len(ratio_list), len(decisions))

        df = DataFrame({"image name": imgnames, "cup to disc ratio": ratio_list, "glaucoma": decisions})
        df = df[["image name", "cup to disc ratio", "glaucoma"]]
        img_num = len(decisions)
        #df.loc[img_num] = ['mean ratio', np.mean(ratio_list), '']
        df.loc[img_num+1] = ['', '', '']
        #df.loc[img_num+2] = ['threshold', threshold, '']
        df.loc[img_num+3] = ['total num', 'healthy num', 'glaucoma num']
        df.loc[img_num+4] = [img_num, img_num - np.sum(decisions), np.sum(decisions)]
        df.loc[img_num+5] = ['rate', 1 - np.sum(decisions)/float(img_num), np.sum(decisions)/float(img_num)]
        df.to_csv(record_file, index=False)


