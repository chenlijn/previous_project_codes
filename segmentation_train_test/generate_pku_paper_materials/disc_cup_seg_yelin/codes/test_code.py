
import os
import numpy as np
import cv2

from compute_cup_disc_ratio import filter_masks, compute_diameter


if __name__ == '__main__':

    img_name = '2fee53d5279a56c66ea16bcdcfdb1e91.png'
    img_disc = '1f93016eba37f31ab4f2e953fff12263_disc.png'
    img_cup = '1f93016eba37f31ab4f2e953fff12263_cup.png'

    #img = cv2.imread(img_name, 0)
    #img_show = cv2.imread(img_name)

    #img = cv2.imread(img_disc, 0)
    #img_show = cv2.imread(img_disc)
    img = cv2.imread(img_cup, 0)
    img_show = cv2.imread(img_cup)
    #disc = cv2.imread(img_disc, 0)

    #im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ## sort contours
    #point_num_list = []
    #for i in range(len(contours)):
    #    point_num_list.append(contours[i].shape[0])
    #contours = [contours[np.argmax(point_num_list)]]

    #cv2.drawContours(img_show, contours, 0, (0, 255, 0), 3)
    #cv2.imwrite('contours.png', img_show)
    #new_cup = np.zeros_like(img)
    #cv2.fillPoly(new_cup, contours, 255)
    new_cup, points, centroid = filter_masks(img)
    print 'points', points.shape
    hull = cv2.convexHull(points)
    new_hull = np.zeros_like(img)
    print 'hull', hull.shape
    cv2.fillPoly(new_hull, [hull], 255)
    cv2.imwrite('new_mask.png', new_cup)
    cv2.imwrite('hull.png', new_hull)

    print 'pp ', points.shape
    #print points
    cv2.line(img_show, tuple(centroid), tuple(centroid), (0,0,255), 3)
    vert_d1, vert_d2, hori_d1, hori_d2 = compute_diameter(points, centroid)
    print "diameters: ", vert_d1, vert_d2, hori_d1, hori_d2
    #vert_pnts, hori_pnts = compute_diameter(points, centroid)
    #for pnt in vert_pnts:
    #    cv2.line(img_show, tuple(pnt), tuple(pnt), (0,255,0), 3)
    #for pnt in hori_pnts:
    #    cv2.line(img_show, tuple(pnt), tuple(pnt), (0,255,0), 3)

    #for i in range(points.shape[0]):
    #    pnt = points[i, :]
    #    print 'iteration: ', pnt 
    #    cv2.line(img_show, tuple(pnt), tuple(pnt), (0,0,255), 1)
    #cv2.drawContours(img_show, [points], 0, (0, 255, 0), 1)
    #cv2.drawContours(img_show, contours, 0, (0, 255, 0), 3)
    cv2.imwrite('img_show.png', img_show)
    cv2.imwrite('new_cup.png', new_cup)

    #print contours[0].shape
    #print im2.shape
    #print len(contours)
    #print hierarchy

