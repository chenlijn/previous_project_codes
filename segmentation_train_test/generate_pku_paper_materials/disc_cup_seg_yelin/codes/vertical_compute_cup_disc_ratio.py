
import os
import shutil
import numpy as np
import cv2

from pandas import DataFrame


#def compute_cup2disc_ratio(disc_mask, cup_mask):
#    disc_mask = np.where(disc_mask==255, 1, 0)
#    cup_mask = np.where(cup_mask==255, 1, 0)
#    disc_area = disc_mask.sum()
#    cup_area = cup_mask.sum()
#    print "disc_area: {}, cup_area: {}, ratio: {}".format(disc_area, cup_area, float(cup_area)/disc_area)
#
#    return float(cup_area)/disc_area


def compute_cup2disc_ratio(disc_mask, cup_mask):
    rows, cols, _ = np.where(disc_mask==255)
    d_height = np.max(rows) - np.min(rows)

    c_rows, c_cols, _ = np.where(cup_mask==255)
    c_height = np.max(c_rows) - np.min(c_rows)

    print "disc_area: {}, cup_area: {}, ratio: {}".format(d_height, c_height, float(c_height)/d_height)

    return float(c_height)/d_height



if __name__ == '__main__':

    img_types = ['healthy', 'light', 'mid', 'serious']
    image_path_root = '../result_data/'
    src_img_path_root = '/mnt/lijian/mount_out/data/glaucomaData_filtered_kan20170825_final/disc_split_cls/'
    bad_case_path = '../bad_case/'

    for img_type in img_types:
        print img_type
        record_file = img_type + '_cup_to_disc_ratio.csv'

        src_img_path = src_img_path_root + img_type
        disc_path = image_path_root + img_type + '/disc/'
        #cup_path = image_path_root + img_type + '/cup/'
        cup_path = image_path_root + img_type + '/cup_yelin/'

        imgnames = os.listdir(disc_path)
        ratio_list = []
        decisions = []
        threshold = 0.6

        for imgname in imgnames[:2]:
            if imgname.endswith('.png'):
                print imgname
                disc = cv2.imread(os.path.join(disc_path, imgname))
                cup = cv2.imread(os.path.join(cup_path, imgname))

                d_rows, d_cols, _ = np.where(disc==255)
                d_height = np.max(d_rows) - np.min(d_rows)

               # c_rows, c_cols, _ = np.where(cup_mask==255)
               # c_height = np.max(c_rows) - np.min(c_rows)
                disc_show = np.copy(disc)
                disc_show[np.min(d_rows), :, :] = 255
                disc_show[np.max(d_rows), :, :] = 255

                rows, cols, _ = np.where(cup == 255)
                disc_show[np.min(rows), :, 0] = 0 
                disc_show[np.min(rows), :, 1] = 255 
                disc_show[np.min(rows), :, 2] = 0 

                disc_show[np.max(rows), :, 0] = 0 
                disc_show[np.max(rows), :, 1] = 255 
                disc_show[np.max(rows), :, 2] = 0 

                disc_show[rows, cols, 0] = 0
                disc_show[rows, cols, 1] = 0
                disc_show[rows, cols, 2] = 255
                cv2.imwrite('./test/'+imgname, disc_show)


                cup2disc_ratio = compute_cup2disc_ratio(disc, cup)
                ratio_list.append(cup2disc_ratio)

                if cup2disc_ratio > threshold:
                    if img_type == 'healthy':
                        srcimg = cv2.imread(os.path.join(src_img_path, imgname))
                        #cv2.imwrite(bad_case_path + img_type + '/' + imgname, srcimg)
                    decisions.append(1)
                else:
                    decisions.append(0)
                    if img_type != 'healthy':
                        srcimg = cv2.imread(os.path.join(src_img_path, imgname))
                        #cv2.imwrite(bad_case_path + img_type + '/' + imgname, srcimg)
                        #shutil.copy(os.path.join(src_img_path, imgname), bad_case_path + img_type + '/')

               
        print "lenght imgnames: {}, ratio_list: {}, decision: {}".format(len(imgnames), len(ratio_list), len(decisions))

        #df = DataFrame({"image name": imgnames, "cup to disc ratio": ratio_list, "glaucoma": decisions})
        #df = df[["image name", "cup to disc ratio", "glaucoma"]]
        #df.loc[len(ratio_list)] = ['mean ratio', np.mean(ratio_list), '/']
        #df.to_csv(record_file, index=False)
    # compute the cup to disc ratio




