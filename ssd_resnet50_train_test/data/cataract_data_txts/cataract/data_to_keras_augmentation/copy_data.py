
import os
import sys
import shutil

illness_types = ['healthy', 'ill', 'after_surgery']  
img_types = ['mydriasis_diffuse_light', 'mydriasis_slit_light', 
             'small_pupil_diffuse_light', 'small_pupil_slit_light']   

txt_dir = './src_test_txt/'
dst_dir = './data_test/'
txt_files = os.listdir(txt_dir)

for txt in txt_files:
    #txt = 'cleaned_healthy_small_pupil_diffuse_light_train.txt'
    for ill_t in illness_types:
        if ill_t in txt:
            temp_dir = dst_dir + ill_t + '/'
    for img_t in img_types:
        if img_t in txt:
            temp_dir += img_t + '/'

    print 'txtfile: ', txt
    print 'dst_dir: ', temp_dir, '\n'
    with open(txt_dir+txt, 'r') as tf:
        lines = tf.readlines()
        for line in lines:
            label = line.split(' ')[-1]
            print "label: ", label
            imgfile = line.strip(label).strip()
            print imgfile
            shutil.copy(imgfile, temp_dir)
    #break

