
import numpy as np
import cv2
import os


def draw_detection(img, pnt1, pnt2, color=(0,255,0)):
    img = cv2.rectangle(img, pnt1, pnt2, color, 5)
    return img


def get_annotation(txt_file):
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    return lines

def get_img_and_pnts(line):
    print(line)
    part0, part1, part2, part3, part4, part5, part6, _ = line.strip('\r\n').split(',')
    imgname = '/root/mount_out/data/' + part1 + '/' + part2
    print(part0)
    print(part1)
    print(part6)
    pnt1 = (int(part3), int(part4))
    pnt2 = (int(part5), int(part6))
    return imgname, pnt1, pnt2

def draw_img(imgname, pnt1, pnt2):
    img = cv2.imread(imgname)
    print(img.shape)
    img = draw_detection(img, pnt1, pnt2) 
    return img 

if __name__=="__main__": 
    work_root = '/root/mount_out/show/cataract_detection_show/'
    lines = get_annotation(work_root + 'cataract_det_ZD.txt')
    for line in lines:
        imgname, pnt1, pnt2 = get_img_and_pnts(line)
        print(imgname)
        print(pnt1)
        print(pnt2)
        img = draw_img(imgname, pnt1, pnt2)
        print(work_root + 'show_annotations/' +imgname.split('/')[-1])
        cv2.imwrite(work_root + 'show_annotations/' +imgname.split('/')[-1], img)


