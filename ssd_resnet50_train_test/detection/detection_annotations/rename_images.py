import os
import sys
import numpy as np
import shutil
import cv2
import shutil
import base64
from hashlib import md5

from xml.etree.ElementTree import Element, SubElement, Comment, tostring
from xml.etree import ElementTree
from xml.dom import minidom

def prettyfy(elem):
    rough_string = tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def generate_xml(imgname, height, width, xmin, ymin, xmax, ymax):
    root = Element('annotation')
    filename = SubElement(root, 'filename')
    filename.text = imgname
    
    size = SubElement(root, 'size')
    width_ = SubElement(size, 'width')
    width_.text = str(width)
    height_ = SubElement(size, 'height')
    height_.text = str(height)
    depth = SubElement(size, 'depth')
    depth.text = '3'

    segmented = SubElement(root, 'segmented')
    segmented.text = '0'

    object_ = SubElement(root, 'object')
    name = SubElement(object_, 'name')
    name.text = 'pupil'
    pose =  SubElement(object_, 'pose')
    pose.text = 'center'
    truncated = SubElement(object_, 'truncated')
    truncated.text = '0'
    difficult = SubElement(object_, 'difficult')
    difficult.text = '0'
    bndbox = SubElement(object_, 'bndbox')
    xmin_ = SubElement(bndbox, 'xmin')
    xmin_.text = str(xmin)
    ymin_ = SubElement(bndbox, 'ymin')
    ymin_.text = str(ymin)
    xmax_ = SubElement(bndbox, 'xmax')
    xmax_.text = str(xmax)
    ymax_ = SubElement(bndbox, 'ymax')
    ymax_.text = str(ymax)
    return prettyfy(root)    


def get_lines(txt_file):
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    return lines

def parse_line(line):
    imgname, xmin, ymin, xmax, ymax = line.split(',')
    return imgname, xmin, ymin, xmax, ymax

def rename_image(imgname, path):
    imgfile = os.path.join(path, imgname)
    print(imgfile)
    with open(imgfile, 'rb') as f:
        imgstr = base64.b64encode(f.read())
    md5_num = md5(imgstr).hexdigest()
    newfile = os.path.join(path, md5_num+'.jpg')
    os.rename(imgfile, newfile)
    return md5_num+'.jpg'



def main(txt_file, new_txt_file, img_path, annot_path):
    lines = get_lines(txt_file)
    new_lines=[]
    for line in lines:
        imgname, xmin, ymin, xmax, ymax = parse_line(line)
        new_name = rename_image(imgname, img_path)
        newline = '{},{},{},{},{}'.format(new_name, xmin, ymin, xmax, ymax)
        new_lines.append(newline)
        image = cv2.imread(os.path.join(img_path, new_name))
        height, width, _ = image.shape
        xml = generate_xml(new_name, height, width, xmin, ymin, xmax, ymax) 
        with open(os.path.join(annot_path, new_name.split('.')[0]+'.xml'), 'w+') as f:
            f.write(xml)
    
    with open(new_txt_file, 'w+') as f:
        f.writelines(new_lines)


if __name__=='__main__':
   
    ## diffuse
    #txt_file = 'picked_diff_box.txt'
    #new_txt_file = 'new_diff_box.txt'
    #img_path = '/root/mount_out/data/cataract_detection_data/diffuse'
    #annot_path = '/root/mount_out/data/cataract_detection_data/diff_annotations'

    # zhongshan slit
    txt_file = 'cataract_det_ZD_box.txt'
    new_txt_file = 'new_cataract_det_ZD_box.txt'
    img_path = '/root/mount_out/data/cataract_det_annotation_for_zhongshan/slit'
    annot_path = '/root/mount_out/data/cataract_det_annotation_for_zhongshan/slit_annotations'

    main(txt_file, new_txt_file, img_path, annot_path) 

