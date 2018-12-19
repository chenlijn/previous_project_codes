#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 18:51:22 2018

@author: gaia
"""
from matplotlib import pyplot as plt
from scipy.misc import imread,imsave,imresize
import os
import numpy as np
import xml.etree.ElementTree as ET
import cv2
from PIL import Image
from skimage.morphology import convex_hull_image
from skimage.transform import rotate,resize
from tqdm import tqdm
def preprocessing_xmls(xml_path):
  tree=ET.parse(xml_path)
  l=[]
  for ele in tree.iter():
    if "point" in ele.tag:
      l.append([int(item) for item in str(ele.text).split(",")])
  return np.array(l)
def clip_img_mask(img,mask,countour,flag=False):
  mask=Image.fromarray(mask)
  img=Image.fromarray(img)
  h,w=img.size
  img_copy=np.array(img.copy())
  img_copy=cv2.cvtColor(img_copy,cv2.COLOR_BGR2GRAY)
  _,thresh=cv2.threshold(img_copy,3,255,cv2.THRESH_BINARY)
#  plt.imshow(thresh)
#  plt.show()
  left=np.argmax(thresh[h/2,:])
  right=np.argmin(thresh[h/2,w/2:])+w/2
  top=np.argmax(thresh[:,w/2])
  bottom=np.argmin(thresh[h/2:,w/2])+h/2
  rect=(left,top,right,bottom)
  img_clean = img.crop(rect)
  mask_clean = mask.crop(rect)
  if flag:
    countour=Image.fromarray(countour)
    countour_clean=countour.crop(rect)
    return np.array(img_clean),np.array(mask_clean),np.array(countour_clean)
  return np.array(img_clean),np.array(mask_clean)
def mkdir(dir_):
  if not os.path.exists(dir_):
    os.makedirs(dir_)

def vis_img_mask(img,mask):
  fig=plt.figure()
  fig.add_subplot(121)
  plt.imshow(img)
  fig.add_subplot(122)
  plt.imshow(mask,cmap=plt.get_cmap("gray"))
  plt.show()   
def vis_img_mask_xml(img,mask,xml):
  fig=plt.figure(figsize=(12,4))
  fig.add_subplot(131)
  plt.imshow(img)
  fig.add_subplot(132)
  plt.imshow(mask,cmap=plt.get_cmap("gray"))
  fig.add_subplot(133)
  plt.imshow(xml,cmap=plt.get_cmap("gray"))
  plt.show()   
labels=["normal","abnormal"]
normal_mask_dir="datasets/glaucoma/201802_data2label_v1_clip/abnormal_and_normal/clip_masks/"
normal_img_dir="datasets/glaucoma/201802_data2label_v1_clip/abnormal_and_normal/clip_imgs/"
save_path="datasets/glaucoma/201802_data2label_v1_clip/abnormal_and_normal/"
img_cols=400;img_rows=400
imgs=[];pimgs=[];masks=[];pmasks=[]
#for label in labels:
#  img_dir="datasets/glaucoma/201802_data2label_v1/glaucoma_data2label_v1-bing/"+label+"/"
#  mask_dir="datasets/glaucoma/201802_data2label_v1/glaucoma_data2label_v1-bing/"+label+"/"+label+"-annotation-bing/"
#  
#  
#  mkdir(normal_mask_dir)
#  mkdir(normal_img_dir)
#  
#  
#  for i,f in enumerate(os.listdir(img_dir)):
#    if f[-3:]!="jpg":
#      continue
#    print(f)
#  #  if f!="14ff81f84a92b0a3ca5a50c027197147.jpg":
#  #    continue
#    img = imread(os.path.join(img_dir,f))
#    if img[0,0,0]!=0:
#      continue
#    xml_path=os.path.join(mask_dir,f[:-3]+"xml")
#    points=preprocessing_xmls(xml_path).reshape((-1,1,2))
#    img_temp=np.zeros(img.shape[:-1],dtype=np.uint8)
#    img_mask=cv2.fillPoly(img_temp,[points],1)
#  #  img_contour=cv2.polylines(img,[points],True,(0,255,255),10,cv2.LINE_AA)
#    clip_mask_path=os.path.join(normal_mask_dir,f)
#    clip_img_path=os.path.join(normal_img_dir,f)
#    if img.shape[0]>1000 or img.shape[1]>1000:
#      
#      img_clip,mask_clip=clip_img_mask(img,img_mask,None)
#  
#    else:
#      img_clip,mask_clip=img,img_mask
#    imsave(clip_mask_path,mask_clip)
#    imsave(clip_img_path,img_clip)
#    vis_img_mask(img_clip,mask_clip)
for f in tqdm(os.listdir(normal_img_dir)):
  img_clip=imread(os.path.join(normal_img_dir,f))
  mask_clip=imread(os.path.join(normal_mask_dir,f))
  img_clip=imresize(img_clip, (img_cols, img_rows))
  mask_clip=imresize(mask_clip, (img_cols, img_rows))
  imgs.append(img_clip)
#  print("max img_clip:",np.max(img_clip))
  masks.append(np.where(np.expand_dims(mask_clip,-1)>0.5,255,0))
#  print("max mask_clip:",np.max(np.where(np.expand_dims(mask_clip,-1)>0.5,255,0)))
  vis_img_mask(img_clip,mask_clip)


  mask_clip = convex_hull_image(mask_clip)
  mask_clip = np.where(mask_clip > 0.5, 255, 0)
  
  img_p = rotate(
      cv2.linearPolar(img_clip, (img_clip.shape[0] / 2, img_clip.shape[1] / 2), img_clip.shape[0] / 2, cv2.WARP_FILL_OUTLIERS),
      -90)
  mask_p = rotate(
      cv2.linearPolar(mask_clip*255, (img_clip.shape[0] / 2, img_clip.shape[1] / 2), img_clip.shape[0] / 2, cv2.WARP_FILL_OUTLIERS),
      -90)
  mask_p = np.where(mask_p >np.min(mask_p), 255, 0)
  pimgs.append(img_p)
  pmasks.append(np.expand_dims(mask_p,-1))
#  print("max img_p",np.max(img_p))
#  print("max mask_p",np.max(mask_p))
  vis_img_mask(img_p,mask_p)
#  break 


imgs_=np.array(imgs)
masks_=np.array(masks)
pimgs_=np.array(pimgs)
pmasks_=np.array(pmasks)

num_images = len(imgs_)
rand_i = np.random.choice(range(num_images), size=num_images, replace=False)
test_i = int(0.2 * num_images)
val_i = int(0.4 * num_images)



np.save(save_path+'/trainImages.npy', imgs_[rand_i[val_i:]])
np.save(save_path+'/trainMasks.npy', masks_[rand_i[val_i:]])
np.save(save_path+'/testImages.npy', imgs_[rand_i[:test_i]])
np.save(save_path+'/testMasks.npy', masks_[rand_i[:test_i]])
np.save(save_path+'/valImages.npy', imgs_[rand_i[test_i:val_i]])
np.save(save_path+'/valMasks.npy', masks_[rand_i[test_i:val_i]])

np.save(save_path+'/ptrainImages.npy', pimgs_[rand_i[val_i:]])
np.save(save_path+'/ptrainMasks.npy', pmasks_[rand_i[val_i:]])
np.save(save_path+'/ptestImages.npy', pimgs_[rand_i[:test_i]])
np.save(save_path+'/ptestMasks.npy', pmasks_[rand_i[:test_i]])
np.save(save_path+'/pvalImages.npy', pimgs_[rand_i[test_i:val_i]])
np.save(save_path+'/pvalMasks.npy', pmasks_[rand_i[test_i:val_i]])





