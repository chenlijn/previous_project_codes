import os
import pandas as pd
import xml.etree.ElementTree as ET
from scipy.misc import imread,imsave,imresize
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from skimage.morphology import convex_hull_image
from skimage.transform import rotate,resize
def vis_img_mask(img,mask):
  fig=plt.figure()
  fig.add_subplot(121)
  plt.imshow(img)
  fig.add_subplot(122)
  plt.imshow(mask,cmap=plt.get_cmap("gray"))
  plt.show()   
def preprocessing_xmls(xml_path):
  tree=ET.parse(xml_path)
  l=[]
  for ele in tree.iter():
    if "point" in ele.tag:
      l.append([int(item) for item in str(ele.text).split(",")])
  return np.array(l)
def clip_by_detection_points(img,mask,pnts):
  mask=Image.fromarray(mask)
  img=Image.fromarray(img)
  miny,maxy,minx,maxx=pnts[0],pnts[1],pnts[2],pnts[3]
  rect=(minx,miny,maxx,maxy)
  img_clean = img.crop(rect)
  mask_clean = mask.crop(rect)
  return np.array(img_clean),np.array(mask_clean)
def mkdir(dir_):
  if not os.path.exists(dir_):
    os.makedirs(dir_)
labels=["normal","abnormal"]
img_cols=400;img_rows=400
imgs=[];pimgs=[];masks=[];pmasks=[]
version="1"
normal_mask_dir="datasets/glaucoma_after_detection/glaucoma_data2label_v"+version+"-bing_detected/abnormal_and_normal/clip_masks/"
normal_img_dir="datasets/glaucoma_after_detection/glaucoma_data2label_v"+version+"-bing_detected/abnormal_and_normal/clip_imgs/"
save_path="datasets/glaucoma_after_detection/glaucoma_data2label_v"+version+"-bing_detected/abnormal_and_normal/"
mkdir(normal_mask_dir)
mkdir(normal_img_dir)
mkdir(save_path)

for label in labels:
  img_dir="datasets/glaucoma/201802_data2label_v"+version+"/glaucoma_data2label_v"+version+"-bing/"+label+"/"
  mask_dir="datasets/glaucoma/201802_data2label_v"+version+"/glaucoma_data2label_v"+version+"-bing/"+label+"/"+label+"-annotation-bing/"
  annot_path="datasets/glaucoma_after_detection/glaucoma_data2label_v"+version+"-bing/"+label+"/"+label+".txt"
  all_annot=pd.read_table(annot_path,header=None).values[:,:5]
  files=all_annot[:,0]
  disc_str=all_annot[:,1]
  macula_str=all_annot[:,2]
  disc_str_orig=all_annot[:,3]
  macula_str_orig=all_annot[:,4]
  for i,f in enumerate(files):
    if i%50==0:
      print(i,"/",len(files))
    #print(f)

    img = imread(os.path.join(img_dir,f))

    xml_path=os.path.join(mask_dir,f[:-3]+"xml")
    points=preprocessing_xmls(xml_path).reshape((-1,1,2))
    img_temp=np.zeros(img.shape[:-1],dtype=np.uint8)
    img_mask=cv2.fillPoly(img_temp,[points],1)
    disc_points=[int(item) for item in disc_str[i].split("_")]
    img_clip,mask_clip=clip_by_detection_points(img,img_mask,disc_points)

    clip_mask_path=os.path.join(normal_mask_dir,f)
    clip_img_path=os.path.join(normal_img_dir,f)

    imsave(clip_mask_path,mask_clip)
    imsave(clip_img_path,img_clip)
for i,f in enumerate(os.listdir(normal_img_dir)):
  if i%50==0:
    print(i,"/",len(os.listdir(normal_img_dir)))
  """???
  img_clip=imread(os.path.join(normal_img_dir,f))
  mask_clip=imread(os.path.join(normal_mask_dir,f))
  img_clip=imresize(img_clip, (img_cols, img_rows))
  mask_clip=imresize(mask_clip, (img_cols, img_rows))
  imgs.append(img_clip)
  masks.append(np.where(np.expand_dims(mask_clip,-1)>0.5,255,0))

  
  mask_clip = convex_hull_image(mask_clip)
  mask_clip = np.where(mask_clip > 0.5, 255, 0)
  
  img_p = rotate(
      cv2.linearPolar(img_clip, (img_clip.shape[0] / 2, img_clip.shape[1] / 2), img_clip.shape[0] / 2, cv2.WARP_FILL_OUTLIERS),
      -90)
  mask_p = rotate(
      cv2.linearPolar(mask_clip*255, (img_clip.shape[0] / 2, img_clip.shape[1] / 2), img_clip.shape[0] / 2, cv2.WARP_FILL_OUTLIERS),
      -90)
  mask_p = np.where(mask_p >np.min(mask_p), 255, 0)
  """
  
  #"""??
  img_clip = cv2.imread(os.path.join(normal_img_dir,f))
  mask_clip = cv2.imread(os.path.join(normal_mask_dir,f),0)
  img_clip=imresize(img_clip, (img_cols, img_rows))
  mask_clip=imresize(mask_clip, (img_cols, img_rows))
  imgs.append(img_clip)
  masks.append(np.where(np.expand_dims(mask_clip,-1)>0.5,255,0))
    
  mask_clip = convex_hull_image(mask_clip)
  
  mask_clip = np.where(mask_clip > 0.5, 255, 0)
  #print(Image.fromarray(img_clip).mode)
  img_p = cv2.linearPolar(img_clip, (img_clip.shape[0] / 2, img_clip.shape[1] / 2), img_clip.shape[0] / 2, cv2.WARP_FILL_OUTLIERS)
  mask_p = cv2.linearPolar(mask_clip*255, (img_clip.shape[0] / 2, img_clip.shape[1] / 2), img_clip.shape[0] / 2, cv2.WARP_FILL_OUTLIERS)

  #"""
  pimgs.append(img_p)
  pmasks.append(np.expand_dims(mask_p,-1))
  #break

imgs_=np.array(imgs)
masks_=np.array(masks)
pimgs_=np.array(pimgs)
pmasks_=np.array(pmasks)

num_images = len(imgs_)
rand_i = np.random.choice(range(num_images), size=num_images, replace=False)
test_i = int(0.2 * num_images)
val_i = int(0.4 * num_images)



np.save(save_path+'/yetrainImages.npy', imgs_[rand_i[val_i:]])
np.save(save_path+'/yetrainMasks.npy', masks_[rand_i[val_i:]])
np.save(save_path+'/yetestImages.npy', imgs_[rand_i[:test_i]])
np.save(save_path+'/yetestMasks.npy', masks_[rand_i[:test_i]])
np.save(save_path+'/yevalImages.npy', imgs_[rand_i[test_i:val_i]])
np.save(save_path+'/yevalMasks.npy', masks_[rand_i[test_i:val_i]])

np.save(save_path+'/yeptrainImages.npy', pimgs_[rand_i[val_i:]])
np.save(save_path+'/yeptrainMasks.npy', pmasks_[rand_i[val_i:]])
np.save(save_path+'/yeptestImages.npy', pimgs_[rand_i[:test_i]])
np.save(save_path+'/yeptestMasks.npy', pmasks_[rand_i[:test_i]])
np.save(save_path+'/yepvalImages.npy', pimgs_[rand_i[test_i:val_i]])
np.save(save_path+'/yepvalMasks.npy', pmasks_[rand_i[test_i:val_i]])

