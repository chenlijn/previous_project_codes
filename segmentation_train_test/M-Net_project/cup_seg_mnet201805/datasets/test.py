
import numpy as np
from skimage import io
import cv2


#prefix = 'train'
prefix = './train_val_v1/train'
#prefix = 'val'
imgs = np.load(prefix + '_imgs.npy')
imgs_polar = np.load(prefix + '_imgs_polar.npy')

masks = np.load(prefix + '_masks.npy')
masks_polar = np.load(prefix + '_masks_polar.npy')


i = 0
kernel = np.ones((3,3), np.uint8)
for img, mask in zip(imgs_polar, masks_polar):
    
    print img.max(), mask.max()
    print img.dtype
    #rows, cols, _ = img.shape
    #img_inv = cv2.linearPolar(img, (cols/2, rows/2), rows/2, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)
    #img_inv = img_inv[:,:,::-1]
    #mask_inv = cv2.linearPolar(mask, (cols/2, rows/2), rows/2, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)
    #erosion = cv2.erode(mask_inv, kernel, iterations=1)
    #edge = mask_inv - erosion
    #rs, cs = np.where(edge==255)
    #img_inv[rs, cs, 0] = 0
    #img_inv[rs, cs, 1] = 255
    #img_inv[rs, cs, 2] = 0
    #cv2.imwrite('./test/img_polar_{}.png'.format(str(i)), img_inv)
    #cv2.imwrite('./test/mask_polar_{}.png'.format(str(i)), mask_inv)
    cv2.imwrite('./test_p/img_polar_{}.png'.format(str(i)), img)
    cv2.imwrite('./test_p/mask_polar_{}.png'.format(str(i)), mask)
    i += 1


