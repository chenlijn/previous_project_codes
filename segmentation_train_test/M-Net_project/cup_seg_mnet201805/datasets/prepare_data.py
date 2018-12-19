
import os 
import numpy as np
import cv2
from skimage import io


def pack_imgs_to_npy(list_file, img_path,  savename, size=(400, 400, 3), is_mask=False):

    rows, cols, chans = size
    with open(list_file, 'r') as lf:
        imgnames = lf.readlines()

    img_num = len(imgnames)
    img_num = 160 # test
    if is_mask:
        chans = 1
    img_arrays = np.zeros((img_num, rows, cols, chans), np.float32)
    polar_img_arrays = np.zeros((img_num, rows, cols, chans), np.float32)

    for idx, name in enumerate(imgnames[:img_num]):
        if is_mask:
            name = name.strip('\n').split('.')[0] + '.png'
            imgfile = '/' + img_path.strip('/') + '/' + name
            image = cv2.imread(imgfile, 0)
        else:
            name = name.strip('\n')
            imgfile = '/' + img_path.strip('/') + '/' + name
            image = cv2.imread(imgfile)
        print image.shape

        # resize
        res_img = cv2.resize(image, (cols, rows))
        
        if is_mask:
            res_img = np.where(res_img > 0, 255, 0)
            img_arrays[idx,:,:,0] = res_img

            polar_mask = cv2.linearPolar(res_img, (cols/2, rows/2), rows/2, cv2.WARP_FILL_OUTLIERS)
            polar_img_arrays[idx,:,:,0] = polar_mask

            cv2.imwrite('mask.png', res_img)
            cv2.imwrite('mask_polar.png', polar_mask)
        else:
            rgb_img = res_img[:,:,::-1]
            print rgb_img.shape
            img_arrays[idx, :, :, :] = rgb_img

            polar_img = cv2.linearPolar(rgb_img, (cols/2, rows/2), rows/2, cv2.WARP_FILL_OUTLIERS)
            polar_img_arrays[idx,:,:,:] = polar_img

            io.imsave('img.png', rgb_img)
            io.imsave('img_polar.png', polar_img)


    np.save(savename, img_arrays)
    np.save(savename.split('.')[0]+'_polar.npy', polar_img_arrays)


if __name__ == '__main__':

    img_type = 'train'
    #img_type = 'val'
    list_file = '/mnt/lijian/mount_out/docker_share/glaucoma/deeplab/lists/{}.txt'.format(img_type)
    img_path = '/mnt/lijian/mount_out/docker_share/glaucoma/deeplab/all_images/'

    pack_imgs_to_npy(list_file, img_path, savename='{}_imgs.npy'.format(img_type))


    img_path = '/mnt/lijian/mount_out/docker_share/glaucoma/deeplab/all_masks/'

    pack_imgs_to_npy(list_file, img_path, savename='{}_masks.npy'.format(img_type), is_mask=True)


