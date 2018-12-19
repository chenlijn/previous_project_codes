from __future__ import print_function

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from skimage.transform import resize,rotate
from skimage.morphology import remove_small_objects
from skimage.measure import label
from scipy.misc import imsave
import numpy as np
import math
import cv2
from PIL import Image
from keras.models import Model
from keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, BatchNormalization, \
    SpatialDropout2D, AveragePooling2D, Average, AtrousConvolution2D, DepthwiseConv2D, concatenate, Activation
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, CSVLogger, ReduceLROnPlateau
from keras.regularizers import l2
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from itertools import izip, imap

from matplotlib import pyplot as plt
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf.Session(config=config)
KTF.set_session(session)

working_path = '/home/gaia/mengji/datasets/glaucoma_after_detection/glaucoma_data2label_v1-bing_detected/abnormal_and_normal/'
#working_path = '/home/gaia/Xiaoye/glaucoma_segmentation_imgs/glaucoma_images_with_cup_disc_annotation_polar2/v3/'
model_path = 'models/aspp_unet_cups_discs.hdf5'
log_path = 'models/aspp_unet_cups_discs.log'
#model_path = 'models/unetp3_da.hdf5'
#log_path = 'models/unetp3_da.log'
K.set_image_dim_ordering('tf')  # Theano dimension ordering in this code

img_rows = 400
img_cols = 400

learning_rate = 5e-5
dropout_rate = 0.
l2factor = 5e-4
batchsize = 8
channelnum = 16
def load_train_data_for_mean_std():
    working_path2='/home/gaia/Xiaoye/glaucoma_segmentation_imgs/glaucoma_images_with_cup_disc_annotation_polar2/v3/'
    imgs_train = np.load(working_path2 + 'ptrainImages.npy')
    imgs_mask_train = np.load(working_path2 + 'ptrainMasks.npy')
    imgs_mask2_train = np.load(working_path2 + 'ptrainMasks2.npy')
    return imgs_train, imgs_mask_train, imgs_mask2_train

def load_train_data():
    imgs_train = np.load(working_path + 'yeptrainImages.npy')
    imgs_mask_train = np.load(working_path + 'yeptrainMasks.npy')
    imgs_mask2_train = np.load(working_path + 'yeptrainMasks_disc_get_by_mnet.npy')
    return imgs_train, imgs_mask2_train, imgs_mask_train


def load_test_data():
    imgs_train = np.load(working_path + 'yeptestImages.npy')
    imgs_mask_train = np.load(working_path + 'yeptestMasks.npy')
    imgs_mask2_train = np.load(working_path + 'yeptestMasks_disc_get_by_mnet.npy')
    oimgs_mask_train = np.load(working_path + 'yetestMasks.npy')
    oimgs_mask2_train = np.load(working_path + 'yetestMasks_disc_get_by_mnet.npy')
    return imgs_train, imgs_mask2_train, imgs_mask_train, oimgs_mask2_train, oimgs_mask_train


def load_val_data():
    imgs_train = np.load(working_path + 'yepvalImages.npy')
    imgs_mask_train = np.load(working_path + 'yepvalMasks.npy')
    imgs_mask2_train = np.load(working_path + 'yepvalMasks_disc_get_by_mnet.npy')
    return imgs_train, imgs_mask2_train, imgs_mask_train

def load_train_data_only_cup():
    imgs_train = np.load(working_path + 'yeptrainImages.npy')
    imgs_mask_train = np.load(working_path + 'yeptrainMasks.npy')
    return imgs_train, imgs_mask_train
    
def load_orig_test_img():
    imgs=np.load(working_path + 'yetestImages.npy')
    return imgs

def load_test_data_only_cup():
    imgs_train = np.load(working_path + 'yeptestImages.npy')
    imgs_mask_train = np.load(working_path + 'yeptestMasks.npy')
    oimgs_mask_train = np.load(working_path + 'yetestMasks.npy')
    return imgs_train, imgs_mask_train, oimgs_mask_train


def load_val_data_only_cup():
    imgs_train = np.load(working_path + 'yepvalImages.npy')
    imgs_mask_train = np.load(working_path + 'yepvalMasks.npy')
    return imgs_train, imgs_mask_train


def aspp(x,filters,img_size_pooled,pooled_times):
    aspp1=Conv2D(filters,(1,1),padding="same")(x)
    aspp1=BatchNormalization()(aspp1)
    aspp1=Activation("elu")(aspp1)
    
    aspp2=DepthwiseConv2D((3,3),padding="same",dilation_rate=(6,6))(x)
    aspp2=BatchNormalization()(aspp2)
    aspp2=Activation("elu")(aspp2)
    aspp2=Conv2D(filters,(1,1),padding="same")(aspp2)
    aspp2=BatchNormalization()(aspp2)
    aspp2=Activation("elu")(aspp2)
    
    aspp3=DepthwiseConv2D((3,3),padding="same",dilation_rate=(12,12))(x)
    aspp3=BatchNormalization()(aspp3)
    aspp3=Activation("elu")(aspp3)
    aspp3=Conv2D(filters,(1,1),padding="same")(aspp3)
    aspp3=BatchNormalization()(aspp3)
    aspp3=Activation("elu")(aspp3)
        
    aspp4=DepthwiseConv2D((3,3),padding="same",dilation_rate=(24,24))(x)
    aspp4=BatchNormalization()(aspp4)
    aspp4=Activation("elu")(aspp4)
    aspp4=Conv2D(filters,(1,1),padding="same")(aspp4)
    aspp4=BatchNormalization()(aspp4)
    aspp4=Activation("elu")(aspp4)
    
    aspp5=AveragePooling2D(pool_size=(img_size_pooled,img_size_pooled))(x)
    aspp5=Conv2D(filters,(1,1),padding="same")(aspp5)
    aspp5=BatchNormalization()(aspp5)
    aspp5=Activation("elu")(aspp5)
    aspp5=UpSampling2D((img_rows/2**pooled_times,img_rows/2**pooled_times))(aspp5)
    
    out=concatenate([aspp1,aspp2,aspp3,aspp4,aspp5])
    
    return out
    

def iou_coef_keras(y_true,y_pred):
  y_true=tf.cast("float32")
  y_pred=tf.cast("float32")
  y_true=tf.cast(K.greater(K.batch_flatten(y_true),0.5),"float32")
  y_pred=tf.cast(K.greater(K.batch_flatten(y_pred),0.5),"float32")
  intersection=K.sum(y_true*y_pred,axis=1)
  union=K.sum(K.maximum(y_true,y_pred),axis=1)
  return K.mean(intersection/K.cast(union,"float32"))
     
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))

def dice_coef2(y_true, y_pred):
    import tensorflow as tf
    y_pred2 = tf.where(tf.greater_equal(y_pred, 0.5), K.ones_like(y_pred), K.zeros_like(y_pred))
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred2)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1e-3) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1e-3)

def dice_coef_np(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1e-3) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1e-3)
def iou_coef_np(y_true,y_pred):
  y_true_f=y_true.flatten()
  y_pred_f=y_pred.flatten()
  intersection=np.sum(y_true_f*y_pred_f)
  union=np.sum(np.maximum(y_true_f,y_pred_f))
  return intersection/(union+1e-3)

def dice_coef_loss(y_true, y_pred):
    return -K.log(dice_coef(y_true, y_pred))


def focal_loss(y_true, y_pred):
    import tensorflow as tf
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    return -K.sum(1 * K.pow(1. - pt, 1) * K.log(pt))


def get_unet():
    print("mnet")
    inputs = Input((img_rows, img_cols, 3))
    inputs2 = AveragePooling2D(pool_size=(2, 2))(inputs)
    inputs4 = AveragePooling2D(pool_size=(2, 2))(inputs2)
    inputs8 = AveragePooling2D(pool_size=(2, 2))(inputs4)
    conv2_i = Conv2D(64, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(inputs2)
    conv3_i = Conv2D(128, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(inputs4)
    conv4_i = Conv2D(256, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(inputs8)

    conv1 = Conv2D(32, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(inputs)
    conv1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv1)
    conv1 = SpatialDropout2D(dropout_rate)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Concatenate(axis=-1)([pool1, conv2_i])
    conv2 = Conv2D(64, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv2)
    conv2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv2)
    conv2 = SpatialDropout2D(dropout_rate)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Concatenate(axis=-1)([pool2, conv3_i])
    conv3 = Conv2D(128, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv3)
    conv3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv3)
    conv3 = SpatialDropout2D(dropout_rate)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Concatenate(axis=-1)([pool3, conv4_i])
    conv4 = Conv2D(256, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv4)
    conv4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv4)
    conv4 = SpatialDropout2D(dropout_rate)(conv4)
    conv4 = Conv2D(256, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(pool4)
    conv5 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv5)
    conv5 = SpatialDropout2D(dropout_rate)(conv5)
    conv5 = Conv2D(512, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv5)

    up6 = Concatenate(axis=-1)(
        [Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=l2(l2factor))(conv5), conv4])
    conv6 = Conv2D(256, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(up6)
    conv6 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv6)
    conv6 = SpatialDropout2D(dropout_rate)(conv6)
    conv6 = Conv2D(256, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv6)

    conv6_s = Conv2DTranspose(channelnum, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=l2(l2factor))(conv6)
    conv6_s = Conv2DTranspose(channelnum, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=l2(l2factor))(conv6_s)
    conv6_s = Conv2DTranspose(channelnum, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=l2(l2factor))(conv6_s)
    conv6_s = Conv2D(2, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv6_s)
    conv7_1 = Conv2D(1, (1, 1), activation='sigmoid', name='disc-3')(conv6_s)
    conv7_2 = Conv2D(1, (1, 1), activation='sigmoid', name='cup-3')(conv6_s)

    up7 = Concatenate(axis=-1)(
        [Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=l2(l2factor))(conv6), conv3])
    conv7 = Conv2D(128, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(up7)
    conv7 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv7)
    conv7 = SpatialDropout2D(dropout_rate)(conv7)
    conv7 = Conv2D(128, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv7)

    conv7_s = Conv2DTranspose(channelnum, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=l2(l2factor))(conv7)
    conv7_s = Conv2DTranspose(channelnum, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=l2(l2factor))(conv7_s)
    conv7_s = Conv2D(2, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv7_s)
    conv8_1 = Conv2D(1, (1, 1), activation='sigmoid', name='disc-2')(conv7_s)
    conv8_2 = Conv2D(1, (1, 1), activation='sigmoid', name='cup-2')(conv7_s)

    up8 = Concatenate(axis=-1)(
        [Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=l2(l2factor))(conv7), conv2])
    conv8 = Conv2D(64, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(up8)
    conv8 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv8)
    conv8 = SpatialDropout2D(dropout_rate)(conv8)
    conv8 = Conv2D(64, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv8)

    conv8_s = Conv2DTranspose(channelnum, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=l2(l2factor))(conv8)
    conv8_s = Conv2D(2, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv8_s)
    conv9_1 = Conv2D(1, (1, 1), activation='sigmoid', name='disc-1')(conv8_s)
    conv9_2 = Conv2D(1, (1, 1), activation='sigmoid', name='cup-1')(conv8_s)

    up9 = Concatenate(axis=-1)(
        [Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=l2(l2factor))(conv8), conv1])
    conv9 = Conv2D(32, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(up9)
    conv9 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv9)
    conv9 = SpatialDropout2D(dropout_rate)(conv9)
    conv9 = Conv2D(32, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv9)

    conv9_s = Conv2D(2, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv9)
    conv10_1 = Conv2D(1, (1, 1), activation='sigmoid', name='disc-0')(conv9_s)
    conv10_2 = Conv2D(1, (1, 1), activation='sigmoid', name='cup-0')(conv9_s)

    output_1 = Average(name='disc')([conv7_1, conv8_1, conv9_1, conv10_1])
    output_2 = Average(name='cup')([conv7_2, conv8_2, conv9_2, conv10_2])


    model = Model(inputs=inputs, outputs=[conv7_1, conv7_2, conv8_1, conv8_2, conv9_1, conv9_2, conv10_1, conv10_2,
                                          output_1, output_2])

    model.compile(optimizer=Adam(lr=learning_rate, clipnorm=1.), loss=dice_coef_loss, loss_weights=[0.5, 0.5, 0.5, 0.5, 0.5,
                                                                                                    0.5, 0.5, 0.5, 0.5, 0.5],
                  metrics=[dice_coef2])

    #model.summary()

    return model
"""
1. add atrous layer in block4
2. remove block5 and block6
"""
def get_atrous_unet():##should be called as atrous_mnet
    print("Atrous mnet")
    inputs = Input((img_rows, img_cols, 3))
    inputs2 = AveragePooling2D(pool_size=(2, 2))(inputs)
    inputs4 = AveragePooling2D(pool_size=(2, 2))(inputs2)
    inputs8 = AveragePooling2D(pool_size=(2, 2))(inputs4)
    conv2_i = Conv2D(64, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(inputs2)
    conv3_i = Conv2D(128, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(inputs4)
    conv4_i = Conv2D(256, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(inputs8)

    conv1 = Conv2D(32, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(inputs)
    conv1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv1)
    conv1 = SpatialDropout2D(dropout_rate)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Concatenate(axis=-1)([pool1, conv2_i])
    conv2 = Conv2D(64, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv2)
    conv2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv2)
    conv2 = SpatialDropout2D(dropout_rate)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Concatenate(axis=-1)([pool2, conv3_i])
    conv3 = Conv2D(128, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv3)
    conv3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv3)
    conv3 = SpatialDropout2D(dropout_rate)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Concatenate(axis=-1)([pool3, conv4_i])
    conv4 = Conv2D(256, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv4)
    conv4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv4)
    conv4 = SpatialDropout2D(dropout_rate)(conv4)
    conv4 = AtrousConvolution2D(256, (3, 3), atrous_rate=2, border_mode="same",kernel_regularizer=l2(l2factor),activation="elu")(conv4)
    conv4 = Conv2D(256, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv4)
    conv4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv4)
    conv4 = SpatialDropout2D(dropout_rate)(conv4)
    

    conv6_s = Conv2DTranspose(channelnum, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=l2(l2factor))(conv4)
    conv6_s = Conv2DTranspose(channelnum, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=l2(l2factor))(conv6_s)
    conv6_s = Conv2DTranspose(channelnum, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=l2(l2factor))(conv6_s)
    conv6_s = Conv2D(2, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv6_s)
    conv7_1 = Conv2D(1, (1, 1), activation='sigmoid', name='disc-3')(conv6_s)
    conv7_2 = Conv2D(1, (1, 1), activation='sigmoid', name='cup-3')(conv6_s)

    up7 = Concatenate(axis=-1)(
        [Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=l2(l2factor))(conv4), conv3])
    conv7 = Conv2D(128, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(up7)
    conv7 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv7)
    conv7 = SpatialDropout2D(dropout_rate)(conv7)
    conv7 = Conv2D(128, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv7)

    conv7_s = Conv2DTranspose(channelnum, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=l2(l2factor))(conv7)
    conv7_s = Conv2DTranspose(channelnum, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=l2(l2factor))(conv7_s)
    conv7_s = Conv2D(2, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv7_s)
    conv8_1 = Conv2D(1, (1, 1), activation='sigmoid', name='disc-2')(conv7_s)
    conv8_2 = Conv2D(1, (1, 1), activation='sigmoid', name='cup-2')(conv7_s)

    up8 = Concatenate(axis=-1)(
        [Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=l2(l2factor))(conv7), conv2])
    conv8 = Conv2D(64, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(up8)
    conv8 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv8)
    conv8 = SpatialDropout2D(dropout_rate)(conv8)
    conv8 = Conv2D(64, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv8)

    conv8_s = Conv2DTranspose(channelnum, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=l2(l2factor))(conv8)
    conv8_s = Conv2D(2, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv8_s)
    conv9_1 = Conv2D(1, (1, 1), activation='sigmoid', name='disc-1')(conv8_s)
    conv9_2 = Conv2D(1, (1, 1), activation='sigmoid', name='cup-1')(conv8_s)

    up9 = Concatenate(axis=-1)(
        [Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=l2(l2factor))(conv8), conv1])
    conv9 = Conv2D(32, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(up9)
    conv9 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv9)
    conv9 = SpatialDropout2D(dropout_rate)(conv9)
    conv9 = Conv2D(32, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv9)

    conv9_s = Conv2D(2, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv9)
    conv10_1 = Conv2D(1, (1, 1), activation='sigmoid', name='disc-0')(conv9_s)
    conv10_2 = Conv2D(1, (1, 1), activation='sigmoid', name='cup-0')(conv9_s)

    output_1 = Average(name='disc')([conv7_1, conv8_1, conv9_1, conv10_1])
    output_2 = Average(name='cup')([conv7_2, conv8_2, conv9_2, conv10_2])


    model = Model(inputs=inputs, outputs=[conv7_1, conv7_2, conv8_1, conv8_2, conv9_1, conv9_2, conv10_1, conv10_2,
                                          output_1, output_2])

    model.compile(optimizer=Adam(lr=learning_rate, clipnorm=1.), loss=dice_coef_loss, loss_weights=[0.5, 0.5, 0.5, 0.5, 0.5,
                                                                                                    0.5, 0.5, 0.5, 0.5, 0.5],
                  metrics=[dice_coef2])

    #model.summary()

    return model
"""
1. add atrous layer in block4
2. remove block5 and block6
"""
def get_aspp_unet():##should be called as atrous_mnet
    print("ASPP mnet")
    inputs = Input((img_rows, img_cols, 3))
    inputs2 = AveragePooling2D(pool_size=(2, 2))(inputs)
    inputs4 = AveragePooling2D(pool_size=(2, 2))(inputs2)
    inputs8 = AveragePooling2D(pool_size=(2, 2))(inputs4)
    conv2_i = Conv2D(64, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(inputs2)
    conv3_i = Conv2D(128, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(inputs4)
    conv4_i = Conv2D(256, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(inputs8)

    conv1 = Conv2D(32, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(inputs)
    conv1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv1)
    conv1 = SpatialDropout2D(dropout_rate)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Concatenate(axis=-1)([pool1, conv2_i])
    conv2 = Conv2D(64, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv2)
    conv2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv2)
    conv2 = SpatialDropout2D(dropout_rate)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Concatenate(axis=-1)([pool2, conv3_i])
    conv3 = Conv2D(128, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv3)
    conv3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv3)
    conv3 = SpatialDropout2D(dropout_rate)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Concatenate(axis=-1)([pool3, conv4_i])
    conv4 = Conv2D(256, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv4)
    conv4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv4)
    conv4 = SpatialDropout2D(dropout_rate)(conv4)
    conv4 = aspp(conv4,256,img_rows/8,3)
    conv4 = Conv2D(256, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv4)
    conv4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv4)
    conv4 = SpatialDropout2D(dropout_rate)(conv4)
    

    conv6_s = Conv2DTranspose(channelnum, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=l2(l2factor))(conv4)
    conv6_s = Conv2DTranspose(channelnum, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=l2(l2factor))(conv6_s)
    conv6_s = Conv2DTranspose(channelnum, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=l2(l2factor))(conv6_s)
    conv6_s = Conv2D(2, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv6_s)
    conv7_1 = Conv2D(1, (1, 1), activation='sigmoid', name='disc-3')(conv6_s)
    conv7_2 = Conv2D(1, (1, 1), activation='sigmoid', name='cup-3')(conv6_s)

    up7 = Concatenate(axis=-1)(
        [Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=l2(l2factor))(conv4), conv3])
    conv7 = Conv2D(128, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(up7)
    conv7 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv7)
    conv7 = SpatialDropout2D(dropout_rate)(conv7)
    conv7 = Conv2D(128, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv7)

    conv7_s = Conv2DTranspose(channelnum, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=l2(l2factor))(conv7)
    conv7_s = Conv2DTranspose(channelnum, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=l2(l2factor))(conv7_s)
    conv7_s = Conv2D(2, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv7_s)
    conv8_1 = Conv2D(1, (1, 1), activation='sigmoid', name='disc-2')(conv7_s)
    conv8_2 = Conv2D(1, (1, 1), activation='sigmoid', name='cup-2')(conv7_s)

    up8 = Concatenate(axis=-1)(
        [Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=l2(l2factor))(conv7), conv2])
    conv8 = Conv2D(64, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(up8)
    conv8 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv8)
    conv8 = SpatialDropout2D(dropout_rate)(conv8)
    conv8 = Conv2D(64, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv8)

    conv8_s = Conv2DTranspose(channelnum, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=l2(l2factor))(conv8)
    conv8_s = Conv2D(2, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv8_s)
    conv9_1 = Conv2D(1, (1, 1), activation='sigmoid', name='disc-1')(conv8_s)
    conv9_2 = Conv2D(1, (1, 1), activation='sigmoid', name='cup-1')(conv8_s)

    up9 = Concatenate(axis=-1)(
        [Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=l2(l2factor))(conv8), conv1])
    conv9 = Conv2D(32, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(up9)
    conv9 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv9)
    conv9 = SpatialDropout2D(dropout_rate)(conv9)
    conv9 = Conv2D(32, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv9)

    conv9_s = Conv2D(2, (3, 3), activation='elu', padding='same', kernel_regularizer=l2(l2factor))(conv9)
    conv10_1 = Conv2D(1, (1, 1), activation='sigmoid', name='disc-0')(conv9_s)
    conv10_2 = Conv2D(1, (1, 1), activation='sigmoid', name='cup-0')(conv9_s)

    output_1 = Average(name='disc')([conv7_1, conv8_1, conv9_1, conv10_1])
    output_2 = Average(name='cup')([conv7_2, conv8_2, conv9_2, conv10_2])


    model = Model(inputs=inputs, outputs=[conv7_1, conv7_2, conv8_1, conv8_2, conv9_1, conv9_2, conv10_1, conv10_2,
                                          output_1, output_2])

    model.compile(optimizer=Adam(lr=learning_rate, clipnorm=1.), loss=dice_coef_loss, loss_weights=[0.5, 0.5, 0.5, 0.5, 0.5,
                                                                                                    0.5, 0.5, 0.5, 0.5, 0.5],
                  metrics=[dice_coef2])

    #model.summary()

    return model
# def preprocess(imgs):
#     imgs_p = np.ndarray((imgs.shape[0], 1, img_rows, img_cols), dtype=np.uint8)
#     for i in range(imgs.shape[0]):
#         imgs_p[i, 0] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)
#
#     return imgs_p

def create_list(a, b):
    return [a, b, a, b, a, b, a, b, a, b]

def train_and_predict():
    print('lr=%f, dr=%f, l2=%f, BS=%d'%(learning_rate, dropout_rate, l2factor, batchsize))
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)
    imgs_train, imgs_mask_train, imgs_mask2_train = load_train_data()
    imsave("train_disc_mask.png",np.squeeze(imgs_mask_train[0]))
    imsave("train_cup_mask.png",np.squeeze(imgs_mask2_train[0]))

    imgs_val, imgs_mask_val, imgs_mask2_val = load_val_data()
    imsave("val_disc_mask.png",np.squeeze(imgs_mask_train[0]))
    imsave("val_cup_mask.png",np.squeeze(imgs_mask2_train[0]))
    
    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization
    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    #imgs_mask_train /= 255.  # scale masks to [0, 1]
    imgs_mask_train=np.where(imgs_mask_train>0,1,0)
    imgs_mask2_train = imgs_mask2_train.astype('float32')
    #imgs_mask2_train /= 255.  # scale masks to [0, 1]
    imgs_mask2_train=np.where(imgs_mask2_train>0,1,0)
    
    imgs_val = imgs_val.astype('float32')
    imgs_val -= mean
    imgs_val /= std

    imgs_mask_val = imgs_mask_val.astype('float32')
    #imgs_mask_val /= 255.  # scale masks to [0, 1]
    imgs_mask_val=np.where(imgs_mask_val>0,1,0)
    imgs_mask2_val = imgs_mask2_val.astype('float32')
    #imgs_mask2_val /= 255.  # scale masks to [0, 1]
    imgs_mask2_val=np.where(imgs_mask2_val>0,1,0)
    
    print(mean, std)
    print(np.mean(imgs_train), np.std(imgs_train))

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)
    model = get_aspp_unet()
    
    # Saving weights to unet.hdf5 at checkpoints
    model_checkpoint = ModelCheckpoint(model_path, monitor='val_cup_dice_coef2', verbose=1, save_best_only=True, mode='max')
    # Stop training when a monitored quantity has stopped improving
    early_stopping = EarlyStopping(monitor='val_loss', patience=8, verbose=1)
    # Callback that streams epoch results to a csv file
    csv_logger = CSVLogger(log_path)
    # Reduce learning rate when a metric has stopped improving
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    # Learning rate scheduler
    # change_lr = LearningRateScheduler(scheduler)

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)
    data_gen_args = dict(width_shift_range=0.1,
                         height_shift_range=0.1,
                         # zoom_range=0.15,
                         # rotation_range=90.
                         # horizontal_flip=True)
                         vertical_flip=True)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    mask2_datagen = ImageDataGenerator(**data_gen_args)
    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    image_datagen.fit(imgs_train, augment=True, seed=seed)
    mask_datagen.fit(imgs_mask_train, augment=True, seed=seed)
    mask2_datagen.fit(imgs_mask2_train, augment=True, seed=seed)
    image_generator = image_datagen.flow(imgs_train, batch_size=batchsize, seed=seed)
    mask_generator = mask_datagen.flow(imgs_mask_train, batch_size=batchsize, seed=seed)
    mask2_generator = mask_datagen.flow(imgs_mask2_train, batch_size=batchsize, seed=seed)
    train_generator = izip(image_generator, imap(create_list, mask_generator, mask2_generator))
    model.fit_generator(train_generator,
                        steps_per_epoch=len(imgs_train) / batchsize * 2,
                        nb_epoch=80,
                        verbose=1,
                        validation_data=(imgs_val, create_list(imgs_mask_val, imgs_mask2_val)),
                        validation_steps=int(len(imgs_val)),
                        callbacks=[model_checkpoint, early_stopping, csv_logger, reduce_lr])

    print('-' * 30)
    print('Loading and preprocessing test data...')
    print('-' * 30)
    imgs_test, imgs_mask_test_true, imgs_mask2_test_true, oimgs_mask_test_true, oimgs_mask2_test_true = load_test_data()
    # imgs_test = preprocess(imgs_test)
    # imgs_mask_test_true = preprocess(imgs_mask_test_true)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    imgs_mask_test_true = imgs_mask_test_true.astype('float32')
    #imgs_mask_test_true /= 255.  # scale masks to [0, 1]
    imgs_mask_test_true=np.where(imgs_mask_test_true>0,1,0)
    imgs_mask2_test_true = imgs_mask2_test_true.astype('float32')
    #imgs_mask2_test_true /= 255.  # scale masks to [0, 1]
    imgs_mask2_test_true=np.where(imgs_mask2_test_true>0,1,0)
    
    oimgs_mask_test_true = oimgs_mask_test_true.astype('float32')
    #oimgs_mask_test_true /= 255.  # scale masks to [0, 1]
    oimgs_mask_test_true=np.where(oimgs_mask_test_true>0,1,0)
    oimgs_mask2_test_true = oimgs_mask2_test_true.astype('float32')
    #oimgs_mask2_test_true /= 255.  # scale masks to [0, 1]
    oimgs_mask2_test_true=np.where(oimgs_mask2_test_true>0,1,0)
    
    print(np.mean(imgs_test), np.std(imgs_test))

    print('-' * 30)
    print('Loading saved weights...')
    print('-' * 30)
    model.load_weights(model_path)

    print('-' * 30)
    print('Predicting masks on test data...')
    print('-' * 30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save(working_path + 'test_result/imgs_mask_testp3_da.npy', imgs_mask_test)
    mean = 0.0
    mean2 = 0.0
    omean = 0.0
    omean2 = 0.0
    num_test = len(imgs_test)
    for i in range(num_test):
        current = np.where(imgs_mask_test[8][i, :, :, 0] > 0.5, 1, 0)
        mean += dice_coef_np(imgs_mask_test_true[i, :, :, 0], current)
        current2 = np.where(imgs_mask_test[9][i, :, :, 0] > 0.5, 1, 0)
        mean2 += dice_coef_np(imgs_mask2_test_true[i, :, :, 0], current2)

        current = current.astype('uint8')*255
        ocurrent = cv2.linearPolar(current, (current.shape[0] / 2, current.shape[1] / 2), current.shape[0] / 2,
                                 cv2.WARP_INVERSE_MAP)/255
        # plt.imshow(ocurrent)
        current2 = current2.astype('uint8') * 255
        ocurrent2 = cv2.linearPolar(current2, (current.shape[0] / 2, current.shape[1] / 2), current.shape[0] / 2,
                                 cv2.WARP_INVERSE_MAP)/255
        # plt.imshow(ocurrent2)
        omean += dice_coef_np(oimgs_mask_test_true[i, :, :, 0], ocurrent)
        omean2 += dice_coef_np(oimgs_mask2_test_true[i, :, :, 0], ocurrent2)
    mean /= num_test
    mean2 /= num_test
    omean /= num_test
    omean2 /= num_test
    print("Mean Dice Coeff : ", mean, mean2, omean, omean2)
def mkdir(dir_):
  if not os.path.exists(dir_):
    os.makedirs(dir_)
def evaluate():
    print('lr=%f, dr=%f, l2=%f, BS=%d'%(learning_rate, dropout_rate, l2factor, batchsize))
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)
    imgs_train, imgs_mask_train, imgs_mask2_train = load_train_data()

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)
    #model = get_atrous_unet()
    #model = get_unet()
    model =  get_aspp_unet()
    model.summary()
    print('-' * 30)
    print('Loading and preprocessing test data...')
    print('-' * 30)
    imgs_test, imgs_mask_test_true, imgs_mask2_test_true, oimgs_mask_test_true, oimgs_mask2_test_true = load_test_data()

    imgs_test_orig=load_orig_test_img()
    # imgs_test = preprocess(imgs_test)
    # imgs_mask_test_true = preprocess(imgs_mask_test_true)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    imgs_mask_test_true = imgs_mask_test_true.astype('float32')
    #imgs_mask_test_true /= 255.  # scale masks to [0, 1]
    imgs_mask_test_true=np.where(imgs_mask_test_true>0,1,0)
    imgs_mask2_test_true = imgs_mask2_test_true.astype('float32')
    #imgs_mask2_test_true /= 255.  # scale masks to [0, 1]
    imgs_mask2_test_true=np.where(imgs_mask2_test_true>0,1,0)

    oimgs_mask_test_true = oimgs_mask_test_true.astype('float32')
    #oimgs_mask_test_true /= 255.  # scale masks to [0, 1]
    oimgs_mask_test_true=np.where(oimgs_mask_test_true>0,1,0)
    oimgs_mask2_test_true = oimgs_mask2_test_true.astype('float32')
    #oimgs_mask2_test_true /= 255.  # scale masks to [0, 1]
    oimgs_mask2_test_true=np.where(oimgs_mask2_test_true>0,1,0)
    print(np.mean(imgs_test), np.std(imgs_test))

    print('-' * 30)
    print('Loading saved weights...')
    print('-' * 30)
    model.load_weights(model_path)

    print('-' * 30)
    print('Predicting masks on test data...')
    print('-' * 30)
    imgs_mask_test = model.predict(imgs_test, verbose=1,batch_size=8)
    np.save(working_path + 'test_result/imgs_mask_testp3_da.npy', imgs_mask_test)
    mean = 0.0
    mean2 = 0.0
    omean = 0.0
    omean2 = 0.0
    iou_mean_disc=0.0
    iou_mean_cup=0.0
    num_test = len(imgs_test)
    for i in range(num_test):
        current = np.where(imgs_mask_test[8][i, :, :, 0] > 0.5, 1, 0)
        
        mean += dice_coef_np(imgs_mask_test_true[i, :, :, 0], current)
        current2 = np.where(imgs_mask_test[9][i, :, :, 0] > 0.5, 1, 0)
        #current2=label(current2,connectivity=1)
        current2=remove_small_objects(current2>0,min_size=1000,connectivity=1)
        mean2 += dice_coef_np(imgs_mask2_test_true[i, :, :, 0], current2)

        current = current.astype('uint8')*255
        ocurrent = cv2.linearPolar(current, (current.shape[0] / 2, current.shape[1] / 2), current.shape[0] / 2,
                                 cv2.WARP_INVERSE_MAP)/255
        
        current2 = current2.astype('uint8') * 255
        ocurrent2 = cv2.linearPolar(current2, (current.shape[0] / 2, current.shape[1] / 2), current.shape[0] / 2,
                                 cv2.WARP_INVERSE_MAP)/255
        #if i==6:imsave("cup06_before_removed3.png",ocurrent2*255)
        #ocurrent2=label(ocurrent2,connectivity=1)
        ocurrent2=remove_small_objects(ocurrent2>0,min_size=1000,connectivity=1)
        #if i==6:imsave("cup06_after_removed3.png",ocurrent2*255)
        ocurrent2=draw_contours_fitellipse_cv2(imgs_test_orig[i],np.array(oimgs_mask2_test_true[i, :, :, 0]).astype('uint8'),\
                                                ocurrent2.astype('uint8'),i,"aspp_cups1000_fit2")
        #draw_contours_cv2(imgs_test_orig[i],np.array(oimgs_mask_test_true[i, :, :, 0]).astype('uint8'),ocurrent.astype('uint8'),i,"aspp_discs1000_2")

        #if i==26:
          #imsave("disc26.png",oimgs_mask_test_true[i, :, :, 0])
        omean += dice_coef_np(oimgs_mask_test_true[i, :, :, 0], ocurrent)
        omean2 += dice_coef_np(oimgs_mask2_test_true[i, :, :, 0], ocurrent2)
        iou_mean_disc+=iou_coef_np(oimgs_mask_test_true[i, :, :, 0], ocurrent)
        iou_mean_cup+=iou_coef_np(oimgs_mask2_test_true[i, :, :, 0], ocurrent2)
    mean /= num_test
    mean2 /= num_test
    omean /= num_test
    omean2 /= num_test
    iou_mean_disc/=num_test
    iou_mean_cup/=num_test
    print("Mean Dice Coeff : ", mean, mean2, omean, omean2,iou_mean_disc,iou_mean_cup)
def get_disc_mask_by_mnet():
    print('lr=%f, dr=%f, l2=%f, BS=%d'%(learning_rate, dropout_rate, l2factor, batchsize))
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)
    imgs_train, imgs_mask_train, imgs_mask2_train = load_train_data_for_mean_std()

    
    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)
    #model = get_atrous_unet()
    model = get_unet()
    #model =  get_aspp_unet()
    model.summary()
    print('-' * 30)
    print('Loading and preprocessing test data...')
    print('-' * 30)
    imgs_test_, imgs_mask_test_true, oimgs_mask_test_true = load_test_data_only_cup()
    #imgs_test_orig=load_orig_test_img()
    #imgs_test_, imgs_mask_test_true = load_val_data_only_cup()

    imgs_test = imgs_test_.astype('float32')
    print("before max img_test",np.max(imgs_test))
    imgs_test -= mean
    imgs_test /= std
    print("after max img_test",np.max(imgs_test))


    model.load_weights(model_path)


    imgs_mask_test = model.predict(imgs_test, verbose=1)

    
    imgs_mask_test_o=np.zeros((len(imgs_mask_test[8]),img_rows,img_cols,1))
    imgs_mask_test_p=np.zeros((len(imgs_mask_test[8]),img_rows,img_cols,1))
    for i in range(len(imgs_mask_test[8])):

      
      current = np.where(imgs_mask_test[8][i, :, :, 0] > 0.5, 1, 0)


      current = current.astype('uint8')*255
      ocurrent = cv2.linearPolar(current, (current.shape[0] / 2, current.shape[1] / 2), current.shape[0] / 2,
                                       cv2.WARP_INVERSE_MAP)/255
      ocurrent = np.where(ocurrent != 0, 255, 0)
      
      imgs_mask_test_o[i]=np.expand_dims(ocurrent,axis=-1)
      imgs_mask_test_p[i]=np.expand_dims(current,axis=-1)
      if i<=-1:
        
        print(ocurrent.shape)
        print("max current",np.max(current))
        print("max ocurrent",np.max(ocurrent))
        imsave("mount_out/mask"+str(i)+".png",ocurrent)
        imsave("mount_out/pmask"+str(i)+".png",current)
        imsave("mount_out/pimg"+str(i)+".png",imgs_test_[i]/255.)
        imsave("mount_out/img"+str(i)+".png",imgs_test_orig[i])
        draw_contours(imgs_test_orig[i]/255.,np.array(ocurrent)/255.,i,"contour")
        draw_contours(imgs_test_[i]/255.,np.array(current)/255.,i,"contour_p")
    np.save(working_path + 'yetestMasks_disc_get_by_mnet.npy', imgs_mask_test_o) 
    np.save(working_path + 'yeptestMasks_disc_get_by_mnet.npy', imgs_mask_test_p) 
    print(working_path + 'yetestMasks_disc_get_by_mnet.npy Saved!')
    print(working_path + 'yeptestMasks_disc_get_by_mnet.npy Saved!')
def draw_contours_np(img,pred_mask,f,save_dir):

  out=np.zeros(img.shape)
  for i in range(pred_mask.shape[0]):
    for j in range(pred_mask.shape[1]):
      if pred_mask[i,j]!=0:
        out[i,j]=0.5*pred_mask[i,j]+0.5*img[i,j]
      else:
        out[i,j]=img[i,j]
        
  imsave("mount_out/"+save_dir+"_"+str(f)+".png",out)
def draw_contours_cv2(img,true_mask,pred_mask,f,save_dir):
  true_mask_contours,cnts,_=cv2.findContours(true_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  for i in range(len(cnts)):
    cv2.drawContours(img,[cnts[i]],-1,(0,0,255),1)
  
  pred_mask_contours,cnts,_=cv2.findContours(pred_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  for i in range(len(cnts)):
    cv2.drawContours(img,[cnts[i]],-1,(255,0,0),1) 
  mkdir("test_result/"+save_dir+"/")
  imsave("test_result/"+save_dir+"/"+save_dir+"_"+str(f)+".png",img)
def draw_contours_fitellipse_cv2(img,true_mask,pred_mask,f,save_dir):
  true_mask_contours,cnts,_=cv2.findContours(true_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  #new_true_mask=np.zeros(img.shape[:2])
  new_pred_mask=np.zeros(img.shape[:2])
  for i in range(len(cnts)):
    #if len(cnts[i])<100:continue
    pnt=cv2.fitEllipse(cnts[i])
    cv2.ellipse(img,pnt,(0,0,255),1)
    #cv2.ellipse(new_true_mask,pnt,1,-1)
    
  
  pred_mask_contours,cnts,_=cv2.findContours(pred_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  for i in range(len(cnts)):
    if len(cnts[i])<20:continue
    pnt=cv2.fitEllipse(cnts[i])
    cv2.ellipse(img,pnt,(255,0,0),1) 
    cv2.ellipse(new_pred_mask,pnt,1,-1)
  
  mkdir("test_result/"+save_dir+"/")
  imsave("test_result/"+save_dir+"/"+save_dir+"_"+str(f)+".png",img)  
  return new_pred_mask
if __name__ == '__main__':
    #train_and_predict()
    evaluate()
    #get_disc_mask_by_mnet()
