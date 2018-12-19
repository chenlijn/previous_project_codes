
import os
import numpy as np
import cv2
import tensorflow as tf

slim = tf.contrib.slim
tfexample_decoder = slim.tfexample_decoder


# Specify how the TF-Examples are decoded.
keys_to_features = {
   #'image/encoded': tf.FixedLenFeature(
   #    (), tf.string, default_value=''),
   'image/tensor': tf.FixedLenSequenceFeature(
           (), tf.float32, default_value=0, allow_missing=True),
   'image/filename': tf.FixedLenFeature(
           (), tf.string, default_value=''),
   #'image/shape': tf.FixedLenFeature(
   #    (), tf.int64, default_value=0),
   #'image/format': tf.FixedLenFeature(
   #    (), tf.string, default_value='jpeg'),
   #'image/format': tf.FixedLenFeature(
   #    (), tf.string, default_value='raw'),
   'image/height': tf.FixedLenFeature(
           (), tf.int64, default_value=0),
   'image/width': tf.FixedLenFeature(
           (), tf.int64, default_value=0),
   'image/channels': tf.FixedLenFeature(
           (), tf.int64, default_value=0),
   'image/segmentation/class/encoded': tf.FixedLenFeature(
           (), tf.string, default_value=''),
   'image/segmentation/class/format': tf.FixedLenFeature(
           (), tf.string, default_value='png'),
}

items_to_handlers = {
 #'image': tfexample_decoder.Image(
 #    image_key='image/encoded',
 #    format_key='image/format',
 #    channels=3),
 'image': tfexample_decoder.Tensor('image/tensor', shape_keys=['image/height', 'image/width', 'image/channels']),
 #'image': tfexample_decoder.Tensor('image/encoded', shape_keys='image/shape'),
 'image_name': tfexample_decoder.Tensor('image/filename'),
 #'shape': tfexample_decoder.Tensor('image/shape'),
       'height': tfexample_decoder.Tensor('image/height'),
       'width': tfexample_decoder.Tensor('image/width'),
       'channels': tfexample_decoder.Tensor('image/channels'),
       'labels_class': tfexample_decoder.Image(
  image_key='image/segmentation/class/encoded',
  format_key='image/segmentation/class/format',
  channels=1),
}

decoder = tfexample_decoder.TFExampleDecoder(
       keys_to_features, items_to_handlers)



print "hello"

tfr_file ='/root/host_share/glaucoma/deeplab/tfrecord2/train-00002-of-00004.tfrecord'
reader = tf.python_io.tf_record_iterator(tfr_file) 
#those_examples = [tf.train.Example().FromString(example_str) for example_str in reader]
those_examples = [example_str for example_str in reader]

print type(those_examples)
one_example = those_examples[0]
print type(one_example)

#feature_name = 'image/width'
#bytes_list = one_example.features.feature[feature_name]#.int64_list.value
#print type(bytes_list)
#print bytes_list
#print type(features)
out = decoder.decode(one_example)
#print "decoded: "

with tf.Session() as sess:
    out_ = sess.run(out)
    print type(out_)
    image = out_[0]
    height = out_[2]
    width = out_[3]
    chanls = out_[4]
    print height, width, chanls
    print image.shape
    print image.dtype
    print type(image)
    rows, clos, _ = np.where(image>=0.5)
    print image.max(), image.min()
    print len(rows)
    save_img = np.zeros_like(image)
    save_img[:,:,0] = image[:,:,2]
    save_img[:,:,1] = image[:,:,1]
    save_img[:,:,2] = image[:,:,0]
    cv2.imwrite('decoded_img.png', save_img)
