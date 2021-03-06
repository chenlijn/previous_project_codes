# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Converts PASCAL VOC 2012 data to TFRecord file format with Example protos.

PASCAL VOC 2012 dataset is expected to have the following directory structure:

  + pascal_voc_seg
    - build_data.py
    - build_voc2012_data.py (current working directory).
    + VOCdevkit
      + VOC2012
        + JPEGImages
        + SegmentationClass
        + ImageSets
          + Segmentation
    + tfrecord

Image folder:
  ./VOCdevkit/VOC2012/JPEGImages

Semantic segmentation annotations:
  ./VOCdevkit/VOC2012/SegmentationClass

list folder:
  ./VOCdevkit/VOC2012/ImageSets/Segmentation

This script converts data into sharded data files and save at tfrecord folder.

The Example proto contains the following fields:

  image/encoded: encoded image content.
  image/filename: image filename.
  image/format: image file format.
  image/height: image height.
  image/width: image width.
  image/channels: image channels.
  image/segmentation/class/encoded: encoded semantic segmentation content.
  image/segmentation/class/format: semantic segmentation file format.
"""
import math
import os.path
import sys
import build_data
import tensorflow as tf
import numpy as np
import cv2

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('image_folder',
                           './VOCdevkit/VOC2012/JPEGImages',
                           'Folder containing images.')

tf.app.flags.DEFINE_string(
    'semantic_segmentation_folder',
    './VOCdevkit/VOC2012/SegmentationClassRaw',
    'Folder containing semantic segmentation annotations.')

tf.app.flags.DEFINE_string(
    'list_folder',
    './VOCdevkit/VOC2012/ImageSets/Segmentation',
    'Folder containing lists for training and validation')

tf.app.flags.DEFINE_string(
    'output_dir',
    './tfrecord',
    'Path to save converted SSTable of TensorFlow examples.')


_NUM_SHARDS = 4


def _convert_dataset(dataset_split):
  """Converts the specified dataset split to TFRecord format.

  Args:
    dataset_split: The dataset split (e.g., train, test).

  Raises:
    RuntimeError: If loaded image and label have different shape.
  """
  dataset = os.path.basename(dataset_split)[:-4]
  print(dataset)
  sys.stdout.write('Processing ' + dataset)
  filenames = [x.strip('\n') for x in open(dataset_split, 'r')]
  num_images = len(filenames)
  num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

  image_reader_png = build_data.ImageReader('png', channels=3)
  image_reader_jpg = build_data.ImageReader('jpg', channels=3)
  label_reader = build_data.ImageReader('png', channels=1)
  #mean_img = np.load('glaucoma_mean.npy')

  for shard_id in range(_NUM_SHARDS):
    output_filename = os.path.join(
        FLAGS.output_dir,
        '%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, _NUM_SHARDS))
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images)
      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
            i + 1, len(filenames), shard_id))
        sys.stdout.flush()
        # Read the image.
        image_filename = os.path.join(
            FLAGS.image_folder, filenames[i])
        image_data = tf.gfile.FastGFile(image_filename, 'rb').read() 
        if filenames[i].split('.')[-1] == 'png':
            decoded_image = image_reader_png.decode_image(image_data)
            height, width = image_reader_png.read_image_dims(image_data)
        elif filenames[i].split('.')[-1] == 'jpg':
            decoded_image = image_reader_jpg.decode_image(image_data)
            height, width = image_reader_jpg.read_image_dims(image_data)

        #centered_img = decoded_image - mean_img
        ##centered_img_str = centered_img.tostring()
        #centered_img_str = centered_img.flatten()

        # image CLAHE 
        b_img = decoded_image[:,:,0]
        g_img = decoded_image[:,:,1]
        r_img = decoded_image[:,:,2]

        #dclahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(10,10))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        #clahe = cv2.createCLAHE(tileGridSize=(10,10))
        res_b_img = clahe.apply(b_img)
        res_g_img = clahe.apply(g_img)
        res_r_img = clahe.apply(r_img)
        norm_img = np.zeros_like(decoded_image)
        norm_img[:,:,0] = res_b_img[:,:]
        norm_img[:,:,1] = res_g_img[:,:]
        norm_img[:,:,2] = res_r_img[:,:]

        norm_img_str = norm_img.flatten()
        #norm_img_str = decoded_image.flatten()


        # Read the semantic segmentation annotation.
        seg_filename = os.path.join(
            FLAGS.semantic_segmentation_folder,
            filenames[i].split('.')[0] + '.' + FLAGS.label_format)
        seg_data = tf.gfile.FastGFile(seg_filename, 'rb').read()
        seg_height, seg_width = label_reader.read_image_dims(seg_data)
        if height != seg_height or width != seg_width:
          raise RuntimeError('Shape mismatched between image and label.')
        # Convert to tf example.
        example = build_data.image_seg_to_tfexample(
            norm_img_str, filenames[i], height, width, seg_data)
        tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()


def main(unused_argv):
  dataset_splits = tf.gfile.Glob(os.path.join(FLAGS.list_folder, '*.txt'))
  for dataset_split in dataset_splits:
    _convert_dataset(dataset_split)


if __name__ == '__main__':
  tf.app.run()
