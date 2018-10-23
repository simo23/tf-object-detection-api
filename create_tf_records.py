import os
import io
import glob
import hashlib
import pandas as pd
import xml.etree.ElementTree as ET
import tensorflow as tf
import random
from PIL import Image
from object_detection.utils import dataset_util

# Define the class names and their weight
class_names = ['class_1', 'class_2', ...]
class_weights = [1.0, 1.0, ...]

def create_example(xml_file):

        tree = ET.parse(xml_file)
        root = tree.getroot()
        image_name = root.find('filename').text
        image_path = root.find('path').text
        file_name = image_name.encode('utf8')
        size=root.find('size')
	# IMPORTANT: assumes "size" to be defined as [width, height] and not [height, width]
        width = int(size[0].text)
        height = int(size[1].text)
        xmin = []
        ymin = []
        xmax = []
        ymax = []
        classes = []
        classes_text = []
        truncated = []
        poses = []
        difficult_obj = []
        weights = [] # Important line

        for member in root.findall('object'):

	   # IMPORTANT: assumes "bbox" to be defined as [xmin, ymin, xmax, ymax]
           xmin.append(float(member[4][0].text) / width)
           ymin.append(float(member[4][1].text) / height)
           xmax.append(float(member[4][2].text) / width)
           ymax.append(float(member[4][3].text) / height)
           difficult_obj.append(0)

           class_name = member[0].text
           class_id = class_names.index(class_name)
           weights.append(class_weights[class_id])

           if class_name == 'class_1':
               classes_text.append('class_1'.encode('utf8'))
               classes.append(1)
           elif class_name == 'class_2':
               classes_text.append('class_2'.encode('utf8'))
               classes.append(2)
           else:
               print('E: class not recognized!')

           truncated.append(0)
           poses.append('Unspecified'.encode('utf8'))

        full_path = image_path 
        with tf.gfile.GFile(full_path, 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        if image.format != 'JPEG':
           raise ValueError('Image format not JPEG')
        key = hashlib.sha256(encoded_jpg).hexdigest()

        #create TFRecord Example
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(file_name),
            'image/source_id': dataset_util.bytes_feature(file_name),
            'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
            'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
            'image/object/truncated': dataset_util.int64_list_feature(truncated),
            'image/object/view': dataset_util.bytes_list_feature(poses),
            'image/object/weight': dataset_util.float_list_feature(weights)
        })) 
        return example  

def main(_):

    weighted_tf_records_output = 'name_of_records_file.record' # output file
    annotations_path = '/path/to/annotations/folder/*.xml' # input annotations

    writer_train = tf.python_io.TFRecordWriter(weighted_tf_records_output)
    filename_list=tf.train.match_filenames_once(annotations_path)
    init = (tf.global_variables_initializer(), tf.local_variables_initializer())
    sess=tf.Session()
    sess.run(init)
    list = sess.run(filename_list)
    random.shuffle(list)  

    for xml_file in list:
      print('-> Processing {}'.format(xml_file))
      example = create_example(xml_file)
      writer_train.write(example.SerializeToString())

    writer_train.close()
    print('-> Successfully converted dataset to TFRecord.')


if __name__ == '__main__':
    tf.app.run()
