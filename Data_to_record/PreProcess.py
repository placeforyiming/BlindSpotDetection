

import tensorflow as tf

import numpy as np



IMAGE_SHAPE=(64,128,3)

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def change_image_to_string(img):
    image=img.astype(np.float32)
    return image.reshape(IMAGE_SHAPE).flatten().tostring()


def write_record(dest_path,df):
    writer=tf.python_io.TFRecordWriter(dest_path)

    for i in range(len(df['data'])):
        #Change format to string
        img = change_image_to_string(df['data'][i])
        example=tf.train.Example(features=tf.train.Features(feature=
                                                                {
                                                                    'example':bytes_feature(img),
                                                                    'label':int64_feature(df['labels'][i])
                                                                }
                                                                ))
        writer.write(example.SerializeToString())
    writer.close()

