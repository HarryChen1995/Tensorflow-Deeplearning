import tensorflow as tf 
import numpy as np 

writer=tf.python_io.TFRecordWriter('output.tfrecord')

A=[b'Harry',b'Chen',b'Lee']
C=[1.1, 2.1, 3.1]


feature_A=tf.train.Feature(bytes_list=tf.train.BytesList(value=A))
feature_C=tf.train.Feature(float_list=tf.train.FloatList(value=C))

features={'A':feature_A, 'C':feature_C}

example=tf.train.Example(features=tf.train.Features(feature=features))



writer.write(example.SerializeToString())
writer.close()