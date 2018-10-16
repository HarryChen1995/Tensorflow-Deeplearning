import tensorflow as tf 

reader=tf.TFRecordReader()


filename_queue = tf.train.string_input_producer(
   ["output.tfrecord"])

_, serialized_example = reader.read(filename_queue)

feature_set = { 'A': tf.FixedLenFeature([3], tf.string),
               'C': tf.FixedLenFeature([3], tf.float32)
           }

features = tf.parse_single_example( serialized_example, features= feature_set )
A=features['A']
C=features['C']


##### inspect tfrecord values ####
for example in tf.python_io.tf_record_iterator('output.tfrecord'):
	result=tf.train.Example.FromString(example)
	print(result.features.feature['A'].bytes_list.value)
	print(result.features.feature['C'].float_list.value)

with tf.Session() as sess:

    # for the queues
    init_op = tf.local_variables_initializer()
    sess.run(init_op)

    # Create a coordinator, launch the queue runner threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
  
    print(sess.run([A,C]))

  
   
    coord.request_stop()
    coord.join(threads)
