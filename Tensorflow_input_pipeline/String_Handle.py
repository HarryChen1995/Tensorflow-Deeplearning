import tensorflow as tf 
import numpy as np 

Train_Data=tf.data.Dataset.from_tensor_slices(np.arange(100)).batch(10)
Test_Data=tf.data.Dataset.from_tensor_slices(np.arange(199,300)).batch(10)

handle=tf.placeholder(tf.string, shape=[])
iterator=tf.data.Iterator.from_string_handle(handle,Train_Data.output_types,Train_Data.output_shapes)

a=tf.placeholder(tf.float32)
b=tf.placeholder(tf.float32)
c=tf.add(a,b)
next_element=iterator.get_next()


train_iterator=Train_Data.make_one_shot_iterator()
test_iterator=Test_Data.make_one_shot_iterator()

with tf.Session() as sess:

	for i in range(10):
		var=sess.run([next_element], feed_dict={handle:sess.run(train_iterator.string_handle())})
		
		print(sess.run(c,feed_dict={a:var,b:var}))


	for i in range(10):
		var=sess.run([next_element], feed_dict={handle:sess.run(test_iterator.string_handle())})
		
		print(sess.run(c,feed_dict={a:var,b:var}))


	

