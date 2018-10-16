import tensorflow as tf 
import numpy as np





x = np.arange(0, 10)
y = np.arange(1, 11)
# create dataset objects from the arrays
dx = tf.data.Dataset.from_tensor_slices(x)
dy = tf.data.Dataset.from_tensor_slices(y)

a=tf.placeholder(tf.float32,[None])
b=tf.placeholder(tf.float32,[None])
c=tf.add(a,b)
# zip the two datasets together
dcomb = tf.data.Dataset.zip((dx, dy)).batch(2)
iterator = dcomb.make_initializable_iterator()
# extract an element
next_element = iterator.get_next()
with tf.Session() as sess:
    sess.run(iterator.initializer)
    for i in range(15):
        val = sess.run(next_element)
        val1=sess.run([c], feed_dict={a:val[0],b:val[1]})
        print (val)
        print(val1)
        if (i+1) % (10//2) == 0 and i > 0:
            sess.run(iterator.initializer)