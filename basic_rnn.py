import numpy as np 
import tensorflow as tf


n_inputs = 3
n_neurons = 5
X1 = tf.placeholder(tf.float32, [None, n_inputs])
X2 = tf.placeholder(tf.float32, [None, n_inputs])
Wx = tf.get_variable("Wx", shape=[n_inputs,n_neurons],dtype=tf.float32, initializer=None, regularizer=None,trainable=True, collections=None)
Wy = tf.get_variable("Wy", shape=[n_neurons,n_neurons], dtype=tf.float32,initializer=None, regularizer=None, trainable=True,collections=None)

b = tf.get_variable("b", shape=[1,n_neurons],dtype=tf.float32, initializer=None, regularizer=None,trainable=True, collections=None)


Y1 = tf.nn.relu(tf.matmul(X1, Wx) + b)
Y2 = tf.nn.relu(tf.matmul(Y1, Wy) + tf.matmul(X2, Wx)+ b)



init_op = tf.global_variables_initializer()
X1_batch = np.array([[0, 2, 3], [2, 8, 9], [5, 3, 8],
[3, 2, 9]]) # t = 0
X2_batch = np.array([[5, 6, 8], [1, 0, 0], [8, 2, 0],
[2, 3, 6]]) # t = 1



with tf.Session() as sess:
	init_op.run()
	Y1_val, Y2_val = sess.run([Y1, Y2], feed_dict={X1: X1_batch, X2: X2_batch})

	print(Y1_val) # output at t = 0
	print(Y2_val) # output at t = 1






#basic_cell =tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
#output_seqs, states =tf.contrib.rnn.static_rnn(basic_cell, [X1, X2],dtype=tf.float32)
#Y1, Y2 = output_seqs
#init_op = tf.global_variables_initializer()
#X1_batch = np.array([[0, 2, 3], [2, 8, 9], [5, 3, 8],
#[3, 2, 9]]) # t = 0
#X2_batch = np.array([[5, 6, 8], [1, 0, 0], [8, 2, 0],
#[2, 3, 6]]) # t = 1
#with tf.Session() as sess:
#	init_op.run()
#	Y1_val, Y2_val = sess.run([Y1, Y2], feed_dict={X1:
#	X1_batch, X2: X2_batch})
#	print(Y1_val) # output at t = 0
#	print(Y2_val) # output at t = 1





n_inputs = 3
n_neurons = 5
n_steps = 2
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
seq_length = tf.placeholder(tf.int32, [None])
basic_cell =tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
output_seqs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
X_batch = np.array([
[[0, 2, 3], [2, 8, 9]], # instance 0
[[5, 6, 8], [0, 0, 0]], # instance 1 (padded with a zero vector)
[[6, 7, 8], [6, 5, 4]], # instance 2
[[8, 2, 0], [2, 3, 6]], # instance 3
])


seq_length_batch = np.array([3, 4, 3, 5])

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
	init_op.run()
	outputs_val, states_val =sess.run([output_seqs, states], feed_dict={X: X_batch, seq_length: seq_length_batch})
	print(outputs_val)

