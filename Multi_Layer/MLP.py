# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
# Parameters
learning_rate = 0.001
training_epochs = 20
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 256 # 2nd layer num features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
# tf Graph input
x = tf.placeholder("float", [None, n_input], name="x")
y = tf.placeholder("float", [None, n_classes], name="y")
#weights layer 1
h = tf.Variable(tf.random_normal([n_input, n_hidden_1]), name="layer_weight_1")
#bias layer 1
bias_layer_1 = tf.Variable(tf.random_normal([n_hidden_1]), name="bias_1")
#layer 1
layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,h),bias_layer_1), name="activation_1")
#weights layer 2
w = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name="layer_weight_2")
#bias layer 2
bias_layer_2 = tf.Variable(tf.random_normal([n_hidden_2]), name="biase_2")
#layer 2
layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,w),bias_layer_2), name="activation_2")
#weights output layer
output = tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name="layer_weight_3")
#biar output layer
bias_output = tf.Variable(tf.random_normal([n_classes]),name="biase_3")
#output layer
output_layer = tf.add(tf.matmul(layer_2, output),bias_output, name="Ouput")
# cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#Plot settings
avg_set = []
epoch_set=[]
# Initializing the variables
init = tf.global_variables_initializer()
# Launch the graph


correct_prediction = tf.equal(tf.argmax(output_layer, 1),tf.argmax(y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
tf.summary.scalar("Accuracy",accuracy)
sum_op=tf.summary.merge_all()
with tf.Session() as sess:
	sess.run(init)
	writer=tf.summary.FileWriter("output",sess.graph)
# Training cycle
	for epoch in range(training_epochs):
		avg_cost = 0.
		total_batch = int(mnist.train.num_examples/batch_size)
		# Loop over all batches
		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
			# Compute average loss
			avg_cost += sess.run(cost, feed_dict={x: batch_xs,y: batch_ys})/total_batch
		# Display logs per epoch step
		if epoch % display_step == 0:
			print ("Epoch:", '%04d' % (epoch+1),"cost=", "{:.9f}".format(avg_cost))
			avg_set.append(avg_cost)
			epoch_set.append(epoch+1)
			_,Accuracy=sess.run([accuracy, sum_op], feed_dict={x: mnist.test.images,y: mnist.test.labels})
			writer.add_summary(Accuracy,epoch)
	writer.close()
	print ("Training phase finished")
	plt.plot(epoch_set,avg_set, 'o', label='MLP Training phase')
	plt.ylabel('cost')
	plt.xlabel('epoch')
	plt.legend()
	plt.show()
# Test model

	print ("Model Accuracy:", accuracy.eval({x: mnist.test.images,y: mnist.test.labels}))
# Fit training using batch data
	img, label = mnist.test.next_batch(1)
	plt.imshow(np.reshape(img,[28,28]))
	plt.show()
	print ("the Prediction digits:",sess.run(tf.argmax(output_layer, 1), feed_dict={x:img}))