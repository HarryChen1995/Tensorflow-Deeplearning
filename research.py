import tensorflow as tf 
import numpy as np 
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

iris = datasets.load_iris()
x_vals = np.array([x[0:4] for x in iris.data])
#x_vals=np.concatenate((x_vals, np.array([x[0:1]+x[1:2] for x in iris.data])), axis=1)
y_vals = np.array([x for x in iris.target])
sess = tf.Session()


seed = 2
tf.set_random_seed(seed)
np.random.seed(seed)


x_vals_train, x_vals_test, y_vals_train, y_vals_test = train_test_split(x_vals, y_vals, test_size=0.33)


scaler.fit(x_vals_train)
x_vals_train=scaler.transform(x_vals_train)
scaler.fit(x_vals_test)
x_vals_test=scaler.transform(x_vals_test)


batch_size = 50
x_data = tf.placeholder(shape=[None, 4], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)



w1=tf.get_variable("W1",[4,5],initializer=tf.random_normal_initializer())
b1=tf.get_variable("b1",[5],initializer=tf.random_normal_initializer())

layer1=tf.nn.relu(tf.add(tf.matmul(x_data,w1),b1))

w2=tf.get_variable("W2",[5,5],initializer=tf.random_normal_initializer())
b2=tf.get_variable("b2",[5],initializer=tf.random_normal_initializer())

layer2=tf.nn.relu(tf.add(tf.matmul(layer1,w2),b2))

w3=tf.get_variable("W3",[5,5],initializer=tf.random_normal_initializer())
b3=tf.get_variable("b3",[5],initializer=tf.random_normal_initializer())

layer3=tf.nn.relu(tf.add(tf.matmul(layer2,w3),b3))


w_output=tf.get_variable("w_ouput",[5,1],initializer=tf.random_normal_initializer())
b_output=tf.get_variable("b_ouput",[1],initializer=tf.random_normal_initializer())



final_output=tf.add(tf.matmul(layer3,w_output),b_output)

loss = tf.reduce_mean(tf.square(y_target - final_output))

my_opt = tf.train.GradientDescentOptimizer(0.001)
train_step = my_opt.minimize(loss)
init = tf.global_variables_initializer()
sess.run(init)


loss_vec = []
test_loss = []
for i in range(10000):
	rand_index = np.random.choice(len(x_vals_train), size=batch_size)
	rand_x = x_vals_train[rand_index]
	rand_y = np.transpose([y_vals_train[rand_index]])
	sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
	temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
	loss_vec.append(np.sqrt(temp_loss))
	test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
	test_loss.append(np.sqrt(test_temp_loss))
	if (i+1)%50==0:
		print('Generation: ' + str(i+1) + '. Loss = ' + str(temp_loss))

plt.plot(loss_vec, 'r-', label='Train Loss')
plt.plot(test_loss, 'b--', label='Test Loss')
plt.title('Least Square Loss')
plt.xlabel('epchos')
plt.ylabel('Error Loss')
plt.legend(loc='upper right')
plt.show()