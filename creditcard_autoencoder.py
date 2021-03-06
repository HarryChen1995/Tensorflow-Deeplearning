import pandas as pd
import numpy as np
import tensorflow as tf
import os
from datetime import datetime
from sklearn.metrics import roc_auc_score as auc
import seaborn as sns # for statistical data visualization
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


df = pd.read_csv('creditcard.csv')

#print(df.shape)

#print(df.columns)

#print(df.head())

#print("Total time spanning: {:.1f} days".format(df['Time'].max() / (3600 * 24.0)))

#print("{:.3f} % of all transactions are fraud.".format(np.sum(df['Class']) / df.shape[0] * 100))


#plt.figure(figsize=(12,5*4))
#gs = gridspec.GridSpec(5, 1)
#for i, cn in enumerate(df.columns[:5]):
#	ax = plt.subplot(gs[i])
#	sns.distplot(df[cn][df.Class == 1], bins=50)
#	sns.distplot(df[cn][df.Class == 0], bins=50)
#	ax.set_xlabel('')
#	ax.set_title('histogram of feature: ' + str(cn))
#plt.show()


TEST_RATIO = 0.20
df.sort_values('Time', inplace = True)
TRA_INDEX = int((1-TEST_RATIO) * df.shape[0])
train_x = df.iloc[:TRA_INDEX, 1:-2].values
train_y = df.iloc[:TRA_INDEX, -1].values
test_x = df.iloc[TRA_INDEX:, 1:-2].values
test_y = df.iloc[TRA_INDEX:, -1].values


print("Total train examples: {}, total fraud cases:{}, equal to {:.5f} % of total cases.".format(train_x.shape[0], np.sum(train_y),(np.sum(train_y)/train_x.shape[0])*100))
print("Total test examples: {}, total fraud cases: {},equal to {:.5f} % of total cases.".format(test_x.shape[0], np.sum(test_y),(np.sum(test_y)/test_y.shape[0])*100))


cols_mean = []
cols_std = []
for c in range(train_x.shape[1]):
	cols_mean.append(train_x[:,c].mean())
	cols_std.append(train_x[:,c].std())
	train_x[:, c] = (train_x[:, c] - cols_mean[-1]) /cols_std[-1]
	test_x[:, c] = (test_x[:, c] - cols_mean[-1]) /cols_std[-1]



learning_rate = 0.001
training_epochs = 1000
batch_size = 256
display_step = 10
n_hidden_1 = 15 # number of neurons is the num features
n_input = train_x.shape[1]

X = tf.placeholder("float", [None, n_input])

weights = {
'encoder_h1': tf.Variable(tf.random_normal([n_input,n_hidden_1])),
'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1,n_input])),
}

biases = {
'encoder_b1':tf.Variable(tf.random_normal([n_hidden_1])),
'decoder_b1':tf.Variable(tf.random_normal([n_input])),
}

def encoder(x):
	layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['encoder_h1']),biases['encoder_b1']))
	return layer_1

def decoder(x):
	layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['decoder_h1']),biases['decoder_b1']))
	return layer_1


encoder_op = encoder(X)
decoder_op = decoder(encoder_op)
y_pred = decoder_op
y_true = X

batch_mse = tf.reduce_mean(tf.pow(y_true - y_pred, 2),1)


cost_op = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer =tf.train.RMSPropOptimizer(learning_rate).minimize(cost_op)

epoch_list = []
loss_list = []
train_auc_list = []
data_dir = 'Training_logs/'

save_model = os.path.join(data_dir,
'autoencoder_model.ckpt')
saver = tf.train.Saver()
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
	now = datetime.now()
	sess.run(init_op)
	total_batch = int(train_x.shape[0]/batch_size)
	# Training cycle
	for epoch in range(training_epochs):
		for i in range(total_batch):
			batch_idx =np.random.choice(train_x.shape[0],batch_size)
			batch_xs = train_x[batch_idx]
# Run optimization op (backprop) and
# cost op (to get loss value)
			_, c = sess.run([optimizer, cost_op],feed_dict={X: batch_xs})
		if epoch % display_step == 0:
			train_batch_mse = sess.run(batch_mse,feed_dict={X: train_x})
			epoch_list.append(epoch+1)
			loss_list.append(c)
			train_auc_list.append(auc(train_y,train_batch_mse))
			print("Epoch:", '%04d,' % (epoch+1),"cost=", "{:.9f},".format(c),"Train auc=", "{:.6f},".format(auc(train_y, train_batch_mse)))
	print("Optimization Finished!")
	save_path = saver.save(sess, save_model)
	print("Model saved in: %s" % save_path)



# Plot Training AUC over time
plt.plot(epoch_list, train_auc_list, 'k--',
label='Training AUC', linewidth=1.0)
plt.title('Training AUC per iteration')
plt.xlabel('Iteration')
plt.ylabel('Training AUC')
plt.legend(loc='upper right')
plt.grid(True)
# Plot train loss over time
plt.plot(epoch_list, loss_list, 'r--', label='Trainingloss', linewidth=1.0)
plt.title('Training loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

save_model = os.path.join(data_dir,'autoencoder_model.ckpt')
saver = tf.train.Saver()
# Initializing the variables
init = tf.global_variables_initializer()
with tf.Session() as sess:
	now = datetime.now()
	saver.restore(sess, save_model)
	test_batch_mse = sess.run(batch_mse, feed_dict={X:test_x})
	print("Test auc score: {:.6f}".format(auc(test_y,test_batch_mse)))



	# Zoom into (0, 30) range
	plt.hist(test_batch_mse[(test_y == 0.0) & (test_batch_mse < 30)], bins = 100)
	plt.title("Fraud score (mse) distribution for nonfraud cases")
	plt.xlabel("Fraud score (mse)")
	plt.show()


	# Display only fraud classes
	plt.hist(test_batch_mse[test_y == 1.0], bins =100)
	plt.title("Fraud score (mse) distribution for fraud cases")
	plt.xlabel("Fraud score (mse)")
	plt.show()
	threshold = 10
	print("Number of detected cases above threshold: {},\n\
	Number of pos cases only above threshold: {}, \n\
	The percentage of accuracy above threshold (Precision): {:0.2f}%. \n\
	Compared to the average percentage of fraud in test set: 0.132%".format( \
	np.sum(test_batch_mse > threshold), \
	np.sum(test_y[test_batch_mse > threshold]),\
	np.sum(test_y[test_batch_mse > threshold])))