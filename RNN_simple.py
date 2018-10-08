import tensorflow as tf 
import numpy as np 
import random
from tensorflow.contrib import rnn
batch_size=50
input_length=10000
input_max=100
input_dim=10
hidden_dim=20
output_dim=1
output_activation=tf.nn.relu
epoch_number=10000
display_step=100
learning_rate=0.001
time_step=5
X=np.random.randint(input_max,size=(input_length,input_dim))
W=np.array([10,20,30,40,40,50,60,50,56,56])
Y=np.matmul(X,np.transpose(W))
print(Y)


inputs=tf.placeholder(tf.float32,shape=[None,time_step,2],name="inputs")
target=tf.placeholder(tf.float32,shape=[None],name="target")

weights=tf.random_normal([hidden_dim,output_dim])
bias=tf.random_normal([output_dim])
A=tf.unstack(inputs,time_step,1)
lstm=rnn.BasicLSTMCell(hidden_dim,forget_bias=1)
h,state=rnn.static_rnn(lstm,A,dtype=tf.float32)
Wo_s=tf.matmul(h[-1],weights)
Wo_s=tf.add(bias,Wo_s,name='Wo_S')
output=output_activation(Wo_s,name='output')
error=tf.square(output-target)
loss=tf.reduce_mean(error,name='loss')

optimizer=tf.train.RMSPropOptimizer(learning_rate=learning_rate)
train_op=optimizer.minimize(loss)

init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epoch_number):
        i=np.random.choice(input_length,batch_size)
        data_dict={inputs:X[i].reshape(batch_size,time_step,2),target:Y[i]}
        t,l,o=sess.run([train_op,loss,output],feed_dict=data_dict)
        if epoch % display_step ==0 or epoch==1:
            print("Step"+str(epoch)+"Minibatch Loss="+"{:.4f}".format(np.sqrt(l)))




