from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class DNN:
    def __init__(self,X,num_layers,num_input,epoch,batch_size,number_classes,num_neuron_each_layers):
        self.num_input=num_input
        self.num_layers=num_layers
        self.number_classes=number_classes
        self.X=X
        self.x=tf.placeholder(dtype="float",shape=[None,self.num_input])
        self.Y=tf.placeholder(dtype="float",shape=[None,self.number_classes])
        self.epoch=epoch
        self.batch_size=batch_size
        self.number_classes=number_classes
        self.num_neuron_each_layers=num_neuron_each_layers


    def layers(self, X, weight_shape,biase_shape):
        
        weight=tf.get_variable("w",weight_shape,initializer=tf.random_normal_initializer())
        bias=tf.get_variable("b",biase_shape,initializer=tf.random_normal_initializer())        

        return tf.nn.sigmoid(tf.matmul(X,weight)+bias)


    def outlayers(self, X, weight_shape,biase_shape):
        
        weight=tf.get_variable("w",weight_shape,initializer=tf.random_normal_initializer())
        bias=tf.get_variable("b",biase_shape,initializer=tf.random_normal_initializer())        

        return tf.add(tf.matmul(X,weight),bias)
    
    

    def construct_layers(self):
        with tf.variable_scope("layers"):
            temp=self.layers(self.x,[self.num_input,self.num_neuron_each_layers],self.num_neuron_each_layers)
        for i in range(self.num_layers-1):
          with tf.variable_scope("layers"+str(i)):
            temp=self.layers(temp,[self.num_neuron_each_layers,self.num_neuron_each_layers],self.num_neuron_each_layers)

        
        with tf.variable_scope("output_layers"):
            y=self.outlayers(temp,[self.num_neuron_each_layers,self.number_classes],self.number_classes)

        return y

    
    def train(self):
      
      y=self.construct_layers()
      cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y,labels=self.Y))
      optimizer=tf.train.AdadeltaOptimizer(learning_rate=0.001).minimize(cost)
      init=tf.global_variables_initializer()
      correct_prediction = tf.equal(tf.argmax(y, 1),tf.argmax(self.Y, 1))
# Calculate accuracy
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
      with tf.Session() as sess:
          sess.run(init)

          for i in range(self.epoch):


              avg_cost=0
              total_batch=int(self.X.train.num_examples/self.batch_size)


              for j in range(total_batch):
                  batch_xs, batch_ys = self.X.train.next_batch(self.batch_size)
                  sess.run(optimizer, feed_dict={self.x: batch_xs, self.Y: batch_ys})
                  avg_cost += sess.run(cost, feed_dict={self.x: batch_xs,self.Y: batch_ys})/total_batch
              if i%10==0:
                  print ("Epoch:%04d  cost= %.9f  Model Accuracy:%f"%(i+1,avg_cost,accuracy.eval({self.x: self.X.test.images,self.Y:self.X.test.labels})))



D=DNN(mnist,3,784,1000,100,10,256)
D.train()
                



        





