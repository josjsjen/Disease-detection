#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: jie
"""
import tensorflow as tf
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import uuid
from tqdm import tqdm

# download image and write to file lung_data
import util.download_lung_data

# access train and test data
class PneumothoraxDataset:
    def __init__(self):
        print("Loading X-Ray Dataset!")

        train = h5py.File(util.download_lung_data.data_dir+'pneumothorax_train.h5','r')
        test = h5py.File(util.download_lung_data.data_dir+'pneumothorax_test.h5','r')

        self.X_train = train['image'][:]
        self.X_test = test['image'][:]
        self.Y_train = train['label'][:]
        self.Y_test = test['label'][:]

        self.num_train = self.X_train.shape[0]
        self.num_test = self.X_test.shape[0]

        self.batch_pointer = 0
        
    def getTotalNumDataPoints(self):
        return self.num_train+self.num_test
    
    def getTrainBatch(self, batch_size):
        inds = np.arange(self.batch_pointer,self.batch_pointer+batch_size)
        inds = np.mod( inds , self.num_train ) 
        batch = (self.X_train[inds], self.Y_train[inds]) 

        self.batch_pointer += batch_size 
        return batch

    def getTestBatch(self, batch_size):
        inds = np.random.choice(self.num_test, size=batch_size)
        return (self.X_test[inds], self.Y_test[inds])

    
data = PneumothoraxDataset() 
print("Dataset consists of {} images".format(data.getTotalNumDataPoints()))

'''
## Visualize data
INDEX = 0

image = data.X_train[INDEX]
label = data.Y_train[INDEX]
pred = np.argmax(label)

print(data.Y_train)
plt.imshow(image[:,:,0], cmap='gray')
print("This X-Ray "+("HAS" if pred else "DOES NOT have")+ " a pneumothorax")


'''

##########################################
############ Define CNN model ############
#downscaled images to  256×256  pixels.
x = tf.placeholder(shape=[None, 256, 256, 1], dtype=tf.float32)         
y = tf.placeholder(shape=[None, 2], dtype=tf.float32)

# reshape our input to a 4-D blob that preserves the spatial nature of the image.
x_image = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), x)

# 64 5x5 filters with stride of 1, ReLU activation
conv1 = tf.layers.conv2d(inputs=x_image, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

# 2x2 max pooling with stride of 2
skip1 = tf.layers.max_pooling2d(conv1+conv2, 2, 2)

conv3 = tf.layers.conv2d(inputs=skip1, filters=128, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
conv4 = tf.layers.conv2d(inputs=conv3, filters=128, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
skip2 = tf.layers.max_pooling2d(conv3+conv4, 2, 2)

conv5 = tf.layers.conv2d(inputs=skip2, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
conv6 = tf.layers.conv2d(inputs=conv5, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
skip3 = tf.layers.max_pooling2d(conv5+conv6, 2, 2)

conv7 = tf.layers.conv2d(inputs=skip3, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
conv8 = tf.layers.conv2d(inputs=conv7, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
skip4 = tf.layers.max_pooling2d(conv7+conv8, 2, 2)

# define last fully connected layer for binary classification
g_avg_pool = tf.reduce_mean(skip4, [1,2])
y_ = tf.layers.dense(g_avg_pool, 2)

# output probabilities of input image belonging to each digit class
probabilities = tf.nn.softmax(y_)



##########################################
############ Define hyperparameters ######

batch_size = 15
learning_rate = 0.05
num_training_steps = int(1e3)



##########################################
## define cost, optimizer and accurancy###
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_)) 
# gradient descent with momentum to prevent oscillations and badly conditioned curvature
optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True).minimize(cost)

correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
prediction = tf.argmax(y_,1) 



##########################################
############ Initialize the data #########

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
saver = tf.train.Saver()
                           
##########################################
############ Train Model #################

sess.run(init)
summary_writer_train = tf.summary.FileWriter('train', graph=tf.get_default_graph())
summary_writer_test = tf.summary.FileWriter('test', graph=tf.get_default_graph())

for step in range(num_training_steps):
    (x_batch, y_batch) = data.getTrainBatch(batch_size) 
    _, trainLoss, summary = sess.run([optimizer, cost, merged_summary_op],
                               feed_dict={x: x_batch, y:y_batch})

    summary_writer_train.add_summary(summary, step) 

    if step % 10 == 0:
        (x_test, y_test) = data.getTestBatch(100) # get a testing batch of data
        testLoss, testAcc, summary = sess.run([cost, accuracy, merged_summary_op], 
                                              feed_dict={x: x_test, y:y_test})

        print("step: {}, train: {}, \t\t test: {}, testAcc: {}".format(
              step, trainLoss, testLoss, int(testAcc*100)))
        summary_writer_test.add_summary(summary, step)

    if step % 100 == 0:
      save_path = saver.save(sess, '/model.ckpt')
      print("Model saved in file: %s" % save_path)


##########################################
########## Reload Train Model ############
#restore the weights from a trained model
#saver.restore(sess, "./saved_model/model.ckpt") #only run this once!



