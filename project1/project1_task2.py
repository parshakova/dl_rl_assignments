
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

print("\n \n \n \n ***MY ENVIRONMENT*** python 2.7.13 tensorflow 1.2.1 \n \n \n \n")


# In[2]:


mnist = input_data.read_data_sets('./MNIST_data', one_hot=False)
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.int64, shape=[None])
keep_prob = tf.placeholder(tf.float32, shape=[])
batch_size = 100


# # Computation graph for classifier on 9 classes and for classifier on 10 classes
#Firstly, I trained a classifier on 9 digits: {0,1,2,3,..,8} until test accuracy of 98.5% was achieved.
#Then I applied a transfer learning by training only the output layer for classification task on 10 digits.


# In[3]:


# ------------------------------------------------------
# Parameters and optimization function for classification on 9 classes
# ------------------------------------------------------

# Convolutional layer
x_image = tf.reshape(x, [-1,28,28,1])
W_conv = tf.Variable(tf.truncated_normal([5, 5, 1, 30], stddev=0.1))
b_conv = tf.Variable(tf.constant(0.1, shape=[30]))
h_conv = tf.nn.conv2d(x_image, W_conv, strides=[1, 1, 1, 1], padding='VALID')
h_relu = tf.nn.relu(h_conv + b_conv)
h_pool = tf.nn.max_pool(h_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],                         padding='SAME')
h_pool = tf.nn.dropout(h_pool, keep_prob)

# Fully-connected layer
W_fc1 = tf.Variable(tf.truncated_normal([12 * 12 * 30, 500], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[500]))
h_pool_flat = tf.reshape(h_pool, [-1, 12*12*30])
h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)
h_fc1 = tf.nn.dropout(h_fc1, keep_prob)

# Output layer for 9 classes
W_fc2 = tf.Variable(tf.truncated_normal([500, 9], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[9]))
logits = tf.matmul(h_fc1, W_fc2) + b_fc2
y_hat=tf.nn.softmax(logits)

# Train and Evaluate the Model for 9 classes
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits                            (logits=logits, labels=y_)
l2 = 0.001* tf.reduce_mean([tf.reduce_mean(tf.nn.l2_loss(W_conv))                            , tf.reduce_mean(tf.nn.l2_loss(W_fc1)),                             tf.reduce_mean(tf.nn.l2_loss(W_fc2))])
loss = cross_entropy+l2
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_hat,1), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# ------------------------------------------------------
# Transfer learning parameters and optimization function 
# for classification on 10 classes
# ------------------------------------------------------

# Output layer for 10 classes
W_fc2_10 = tf.Variable(tf.truncated_normal([500, 10], stddev=0.1))
b_fc2_10 = tf.Variable(tf.constant(0.1, shape=[10]))
logits_10 = tf.matmul(h_fc1, W_fc2_10) + b_fc2_10
y_hat_10 =tf.nn.softmax(logits_10)

# Train and Evaluate the Model for 10 classes
cross_entropy_10 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_10, labels=y_)
l2_10 = 0.001* tf.reduce_mean([tf.reduce_mean(tf.nn.l2_loss(W_fc2_10))])
loss_10 = cross_entropy_10+l2_10
train_step_10 = tf.train.AdamOptimizer(1e-4).minimize(loss_10,                                         var_list=[W_fc2_10, b_fc2_10])
correct_prediction_10 = tf.equal(tf.argmax(y_hat_10,1), y_)
accuracy_10 = tf.reduce_mean(tf.cast(correct_prediction_10, tf.float32))


# # Filter train/valid/test dataset to include all digits except "9"

# In[ ]:


train_l = mnist.train.labels[np.where(mnist.train.labels!=9)]
train_i = mnist.train.images[np.where(mnist.train.labels!=9)]
val_l = mnist.validation.labels[np.where(mnist.validation.labels!=9)]
val_i = mnist.validation.images[np.where(mnist.validation.labels!=9)]
test_l = mnist.test.labels[np.where(mnist.test.labels!=9)]
test_i = mnist.test.images[np.where(mnist.test.labels!=9)]
sess.run(tf.global_variables_initializer())



# In[5]:


n_train = train_i.shape[0]
batches = zip(range(0, n_train-batch_size, batch_size),               range(batch_size, n_train, batch_size))
batches = [(start, end) for start, end in batches]


# # Train classifier on 9 classes

# In[6]:


print("=================================")
print("|    Training for 9 classes     |")
print("|Epoch\tBatch\t|Train\t|Val\t|")
print("|===============================|")
for j in range(10):
    for i, inds in enumerate(batches):
        start, end = inds
        batch_i = train_i[start:end]
        batch_l = train_l[start:end]
        train_step.run(feed_dict={x: batch_i, y_: batch_l, keep_prob:0.7})
        if i%50 == 49:
            train_accuracy = accuracy.eval(feed_dict={x:batch_i,                                                y_: batch_l, keep_prob:1})
            val_accuracy = accuracy.eval(feed_dict=                {x: val_i, y_:val_l, keep_prob:1})
            print("|%d\t|%d\t|%.4f\t|%.4f\t|"%(j+1, i+1,                                             train_accuracy, val_accuracy))
print("|===============================|")
test_accuracy = accuracy.eval(feed_dict=    {x: test_i, y_:test_l, keep_prob:1})
print("test accuracy=%.4f"%(test_accuracy))


# In[7]:


# assign values from learned 9 neurons to the output layer 
# of transfer learning matrix on 10 classes
sess.run(W_fc2_10[:,:9].assign(W_fc2))


# # Transfer learning from classifier on 9 classes to classifier on 10 classes

# In[8]:


print("=================================")
print("|    Training for 10 classes    |")
print("|Epoch\tBatch\t|Train\t|Val\t|")
print("|===============================|")
for j in range(8):
    for i in range(550):
        batch = mnist.train.next_batch(batch_size)
        train_step_10.run(feed_dict={x: batch[0], y_: batch[1],                                      keep_prob:0.7})
        if i%50 == 49:
            train_accuracy = accuracy_10.eval(feed_dict={x:batch[0],                                                y_: batch[1], keep_prob:1})
            val_accuracy = accuracy_10.eval(feed_dict=                {x: mnist.validation.images, y_:mnist.validation.labels,                                                      keep_prob:1})
            print("|%d\t|%d\t|%.4f\t|%.4f\t|"%(j+1, i+1, train_accuracy,                                                        val_accuracy))
print("|===============================|")
test_accuracy = accuracy_10.eval(feed_dict=    {x: mnist.test.images, y_:mnist.test.labels, keep_prob:1})
print("test accuracy=%.4f"%(test_accuracy))

# # Transfer Learning Classification Results
# 
# Since the final test accuracy on the transfer learning for 10 classes is 97.88% which is good but not as high as it could be while all layers are trained 98.5%.
# 
# We can make a conclusion that learning just weights from a final layer and trasfer weights on previous layers from classifier on 9 classes is not absolutely enough and some useful modifications which are made in the convolution layer and first fully connected layer are useful for improving the performance of the model.
