# EE488B Special Topics in EE <Deep Learning and AlphaGo>
# Fall 2017, School of EE, KAIST

import tensorflow as tf
import numpy as np 

#-------------------- 
# Parameter Specification
#--------------------
NUM_ITER = 200      # Number of iterations
nh = 2              # Number of hidden neurons
lr_init = 0.2       # Initial learning rate for gradient descent algorithm
lr_final = 0.01     # Final learning rate
var_init = 0.1      # Standard deviation of initializer

# Symbolic variables
x_ = tf.placeholder(tf.float32, shape=[None,2])
y_ = tf.placeholder(tf.float32, shape=[None,1])

# Training data
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [[0], [1], [1], [0]]

# Weight initialization
# Chosen to be an optimal point to demonstrate convergence to global optimum
W_init = [[1.0,-1.0],[-1.0,1.0]]
w_init = [[1.0],[1.0]]
c_init = [[0.0,0.0]]
b_init = 0.

#--------------------
# Layer setting
#--------------------
# Weights and biases
W = tf.Variable(W_init+tf.truncated_normal([2, nh], stddev=var_init))
w = tf.Variable(w_init+tf.truncated_normal([nh, 1], stddev=var_init))
c = tf.Variable(c_init+tf.truncated_normal([nh], stddev=var_init))
b = tf.Variable(b_init+tf.truncated_normal([1], stddev=var_init))

#-- Activation setting --
h = tf.nn.relu(tf.matmul(x_, W)+c)
yhat = tf.matmul(h, w)+b

#-- MSE cost function --
cost = tf.reduce_mean((y_-yhat)**2)

lr = tf.placeholder(tf.float32, shape=[])
train_step = tf.train.GradientDescentOptimizer(lr).minimize(cost)

#--------------------
# Run optimization
#--------------------
sess = tf.Session()

sess.run(tf.initialize_all_variables())
for i in range(NUM_ITER):
    j=np.random.randint(4)    # random index 0~3
    lr_current=lr_init + (lr_final - lr_init) * i / NUM_ITER
    a=sess.run(train_step, feed_dict={x_:[x_data[j]], y_:[y_data[j]], lr: lr_current})
    deploy_cost = sess.run(cost, feed_dict={x_:x_data, y_:y_data})
    deploy_yhat = sess.run(yhat, feed_dict={x_:x_data})
    print('{:2d}: XOR(0,0)={:7.4f}   XOR(0,1)={:7.4f}   XOR(1,0)={:7.4f}   XOR(1,1)={:7.4f}   cost={:.5g}'.\
        format(i+1, float(deploy_yhat[0]), float(deploy_yhat[1]), float(deploy_yhat[2]), float(deploy_yhat[3]),float(deploy_cost)))

print("W: ", sess.run(W))
print("w: ", sess.run(w))
print("c: ", sess.run(c))
print("b: ", sess.run(b))

