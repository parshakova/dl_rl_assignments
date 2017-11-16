# EE488B Special Topics in EE <Deep Learning and AlphaGo>
# Fall 2017, School of EE, KAIST

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

for i in range(10):
    for j in range(10):
        img=mnist.train.images[i*10+j]
        img.shape=(28,28)
        plt.subplot(10,10,i*10+j+1)
        plt.imshow(img,cmap='gray')
        plt.axis('off')
plt.savefig('mnist_train_images.png',dpi=300,bbox_inches='tight')
plt.show()

