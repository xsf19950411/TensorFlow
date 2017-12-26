# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import PIL

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  sess = tf.InteractiveSession()
  with tf.name_scope('inputs'):
    x = tf.placeholder(tf.float32, [None, 784])
    #tf.summary.scalar('x', x)
  with tf.name_scope('params'):
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    tf.summary.histogram('Weight', W)
    tf.summary.histogram('Bias', b)
  with tf.name_scope('Output'):
    y = tf.matmul(x, W) + b
    #tf.summary.scalar('Prediction', y)

  # Define loss and optimizer
  with tf.name_scope('Labels'):
    y_ = tf.placeholder(tf.float32, [None, 10])
    #tf.summary.scalar('Labels', y_)
  

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    tf.summary.scalar('loss', cross_entropy)

  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
  
  merged=tf.summary.merge_all()
  writer = tf.summary.FileWriter("C:/Files/Program/py/TensorFlow/MNIST", sess.graph) 
  init=tf.global_variables_initializer()
  sess.run(init)
  # Train
  for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if _%50==0:    
        result = sess.run(merged,feed_dict={x: batch_xs,y_: batch_ys}) #merged也是需要run的    
        writer.add_summary(result,_) #result是summary类型的，需要放入writer中，i步数（x轴） 

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))
  #dat=sess.run(batch_xs[10])
  #print(dat)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)