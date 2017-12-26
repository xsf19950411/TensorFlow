import tensorflow as tf    
import numpy as np    
    
sess=tf.Session()  


with tf.name_scope('Xor_Nets'):
    xs=tf.placeholder(tf.float32, [None,2], name='x_input')
    with tf.name_scope('weights'):    
        W = tf.Variable(tf.zeros([2, 2])+0.2) #Weight中都是随机变量    
        tf.summary.histogram("weights", W) #可视化观看变量    
    with tf.name_scope('biases'):    
        b = tf.Variable(tf.zeros([2])-0.5) #biases推荐初始值不为0    
        tf.summary.histogram('/biases', b) #可视化观看变量  
    with tf.name_scope('Wx_b'):
        Wx_plus_b= tf.matmul(xs,W)+b
        tf.summary.histogram('Wx+b', Wx_plus_b)
    outputs= tf.nn.relu(Wx_plus_b)
    with tf.name_scope('weights2'):    
        W2 = tf.Variable(tf.zeros([2, 1])+0.2) #Weight中都是随机变量    
        tf.summary.histogram("weights2", W2) #可视化观看变量    
    with tf.name_scope('biases2'):    
        b2 = tf.Variable(tf.zeros([1])) #biases推荐初始值不为0    
        tf.summary.histogram('/biases2', b2) #可视化观看变量  
    outputs2=tf.matmul(outputs, W2)+ b2

y_=tf.placeholder(tf.float32, [None,1], name='y_train')

with tf.name_scope('Loss'):
    Loss=tf.reduce_mean(tf.reduce_sum(tf.square(outputs2-y_),reduction_indices=[1]))
    tf.summary.scalar('Loss',Loss)

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(Loss)
fixb=tf.assign(b, [0, -1])

merged=tf.summary.merge_all()
writer=tf.summary.FileWriter("C:/Files/Program/py/TensorFlow/Summary", sess.graph)
init = tf.global_variables_initializer() 

sess.run(init)
sess.run(fixb)
for i in range(300):
    sess.run(train_step, feed_dict={xs: [[0, 0], [0, 1], [1, 0], [1, 1]], y_: [[0], [1], [1], [0]]})
    result=sess.run(merged, feed_dict={xs: [[0, 0], [0, 1], [1, 0], [1, 1]], y_: [[0], [1], [1], [0]]})
    writer.add_summary(result, i)
print('xs:', sess.run(xs,{xs: [[0, 1]]} ))
print('W:', sess.run(W))
print('b:', sess.run(b))
print('Wx_b', sess.run(Wx_plus_b,{xs: [[0, 1]]} ))
print('output1:', sess.run(outputs,{xs: [[0, 1]]} ))
print('W2:', sess.run(W2))
print('b2:', sess.run(b2))
print('output2:', sess.run(outputs2,{xs: [[0, 1]]} ))
