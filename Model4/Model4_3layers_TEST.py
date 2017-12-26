import tensorflow as tf
import numpy as np
import dataproc

dataLen=500
noise_amp=0.03

##########################################  MODEL  ###############################################
sess= tf.Session()

with tf.name_scope('Detor'):
	x= tf.placeholder(tf.float32, [None, dataLen], name = 'detor')
	y= tf.placeholder(tf.float32, [None, dataLen], name = 'ori')

with tf.name_scope('layer1'):
	with tf.name_scope('Weight1'):
		W1= tf.Variable(tf.random_normal([dataLen, 2*dataLen], stddev=0.3))
	with tf.name_scope('Biase1'):
		b1= tf.Variable(tf.zeros([2* dataLen]))
	L1_mul= tf.matmul(x, W1)
	L1_mul= tf.nn.bias_add(L1_mul, b1)
	L1_features= tf.nn.softplus(L1_mul)

with tf.name_scope('output1'):
	with tf.name_scope('Weight_o_1'):
		W_o_1= tf.Variable(tf.random_normal([2*dataLen, dataLen], stddev=0.3))
	with tf.name_scope('Biase_o_1'):
		b_o_1= tf.Variable(tf.zeros([dataLen]))
	L1_o_mul= tf.matmul(L1_features, W_o_1)
	L1_o_mul= tf.nn.bias_add(L1_o_mul, b_o_1)
	L1_output= tf.nn.softplus(L1_o_mul)

with tf.name_scope('loss1'):
	Loss1= tf.reduce_mean(tf.abs(y- L1_output))
	tf.summary.scalar('Loss1', Loss1)

with tf.name_scope('layer2'):
	with tf.name_scope('Weight2'):
		W2= tf.Variable(tf.random_normal([2* dataLen, 2* dataLen], stddev=0.3))
	with tf.name_scope('Biase2'):
		b2= tf.Variable(tf.zeros([2* dataLen]))
	L2_mul= tf.matmul(L1_features, W2)
	L2_mul= tf.nn.bias_add(L2_mul, b2)
	L2_features= tf.nn.softplus(L2_mul)

with tf.name_scope('output2'):
	with tf.name_scope('Weight_o_2'):
		W_o_2= tf.Variable(tf.random_normal([2*dataLen, dataLen], stddev=0.3))
	with tf.name_scope('Biase_o_2'):
		b_o_2= tf.Variable(tf.zeros([dataLen]))
	L2_o_mul= tf.matmul(L2_features, W_o_2)
	L2_o_mul= tf.nn.bias_add(L2_o_mul, b_o_2)
	L2_output= tf.nn.softplus(L2_o_mul)

with tf.name_scope('loss2'):
	Loss2= tf.reduce_mean(tf.abs(y- L2_output))
	tf.summary.scalar('Loss2', Loss2)

with tf.name_scope('layer3'):
	with tf.name_scope('Weight3'):
		W3= tf.Variable(tf.random_normal([2* dataLen, 2* dataLen], stddev=0.3))
	with tf.name_scope('Biase3'):
		b3= tf.Variable(tf.zeros([2* dataLen]))
	L3_mul= tf.matmul(L2_features, W3)
	L3_mul= tf.nn.bias_add(L3_mul, b3)
	L3_features= tf.nn.softplus(L3_mul)

with tf.name_scope('output3'):
	with tf.name_scope('Weight_o_3'):
		W_o_3= tf.Variable(tf.random_normal([2*dataLen, dataLen], stddev=0.3))
	with tf.name_scope('Biase_o_3'):
		b_o_3= tf.Variable(tf.zeros([dataLen]))
	L3_o_mul= tf.matmul(L3_features, W_o_3)
	L3_o_mul= tf.nn.bias_add(L3_o_mul, b_o_3)
	L3_output= tf.nn.softplus(L3_o_mul)

with tf.name_scope('loss3'):
	Loss3= tf.reduce_mean(tf.abs(y- L3_output))
	tf.summary.scalar('Loss3', Loss3)

writer= tf.summary.FileWriter("F:/Files/Program/py/TensorFlow/Model4/summary", sess.graph)
######################################  DATA  #########################################################
# NO TEST SET, JUST SINGLE TEST 
y_singleTest= np.zeros((1, dataLen))
x_singleTest= np.zeros((1, dataLen))
noise= np.random.normal(loc= 0, scale= noise_amp, size=(dataLen))
km=1                   #频率范围0.01~0.05，幅度范围0.1~1.5
frequency=0.02
y_singleTest[0, :]= np.sin(2*3.14*frequency*np.linspace(0, dataLen, dataLen))
y_singleTest[0, :]= km* y_singleTest[0, :]
x_singleTest[0, :]= np.sin(y_singleTest[0, :])+ noise

######################################  PARAMS RESTORE ################################################
saver=tf.train.Saver()
saver.restore(sess, 'F:\Files\Program\py\TensorFlow\Model4\Model4.ckpt')
print('Model restored.')

######################################  SINGLE TEST & DATA PROC ########################################
with open('F:/Files/Program/py/TensorFlow/Model4/ori.txt', 'w') as f:
	f.write(str(np.reshape(y_singleTest, dataLen)))

with open('F:/Files/Program/py/TensorFlow/Model4/detor.txt', 'w') as f:
	f.write(str(np.reshape(x_singleTest, dataLen)))
flattenResult= tf.reshape(L1_output, [-1])
with open('F:/Files/Program/py/TensorFlow/Model4/calc.txt', 'w') as f:
	f.write(str(sess.run(flattenResult, feed_dict={x: x_singleTest})))
dataproc.dataproc()
print('single test finish. check files for results.')

