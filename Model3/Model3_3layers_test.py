import tensorflow as tf
import numpy as np
import dataproc

sess=tf.Session()
dataLength= 500
noise_amp=0.001

filterWidth= 4     #HyperParameters
numberOfLayerNodes=10
alpha=1

with tf.name_scope('inputs'):
	x= tf.placeholder(dtype=tf.float32, shape= [None, 1, dataLength, 1])
	y= tf.placeholder(dtype=tf.float32, shape= [None, 1, dataLength, 1])

with tf.name_scope('Layer1'):
	L1_filters= tf.Variable(tf.random_normal([numberOfLayerNodes, 1, filterWidth, 1, 1], stddev= 3))
	L1_biases= tf.Variable(tf.zeros([numberOfLayerNodes, 1])+0.1)
	L1_convs= []
	for i in range(numberOfLayerNodes):
		conv_result= tf.nn.conv2d(x, L1_filters[i,], [1, 1, 1, 1], 'SAME')+ L1_biases[i]
		L1_convs.append(conv_result)
	L1_convs= tf.stack(L1_convs)
	L1_outputs= tf.nn.softplus(L1_convs)

with tf.name_scope('outputLayer1'):
	x_wave1= tf.zeros(tf.shape(x))
	addWeight= tf.Variable(tf.random_normal([numberOfLayerNodes], stddev=1./numberOfLayerNodes))
	for i in range(numberOfLayerNodes):
		x_wave1= x_wave1+addWeight[i]* L1_outputs[i] 

with tf.name_scope('Loss1'):
	Loss1= tf.reduce_mean(tf.abs(y- x_wave1))#+ alpha* (tf.reduce_sum(tf.square(L1_filters))+ tf.reduce_sum(tf.square(addWeight)))
	tf.summary.scalar('Loss1', Loss1)

with tf.name_scope('Layer2'):
	L2_filters= tf.Variable(tf.random_normal([numberOfLayerNodes, 4, 1, filterWidth, 1, 1], stddev= 3))
	L2_biases= tf.Variable(tf.zeros([numberOfLayerNodes, 4, 1])+ 0.1)
	L2_convs= []
	for i in range(numberOfLayerNodes):
		for k in range(4):
			conv_result= tf.zeros(tf.shape(x))
			conv_result= conv_result+ tf.nn.conv2d(L1_outputs[(i+ k)%numberOfLayerNodes], L2_filters[i, k], [1, 1, 1, 1], 'SAME')+ L2_biases[i, k]
		L2_convs.append(conv_result)
	L2_convs= tf.stack(L2_convs)
	L2_outputs= tf.nn.softplus(L2_convs)

with tf.name_scope('outputLayer2'):
	x_wave2= tf.zeros(tf.shape(x))
	addWeight2= tf.Variable(tf.random_normal([numberOfLayerNodes], stddev=1./numberOfLayerNodes))
	#addBiases2= tf.Variable()
	for i in range(numberOfLayerNodes):
		x_wave2= x_wave2+addWeight2[i]* L2_outputs[i]

with tf.name_scope('Loss2'):
	Loss2= tf.reduce_mean(tf.abs(y- x_wave2))#+ alpha* (tf.reduce_sum(tf.square(L1_filters))+ tf.reduce_sum(tf.square(addWeight)))
	tf.summary.scalar('Loss2', Loss2)

with tf.name_scope('Layer3'):
	L3_filters= tf.Variable(tf.random_normal([numberOfLayerNodes, 4, 1, filterWidth, 1, 1], stddev= 3))
	L3_biases= tf.Variable(tf.zeros([numberOfLayerNodes, 4, 1])+ 0.1)
	L3_convs= []
	for i in range(numberOfLayerNodes):
		for k in range(4):
			conv_result= tf.zeros(tf.shape(x))
			conv_result= conv_result+ tf.nn.conv2d(L2_outputs[(i+ k)%numberOfLayerNodes], L3_filters[i, k], [1, 1, 1, 1], 'SAME')+ L3_biases[i, k]
		L3_convs.append(conv_result)
	L3_convs= tf.stack(L3_convs)
	L3_outputs= L3_convs

with tf.name_scope('outputLayer3'):
	x_wave3= tf.zeros(tf.shape(x))
	addWeight3= tf.Variable(tf.random_normal([numberOfLayerNodes], stddev=1./numberOfLayerNodes))
	#addBiases2= tf.Variable()
	for i in range(numberOfLayerNodes):
		x_wave3= x_wave3+addWeight3[i]* L3_outputs[i]

with tf.name_scope('Loss3'):
	Loss3= tf.reduce_mean(tf.abs(y- x_wave3))#+ alpha* (tf.reduce_sum(tf.square(L1_filters))+ tf.reduce_sum(tf.square(addWeight)))
	tf.summary.scalar('Loss3', Loss3)

# writer= tf.summary.FileWriter("F:/Files/Program/py/TensorFlow/Model3", sess.graph)
saver=tf.train.Saver()

saver.restore(sess, 'F:\Files\Program\py\TensorFlow\Model3\Model3.ckpt')
print('Model restored.')

###############单次测试##################
x_singleTest= np.zeros((1, 1, dataLength, 1))
y_singleTest= np.zeros((1, 1, dataLength, 1))
noise= np.random.normal(loc= 0, scale= noise_amp, size=(dataLength))
km=1
frequency=0.02
y_singleTest[0, 0, :, 0]= np.sin(2*3.14*frequency*np.linspace(0, dataLength, dataLength))
x_singleTest[0, 0, :, 0]= np.sin(km*y_singleTest[0, 0, :, 0])+ noise
y_singleTest[0, 0, :, 0]=km* y_singleTest[0, 0, :, 0]

with open('F:/Files/Program/py/TensorFlow/Model3/ori.txt', 'w') as f:
	f.write(str(np.reshape(y_singleTest, dataLength)))
print('Origin data:\n', np.reshape(y_singleTest, dataLength))
with open('F:/Files/Program/py/TensorFlow/Model3/detor.txt', 'w') as f:
	f.write(str(np.reshape(x_singleTest, dataLength)))
print('Detor data:\n', np.reshape(x_singleTest, dataLength))
flattenResult= tf.reshape(x_wave3, [-1])
with open('F:/Files/Program/py/TensorFlow/Model3/calc.txt', 'w') as f:
	f.write(str(sess.run(flattenResult, feed_dict={x: x_singleTest})))
print('calc result:\n', sess.run(flattenResult, feed_dict={x: x_singleTest}))

dataproc.dataproc()

