import tensorflow as tf
import numpy as np
import dataproc

sess=tf.Session()
dataLength= 500
noise_amp=0.5

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
	L1_outputs=L1_convs

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
	L2_outputs= L2_convs

with tf.name_scope('outputLayer2'):
	x_wave2= tf.zeros(tf.shape(x))
	addWeight2= tf.Variable(tf.random_normal([numberOfLayerNodes], stddev=1./numberOfLayerNodes))
	#addBiases2= tf.Variable()
	for i in range(numberOfLayerNodes):
		x_wave2= x_wave2+addWeight2[i]* L2_outputs[i]

with tf.name_scope('Loss2'):
	Loss2= tf.reduce_mean(tf.abs(y- x_wave2))#+ alpha* (tf.reduce_sum(tf.square(L1_filters))+ tf.reduce_sum(tf.square(addWeight)))
	tf.summary.scalar('Loss2', Loss2)

params= tf.trainable_variables()
opt=tf.train.GradientDescentOptimizer(0.03)
gradients1=tf.gradients(Loss1, params)
Train_op1= opt.apply_gradients(zip(gradients1, params))
gradients2=tf.gradients(Loss2, params)
Train_op2= opt.apply_gradients(zip(gradients2, params))

init= tf.global_variables_initializer()
merged= tf.summary.merge_all()
writer= tf.summary.FileWriter("F:/Files/Program/py/TensorFlow/Model2", sess.graph)

###########生成数据####################
phaseStep= 2* 3.14 /20
y_train= np.zeros((60, 1, dataLength, 1))
x_train= np.zeros((60, 1, dataLength, 1))

y_test= np.zeros((1000, 1, dataLength, 1))
x_test= np.zeros((1000, 1, dataLength, 1))

for i in range(60):
	y_train[i, 0, :, 0]= np.sin(2*3.14*0.015*np.linspace(0, dataLength, dataLength))
	noise= np.random.normal(loc= 0, scale= noise_amp, size=(dataLength))
	x_train[i, 0, :, 0]= y_train[i, 0, :, 0]+ noise

for i in range(100):
	y_test[i, 0, :, 0]= np.sin(2*3.14*0.015*np.linspace(0, dataLength, dataLength))
	noise= np.random.normal(loc= 0, scale= noise_amp, size=(dataLength))
	x_test[i, 0, :, 0]= y_test[i, 0, :, 0]+ noise


##############模型训练--第一层####################
sess.run(init)
for i in range(200):
	sess.run(Train_op1, feed_dict={x: x_train, y: y_train})
	if i%10 ==0:
		result=sess.run(merged, feed_dict={x: x_train, y: y_train})
		writer.add_summary(result, i)
############模型训练--第二层#################
for i in range(1400):
	sess.run(Train_op2, feed_dict={x: x_train, y: y_train})
	if i%10 ==0:
		result=sess.run(merged, feed_dict={x: x_train, y: y_train})
		writer.add_summary(result, 300+ i)

###############单次测试##################
x_singleTest= np.zeros((1, 1, dataLength, 1))
y_singleTest= np.zeros((1, 1, dataLength, 1))
y_singleTest[0, 0, :, 0]= np.sin(2*3.14*0.015*np.linspace(0, dataLength, dataLength))
noise= np.random.normal(loc= 0, scale= noise_amp, size=(dataLength))
x_singleTest[0, 0, :, 0]= y_singleTest[0, 0, :, 0]+noise

with open('F:/Files/Program/py/TensorFlow/Model2/ori.txt', 'w') as f:
	f.write(str(np.reshape(y_singleTest, dataLength)))
print('Origin data:\n', np.reshape(y_singleTest, dataLength))
with open('F:/Files/Program/py/TensorFlow/Model2/detor.txt', 'w') as f:
	f.write(str(np.reshape(x_singleTest, dataLength)))
print('Detor data:\n', np.reshape(x_singleTest, dataLength))
flattenResult= tf.reshape(x_wave2, [-1])
with open('F:/Files/Program/py/TensorFlow/Model2/calc.txt', 'w') as f:
	f.write(str(sess.run(flattenResult, feed_dict={x: x_singleTest})))
print('calc result:\n', sess.run(flattenResult, feed_dict={x: x_singleTest}))

dataproc.dataproc()


