import tensorflow as tf
import numpy as np
import dataproc

sess=tf.Session()
dataLength= 2000
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
	L3_outputs= tf.nn.softplus(L3_convs)

with tf.name_scope('outputLayer3'):
	x_wave3= tf.zeros(tf.shape(x))
	addWeight3= tf.Variable(tf.random_normal([numberOfLayerNodes], stddev=1./numberOfLayerNodes))
	#addBiases2= tf.Variable()
	for i in range(numberOfLayerNodes):
		x_wave3= x_wave3+addWeight3[i]* L3_outputs[i]

with tf.name_scope('Loss3'):
	Loss3= tf.reduce_mean(tf.abs(y- x_wave3))#+ alpha* (tf.reduce_sum(tf.square(L1_filters))+ tf.reduce_sum(tf.square(addWeight)))
	tf.summary.scalar('Loss3', Loss3)

params= tf.trainable_variables()
opt=tf.train.GradientDescentOptimizer(0.03)
gradients1=tf.gradients(Loss1, params)
Train_op1= opt.apply_gradients(zip(gradients1, params))
gradients2=tf.gradients(Loss2, params)
Train_op2= opt.apply_gradients(zip(gradients2, params))
gradients3=tf.gradients(Loss3, params)
Train_op3= opt.apply_gradients(zip(gradients3, params))

init= tf.global_variables_initializer()
merged= tf.summary.merge_all()
writer= tf.summary.FileWriter("F:/Files/Program/py/TensorFlow/Model3_v2/summary", sess.graph)
saver=tf.train.Saver()

###########生成数据####################
km=1
y_train= np.zeros((6*15, 1, dataLength, 1))
x_train= np.zeros((6*15, 1, dataLength, 1))
noise= np.random.normal(loc= 0, scale= noise_amp, size=(dataLength))

for i in range(6):             #频率范围0.01~0.05，幅度范围0.1~1.5
	for k in range(15):
		km= 0.1 + 0.1 * k
		frequency= 0.02 + 0.005 * i 
		y_train[9*i+k, 0, :, 0]= np.sin(2*3.14*frequency*np.linspace(0, dataLength, dataLength))
		y_train[9*i+k, 0, :, 0]=km* y_train[9*i+k, 0, :, 0]
		x_train[9*i+k, 0, :, 0]= np.sin(y_train[9*i+k, 0, :, 0])
		


##############模型训练--第一层####################
sess.run(init)
for i in range(500):
	sess.run(Train_op1, feed_dict={x: x_train, y: y_train})
	if i%50 ==0:
		result=sess.run(merged, feed_dict={x: x_train, y: y_train})
		writer.add_summary(result, i)
		print('epoch: ', i)
############模型训练--第二层#################
for i in range(500):
	sess.run(Train_op2, feed_dict={x: x_train, y: y_train})
	if i%50 ==0:
		result=sess.run(merged, feed_dict={x: x_train, y: y_train})
		writer.add_summary(result, 500+ i)
		print('epoch: 500+', i)
############模型训练--第三层#################
for i in range(100000):
	sess.run(Train_op3, feed_dict={x: x_train, y: y_train})
	if i%100 ==0:
		result=sess.run(merged, feed_dict={x: x_train, y: y_train})
		writer.add_summary(result, 1000+ i)
		print('epoch: 1000+', i)

savePath=saver.save(sess, 'F:\Files\Program\py\TensorFlow\Model3_v2\Model3.ckpt')
print('Model params saved in: ', savePath)

###############单次测试##################
print('Training Finished!')


