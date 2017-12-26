import tensorflow as tf
import numpy as np

dataLen=500
noise_amp=0.03

##########################################  MODEL  ###############################################
sess= tf.Session()

with tf.name_scope('Detor'):
	x= tf.placeholder(tf.float32, [None, dataLen], name = 'detor')
	y= tf.placeholder(tf.float32, [None, dataLen], name = 'ori')

with tf.name_scope('layer1'):
	with tf.name_scope('Weight1'):
		W1= tf.Variable(tf.random_normal([dataLen, int(1/2*dataLen)], stddev=0.3))
	with tf.name_scope('Biase1'):
		b1= tf.Variable(tf.zeros([int(1/2* dataLen)]))
	L1_mul= tf.matmul(x, W1)
	L1_mul= tf.nn.bias_add(L1_mul, b1)
	L1_features= tf.nn.softplus(L1_mul)

with tf.name_scope('output1'):
	with tf.name_scope('Weight_o_1'):
		W_o_1= tf.Variable(tf.random_normal([int(1/2*dataLen), dataLen], stddev=0.3))
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
		W2= tf.Variable(tf.random_normal([int(1/2* dataLen), dataLen], stddev=0.3))
	with tf.name_scope('Biase2'):
		b2= tf.Variable(tf.zeros([dataLen]))
	L2_mul= tf.matmul(L1_features, W2)
	L2_mul= tf.nn.bias_add(L2_mul, b2)
	L2_features= tf.nn.softplus(L2_mul)

with tf.name_scope('output2'):
	with tf.name_scope('Weight_o_2'):
		W_o_2= tf.Variable(tf.random_normal([dataLen, dataLen], stddev=0.3))
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
		W3= tf.Variable(tf.random_normal([dataLen, dataLen], stddev=0.3))
	with tf.name_scope('Biase3'):
		b3= tf.Variable(tf.zeros([dataLen]))
	L3_mul= tf.matmul(L2_features, W3)
	L3_mul= tf.nn.bias_add(L3_mul, b3)
	L3_features= tf.nn.softplus(L3_mul)

with tf.name_scope('output3'):
	with tf.name_scope('Weight_o_3'):
		W_o_3= tf.Variable(tf.random_normal([dataLen, dataLen], stddev=0.3))
	with tf.name_scope('Biase_o_3'):
		b_o_3= tf.Variable(tf.zeros([dataLen]))
	L3_o_mul= tf.matmul(L3_features, W_o_3)
	L3_o_mul= tf.nn.bias_add(L3_o_mul, b_o_3)
	L3_output= tf.nn.softplus(L3_o_mul)

with tf.name_scope('loss3'):
	Loss3= tf.reduce_mean(tf.abs(y- L3_output))
	tf.summary.scalar('Loss3', Loss3)

params= tf.trainable_variables()
opt=tf.train.GradientDescentOptimizer(0.02)
gradients1=tf.gradients(Loss1, params)
Train_op1= opt.apply_gradients(zip(gradients1, params))
gradients2=tf.gradients(Loss2, params)
Train_op2= opt.apply_gradients(zip(gradients2, params))
gradients3=tf.gradients(Loss3, params)
Train_op3= opt.apply_gradients(zip(gradients3, params))

init= tf.global_variables_initializer()
merged= tf.summary.merge_all()
writer= tf.summary.FileWriter("F:/Files/Program/py/TensorFlow/Model4/summary", sess.graph)

######################################  DATA  #########################################################
y_train= np.zeros((135, dataLen))
x_train= np.zeros((135, dataLen))
noise= np.random.normal(loc= 0, scale= noise_amp, size=(dataLen))

for i in range(9):             #频率范围0.01~0.05，幅度范围0.1~1.5
	for k in range(15):
		km= 0.1 + 0.1 * k
		frequency= 0.01 + 0.005 * i 
		y_train[9*i+k, :]= np.sin(2*3.14*frequency*np.linspace(0, dataLen, dataLen))
		y_train[9*i+k, :]=km* y_train[9*i+k, :]
		x_train[9*i+k, :]= np.sin(y_train[9*i+k, :])
		
#####################################  TRAIN  ##########################################################
sess.run(init)
for i in range(2000):
	sess.run(Train_op1, feed_dict={x: x_train, y: y_train})
	if i%50 ==0:
		result = sess.run(merged, feed_dict={x: x_train, y: y_train})
		writer.add_summary(result, i)
for i in range(500):
	sess.run(Train_op2, feed_dict={x: x_train, y: y_train})
	if i%20 ==0:
		result = sess.run(merged, feed_dict={x: x_train, y: y_train})
		writer.add_summary(result, 2000+i)
for i in range(500):
	sess.run(Train_op3, feed_dict={x: x_train, y: y_train})
	if i%20 ==0:
		result = sess.run(merged, feed_dict={x: x_train, y: y_train})
		writer.add_summary(result, 2500+i)

print('Train finished.')

####################################  SAVE PARAMS  #####################################################
saver= tf.train.Saver()
savePath=saver.save(sess, 'F:\Files\Program\py\TensorFlow\Model4\Model4.ckpt')
print('Model params saved in: ', savePath)


