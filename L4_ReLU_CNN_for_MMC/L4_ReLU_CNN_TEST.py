import tensorflow as tf
import numpy as np



dataLength=1000
filterWidth=4
numOfNodes=[32, 34, 38, 1]
learningRate=0.03


def weight_variable(shape):
	initial=tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial=tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
sess=tf.Session()

with tf.name_scope('Inputs'):
	x= tf.placeholder(dtype=tf.float32, shape=[None, dataLength])
	y= tf.placeholder(dtype=tf.float32, shape=[None, dataLength])
	x_reshape=tf.reshape(x, [-1, 1, dataLength, 1])
	y_reshape=tf.reshape(y, [-1, 1, dataLength, 1])
with tf.name_scope('Layer1'):
	W_1=weight_variable([1, filterWidth, 1, numOfNodes[0]])
	b_1=bias_variable([numOfNodes[0]])
	h1=tf.nn.relu(conv(x_reshape, W_1)+b_1)
with tf.name_scope('Layer2'):
	W_2=weight_variable([1, filterWidth, numOfNodes[0], numOfNodes[1]])
	b_2=bias_variable([numOfNodes[1]])
	h2=tf.nn.relu(conv(h1, W_2)+b_2)
with tf.name_scope('Layer3'):
	W_3=weight_variable([1, filterWidth, numOfNodes[1], numOfNodes[2]])
	b_3=bias_variable([numOfNodes[2]])
	h3=tf.nn.relu(conv(h2, W_3)+b_3)
with tf.name_scope('Output_Layer'):
	W_4=weight_variable([1, filterWidth, numOfNodes[2], numOfNodes[3]])
	b_4=bias_variable([numOfNodes[3]])
	output=conv(h3, W_4)+b_4

saver=tf.train.Saver()
saver.restore(sess, 'F:\Files\Program\py\TensorFlow\L4_ReLU_CNN_for_MMC\parameters.ckpt')
print('Model restored.')



###############单次测试##################
x_singleTest= np.zeros((1, dataLength))
y_singleTest= np.zeros((1, dataLength))
frequency=0.033
Amplitude_mismatch=0.01
y_singleTest[0, :]= np.sin(2*3.14*frequency*np.linspace(0, dataLength, dataLength))
x_singleTest[0, :]= y_singleTest[0, :]
for i in range(dataLength):
	if i%2 ==0:
		x_singleTest[0, i]= (1+Amplitude_mismatch)* x_singleTest[0, i]

#############################文件操作##############################
with open('F:/Files/Program/py/TensorFlow/L4_ReLU_CNN_for_MMC/ori.txt', 'w') as f:
	f.write('')
with open('F:/Files/Program/py/TensorFlow/L4_ReLU_CNN_for_MMC/ori.txt', 'a') as f:
	y_reshape=np.reshape(y_singleTest, -1)
	for i in y_reshape:
		f.write(str(i))
		f.write('\n')
with open('F:/Files/Program/py/TensorFlow/L4_ReLU_CNN_for_MMC/detor.txt', 'w') as f:
	f.write('')
with open('F:/Files/Program/py/TensorFlow/L4_ReLU_CNN_for_MMC/detor.txt', 'a') as f:
	x_reshape=np.reshape(x_singleTest, -1)
	for i in x_reshape:
		f.write(str(i))
		f.write('\n')
with open('F:/Files/Program/py/TensorFlow/L4_ReLU_CNN_for_MMC/calc.txt', 'w') as f:
	f.write('')
with open('F:/Files/Program/py/TensorFlow/L4_ReLU_CNN_for_MMC/calc.txt', 'w') as f:
	flattenResult= tf.reshape(output, [-1])
	for i in sess.run(flattenResult, feed_dict={x: x_singleTest}):
		f.write(str(i))
		f.write('\n')
print('single test finished, check files for results')