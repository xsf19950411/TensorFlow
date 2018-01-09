import tensorflow as tf
import numpy as np



dataLength=1000
filterWidth=4
numOfNodes=[32, 34, 38, 1]
learningRate=0.03

y_train= np.zeros((6*15, dataLength))
x_train= np.zeros((6*15, dataLength))

for i in range(6):             #频率范围0.01~0.05，幅度范围0.1~1.5
	for k in range(15):
		km= 0.1 + 0.1 * k
		frequency= 0.02 + 0.005 * i 
		y_train[9*i+k, :]= np.sin(2*3.14*frequency*np.linspace(0, dataLength, dataLength))
		y_train[9*i+k, :]=km* y_train[9*i+k, :]
		x_train[9*i+k, :]= np.sin(y_train[9*i+k, :])

y_test= np.zeros((6*15, dataLength))
x_test= np.zeros((6*15, dataLength))

for i in range(6):             #频率范围0.01~0.05，幅度范围0.1~1.5
	for k in range(15):
		km_test= 0.13 + 0.09 * k
		frequency= 0.02 + 0.004 * i 
		y_test[9*i+k, :]= np.sin(2*3.14*frequency*np.linspace(0, dataLength, dataLength))
		y_test[9*i+k, :]=km_test* y_test[9*i+k, :]
		x_test[9*i+k, :]= np.sin(y_test[9*i+k, :])


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
with tf.name_scope('Loss'):
	Training_Loss= tf.reduce_mean(tf.square(tf.abs(output-y_reshape)))
	Valid_Loss= tf.reduce_mean(tf.square(tf.abs(output-y_reshape)))
	tf.summary.scalar('TrainLoss', Training_Loss)


train_step=tf.train.GradientDescentOptimizer(learningRate).minimize(Training_Loss)
saver=tf.train.Saver()


writer=tf.summary.FileWriter('F:/Files/Program/py/TensorFlow/L4_ReLU_CNN_for_NLS/summary', sess.graph)
merged=tf.summary.merge_all()
sess.run(tf.global_variables_initializer())
f=open('F:\Files\Program\py\TensorFlow\L4_ReLU_CNN\Valid_Loss.txt','w')
for i in range(100):
	if i % 5 ==0:
		trainResult=sess.run(merged, feed_dict={x: x_train, y: y_train})
		validResult=sess.run(Valid_Loss, feed_dict={x: x_test, y: y_test})
		f.write('Valid Loss:'+str(validResult)+'\n')
		writer.add_summary(trainResult)
	sess.run(train_step, feed_dict={x: x_train, y: y_train})
f.close()
savePath=saver.save(sess, 'F:\Files\Program\py\TensorFlow\L4_ReLU_CNN_for_NLS\parameters.ckpt')
print('Model params saved in: ', savePath)
print('Training Finished')
