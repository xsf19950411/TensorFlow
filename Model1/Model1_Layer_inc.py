import tensorflow as tf    
import numpy as np   

    
sess= tf.Session()  

#####################生成模型####################
with tf.name_scope('Input'):
	xs= tf.placeholder(tf.float32, [None, 2000], name= 'x_inputs')
	y_= tf.placeholder(tf.float32, [None, 10], name= 'y_labels')

with tf.name_scope('Layer1'):
	W1=tf.Variable(tf.random_normal([2000, 2000], 0, 0.01))
	b1=tf.Variable(tf.zeros([1, 2000]))
	Wx_plus_b1=tf.matmul(xs, W1)+ b1
	Output1=tf.nn.relu(Wx_plus_b1)

with tf.name_scope('Layer2'):
	W2=tf.Variable(tf.random_normal([2000, 2000], 0, 0.01))
	b2=tf.Variable(tf.zeros([1, 2000]))
	Wx_plus_b2=tf.matmul(Output1, W2)+ b2
	Output2=tf.nn.relu(Wx_plus_b2)

with tf.name_scope('Layer3'):
	W3=tf.Variable(tf.random_normal([2000, 10], 0, 0.01))
	b3=tf.Variable(tf.zeros([1, 10]))
	Output3=tf.matmul(Output2, W3)+ b3

with tf.name_scope('Loss'):
	Loss= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=Output3))
	tf.summary.scalar('Loss', Loss)

with tf.name_scope('Train'):
	Train_step= tf.train.GradientDescentOptimizer(0.02).minimize(Loss)

init= tf.global_variables_initializer()
merged= tf.summary.merge_all()
writer= tf.summary.FileWriter("F:/Files/Program/py/TensorFlow/Model1_layer_inc", sess.graph)

##########生成数据###############################
labels= np.linspace(0, 2*3.1415, 10)
x_data= np.zeros((20,2000))
y_data= np.zeros((20,10))
x_test= np.zeros((1000, 2000))
y_test= np.zeros((1000,10))

for i in range(20):       #生成训练数据
	x_data[i, :]= np.sin(2*3.14*0.05*np.linspace(0, 2000, 2000)+ labels[(13*i)%10])
	y_data[i, (13*i)%10]= 1

for i in range(1000):      #生成测试数据
	index= np.random.randint(10)
	x_test[i, :]= np.sin(2*3.14*0.05*np.linspace(0, 2000, 2000)+ labels[index])
	y_test[i, index]= 1


##########模型训练###############################
sess.run(init)
for i in range(1000):
	sess.run(Train_step, feed_dict={xs: x_data, y_: y_data})
	if i%10 ==0:
		result=sess.run(merged, feed_dict={xs: x_data, y_: y_data})
		writer.add_summary(result, i)

#训练完成后，训练数据输出
#print('Trained result when input is training data:\n', sess.run(Output2, feed_dict={xs: x_data}))


##########测试######################
correct_prediction = tf.equal(tf.argmax(Output2, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Test Accuracy:', sess.run(accuracy, feed_dict={xs: x_test,
                                  y_: y_test}))

# 单次测试结果
x_test= np.zeros((1,2000))
x_test[0,:]= np.sin(2*3.14*0.05*np.linspace(0, 2000, 2000)+ labels[2])
print('Test result (label=2):\n', sess.run(Output2, feed_dict={xs: x_test}))
