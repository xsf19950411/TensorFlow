import tensorflow as tf

x=tf.constant([[1,2,3],[4,5,6]])

x_r=tf.reshape(x, [-1, 1, 3, 1])

sess=tf.Session()
print(x, sess.run(x), '\n', x_r, sess.run(x_r))