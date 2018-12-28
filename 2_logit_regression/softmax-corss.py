import tensorflow as tf 
import numpy as np 
#模拟softmax和交叉熵
table = np.array([[1,0],[0,1],[0,0],[0,1],[1,0]])
input_a = np.array([[0.1,0.3],[0.3,0.8],[0.5,0.4],[0.6,0.1],[0.8,0.2]])
#分步进行计算
sm_x = tf.nn.softmax(input_a)
cross_entropy = -tf.reduce_sum(table*tf.log(sm_x))

#sparse_softmax和softmax 交叉熵函数最大的区别在与输入的labels是否是one-hot结构
#如果是选择softmax，如果不是选择sparse_softmax，它会先转换成one-hot
input_b = np.array([0,1,0,0,1])
input_a_one_hot = tf.one_hot(input_b,2)
loss_sf = tf.losses.softmax_cross_entropy(onehot_labels=input_a_one_hot,logits=input_a)
loss_sp = tf.losses.sparse_softmax_cross_entropy(labels=input_b,logits=input_a)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    sm_xx,cross_xx,loss_sp_xx,loss_sf_xx=sess.run([sm_x,cross_entropy,loss_sp,loss_sf])
    print("sm_xx:",sm_xx,"cross_xx:",cross_xx,"loss_sp_xx:",loss_sp_xx,"loss_sf_xx",loss_sf_xx)
