import tensorflow as tf 
import numpy as np  
#初始化训练数据，2维数据结构，三个变量，训练集一共有6组
#这里重点是把所有数据的shape搞对，不然在进行矢量乘法时会出问题
x_org = np.random.rand(6,3)
W_org = np.array([[1],[2],[3]])
b_org = 4
y_org = np.add(np.matmul(x_org,W_org),b_org)
#Y=W1*X1+W2*X2+W3*X3+b
#print(x_org,W_org,y_org)
#这里主要是统一所有变量的数据类型
tf.cast(x_org,tf.float32)
tf.cast(W_org,tf.float32)
tf.cast(x_org,tf.float32)
tf.cast(b_org,tf.float32)
#占位符，最好习惯吧shape表示出来，养成好习惯
X = tf.placeholder(tf.float32,[None,3])
Y = tf.placeholder(tf.float32,[None,1])
#初始化变量，关键是注意shape
W = tf.Variable(tf.random_normal([3,1],name="Weight"))
b = tf.Variable(tf.random_normal([1]),name="bias")
#定义预测值、损失函数、梯度下降算法
Y_pred = tf.add(tf.matmul(X,W),b)
cost = tf.reduce_mean(tf.square(Y_pred-Y))
learning_rate = 0.1#数据比较少，所以学习速率这里选的是0.1
opimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()
#训练1000次,最后的训练结果就非常接近W_org和b_org
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        _,loss=sess.run([opimizer,cost],feed_dict={X:x_org,Y:y_org})
        if(i%100==0):
            print("Cost:",loss,"W:",sess.run(W),"b:",sess.run(b))
    print("Finished")
    print("Cost:",loss,"W:",sess.run(W),"b:",sess.run(b))
#Cost: 6.12786e-08 W: [[ 1.00071549][ 2.00028825][ 3.0014329 ]] b: [ 3.99881554]