import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#模拟训练集合总数是m，变量只有一个的训练集合
train_x = np.linspace(1.,2.,20)
#print(train_x)
#print(*train_x.shape)
train_y = np.array([20.,23.,21.,35.,40.,35.,38.,44.,50.,47.,56.,55.,60.,56.,65.,63.,68.,65.,74.,79.])
#print(train_y)
#print(*train_y.shape)

#X/Y占位符，在tensorflow中已知的变量，使用占位符表示，这样就可以建立一个完整的graph
#如果不明白tensorflow中的API可以使用help命令
#help(tf.placeholder)
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#定义未知变量W/b，这也是想要得到的参数
W = tf.Variable(tf.random_normal([1]),name="weight")
b = tf.Variable(tf.random_normal([1]),name="bias")

#预测值Y_=W*X+b是最终想要求得的值
Y_ = tf.add(tf.multiply(X,W),b)
#help(tf.reduce_mean)

#损失函数，加入tensorboard可以观察损失函数的下降趋势
with tf.name_scope('cost'):
    cost=tf.reduce_mean(tf.square(Y_-Y))
    tf.summary.scalar('cost',cost)
#定义学习速率
learning_rate = 0.01
#tensorflow中定义的梯度下降法，通过向后传播不断迭代更新参数W和b，在损失函数最小时求出W和b
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#训练步骤
training_epochs = 100
display_step = 2
#保存训练模型
saver = tf.train.Saver()
savedir = "D:\\Tensorflow\\ronny-git-master\\tf-tutorials\\1_linear_regression\\"
#保存tensorboard graph
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(savedir+"tb_logs\\",tf.Session().graph)
#初始化所有变量
init = tf.global_variables_initializer()
#创建session所有的tensorflow的graph都必须在一个session中运行
with tf.Session() as sess: 
    sess.run(init)
    for epoch in range(training_epochs):
        #这里使用zip将train_x, train_y打包一一对应，其实与batch类似
        for (x, y) in zip(train_x, train_y):
            #run optimizer，feed_dict给占位符是X和Y喂数据，运行优化器
            sess.run(optimizer, feed_dict={X: x, Y: y})
        if epoch % display_step == 0:
            #run tensorboard graph和损失函数    
            summary,loss = sess.run([merged,cost],feed_dict={X : train_x,Y : train_y})
            #tensorboard summary
            writer.add_summary(summary, epoch)
            #打印出参数
            print("Epoch:", epoch, "cost=", loss,"W=", sess.run(W), "b=", sess.run(b))
    #到这里就完成了线性回归模型的训练
    print (" Finished!")
    #保存模型
    saver.save(sess, savedir+"logs\\linermodel.cpkt")
    print ("cost=", sess.run(cost, feed_dict={X: train_x, Y: train_y}), "W=", sess.run(W), "b=", sess.run(b))
    #绘图，原始数据和线性回归模型
    plt.plot(train_x, train_y, 'ro', label='Original data')
    plt.plot(train_x, sess.run(W) * train_x + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
#给出一个X值，用刚刚训练完成的模型得出预测值Y_
with tf.Session() as sess_logs:
    sess_logs.run(tf.global_variables_initializer())
    saver.restore(sess_logs,savedir+"logs\\linermodel.cpkt")
    print("x=1.4,y=",sess_logs.run(Y_,feed_dict={X:1.4}))
