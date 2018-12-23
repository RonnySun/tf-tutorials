import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

train_x = np.linspace(1.,2.,20)
#print(train_x)
#print(*train_x.shape)
train_y = np.array([20.,23.,21.,35.,40.,35.,38.,44.,50.,47.,56.,55.,60.,56.,65.,63.,68.,65.,74.,79.])
#print(train_y)
#print(*train_y.shape)

#help(tf.placeholder)
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1]),name="weight")
b = tf.Variable(tf.random_normal([1]),name="bias")

Y_ = tf.add(tf.multiply(X,W),b)
#help(tf.reduce_mean)

with tf.name_scope('cost'):
    cost=tf.reduce_mean(tf.square(Y_-Y))
    tf.summary.scalar('cost',cost)

learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

training_epochs = 100
display_step = 2
plotdata = { "batchsize":[], "loss":[] }
saver = tf.train.Saver()
savedir = "D:\\Tensorflow\\tf-tutorials\\1_linear_regression\\"

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(savedir+"tb_logs\\",tf.Session().graph)

init = tf.global_variables_initializer()
with tf.Session() as sess: 
    sess.run(init)
    for epoch in range(training_epochs):
        for (x, y) in zip(train_x, train_y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
        if epoch % display_step == 0:    
            summary,loss = sess.run([merged,cost],feed_dict={X : train_x,Y : train_y})
            writer.add_summary(summary, epoch)
            print("Epoch:", epoch, "cost=", loss,"W=", sess.run(W), "b=", sess.run(b))
    
    print (" Finished!")
    saver.save(sess, savedir+"logs\\linermodel.cpkt")
    print ("cost=", sess.run(cost, feed_dict={X: train_x, Y: train_y}), "W=", sess.run(W), "b=", sess.run(b))
    plt.plot(train_x, train_y, 'ro', label='Original data')
    plt.plot(train_x, sess.run(W) * train_x + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

with tf.Session() as sess_logs:
    sess_logs.run(tf.global_variables_initializer())
    saver.restore(sess_logs,savedir+"logs\\linermodel.cpkt")
    print("x=1.4,y=",sess_logs.run(Y_,feed_dict={X:1.4}))
