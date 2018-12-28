from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import tensorflow as tf 
import numpy as np 

n_features = 2
n_classes = 2
batch_size = 32
#help(plt.scatter)
#采用sklearn生成特征值是2,2分类，一共1000个样本，每个类别一个簇
x,y = datasets.make_classification(n_samples=1000,n_features=n_features,n_redundant=0,n_informative=1,
                                    n_classes=n_classes,n_clusters_per_class=1)
#print(x,y)
#按照7:3分成训练集合和测试集合
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3)
#可以用图表示出来
plt.scatter(train_x[:,0],train_x[:,1], marker='o', c=train_y,
            s=25, edgecolor='k')
#plt.show()

#print("Train----",train_x,train_y)
#yield构建一个batch生成器
def get_batch(x_b,y_b,batch):
    n_samples = len(x_b)
    #print(n_samples)
    for i in range(batch,n_samples,batch):
        #print(i,batch,n_samples)
        yield x_b[i-batch:i],y_b[i-batch:i]

#注意y_input是int型
x_input = tf.placeholder(tf.float32,shape=[None,n_features],name='X_IPNUT')
y_input = tf.placeholder(tf.int32,shape=[None],name='Y_INPUT')
#注意W和b的shape
W = tf.Variable(tf.truncated_normal([n_features,n_classes]),name='W')
b = tf.Variable(tf.zeros([n_classes],name='b'))
#二分类问题采用sigmoid函数，logits.shape(?,2)
logits = tf.sigmoid(tf.matmul(x_input,W)+b)
#这里直接调用了sparse_softmax_cross_entropy函数，包含了先将logits用softmax函数表示
#然后计算交叉熵，logits.shape(n,2) y_input.shape(n,),要先将table转成one-hot结构
loss = tf.losses.sparse_softmax_cross_entropy(labels=y_input,logits=logits)
loss = tf.reduce_mean(loss)
#学习速率，采用Adam算法
learning_rate = 0.01
opitimer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
#计算准确度
predict = tf.arg_max(logits,1,name='predict')
#tf.metrics.accuracy返回值局部变量
acc, acc_op = tf.metrics.accuracy(labels=y_input,predictions=predict)
#训练模型
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for epoch in range(200):
        #batch的形式训练
        for tx,ty in get_batch(train_x,train_y,batch_size):
            loss_value,_,acc_value=sess.run([loss,opitimer,acc_op],feed_dict={x_input:tx,y_input:ty})
        if epoch%10==0:
            print('loss = {}, acc = {}'.format(loss_value,acc_value))
    #测试模型
    acc_value_test = sess.run([acc_op],feed_dict={x_input:test_x ,y_input:test_y}) 
    print('val acc = {}'.format(acc_value_test),"W:",sess.run(W),"b:",sess.run(b)) 
 