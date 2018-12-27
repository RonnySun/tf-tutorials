from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import tensorflow as tf 
import numpy as np 

n_features = 2
n_classes = 2
batch_size = 32
#help(plt.scatter)
x,y = datasets.make_classification(n_samples=500,n_features=n_features,n_redundant=0,n_informative=1,
                                    n_classes=n_classes,n_clusters_per_class=1)
#print(x,y)
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3)
plt.scatter(train_x[:,0],train_x[:,1], marker='o', c=train_y,
            s=25, edgecolor='k')
#plt.show()

#print("Train----",train_x,train_y)

def get_batch(x_b,y_b,batch):
    n_samples = len(x_b)
    print(n_samples)
    for i in range(batch,n_samples,batch):
        print(i,batch,n_samples)
        yield x_b[i-batch:i],y_b[i-batch:i]

x_input = tf.placeholder(dtype=tf.float32,shape=[None,n_features],name='X_IPNUT')
y_input = tf.placeholder(dtype=tf.int32,shape=[None],name='Y_INPUT')

W = tf.Variable(tf.truncated_normal([n_features,n_classes]),name='W')
b = tf.Variable(tf.zeros([n_classes],name='b'))

logits = tf.sigmoid(tf.matmul(x_input,W)+b)
loss = tf.losses.sparse_softmax_cross_entropy(labels=y_input,logits=logits)
loss = tf.reduce_mean(loss)
learning_rate = 0.01
opitimer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    step = 0
    for epoch in range(100):
        for tx,ty in get_batch(train_x,train_y,batch_size):
            #print(tx,ty)
            step +=1