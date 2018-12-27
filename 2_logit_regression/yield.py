#本案例其实也是解释了为什么在模型训练时采用batch的方法会更加有效率
#在训练数据十分庞大时，如果只是简单的将数据全部轮训一遍做法很低效，把数据切分会变得有效率
import numpy as np 
def get_batch(x,y,batch):
    n_samples = len(x)
    print("n_samples:",n_samples)
    #n_samples=10,for i in range(3,10,3) 
    #i的值分别是3,6,9，这样实际上只会取到数组[0-9]第10个取不到的
    for i in range(batch,n_samples,batch):
        print("i:",i,"batch:",batch)
        yield x[i-batch:i],y[i-batch:i]
#yield用在函数中，把这函数封装成一个generator(生成器)，在调用for i in fun(param)起作用
ma = np.array([[0,1],[1,2],[2,3],[3,4],[4,3],[5,5],[6,2],[7,4],[8,3],[9,5]])
#ma.shape(10,2)
print("ma:",ma[0:3])
#[[0 1][1 2][2 3]]
mb = np.array([0,1,2,3,4,5,6,7,8,9])
#mb.shape(10,)
for j in range(3):  
    for tx,ty in get_batch(ma,mb,3):
        print("tx:",tx,"ty:",ty)
        print("over.")
print("Finished.",tx,ty)

