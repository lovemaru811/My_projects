import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd
import numpy as np
"""
year,avgTemp,minTemp,maxTemp,rainFall,avgPrice
년도, 평균기온, 최저온도, 최고온도, 강수량, 평균가격
"""
class Cabbage:
    def __init__(self):
        pass

    def new_model(self):
        model = tf.global_variables_initializer()
        data = pd.read_csv('./data/price_data.csv', sep=',')
        xy = np.array(data, dtype=np.float32)
        x_data = xy[:,1:-1] # 가격결정요인 : 평균기온, 최저온도, 최고온도, 강수량
        y_data = xy[:, [-1]] # 가격
        X = tf.placeholder(tf.float32, shape=[None, 4])
        Y = tf.placeholder(tf.float32, shape=[None, 1])
        W = tf.Variable(tf.random_normal([4,1]), name='weight')
        b = tf.Variable(tf.random_normal([1]), name='bias')
        hypothesis = tf.matmul(X, W) + b
        cost = tf.reduce_mean(tf.square(hypothesis - Y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000005)
        train = optimizer.minimize(cost)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for step in range(100000):
            cost_, hypo_, _ = sess.run([cost, hypothesis, train],
                                       {X: x_data,
                                        Y: y_data})
            if step % 500 == 0:
                print(f'{step}번째 손실비용 {cost_}')
                print(f' 배추가격 : {hypo_[0]}')

        saver = tf.train.Saver()
        saver.save(sess, 'saved_model/machine.ckpt')
        print('저장완료')





