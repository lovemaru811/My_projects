import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import pandas as pd
import numpy as np

class Blood:

    # 새로 입력할 Features 초기화 설정
    def initialize(self, weight, age):
        self._weight = weight
        self._age = age

    # text 파일 읽는 법
    def raw_data(self):
        tf.set_random_seed(777)
        return np.genfromtxt('blood.txt' , skip_header=36)

    def model(self):
        x_data = None
        y_data = None

        tf.global_variables_initializer()
        data = pd.read_csv('./blood.csv')
        data.columns = ['col1']
        data = data.col1.str.split('  ', expand=True)
        data = data.iloc[:, -3:]
        data.columns = ['Weight', 'Age', 'Blood_fat_content']

        xy = np.array(data, dtype=np.float32)
        x_data = xy[: ,  0:2]
        y_data = xy[: , -1:]

        # Neuron 구조화(초기화) , Features 상수화 시킴
        X = tf.placeholder(tf.float32, shape=[None, 2])
        Y = tf.placeholder(tf.float32, shape=[None, 1] )
        W = tf.Variable(tf.random_normal([2,1]) , name='weight')
        b = tf.Variable(tf.random_normal([1]) , name='bias')

        # 1차 선형회귀식 가정 , hypothesis가 yhat 이라고 보면 됨.
        hypothesis = tf.matmul(X , W) + b

        # 비용함수 설정 - 최소제곱 ( SSR )
        loss = tf.reduce_mean(tf.square(hypothesis - Y))

        # 최적화 함수 설정
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0005)

        # 학습 / 저장(Archived)
        train = optimizer.minimize(loss)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for step in range(100000):
                loss_ , hypo_, _ = sess.run([loss, hypothesis, train], {X:x_data , Y:y_data})
                if step % 1000 ==0:
                    print(f' step : {step}, loss : {loss_}')
                    print(f' price : {hypo_}')
            saver = tf.train.Saver()
            saver.save(sess, 'blood.ckpt')

    def service(self):
        X = tf.placeholder(tf.float32, shape=[None, 2])
        W = tf.Variable(tf.random_normal([2, 1]) , name='weight')
        b = tf.Variable(tf.random_normal([1]), name='bias')
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, './blood/blood.ckpt')  # ckpt 까지만 해주면, ,ckpt 확장자 파일 다 들어옴 (≒ ckpt는 학습된 결과)
            data = [[self._weight, self._age],]   # [] 가 리스트 구조,  [[],] 는 텐서구조  [[],] 에서 , 찍은 것에 주의할 것.
            arr = np.array(data, dtype=np.float32)
            dict = sess.run(tf.matmul(X, W) + b, {X: arr[0:2]})
            print('dict 확인 : ' , dict)
        return int(dict[0])

if __name__ == '__main__':
    pass
    # blood = Blood()
    # Blood.model()