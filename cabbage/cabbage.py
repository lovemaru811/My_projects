import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import pandas as pd
import numpy as np

class Cabbage:
    def model(self):
        tf.global_variables_initializer()  # 맨 먼저 초기화
        data = pd.read_csv('./cabbage_price.csv')  # 데이터 로딩

        """
        ### Features ### 
        avgTemp,
        minTemp,
        maxTemp,
        rainFall,
        
        ### Label ###
        avgPrice
        """
        xy = np.array(data, dtype=np.float32)
        x_data = xy[: ,  1:-1]
        y_data = xy[: , -1:]

        # Neuron 구조화(초기화) , Features 상수화 시킴
        X = tf.placeholder(tf.float32, shape=[None, 4])   # 머신러닝 입장에서는 W 를 계속 찾는 거라, X 의 Features 는 상수로 역할함
        Y = tf.placeholder(tf.float32, shape=[None, 1])
        W = tf.Variable(tf.random_normal([4, 1]) , name='weight')   # 정규분포상의 random 값을 적용. Features 를 4개 썻으니, [4,1] 형이 됨
        b = tf.Variable(tf.random_normal([1]) , name='bias')  # 정규분포상의 random 값을 적용. 절편은 1개이니, [1] 형이 됨


        # 1차 선형회귀식 가정 , hypothesis가 yhat 이라고 보면 됨.
        hypothesis = tf.matmul(X, W) + b

        # 비용함수 설정 - 최소제곱 ( SSR )
        loss = tf.reduce_mean(tf.square(hypothesis - Y))

        # 최적화 함수 설정
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0005)

        # 학습 / 저장(archived)
        train = optimizer.minimize(loss)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer()) # 학습할 때마다, 이전 값 초기화
            for step in range(100000):
                loss_ , hypo_ , _ = sess.run([loss, hypothesis, train] , {X:x_data , Y:y_data})
                if step % 1000 ==0:
                    print(f' step : {step}, loss : {loss_}')
                    print(f' price : {hypo_}')
            saver = tf.train.Saver()
            saver.save(sess, 'cabbage.ckpt')  # 매번 학습한다면 비효율 -> 예측결과를 ckpt 형식으로 저장

    # 새로 예측 수행할 Features에 해당하는 값들 주입
    def initialize(self, avgTemp, minTemp, maxTemp, rainFall):
        self.avgTemp = avgTemp
        self.minTemp = minTemp
        self.maxTemp = maxTemp
        self.rainFall = rainFall

    def service(self):
        X = tf.placeholder(tf.float32, shape=[None, 4])
        W = tf.Variable(tf.random_normal([4, 1]) , name='weight')
        b = tf.Variable(tf.random_normal([1]), name='bias')
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, './cabbage/cabbage.ckpt')  # ckpt 까지만 해주면, ,ckpt 확장자 파일 다 들어옴 (≒ ckpt는 학습된 결과)
            data = [[self.avgTemp, self.minTemp, self.maxTemp, self.rainFall],]   # [] 가 리스트 구조,  [[],] 는 텐서구조  [[],] 에서 , 찍은 것에 주의할 것.
            arr = np.array(data, dtype=np.float32)
            dict = sess.run(tf.matmul(X, W) + b, {X: arr[0:4]})
            print('dict 확인 : ' , dict)
        return int(dict[0])

if __name__ == '__main__':
    pass
    # cabbage = Cabbage()
    # cabbage.model()
    # cabbage.service()
