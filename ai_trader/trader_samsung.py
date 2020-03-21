import pandas as pd
import plotly.offline as offline
import plotly.graph_objs as go
from datetime import datetime, timedelta
import time
import re
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np


class Trader:
    def __init__(self):
        self.code_df = pd.DataFrame({'name':[], 'code':[]})

    def crawling(self):
        ts = time.time()
        """ 라이브러리 호출 """
        import urllib
        from bs4 import BeautifulSoup
        import requests

        """ 회사코드 및 조회기간 설정 """
        symbol = '005930'
        startTime = (datetime.today() - timedelta(days=1)).strftime('%Y%m%d')
        count = str(1000)

        """ url 설정 """
        url = 'https://fchart.stock.naver.com/sise.nhn?symbol={}&timeframe=day&startTime={}&count={}&requestType=2'.format(
            symbol, startTime, count)

        """ 크롤링 & 전처리 """
        r = requests.get(url)
        html = r.content
        soup = BeautifulSoup(html, 'html.parser')
        tr = soup.find_all('item')
        cols = ['일자', '시가', '고가', '저가', '종가', '거래량']
        list = []
        for i in range(0, len(soup.find_all('item'))):
            list.append(re.search(r'"(.*)"', str(tr[i])).group(1).split('|'))
        df = pd.DataFrame(list, columns=cols)
        df['일자'] = pd.to_datetime(df['일자'].str[:4] + '-' + df['일자'].str[4:6] + '-' + df['일자'].str[6:])
        df.set_index(df['일자'], inplace=True)
        df = df.drop(columns='일자')
        print('작동소요시간 :', round(time.time() - ts, 1), '초')
        df.to_csv('./stock_price_samsung.csv' , encoding='utf-8')

    def model(self):
        tf.global_variables_initializer()  # 맨 먼저 초기화
        data = pd.read_csv('./stock_price_samsung.csv')  # 데이터 로딩
        data = data.iloc[:, 1:6]
        # data = data.iloc[:, [4]]
        xy = np.array(data, dtype=np.float32)
        x_data = xy[:, 1:-1]
        y_data = xy[:, [3]]

        print('type:{}'.format(type(xy)))
        print('shape:{}, dimension:{}, dtype:{}'.format(xy.shape, xy.ndim, xy.dtype))
        print("Array's Data:\n", xy)

        # Neuron 구조화(초기화) , Features 상수화 시킴
        X = tf.placeholder(tf.float32, shape=[None, 4])  # 머신러닝 입장에서는 W 를 계속 찾는 거라, X 의 Features 는 상수로 역할함
        Y = tf.placeholder(tf.float32, shape=[None, 1])
        W = tf.Variable(tf.random_normal([4, 1]), name='weight')  # 정규분포상의 random 값을 적용. Features 를 4개 썻으니, [4,1] 형이 됨
        b = tf.Variable(tf.random_normal([1]), name='bias')  # 정규분포상의 random 값을 적용. 절편은 1개이니, [1] 형이 됨

        # 1차 선형회귀식 가정 , hypothesis가 yhat 이라고 보면 됨.
        hypothesis = tf.matmul(X, W) + b

        # 비용함수 설정 - 최소제곱 ( SSR )
        loss = tf.reduce_mean(tf.square(hypothesis - Y))

        # 최적화 함수 설정
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0005)

        # 학습 / 저장(archived)
        train = optimizer.minimize(loss)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())  # 학습할 때마다, 이전 값 초기화
            for step in range(100000):
                loss_, hypo_, _ = sess.run([loss, hypothesis, train], {X: x_data, Y: y_data})
                if step % 1000 == 0:
                    print(f' step : {step}, loss : {loss_}')
                    print(f' price : {hypo_}')
            saver = tf.train.Saver()
            saver.save(sess, 'trader.ckpt')  # 매번 학습한다면 비효율 -> 예측결과를 ckpt 형식으로 저장






    def krx_crawl(self):
        self.code_df = pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13',header=0)[0]
        self.code_df.종목코드 = self.code_df.종목코드.map('{:06d}'.format())
        self.code_df = self.code_df[['회사명','종목코드']]
        self.code_df = self.code_df.rename(columns={'회사명':'name', '종목코드':'code'})
    def code_df_head(self):
        print(self.code_df.head())
    def get_url(self, item_name, code_df):
        code = code_df.query("name=='{}'".format(item_name))['code'].to_string(index=False)
        url = 'http://finance.naver.com/item/sise_day.nhn?code={code}'.format(code='005930') # 'code'로 대체
        print('요청 URL = {}'.format(url))
        return url
    def test(self, code):
        # item_name = '삼성전자'
        # url = self.get_url(item_name, self.code_df)
        df = pd.DataFrame()
        for page in range(1, 21):
            pg_url = 'https://finance.naver.com/item/sise_day.nhn?code={code}&page={page}'.format(code=code, page=page)
            df = df.append(pd.read_html(pg_url, header=0)[0], ignore_index=True)
            df.dropna(inplace = True) # na 결측값 있는 행 제거
        return df
    def rename_item_name(self, param):
        df = param.rename(columns = {'날짜':'date', '종가':'close','전일비':'diff',
        '시가':'open','고가':'high','저가':'low','거래량':'volumn'})
        df[['close','diff','open','high','low','volumn']] =\
        df[['close','diff','open','high','low','volumn']].astype(int)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by=['date'], ascending=True)
        return df

if __name__ == '__main__':
    def print_menu():
        print('0. EXIT\n'
        '1. 종목헤드\n'
        '2. 종목컬럼명 보기\n'
        '3. 전처리결과 보기\n'
        '4. 종목 크롤링 및 결과저장\n'
        '5. 모델링')
        return input('CHOOSE ONE \n')

    m = Trader()
    while 1:
        menu = print_menu()
        print('MENU %s \n' % menu)
        if menu == '0':
            break
        elif menu == '1':
            m.code_df_head()
        elif menu == '2':
            print(m.test('005930'))
        elif menu == '3':
            print(m.rename_item_name(m.test('005930')))
        elif menu == '4':
            m.crawling()
        elif menu == '5':
            m.model()