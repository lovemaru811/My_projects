import sqlite3
class Member:
    def __init__(self):
        self.conn = sqlite3.connect('sqlite.db')

    def create(self):
        query = """
            create table if not exists member(\
                username varchar(10) primary key,
                password varchar(10),
                phone varchar(15),
                regdate date default current_timestamp
            )
        """
        self.conn.execute(query)
        self.conn.commit()

    def insert_many(self):
        data = [
            ('lee', '1', '010-1234-1234'),
            ('kim', '1', '010-1234-5567'),
            ('park', '1', '010-1234-4534')
        ]
        query = """
            insert into member(username, password,phone)
            values(?, ?, ?)
        """
        self.conn.executemany(query, data)
        self.conn.commit()

    def fetch_one(self, username):
        query = f'select * from  member where username like  \'{username}\''
        print('--->'+ query)
        cursor = self.conn.execute(query)
        row = cursor.fetchone()
        print(f'검색 결과: {row}')
        return row

    def fetch_all(self):
        query = f'select * from  member'
        cursor = self.conn.execute(query)
        rows = cursor.fetchall()
        count = 0
        for i in rows:
            count += 1
        print(f'총인원수 : {count}')

    def login(self, username, password):
        query = """
            select * from member
            where username like ?
            and password like ?
        """
        data = [username, password]
        cursor = self.conn.execute(query, data)
        row = cursor.fetchone()
        print(f' 로그인한 회원 정보: {row}')
        return row


