from flask import Flask
from flask import render_template, request

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/login", methods=['post'])
def login():
    username = request.form['username']
    password = request.form['password']
    print(f'입력된 id: {username}, pw: {password}')
    return render_template('index.html')

if __name__ == '__main__':
    app.debug = True
    app.run()