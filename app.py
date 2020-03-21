
from flask import Flask
from flask import render_template , request , jsonify
import re
from calculator.ai_calculator import Calculator
from cabbage.cabbage import Cabbage
from blood.blood import Blood

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/move/<path>')
def move(path):
    return render_template('{}.html'.format(path))

@app.route('/calculator')
def ui_calculator():
    stmt = request.args.get('stmt', 'NONE')
    if(stmt == 'NONE'):
        print('넘어온 값이 없음')
    else:
        print(f'넘어온 식: {stmt}')
        patt = '[0-9]+'
        op = re.sub(patt, '', stmt)
        nums = stmt.split(op)
        result = 0
        n1 = int(nums[0])
        n2 = int(nums[1])
        if op == '+': result = n1 + n2
        elif op == '-': result = n1 - n2
        elif op == '*': result = n1 * n2
        elif op == '/': result = n1 / n2
    return jsonify(result = result)

@app.route('/ai_calculator', methods=['POST'])
def ai_calculator():
    num1 = request.form['num1']
    num2 = request.form['num2']
    opcode = request.form['opcode']

    result = Calculator.service(num1, num2, opcode)
    render_params = {}
    render_params['result'] = int(result)
    return render_template('ai_calculator.html', **render_params)

@app.route('/cabbage', methods=['POST'])
def cabbage():
    avgTemp = request.form['avg_temp']
    minTemp = request.form['min_temp']
    maxTemp = request.form['max_temp']
    rainFall = request.form['rain_fall']

    # self 로 반영되어있을때 이렇게 부어주는 건가?
    cabbage = Cabbage()
    cabbage.initialize(avgTemp, minTemp, maxTemp, rainFall)
    result = cabbage.service()

    render_params = {}
    render_params['result'] = result
    return render_template('cabbage.html', **render_params)

@app.route('/blood', methods=['POST'])
def blood():
    weight = request.form['weight']
    age = request.form['age']

    blood = Blood()
    blood.initialize(weight, age)
    result = blood.service()

    render_params = {}
    render_params['result'] = int(result)
    return render_template('blood.html', **render_params)

if __name__ == '__main__':
    app.run()