# web flask library
import os
from flask import Flask, render_template, url_for, request
from pickle import load

# load model
scaler = load(open('model/standard_scaler.pkl', 'rb'))
model = load(open('model/linear_regression_model.pkl', 'rb'))

app = Flask(__name__)

# Halaman awal


@app.route('/', methods=['POST', 'GET'])
def index():
    sex, height, pred = None, None, None
    if request.method == 'POST':
        sex = request.form['sex']
        height = request.form['height']

    if sex:
        pred_data = [[int(sex), float(height)]]
        pred_data = scaler.transform(pred_data)
        pred = model.predict(pred_data)

    return render_template('index.html', pred=pred)


if __name__ == '__main__':
    app.run(debug=True)
