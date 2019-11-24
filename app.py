import flask
from flask import render_template
import pickle
import pandas as pd
import sklearn

app = flask.Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return render_template('main.html' )

    if flask.request.method == 'POST':
        temp = 1
        with open('model_titanic.pkl', 'rb') as fh:
            loaded_model = pickle.load(fh)
        pclass = int(flask.request.form['pclass'])
        sex = 1 if (flask.request.form['sex']) == '1' else 0
        age = float(flask.request.form['age'])
        sibsp = int(flask.request.form['sibsp'])
        parch = int(flask.request.form['parch'])
        fare = float(flask.request.form['fare'])
        emb_c = 0
        emb_q = 0
        emb_s = 0
        embarked = flask.request.form['embarked']
        if embarked == 'C':
            emb_c = 1
        if embarked == 'Q':
            emb_q = 1
        if embarked == 'S':
            emb_s = 1
        df = pd.DataFrame([[3, 1, 34.5, 0, 0, 7.8292, 0, 1, 0]], columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Emb_C', 'Emb_Q', 'Emb_S'])

        # data = [pclass, sex, age, sibsp, parch, fare, emb_c, emb_q, emb_s]
        # df = pd.DataFrame([data], columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Emb_C', 'Emb_Q', 'Emb_S'])
        print(df)

        temp = loaded_model.predict(df)

        return render_template('main.html', result='Вы выжили!!!' if temp[0] else 'Вы утонули...')


if __name__ == '__main__':
    app.run()
