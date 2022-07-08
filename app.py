from flask import Flask, render_template, url_for, request, redirect, jsonify

import CardRecog
import json


app = Flask(__name__)
lastCard = "failed"

@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/readImage', methods=['GET', 'POST'])
def testing():
    global lastCard
    response = lastCard
    print('LastCard: ' + lastCard)
    if request.method == 'POST':
        print("got to POST Flask")
        newdata = request.get_json(True)
        lastCard = newdata['data'][22:]
    if request.method == 'GET':
        print("got to GET Flask")
        if lastCard != "failed":
            response = CardRecog.identifyimage(lastCard)
    return response
    # return testFunction.pythontrial()


if __name__ == '__main__':
    app.run()
