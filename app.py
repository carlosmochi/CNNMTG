from flask import Flask, render_template, url_for, request, redirect, jsonify

import CardRecog
import json


app = Flask(__name__)
#Salva a última carta lida pelo sistema "Deve ser alterado para permitir múltiplas pesquisas ao mesmo tempo"
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
        #Quando receber um POST, extrai o ID da carta do pedido para o reconhecimento
        newdata = request.get_json(True)
        lastCard = newdata['data'][22:]
    if request.method == 'GET':
        print("got to GET Flask")
        if lastCard != "failed":
            #Encvia o ID recebido em POST para o reconhecimento e devolve a resposta à página 
            response = CardRecog.identifyimage(lastCard)
    return response


if __name__ == '__main__':
    app.run()
