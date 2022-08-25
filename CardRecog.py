import asyncio
import json
import os.path
import base64
import time

import cv2
import keras_preprocessing.image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image

#Script principal de reconhecimento de imagem


def identifyimage(imgb64):
    #Leitura do modelo e da lista de cartas já conhecidas pela máquina
    #Alterar os PATHs para que eles coincidam com o de sua máquina
    model = keras.models.load_model('/Users/"seu_computador"/PycharmProjects/flaskIntegration/model306')
    path = os.path.join("/Users/"seu_computador"/PycharmProjects/flaskIntegration/dataset", "khm")
    
    #img_list possui os caminhos das imagens a serem mostradas na conclusão do código
    img_list = os.listdir(path)
    img_list.sort()
    oldk = -1
    card0 = np.zeros((457, 626, 3), np.uint8)
    card = card0
    selectcard = "1"
    
    #Leitura e tratamento da imagem recebida antes do sistema analiza-la
    im_bytes = base64.b64decode(imgb64)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img_array = keras_preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    #Envia a imagem para a rede a adivinhar a carta
    predictions = model.predict(img_array)

    k = np.argmax(predictions[0])
    score = predictions[0][k]
    print(k, score)
    #Prediçoes com mais de 90% de certeza tem seu ID salvo para ser enviado ao servidor
    if score > 0.9:
        print(k)
        print(img_list[k])
        selectcard = img_list[k]
    #Quando a predição for diferente da anterior, é extraida a nova imagem usando os caminhos salvos em img_list    
        if k != oldk:
            # if img_list[k] == '.DS_Store':
            card = os.path.join(path, img_list[k], 'card_front.jpg')
            selectcard = img_list[k]
            # print(k, class_json[k]['name'])
            # oldk = k
    elif score < 0.6:
        card = card0
        oldk = -1
    return selectcard
