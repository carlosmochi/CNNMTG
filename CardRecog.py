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


#
# predictions = model.predict(img_array)
# score = tf.nn.softmax(predictions[0])
# k = np.argmax(score)
# print(k, class_names[k])


def testing(data):
    print(data)
    return "yep, its a " + data


def identifyimage(imgb64):
    model = keras.models.load_model('/Users/carlosmochi/PycharmProjects/flaskIntegration/model306')
    path = os.path.join("/Users/carlosmochi/PycharmProjects/flaskIntegration/dataset", "khm")
    # with open(path + '.json', 'r') as file:
    #     class_json = json.load(file)
    #     class_json = class_json['data']
    #     class_json.sort(key=lambda x: x['id'])

    img_list = os.listdir(path)
    img_list.sort()
    oldk = -1
    card0 = np.zeros((457, 626, 3), np.uint8)
    card = card0
    selectcard = "1"

    im_bytes = base64.b64decode(imgb64)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img_array = keras_preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    # print('image hsape: ',img_array.shape)
    predictions = model.predict(img_array)

    k = np.argmax(predictions[0])
    score = predictions[0][k]
    print(k, score)
    if score > 0.9:
        print(k)
        print(img_list[k])
        selectcard = img_list[k]
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

#
# # After the loop release the cap object
# # Destroy all the windows
