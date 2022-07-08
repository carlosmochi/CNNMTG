import json
import os.path
import cv2
import keras_preprocessing.image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image


model = keras.models.load_model('/Users/carlosmochi/PycharmProjects/flaskIntegration/model306')
path = os.path.join("dataset", "khm")
with open(path+'.json', 'r') as file:
    class_json = json.load(file)
    class_json = class_json['data']
    class_json.sort(key=lambda x: x['id'])

img_list = os.listdir(path)
img_list.sort()
#
# predictions = model.predict(img_array)
# score = tf.nn.softmax(predictions[0])
# k = np.argmax(score)
# print(k, class_names[k])


vid = cv2.VideoCapture(0)
oldk = -1
card0 = np.zeros((457, 626, 3), np.uint8)
card = card0

while True:
    def catchId():
        return img_list[k]
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    frame = frame[0:457, 0:626]
    img = cv2.resize(frame, (224, 224))
    img_array = keras_preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    #print('image hsape: ',img_array.shape)
    predictions = model.predict(img_array)

    k = np.argmax(predictions[0])
    score = predictions[0][k]
    print(k, score)
    if score > 0.9:
        print(k)
        print(img_list[k])
        if k != oldk:
            if img_list[k] == '.DS_Store':
                continue
            card = os.path.join(path, img_list[k], 'art_front.jpg')
            card = cv2.imread(card)
            card = cv2.resize(card, (626, 457))
            print(k, class_json[k]['name'])
            oldk = k
    elif score < 0.6:
        card = card0
        oldk = -1

    # Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.imshow('card', card)
    cv2.imshow('view', img)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
