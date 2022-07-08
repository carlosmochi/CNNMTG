import json
import os.path
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image


model = keras.models.load_model('model306')
path = os.path.join("dataset", "khm")
with open(path+'.json', 'r') as file:
    class_json = json.load(file)
    class_json = class_json['data']
    class_json.sort(key=lambda x: x['id'])

class_list = os.listdir(path)
class_list.sort()
for card_id in class_list:
    if card_id == '.DS_Store':
        continue
    class_path = os.path.join(path,card_id)
    img_list = os.listdir(class_path)
    for img in img_list:
        if img == '.DS_Store':
            continue
        img_path = os.path.join(class_path, img)
        img_array = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        img_array = keras.preprocessing.image.img_to_array(img_array)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch
        predictions = model.predict(img_array)
        #print(tf.nn.softmax(predictions[0]))
        k = np.argmax(predictions[0])
        score = predictions[0][k]
        print(k, score)
