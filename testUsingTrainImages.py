import json
import os.path

import numpy as np
from matplotlib import pyplot as plt
from tensorflow import expand_dims
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.models import load_model

IMG_WIDTH = 224
IMG_HEIGHT = 224

model = load_model('model306')
path = os.path.join("dataset", "khm")
class_list = os.listdir(path)

with open(path + '.json', 'r') as file:
    class_json = json.load(file)
    class_json = class_json['data']
    class_json.sort(key=lambda x: x['id'])
    class_json = list(filter(lambda x: x['id'] in class_list, class_json))

for idx, card in enumerate(class_json):
    card_id = card['id']
    class_path = os.path.join(path, card['id'])
    img_list = os.listdir(class_path)

    for file in img_list:
        if file[0] == '.':  # hidden files
            continue
        img_path = os.path.join(class_path, file)
        img_card = load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
        img_array = img_to_array(img_card)
        img_array = expand_dims(img_array, 0)  # Create a batch

        predictions = model.predict(img_array)
        k = np.argmax(predictions[0])
        score = predictions[0][k] * 100

        if idx == k:
            print('\033[94m', f'{k:3}', f'{score:6.2f}%', card['id'], card['name'], file, '\033[0m')
            continue

        # print(j, class_json[j]['id'], class_json[j]['name'])
        # print(k, class_json[k]['id'], class_json[k]['name'], score)
        print('\033[91m', f'{k:3}', f'{score:6.2f}%', card['id'], card['name'], file, '\033[0m')

        predit_card_id = class_json[k]['id']
        predit_path = os.path.join(path, predit_card_id)
        ids = [card_id, predit_card_id]
        rows = [class_path, predit_path]
        cols = ['art_front.jpg', 'art_back.jpg']

        fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))
        for i, xpath in enumerate(rows):
            for j, xfile in enumerate(cols):
                xpathfile = os.path.join(xpath, xfile)
                try:
                    image = load_img(xpathfile, target_size=(IMG_WIDTH, IMG_HEIGHT))
                    axarr[j, i].imshow(image)
                    axarr[j, i].set_title(ids[i])
                except:
                    pass
        plt.show()


