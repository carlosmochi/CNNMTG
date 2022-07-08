import json
import random

import numpy as np
import requests
import os

import tensorflow as tf
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import cv2

# CONFIG
MAGIC_SET = "khm"
IMG_WIDTH = 224
IMG_HEIGHT = 224
MODEL_NAME = "model306"
NEW_MODEL = False
DOWNLOAD_DATASET = False
DATASET_PATH = "dataset"

def create_model(img_width, img_height, num_classes, model_name='modelNoName'):

    model = tf.keras.Sequential()
    model.add(layers.InputLayer(input_shape=(img_width, img_height, 3)))
    model.add(layers.experimental.preprocessing.Rescaling(1. / 255))
    model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1,1),  activation='relu'))
    model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), strides=1,  activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1,  activation='relu'))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1,  activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1,  activation='relu'))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1,  activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=128, kernel_size=(2, 2), strides=1, activation='relu'))
    model.add(layers.Conv2D(filters=128, kernel_size=(2, 2), strides=1, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
#Reteste em andamento 20/01/2020

# Modelo obteve 100% de acerto nos testes de validação, porém errou todos os testes das classes main
# 7 classes com uma imagem cada; ImageDataGenerator alterado para testar claridade e espelhamento horizontal
# 19/01/2022
            #layers.Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu'),
            #layers.MaxPooling2D(pool_size=(2, 2)),
            #layers.Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu'),
            #layers.Conv2D(filters=32, kernel_size=(2, 2), strides=1, padding='same', activation='relu'),
            #layers.MaxPooling2D(pool_size=(2, 2)),
            #layers.Conv2D(filters=64, kernel_size=(2, 2), strides=1, padding='same', activation='relu'),
            #layers.Conv2D(filters=64, kernel_size=(2, 2), strides=1, padding='same', activation='relu'),
            #layers.MaxPooling2D(pool_size=(2, 2)),
            #layers.Conv2D(filters=128, kernel_size=(2, 2), strides=1, padding='same', activation='relu'),
            #layers.Flatten(),
            #layers.Dense(2000, activation='relu'),
            #layers.Dropout(0.5),
            #layers.Dense(1000, activation='relu'),
            #layers.Dropout(0.5),
            #layers.Dense(343, activation='relu'),
            #layers.Dense(7, activation='softmax')
# Modelo com sucesso no teste com imagens no computador, 0% de acerto com imagens da camera
# 7 classes, com uma imagem por classe
# 17/01/2022
# Modelo baseado nos testes de 13/01/2022
            #layers.MaxPooling2D(pool_size=(2,2)),
            #layers.Conv2D(filters=16, kernel_size=(2, 2), strides=(1,1), padding='same', activation='relu'),
            #layers.Conv2D(filters=32, kernel_size=(2, 2), strides=1, padding='same', activation='relu'),
            #layers.MaxPooling2D(pool_size=(2, 2)),
            #layers.Conv2D(filters=64, kernel_size=(2, 2), strides=1, padding='same', activation='relu'),
            #layers.Conv2D(filters=64, kernel_size=(2, 2), strides=1, padding='same', activation='relu'),
            #layers.MaxPooling2D(pool_size=(2, 2)),
            #layers.Conv2D(filters=128, kernel_size=(2, 2), strides=1, padding='same', activation='relu'),
            #layers.Flatten(),
            #layers.Dense(2000, activation='relu'),
            #layers.Dropout(0.5),
            #layers.Dense(343, activation='relu'),
            #layers.Dense(7, activation='softmax')
# Já testado, resultados foram horríveis, 0% de acurácia em testes de validação com 1.9 de loss constante
# 7 classes, com uma imagem por classe
# 14/01/2022
            #layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu'),
            #layers.MaxPooling2D(pool_size=(2,2)),
            #layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1,1), padding='same', activation='relu'),
            #layers.MaxPooling2D(pool_size=(2, 2)),
            #layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            #layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            #layers.MaxPooling2D(pool_size=(2, 2)),
            #layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            #layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            #layers.MaxPooling2D(pool_size=(2, 2)),
            #layers.Flatten(),
            #layers.Dense(3276, activation='relu'),
            #layers.Dropout(0.5),
            #layers.Dense(3276, activation='relu'),
            #layers.Dropout(0.6),
            #layers.Dense(3276, activation='relu'),
            #layers.Dense(7, activation='softmax')

# Já testado, resultados não foram bons, após 5 treinos de 50 epocas, errou quase todas as predições
# 7 classes, com uma imagem por classe
# 13/01/2022
# https://towardsdatascience.com/classify-butterfly-images-with-deep-learning-in-keras-b3101fe0f98
            #layers.Conv2D(fiylters=32, kernel_size=(2, 2), input_shape=(img_width, img_height, 3), strides=(1,1),
           #               padding='same', activation='relu'),
           # layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            #layers.Conv2D(filters=32, kernel_size=(2, 2), strides=(1,1), padding='same', activation='relu'),
            #layers.Conv2D(filters=64, kernel_size=(2, 2), strides=1, padding='same', activation='relu'),
            #layers.MaxPooling2D(pool_size=(2, 2)),
            #layers.Conv2D(filters=64, kernel_size=(2, 2), strides=1, padding='same', activation='relu'),
            #layers.Conv2D(filters=128, kernel_size=(2, 2), strides=1, padding='same', activation='relu'),
            #layers.MaxPooling2D(pool_size=(2, 2)),
            #layers.Conv2D(filters=128, kernel_size=(2, 2), strides=1, padding='same', activation='relu'),
            #layers.MaxPooling2D(pool_size=(2,2)),
            #layers.Flatten(),
            #layers.Dense(3276, activation='relu'),
            #layers.Dropout(0.5),
            #layers.Dense(3276, activation='relu'),
            #layers.Dense(7, activation='softmax')
    # Compile the model
    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=tf.keras.metrics.SparseCategoricalAccuracy(
                    name='sparse_categorical_accuracy', dtype=None)
                  )
    print("compile OK")
    model.save(model_name)
    print("model save on %s" % model_name)
    return model


def load_dataset(path, img_width, img_height, batch_size=1, img_data_gen=None):
    # DATASET (TRAIN, VALIDATION), make on the nail, lol
    if img_data_gen is None:
        train_datagen = ImageDataGenerator(
            brightness_range=[0.3, 2.0],
            #rotation_range=45,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=False,
            preprocessing_function=add_noise,
            fill_mode='constant')
    else:
        train_datagen = img_data_gen
    test_datagen = ImageDataGenerator()
    print("ImageDataGenerator OK")

    my_train_ds = train_datagen.flow_from_directory(path,
                                                    #save_to_dir='dataset/gen', save_format='jpg', save_prefix='gen',
                                                    batch_size=batch_size,
                                                    target_size=(img_width, img_height),
                                                    class_mode='sparse')

    my_valid_ds = test_datagen.flow_from_directory(path,
                                                   batch_size=batch_size,
                                                   target_size=(img_width, img_height),
                                                   class_mode='sparse')
    print("flow form directory OK")
    return my_train_ds, my_valid_ds


def start_trainning(model, train_ds, valid_ds, batch_size=1, epochs=10):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15, restore_best_weights=True)
    history = model.fit(train_ds, shuffle=True,
                        steps_per_epoch=train_ds.n // batch_size,
                        epochs=epochs, batch_size=batch_size,
                        validation_data=valid_ds, callbacks=[es])

    scores = model.evaluate(valid_ds, verbose=1)
    print("Accuracy: %.2f%%" % (scores[1] * 100), "| Loss: %.5f" % (scores[0]))
    return history


def show_graphics_from_history(history):
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = history.epoch

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def download_from_scryfall(magic_set, dataset_path):

    my_path = os.path.join(dataset_path, magic_set)
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    if not os.path.exists(my_path):
        os.mkdir(my_path)

    page = 1
    contents = requests.get("https://api.scryfall.com/cards/search?q=s:%s&page=%d" % (magic_set, page))
    my_json = json.loads(contents.text)
    aux = my_json
    while aux['has_more']:
        page += 1
        contents = requests.get("https://api.scryfall.com/cards/search?q=s:%s&page=%d" % (magic_set, page))
        aux = json.loads(contents.text)
        my_json['data'].extend(aux['data'])

    with open(my_path + '.json', 'w') as my_file:
        json.dump(my_json, my_file)
        my_file.flush()
        my_file.close()

    my_list = my_json['data']
    my_list.sort(key=lambda x: x['id'])
    size = len(my_list)
    cont = 1

    for card in my_list:
        if card['name'][0] == 'A' and card['name'][1] == '-':
            continue  # hide arena's card
        my_path = os.path.join(dataset_path, magic_set, card['id'])
        if not os.path.exists(my_path):
            os.mkdir(my_path)

        if card['layout'] == 'transform' or card['layout'] == 'modal_dfc':

            print("{:5.1f}%".format(cont / size * 100), card['id'], card['name'], 'front',
                  card['card_faces'][0]['image_uris']['art_crop'])
            page = requests.get(card['card_faces'][0]['image_uris']['art_crop'])
            my_file = os.path.join(my_path, 'art_front.jpg')
            my_file = open(my_file, 'bw')
            my_file.write(page.content)
            my_file.flush()
            # my_file.close()
            page = requests.get(card['card_faces'][0]['image_uris']['border_crop'])
            my_file = os.path.join(my_path, 'card_front.jpg')
            my_file = open(my_file, 'bw')
            my_file.write(page.content)
            my_file.flush()
            my_file.close()

            print("{:5.1f}%".format(cont / size * 100), card['id'], card['name'], 'back',
                  card['card_faces'][1]['image_uris']['art_crop'])
            my_file = os.path.join(my_path, 'art_back.jpg')
            page = requests.get(card['card_faces'][1]['image_uris']['art_crop'])
            my_file = open(my_file, 'bw')
            my_file.write(page.content)
            my_file.flush()
            my_file.close()
            # page = requests.get(card['card_faces'][1]['image_uris']['border_crop'])
            # my_file = os.path.join(my_path, 'card_back.jpg')
            # my_file = open(my_file, 'bw')
            # my_file.write(page.content)
            # my_file.flush()
            # my_file.close()
        else:
            print("{:5.1f}%".format(cont / size * 100), card['id'], card['name'], 'front',
                  card['image_uris']['art_crop'])
            page = requests.get(card['image_uris']['art_crop'])
            my_file = os.path.join(my_path, 'art_front.jpg')
            my_file = open(my_file, 'bw')
            my_file.write(page.content)
            my_file.flush()
            # my_file.close()
            page = requests.get(card['image_uris']['border_crop'])
            my_file = os.path.join(my_path, 'card_front.jpg')
            my_file = open(my_file, 'bw')
            my_file.write(page.content)
            my_file.flush()
            my_file.close()
        cont = cont + 1
    return my_list


# MAIN
path = os.path.join(DATASET_PATH, MAGIC_SET)
if DOWNLOAD_DATASET:
    classes_json = download_from_scryfall(MAGIC_SET, DATASET_PATH)
else:
    file = open(path + '.json', 'r')
    classes_json = json.load(file)['data']
    classes_json.sort(key=lambda x: x['id'])
    file.close()

# MODEL
if os.path.exists(MODEL_NAME):
    print('OLD MODEL LOAD')
    model = load_model(MODEL_NAME)
else:
    model = create_model(IMG_WIDTH, IMG_HEIGHT, num_classes=len(classes_json), model_name=MODEL_NAME)
    print('NEW MODEL CREATED')

model.summary()

def add_noise(img):
    # """Add random noise to an image"""
    VARIABILITY = 15
    deviation = VARIABILITY*random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    np.clip(img, 0., 255.)
    return img

def add_blur(img):
    img = cv2.GaussianBlur(img,(9,9),0)
    return img

# DATASET FOR TRAINING
idg = ImageDataGenerator(
    brightness_range=[0.3, 2.0],
    #rotation_range=15,
    #width_shift_range=0.2,
    #height_shift_range=0.2,
    #shear_range=0.2,
    zoom_range=0.2,
    #horizontal_flip=True,
    #vertical_flip=False,
    preprocessing_function=add_blur,
    fill_mode='constant')
train_ds, valid_ds = load_dataset(path, IMG_WIDTH, IMG_HEIGHT, batch_size=1, img_data_gen=idg)
scores = model.evaluate(valid_ds, verbose=1)
print("Accuracy: %.2f%%" % (scores[1] * 100), "| Loss: %.5f" % (scores[0]))

history = start_trainning(model, train_ds, valid_ds,batch_size=1, epochs=50)
show_graphics_from_history(history)

print('Deseja salvar essa joça? S/N')
a = input()
if a == 's' or a == 'S':
    model.save(MODEL_NAME)
    print('Esta joça foi salva com sucesso')
else:
    print('Você perdeu essa joça')
