#!/usr/bin/env python
# coding: utf-8

# # Keras Benchmark 

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
from time import time
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D,    Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.backend import epsilon
from keras import backend
from keras.metrics import AUC
import os
import csv
import pandas as pd


def unison_shuffled_copies(a, b):
    # Shuffles two lists keeping orders
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def crop(img):
    if img.shape[0]<img.shape[1]:
        x = img.shape[0]
        y = img.shape[1]
        crop_img = img[: , int(y/2-x/2):int(y/2+x/2)]
    else:
        x = img.shape[1]
        y = img.shape[0]
        crop_img = img[int(y/2-x/2):int(y/2+x/2) , :]

    return crop_img

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+backend.epsilon()))

def recall_m(y_true, y_pred):
        true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
        possible_positives = backend.sum(backend.round(backend.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + backend.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
        predicted_positives = backend.sum(backend.round(backend.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + backend.epsilon())
        return precision

sex = ['men', 'women']
models = ['DC-GAN','wasserstein gan','vae_800_women','softmax gan','adversarial auto encoder',"Oversample","None"]
for model_name in models:
    nb_epochs = 50

    if model_name != "None" and model_name != "Oversample":
        # Gennerated images
        files_generated = glob('generated_images/'+model_name+'/*.png')
        if len(files_generated) == 0:
            files_generated = glob('generated_images/'+model_name+'/*.jpg')
        files_generated = np.random.permutation(files_generated)

    # ##### Load 800 women dataset (data used to train generator models)
    with open('800_women.csv', 'r') as f:
        reader = csv.reader(f)
        women_real_to_generate = np.array(list(reader)).flatten()
    women_real_to_generate = [x.replace('combined\\','UTKFace/') for x in women_real_to_generate]
    women_real_to_generate = list(set(women_real_to_generate))

    # ##### Add 7200 random generated women to 800 real women and create train/ test data

    training_class_size = 6000

    # Import pictures names. This is to keep them all the same
    train_men = pd.read_csv('train_men.csv',index_col=0).values.flatten()
    test_men = pd.read_csv('test_men.csv',index_col=0).values.flatten()
    test_women = pd.read_csv('test_women.csv',index_col=0).values.flatten()

    if model_name != "None" and model_name != "Oversample":
        # Choose 7200 from the generated women
        train_women = np.random.choice(files_generated, training_class_size-len(women_real_to_generate),replace=False)
        # Add generated woment to training_women
        train_women = np.concatenate((train_women, np.array(women_real_to_generate)),axis=0)
    elif model_name == "Oversample":
        train_women = np.random.choice(women_real_to_generate,6000-len(women_real_to_generate))
        train_women = np.concatenate((train_women, np.array(women_real_to_generate)),axis=0)
    else:
        train_women = women_real_to_generate

    dim = 60

    print('Scaling...', end='')
    start = time()
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    if not (training_class_size == len(train_men) and len(train_men) == len(train_women)):
        print('Length mismatch in training set')
    # assert training_class_size == len(train_men) and len(train_men) == len(train_women) , 
    assert len(test_men) == len(test_women) , 'Length mismatch in test set'

    # Men
    for ix, file in enumerate(train_men): 
        image = plt.imread(file, 'jpg')
        image = Image.fromarray(image).resize((dim, dim)).convert('L')
        image = crop(np.array(image))
        x_train.append(image)
        y_train.append(0)
    # Women
    for ix, file in enumerate(train_women): 
        image = plt.imread(file, 'jpg')
        image = Image.fromarray(image).resize((dim, dim)).convert('L')
        image = crop(np.array(image))
        x_train.append(image)
        y_train.append(1)

    # Men (test)
    for ix, file in enumerate(test_men): 
        image = plt.imread(file, 'jpg')
        image = Image.fromarray(image).resize((dim, dim)).convert('L')
        image = crop(np.array(image))
        x_test.append(image)
        y_test.append(0)

    # Women (test)
    for ix, file in enumerate(test_women): 
        image = plt.imread(file, 'jpg')
        image = Image.fromarray(image).resize((dim, dim)).convert('L')
        image = crop(np.array(image))
        x_test.append(image)
        y_test.append(1)
        
    print(f'\rDone in {int(time() - start)} seconds')


    # ##### Turning the pictures into arrays
    # Train
    x_train = np.array(x_train, dtype=np.float32).reshape(-1, 60, 60, 1)
    y_train = np.array(y_train, dtype=np.float32)
    # Test
    x_test = np.array(x_test, dtype=np.float32).reshape(-1, 60, 60, 1)
    y_test = np.array(y_test, dtype=np.float32)
    labels_test = y_test.copy()

    # ##### Shuffle train sets
    x_train, y_train = unison_shuffled_copies(x_train, y_train)
    labels_train = y_train.copy()

    # ##### Turning the targets into a 2D matrix
    y_train = to_categorical(y_train, 2)
    y_test = to_categorical(y_test, 2)

    assert x_train.ndim == 4, 'The input is the wrong shape!'

    yy, xx = y_train.nbytes, x_train.nbytes

    print(f'The size of X is {xx:,} bytes and the size of Y is {yy:,} bytes.')

    files, faces = None, None

    # ##### Displaying the pictures
    show_images = False
    if show_images:
        fig = plt.figure(figsize=(12, 12))
        for i in range(1, 5):
            plt.subplot(1, 5, i)
            rand = np.random.randint(0, x_train.shape[0])
            ax = plt.imshow(x_train[rand][:, :, 0], cmap='gray')
            plt.title('<{}>'.format(sex[int(labels_train[rand])].capitalize()))
            yticks = plt.xticks([])
            yticks = plt.yticks([])

        plt.show()

    trainsize, testsize = x_train.shape[0], x_test.shape[0]
    print(f'The size of the training set is {trainsize:,} and the '     f'size of the test set is {testsize:,}.')

    # ##### Scaling, casting the arrays
    print('Scaling...', end='')
    image_size = x_train.shape[1] * x_train.shape[1] 
    x_train = x_train.astype('float32') / 255 
    x_test = x_test.astype('float32') / 255
    print('\rDone.     ')

    model = Sequential([
        Conv2D(16*4, (3, 3), input_shape=(60, 60, 1), activation='relu'),
        MaxPooling2D(),
        
        Conv2D(32*4, (3, 3), activation='relu'),
        MaxPooling2D(),
        
        Conv2D(64*4, (3, 3), activation='relu'),
        MaxPooling2D(),
        
        Conv2D(128*4, (3, 3), activation='relu'),
        MaxPooling2D(),
        
        Flatten(),
        
        Dense(1024, activation='relu'),
        Dense(2048, activation='relu'),
        Dense(2, activation='sigmoid')
    ])

    # model.summary()

    model.compile(optimizer=Adam(lr=0.001), 
                                 loss='binary_crossentropy', 
                                 metrics=['accuracy', AUC(),f1_m])

    e_s = EarlyStopping(monitor='val_loss', patience=10)

    hist = model.fit(x_train, y_train,
                     epochs=nb_epochs,
                     validation_data=[x_test, y_test],
                     batch_size=32,
                     callbacks=[e_s])


    pd.DataFrame(hist.history).to_csv(model_name+'_history.csv')

    test_loss, test_acc, test_AUC, test_f1 = model.evaluate(x_test, y_test)

    print("-------------------")
    print(model_name)
    print(f'Test loss: {np.round(test_loss, 4)} — Test accuracy: {np.round(test_acc*100,2)}%')
    print(f'Test AUC: {np.round(test_AUC, 4)} — Test F1: {np.round(test_f1,4)}%')

