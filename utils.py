# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 13:14:54 2018

@author: tgill
"""
import os
import numpy as np
from keras.preprocessing.image import load_img,img_to_array
import cv2

def producer(queue, stop_event, X_paths, y, batch_size, target_size=None, prep_func=None):
    npts = len(X_paths)
    while not stop_event.is_set():
        X_batch=[]
        y_batch=[]
        c = 0
        while c<batch_size:
#            print(c)
            try:
                idx = np.random.randint(0, npts)
#                img = cv2.imread(X_paths[idx])
#                img = cv2.resize(img, target_size)
                img = img_to_array(load_img(X_paths[idx], target_size = target_size))
                if img is not None:
#                    if prep_func is not None:
#                        img = prep_func(img)
                    #Augmentation
                    flip = np.random.rand()
                    if flip<0.5:
                        img = np.flip(img, axis=1)
                    X_batch.append(img)
                    y_batch.append(y[idx])
                    c+=1
            except:
                pass
        X_batch = np.ascontiguousarray(X_batch)
        #♣X_batch = X_batch[...,::-1]
        if prep_func is not None:
            X_batch = prep_func(X_batch)
        queue.put((X_batch, np.ascontiguousarray(y_batch)))
        
def getPaths(data):
    X = []
    y = []
    classes = []
    for i, style in enumerate(os.listdir(data)):
        classes.append((i, style))
        path = os.path.join(data, style)
        for pic in os.listdir(path):
            path_pic = os.path.join(path, pic)
            #image = cv2.imread(path_pic)
            #image = Image.open(path_pic)
            X.append(path_pic)
            y.append(i)
    return np.ascontiguousarray(X), np.ascontiguousarray(y), classes

def scale(x):
    return x/255

def mean(x):
    return x/127.5-1.0
