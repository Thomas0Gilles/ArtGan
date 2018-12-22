# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 10:24:13 2018

@author: tgill
"""
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import cv2

from tqdm import tqdm

from networks import get_models, adversarial, bin_accuracy


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train / 255
X_test = X_test / 255


y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

n_epochs = 20
epoch_size = 500
batch_size = 128

target_size = (64, 64)
x_m = []
for i in range(len(X_train)):
    x = cv2.resize(X_train[i], target_size)
    x = x.reshape(target_size[0], target_size[1], 1)
    x_m.append(x)
X_train = np.asarray(x_m)

noise_dim=100

gen, disc = get_models(64, 64, 256, 256,  noise_dim, 10, n_channels=1)
opt = Adam(0.0002, 0.5)
disc.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer=opt, metrics=['accuracy'])
adv = adversarial(gen, disc, opt)
plt.ion()
plt.figure(figsize=(10, 10))
for i in range(n_epochs):
    print("Epoch ", i)
    loss_disc=[]
    loss_gen=[]
    for j in tqdm(range(epoch_size)):
        #print("Iteration ", j)
        idxs_batch = np.random.randint(0, len(X_train), batch_size)
        X_true = X_train[idxs_batch]
        label_true = y_train[idxs_batch]
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, noise_dim])
        sampled_labels = np.random.randint(0, 10, (batch_size, 1))
        sampled_labels = to_categorical(sampled_labels, num_classes=10)
        X_fake = gen.predict([noise, sampled_labels], batch_size=batch_size)
#        label_fake = np.zeros_like(label_true)
#        label_fake[:,10]=1
        
        X_batch = np.concatenate([X_true, X_fake])
        y_batch = np.ones([2*batch_size, 1])
        y_batch[batch_size:] = 0
        labels_batch = np.concatenate([label_true, sampled_labels])
        y_true = np.ones([batch_size, 1])
        y_fake = np.zeros([batch_size, 1])
        
        
#        shuffle = np.random.permutation(2*batch_size)
#        X_batch = X_batch[shuffle]
#        y_batch = y_batch[shuffle]
        
#        disc.trainable = True
#        disc.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        disc_loss1 = disc.train_on_batch(X_true, [y_true, label_true])
        disc_loss2 = disc.train_on_batch(X_fake, [y_fake, sampled_labels])
        loss_disc.append(np.add(disc_loss1,disc_loss2)/2)
#        disc_loss = disc.train_on_batch(X_batch, [y_batch, labels_batch])
#        loss_disc.append(disc_loss)
        #print(disc_loss)
        
        y_noise = np.ones([batch_size, 1])
        X_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
#        label_fake = np.zeros_like(label_true)
#        label_fake[:,:10]=sampled_labels
        gen_loss = adv.train_on_batch([X_noise, sampled_labels], [y_noise, sampled_labels])
        loss_gen.append(gen_loss)
        #print(gen_loss)
    print("Disc", np.mean(loss_disc, axis=0))
    print("Gen", np.mean(loss_gen, axis=0))
    #plt.imshow(X_fake[1][:,:,0])
    #plt.title(sampled_labels[1])
    
    examples=100
    noise = np.random.uniform(-1.0, 1.0, size=[examples, noise_dim])
    labels = np.arange(10)
    l = labels.reshape(-1, 1)
    l = np.tile(l, 10)
    l = l.reshape(100)
    labels = np.tile(labels, 10)
    labels = to_categorical(labels)
    l = to_categorical(l)
    labels = labels
    images = gen.predict([noise, labels], batch_size=batch_size)
    images = images.reshape(-1, target_size[0],target_size[1])
    for i in range(images.shape[0]):
        plt.subplot(10, 10, i+1)
        plt.imshow(images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.pause(0.05)
