# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 10:24:13 2018

@author: tgill
"""
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import  mnist
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Input
from keras.models import Model
import cv2

from tqdm import tqdm

from networks import   generator, discriminator, RandomWeightedAverage, gradient_penalty_loss, wasserstein_loss
from functools import partial


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train / 255
X_test = X_test / 255

names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

#X_train = np.expand_dims(X_train, axis=-1)
#X_test = np.expand_dims(X_test, axis=-1)

n_epochs = 100
epoch_size = 500
batch_size = 128

target_size = (64, 64, 1)

x_m = []
for i in range(len(X_train)):
    x = cv2.resize(X_train[i], target_size[:2])
    x = x.reshape(target_size[0], target_size[1], target_size[2])
    x_m.append(x)
X_train = np.asarray(x_m)

noise_dim=100
n_critic = 5
n_labels=10
n_channels=3

gen = generator(target_size[0], target_size[1], 512, noise_dim, n_labels, target_size[2])
disc = discriminator(target_size[0], target_size[1], 256, n_labels, target_size[2], wgan=True)
opt = Adam(0.0002, 0.5)

#-------------------------------
# Construct Computational Graph
#       for the Critic
#-------------------------------
# Freeze generator's layers while training critic
gen.trainable = False
# Image input (real sample)
real_img = Input(shape=target_size)
# Noise input
z_disc = Input(shape=(noise_dim,))
label_inp = Input(shape=(n_labels,))
# Generate image based of noise (fake sample)
fake_img = gen([z_disc, label_inp])
# Discriminator determines validity of the real and fake images
fake, fake_label = disc(fake_img)
valid, valid_label = disc(real_img)
# Construct weighted average between real and fake images
interpolated_img = RandomWeightedAverage(batch_size)([real_img, fake_img])
# Determine validity of weighted sample
validity_interpolated, _ = disc(interpolated_img)

# Use Python partial to provide loss function with additional
# 'averaged_samples' argument
partial_gp_loss = partial(gradient_penalty_loss,
                  averaged_samples=interpolated_img)
partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

disc_model = Model(inputs=[real_img, z_disc, label_inp],
                            outputs=[valid, valid_label, fake, fake_label, validity_interpolated])
disc_model.compile(loss=[wasserstein_loss, 'categorical_crossentropy',
                              wasserstein_loss, 'categorical_crossentropy',
                              partial_gp_loss],
                        optimizer=opt,
                        loss_weights=[1, 1, 1, 1, 10])

#-------------------------------
# Construct Computational Graph
#         for Generator
#-------------------------------
# For the generator we freeze the critic's layers
disc.trainable = False
gen.trainable = True
# Sampled noise for input to generator
z_gen = Input(shape=(100,))
# Generate images based of noise
img = gen([z_gen, label_inp])
# Discriminator determines validity
valid, valid_label = disc(img)
# Defines generator model
gen_model = Model([z_gen, label_inp], [valid, valid_label])
gen_model.compile(loss=[wasserstein_loss, 'categorical_crossentropy'], optimizer=opt)



#Fixed targets
y_true = -np.ones([batch_size, 1])
y_fake = np.ones([batch_size, 1])
y_dummy = np.zeros([batch_size, 1])

for i in range(n_epochs):
    print("Epoch ", i)
    loss_disc=[]
    loss_gen=[]
    for j in tqdm(range(epoch_size)):
        for k in range(n_critic):
            idxs_batch = np.random.randint(0, len(X_train), batch_size)
            X_true = X_train[idxs_batch]
            label_true = y_train[idxs_batch]
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, noise_dim])
            #noise = np.random.randn(batch_size, noise_dim)
            sampled_labels = np.random.randint(0, 10, (batch_size, 1))
            sampled_labels = to_categorical(sampled_labels, num_classes=10)
            #X_fake = gen.predict([noise, sampled_labels], batch_size=batch_size)
            d_loss = disc_model.train_on_batch([X_true, noise, sampled_labels],
                                                                [y_true, label_true, y_fake, sampled_labels, y_dummy])
            loss_disc.append(d_loss)

        
        X_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, noise_dim])
        #X_noise = np.random.randn(batch_size, noise_dim)
        sampled_labels = np.random.randint(0, 10, (batch_size, 1))
        sampled_labels = to_categorical(sampled_labels, num_classes=10)
        gen_loss = gen_model.train_on_batch([X_noise, sampled_labels], [y_true, sampled_labels])
        loss_gen.append(gen_loss)
        #print(gen_loss)
    print("Disc", np.mean(loss_disc, axis=0))
    print("Gen", np.mean(loss_gen, axis=0))
    #plt.imshow(X_fake[1][:,:,0])
    #plt.title(sampled_labels[1])
    if i%1==0:
        examples=9
        noise = np.random.uniform(-1.0, 1.0, size=[examples, noise_dim])
        labels = np.arange(9)
    #    l = labels.reshape(-1, 1)
    #    l = np.tile(l, 10)
    #    l = l.reshape(100)
    #    labels = np.tile(labels, 10)
        labels = to_categorical(labels, num_classes=10)
    #    l = to_categorical(l)
    #    labels = labels
        images = gen.predict([noise, labels], batch_size=batch_size)
        images = images.reshape(-1, target_size[0],target_size[1])
        plt.figure(figsize=(10, 10))
        for i in range(images.shape[0]):
            ax=plt.subplot(3, 3, i+1)
            ax.set_title(names[i], fontsize=10)
            plt.imshow(images[i], interpolation='nearest', cmap='gray_r')
            plt.axis('off')
        plt.tight_layout()
        plt.show()
        plt.pause(0.05)
