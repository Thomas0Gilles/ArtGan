# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 10:24:13 2018

@author: tgill
"""
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist, cifar10
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from keras.utils.np_utils import to_categorical
import cv2
from functools import partial

from tqdm import tqdm

from wasserstein import get_models, wasserstein_loss, RandomWeightedAverage, gradient_penalty_loss


(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train / 255
X_test = X_test / 255

names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

#X_train = np.expand_dims(X_train, axis=-1)
#X_test = np.expand_dims(X_test, axis=-1)

n_epochs = 20
epoch_size = 500
batch_size = 128

target_size = (64, 64)
x_m = []
for i in range(len(X_train)):
    x = cv2.resize(X_train[i], target_size)
    x = x.reshape(target_size[0], target_size[1], 3)
    x_m.append(x)
X_train = np.asarray(x_m)

noise_dim=100
GRADIENT_PENALTY_WEIGHT=10

gen, disc = get_models(64, 64, 256, 512,  noise_dim, 10, n_channels=3)

disc.trainable = False
generator_input = Input(shape=(100,))
generator_layers = gen(generator_input)
discriminator_layers_for_generator = disc(generator_layers)
generator_model = Model(inputs=[generator_input], outputs=[discriminator_layers_for_generator])
# We use the Adam paramaters from Gulrajani et al.
generator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9), loss=wasserstein_loss)

disc.trainable = True
gen.trainable = False
real_samples = Input(shape=X_train.shape[1:])
generator_input_for_discriminator = Input(shape=(100,))
generated_samples_for_discriminator = gen(generator_input_for_discriminator)
discriminator_output_from_generator = disc(generated_samples_for_discriminator)
discriminator_output_from_real_samples = disc(real_samples)

averaged_samples = RandomWeightedAverage(batch_size)([real_samples, generated_samples_for_discriminator])
averaged_samples_out = disc(averaged_samples)

partial_gp_loss = partial(gradient_penalty_loss,
                          averaged_samples=averaged_samples,
                          gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names or Keras will throw an error

discriminator_model = Model(inputs=[real_samples, generator_input_for_discriminator],
                            outputs=[discriminator_output_from_real_samples,
                                     discriminator_output_from_generator,
                                     averaged_samples_out])
discriminator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                            loss=[wasserstein_loss,
                                  wasserstein_loss,
                                  partial_gp_loss])

for ite in range(10):
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
        images = images.reshape(-1, target_size[0],target_size[1], 3)
        for i in range(images.shape[0]):
            ax=plt.subplot(3, 3, i+1)
            ax.set_title(names[i], fontsize=10)
            plt.imshow(images[i], interpolation='nearest', cmap='gray_r')
            plt.axis('off')
        plt.tight_layout()
        plt.show()
        plt.pause(0.05)
