# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 10:24:13 2018

@author: tgill
"""
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import  cifar10
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Input
from keras.models import Model
from keras import backend as K
import cv2

from tqdm import tqdm
import queue
import threading

from networks import   generator, discriminator, RandomWeightedAverage, gradient_penalty_loss, wasserstein_loss, null_loss, res_generator, res_discriminator
from functools import partial
from utils import producer, getPaths, scale, mean


data_train = r"C:\Users\tgill\OneDrive\Documents\GD_AI\ArtGAN\wikipaintings_full\wikipaintings_train"
data_test = r"C:\Users\tgill\OneDrive\Documents\GD_AI\ArtGAN\wikipaintings_full\wikipaintings_train"

train_paths, y_train, classes = getPaths(data_train)
test_paths, y_test, classes = getPaths(data_test)

ls = [-np.sum(y_train==i) for i in range(25)]
arg = np.argsort(ls)
classement = np.argsort(arg)

nb_select = 9
select = arg[:nb_select]
idx_select = np.isin(y_train, select)
train_paths = train_paths[idx_select]
y_train = y_train[idx_select]
y_train = classement[y_train]
print(train_paths.shape)

n_epochs = 100
epoch_size = 1400
batch_size = 32

target_size = (64, 64, 3)


noise_dim=100
n_critic = 5
n_labels=10
n_channels=3
n_labels=nb_select
prep_func = mean

gen = res_generator(target_size[0], target_size[1], 128, noise_dim, n_labels, target_size[2], tanh=True)
disc = res_discriminator(target_size[0], target_size[1], 128, n_labels, target_size[2], wgan=True)
lr = 0.0002
opt = Adam(lr, 0.0, 0.9)
label_loss  = 'categorical_crossentropy' #null_loss()#    'categorical_crossentropy'
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
disc_model.compile(loss=[wasserstein_loss, label_loss,
                              wasserstein_loss, label_loss,
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
gen_model.compile(loss=[wasserstein_loss, label_loss], optimizer=opt)



#Fixed targets
y_true = -np.ones([batch_size, 1])
y_fake = np.ones([batch_size, 1])
y_dummy = np.zeros([batch_size, 1])

q = queue.Queue(maxsize=100)
stop_event = threading.Event()
writer = threading.Thread(target=producer, args=(q, stop_event, train_paths, y_train, batch_size, target_size[:2], prep_func))
writer2 = threading.Thread(target=producer, args=(q, stop_event, train_paths, y_train, batch_size, target_size[:2], prep_func))
writer3 = threading.Thread(target=producer, args=(q, stop_event, train_paths, y_train, batch_size, target_size[:2], prep_func))
writer4 = threading.Thread(target=producer, args=(q, stop_event, train_paths, y_train, batch_size, target_size[:2], prep_func))
writer5 = threading.Thread(target=producer, args=(q, stop_event, train_paths, y_train, batch_size, target_size[:2], prep_func))
writer6 = threading.Thread(target=producer, args=(q, stop_event, train_paths, y_train, batch_size, target_size[:2], prep_func))
writer7 = threading.Thread(target=producer, args=(q, stop_event, train_paths, y_train, batch_size, target_size[:2], prep_func))
writer8 = threading.Thread(target=producer, args=(q, stop_event, train_paths, y_train, batch_size, target_size[:2], prep_func))
#writer9 = threading.Thread(target=producer, args=(q, stop_event, train_paths, y_train, batch_size, target_size[:2], prep_func))
#writer10 = threading.Thread(target=producer, args=(q, stop_event, train_paths, y_train, batch_size, target_size[:2], prep_func))
#writer11 = threading.Thread(target=producer, args=(q, stop_event, train_paths, y_train, batch_size, target_size[:2], prep_func))
#writer12 = threading.Thread(target=producer, args=(q, stop_event, train_paths, y_train, batch_size, target_size[:2], prep_func))
#writer13 = threading.Thread(target=producer, args=(q, stop_event, train_paths, y_train, batch_size, target_size[:2], prep_func))
#writer14 = threading.Thread(target=producer, args=(q, stop_event, train_paths, y_train, batch_size, target_size[:2], prep_func))
#writer15 = threading.Thread(target=producer, args=(q, stop_event, train_paths, y_train, batch_size, target_size[:2], prep_func))
#writer16 = threading.Thread(target=producer, args=(q, stop_event, train_paths, y_train, batch_size, target_size[:2], prep_func))

writer.start()
writer2.start()
writer3.start()
writer4.start()
writer5.start()
writer6.start()
writer7.start()
writer8.start()
#writer9.start()
#writer10.start()
#writer11.start()
#writer12.start()
#writer13.start()
#writer14.start()
#writer15.start()
#writer16.start()
#examples=nb_select
examples=nb_select
noise_disp = np.random.randn(examples, noise_dim)

for i in range(n_epochs):
    print("Epoch ", i)
    loss_disc=[]
    loss_gen=[]
    print(q.qsize())
    decay = max(0.0, 1-i/n_epochs)
    K.set_value(opt.lr, lr*decay)
    for j in tqdm(range(epoch_size)):
        for k in range(n_critic):
            X_true, label_true = q.get()
            #print(q.qsize())
            label_true = to_categorical(label_true, n_labels)
            #noise = np.random.uniform(-1.0, 1.0, size=[batch_size, noise_dim])
            noise = np.random.randn(batch_size, noise_dim)
            sampled_labels = np.random.randint(0, n_labels, (batch_size, 1))
            sampled_labels = to_categorical(sampled_labels, num_classes=n_labels)

            d_loss = disc_model.train_on_batch([X_true, noise, sampled_labels],
                                                                [y_true, label_true, y_fake, sampled_labels, y_dummy])
            loss_disc.append(d_loss)

        
        #X_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, noise_dim])
        X_noise = np.random.randn(batch_size, noise_dim)
        sampled_labels = np.random.randint(0, n_labels, (batch_size, 1))
        sampled_labels = to_categorical(sampled_labels, num_classes=n_labels)
        gen_loss = gen_model.train_on_batch([X_noise, sampled_labels], [y_true, sampled_labels])
        loss_gen.append(gen_loss)
        #print(gen_loss)
    print("Disc", np.mean(loss_disc, axis=0))
    print(np.mean(loss_disc, axis=0)[3]+np.mean(loss_disc, axis=0)[1])
    print("Gen", np.mean(loss_gen, axis=0))
    #plt.imshow(X_fake[1][:,:,0])
    #plt.title(sampled_labels[1])
    if i%1==0:
        plt.figure(figsize=(10, 10))
   #     noise_disp = np.random.randn(examples, noise_dim)
        #noise = np.random.uniform(-1.0, 1.0, size=[examples, noise_dim])
#        noise_disp = np.random.randn(1, noise_dim)
#        noise_disp = np.tile(noise_disp, (examples,1))
        labels = np.unique(y_train)
        dim = int(np.sqrt(nb_select))
#        l = labels.reshape(-1, 1)
#        l = np.tile(l, dim)
#        l = l.reshape(examples)
        labels = np.tile(labels, dim)
        labels = to_categorical(labels, num_classes = n_labels)
    #    l = to_categorical(l)
    #    labels = labels+l
        images = gen.predict([noise_disp, labels], batch_size=batch_size)
        images = images.reshape(-1, target_size[0],target_size[1], 3)
        
        for i in range(images.shape[0]):
            ax=plt.subplot(dim, dim, i+1)
            #ax.title.set_text(classes[i][1], fontsize=5)
            ax.set_title(classes[arg[i]][1], fontsize=5)
            plt.imshow((images[i]+1)/2, interpolation='nearest', cmap='gray_r')
            plt.axis('off')
        plt.tight_layout()
        plt.show()
        plt.pause(0.05)
        
#        sampled_labels = np.random.randint(0, n_labels, (examples, 1))
#        images = gen.predict([noise_disp, sampled_labels], batch_size=batch_size)
#        dim = int(np.sqrt(examples))
#        for i in range(images.shape[0]):
#            ax=plt.subplot(dim, dim, i+1)
#            #ax.title.set_text(classes[i][1], fontsize=5)
#            plt.imshow((images[i]+1.)/2., interpolation='nearest', cmap='gray_r')
#            plt.axis('off')
#        plt.tight_layout()
#        plt.show()
#        plt.pause(0.05)
        
stop_event.set()

plt.figure(figsize=(20, 20))
examples=81
noise = np.random.randn(examples, noise_dim)
labels = np.arange(9)
l = labels.reshape(-1, 1)
l = np.tile(l, 9)
l = l.reshape(81)
labels = np.tile(labels, 9)
labels = to_categorical(labels)
l = to_categorical(l)
labels = (labels+l)/2
images = gen.predict([noise, labels], batch_size=batch_size)
images = images.reshape(-1, target_size[0],target_size[1], 3)
for i in range(images.shape[0]):
    ax=plt.subplot(9, 9, i+1)
    if i//9==0:
        ax.set_title(classes[arg[i%9]][1], fontsize=5)
    if i%9==0:
        plt.set_ylabel(classes[arg[i//9]][1])
    plt.imshow((images[i]+1)/2, interpolation='nearest', cmap='gray_r')
    plt.axis('off')
plt.tight_layout()
plt.show()
plt.pause(0.05)
