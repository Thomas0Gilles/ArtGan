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
from keras.applications.resnet50 import preprocess_input
from keras.models import Model

import queue
import threading

from tqdm import tqdm

from networks import get_models, adversarial, null_loss, generator, discriminator
from utils import producer, getPaths, scale, mean

data_train = r"C:\Users\tgill\OneDrive\Documents\GD_AI\ArtGAN\wikipaintings_full\wikipaintings_train"
data_test = r"C:\Users\tgill\OneDrive\Documents\GD_AI\ArtGAN\wikipaintings_full\wikipaintings_train"

train_paths, y_train, classes = getPaths(data_train)
test_paths, y_test, classes = getPaths(data_test)
target_size = (128, 128)

ls = [-np.sum(y_train==i) for i in range(25)]
arg = np.argsort(ls)
classement = np.argsort(arg)

nb_select = 1
select = arg[:nb_select]
idx_select = np.isin(y_train, select)
train_paths = train_paths[idx_select]
y_train = y_train[idx_select]
y_train = classement[y_train]
print(train_paths.shape)

n_epochs = 20
epoch_size = 300
batch_size = 64

noise_dim=100
n_labels=nb_select
prep_func = mean


q = queue.Queue(maxsize=40)
stop_event = threading.Event()
writer = threading.Thread(target=producer, args=(q, stop_event, train_paths, y_train, batch_size, target_size, prep_func))
writer2 = threading.Thread(target=producer, args=(q, stop_event, train_paths, y_train, batch_size, target_size, prep_func))
writer3 = threading.Thread(target=producer, args=(q, stop_event, train_paths, y_train, batch_size, target_size, prep_func))
writer4 = threading.Thread(target=producer, args=(q, stop_event, train_paths, y_train, batch_size, target_size, prep_func))
writer5 = threading.Thread(target=producer, args=(q, stop_event, train_paths, y_train, batch_size, target_size, prep_func))
writer6 = threading.Thread(target=producer, args=(q, stop_event, train_paths, y_train, batch_size, target_size, prep_func))
#writer7 = threading.Thread(target=producer, args=(q, stop_event, train_paths, y_train, batch_size, target_size, prep_func))
#writer8 = threading.Thread(target=producer, args=(q, stop_event, train_paths, y_train, batch_size, target_size, prep_func))

writer.start()
writer2.start()
writer3.start()
writer4.start()
writer5.start()
writer6.start()
#writer7.start()
#writer8.start()
#
gen = generator(target_size[0], target_size[1], 1024, noise_dim, n_labels, target_size[2], tanh=True)
disc = discriminator(target_size[0], target_size[1], 512, n_labels, target_size[2], wgan=True)
opt = Adam(0.0002, 0.5)
disc.compile(loss=['binary_crossentropy', null_loss()], optimizer=opt, metrics=['accuracy'])
frozen_disc = Model(inputs = disc.inputs, outputs=disc.outputs)
frozen_disc.trainable = False
adv = Model(inputs = gen.input, outputs = frozen_disc(gen.output))
adv.compile(loss=['binary_crossentropy', null_loss()], optimizer=opt, metrics=['accuracy'])
for repeat in range(8):
    plt.ion()
    plt.figure(figsize=(10, 10))
    for i in range(n_epochs):
        print("Epoch ", i)
        loss_disc=[]
        loss_gen=[]
        for j in tqdm(range(epoch_size)):
            X_true, label_true = q.get()
            label_true = to_categorical(label_true, n_labels)
            noise = np.random.randn(batch_size, noise_dim)
            sampled_labels = np.random.randint(0, n_labels, (batch_size, 1))
            sampled_labels = to_categorical(sampled_labels, num_classes=n_labels)
            X_fake = gen.predict([noise, sampled_labels], batch_size=batch_size)
            
            y_true = np.ones([batch_size, 1])
            y_fake = np.zeros([batch_size, 1])
            
            disc_loss1 = disc.train_on_batch(X_true, [y_true, label_true])
            disc_loss2 = disc.train_on_batch(X_fake, [y_fake, sampled_labels])
            loss_disc.append(np.add(disc_loss1,disc_loss2)/2)
            #print(disc_loss)
            
            y_noise = np.ones([batch_size, 1])
            X_noise = np.random.randn(batch_size, 100)
            label_fake = np.zeros_like(label_true)
            label_fake[:,:n_labels]=sampled_labels
            #disc.trainable=False
            gen_loss = adv.train_on_batch([X_noise, sampled_labels], [y_noise, label_fake])
            loss_gen.append(gen_loss)
            #print(gen_loss)
        print("Disc", np.mean(loss_disc, axis=0))
        print("Gen", np.mean(loss_gen, axis=0))
        #plt.imshow(X_fake[1][:,:,0])
        #plt.title(sampled_labels[1])
        if i%1==0:
            plt.figure(figsize=(10, 10))
            examples=nb_select
            #noise = np.random.uniform(-1.0, 1.0, size=[examples, noise_dim])
            noise = np.random.randn(examples, noise_dim)
            labels = np.unique(y_train)
        #    l = labels.reshape(-1, 1)
        #    l = np.tile(l, 10)
        #    l = l.reshape(100)
        #    labels = np.tile(labels, 10)
            labels = to_categorical(labels, num_classes = n_labels)
        #    l = to_categorical(l)
        #    labels = labels+l
            images = gen.predict([noise, labels], batch_size=batch_size)
            images = images.reshape(-1, target_size[0],target_size[1], 3)
            dim = int(np.sqrt(nb_select))
            for i in range(images.shape[0]):
                ax=plt.subplot(dim, dim, i+1)
                #ax.title.set_text(classes[i][1], fontsize=5)
                ax.set_title(classes[arg[i]][1], fontsize=5)
                plt.imshow(images[i], interpolation='nearest', cmap='gray_r')
                plt.axis('off')
            plt.tight_layout()
            plt.show()
            plt.pause(0.05)
    
stop_event.set()
