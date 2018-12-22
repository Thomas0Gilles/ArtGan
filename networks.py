# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 10:27:02 2018

@author: tgill
"""
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, Reshape, UpSampling2D, Conv2DTranspose, BatchNormalization, Concatenate, LeakyReLU, ReLU, Add, AveragePooling2D, GlobalAveragePooling2D
from keras.layers.merge import _Merge
from keras_layer_normalization import LayerNormalization
import tensorflow as tf
from keras import backend as K
import numpy as np

def discriminator(height, width, n_filters=16, n_labels=10, n_channels=3, wgan=False):
    inp = Input((height, width, n_channels))
#    x = Conv2D(filters=n_filters//16, kernel_size=(3,3), strides=2, padding='same')(inp) #64*64
#    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=n_filters//8, kernel_size=(3,3), strides=2, padding='same')(inp)   #32*32
    #x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=n_filters//4, kernel_size=(3,3), strides=2, padding='same')(x)  #16*16
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=n_filters//2, kernel_size=(3,3), strides=2, padding='same')(x)  #8*8
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=n_filters, kernel_size=(3,3), strides=2, padding='same')(x)    #4*4
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)
    #x = Conv2D(filters=8*n_filters, kernel_size=(3,3), strides=1, padding='valid', activation='relu')(x)
    x = Flatten()(x)
    #x = Dense(units=n_filters, activation='relu')(x)
    if wgan:
        validity = Dense(units=1)(x)
    else:
        validity = Dense(units=1, activation='sigmoid')(x)
    label = Dense(units=n_labels, activation='softmax')(x)
    model = Model(inputs=inp, outputs=[validity, label])
    return model

def generator(height, width, n_filters, noise_dim, n_labels, n_channels=3, tanh=False):
    inp = Input((noise_dim,))
    label = Input((n_labels,))
    conc = Concatenate()([inp, label])
    x = Dense(units=(height//16)*(width//16)*n_filters)(conc)
    x = Reshape((height//16, width//16, n_filters))(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = ReLU()(x)
#    x = UpSampling2D((2,2))(x)
#    x = Conv2D(n_filters//2, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = Conv2DTranspose(n_filters//2, kernel_size=(5,5), strides=2, padding='same')(x) #/8
    x = BatchNormalization(momentum=0.8)(x)
    x = ReLU()(x)
#    x = Conv2D(n_filters//2, kernel_size=(3,3), padding='same')(x)
#    x = BatchNormalization(momentum=0.8)(x)
#    x = ReLU()(x)
#    x = UpSampling2D((2,2))(x)
#    x = Conv2D(n_filters//4, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = Conv2DTranspose(n_filters//4, kernel_size=(5,5), strides=2, padding='same')(x) #/4
    x = BatchNormalization(momentum=0.8)(x)
    x = ReLU()(x)
#    x = Conv2D(n_filters//2, kernel_size=(3,3), padding='same')(x)
#    x = BatchNormalization(momentum=0.8)(x)
#    x = ReLU()(x)
    
    
    x = Conv2DTranspose(n_filters//8, kernel_size=(5,5), strides=2, padding='same')(x) #/2
    x = BatchNormalization(momentum=0.8)(x)
    x = ReLU()(x)
    
#    x = Conv2D(n_filters//8, kernel_size=(3,3), padding='same')(x)
#    x = BatchNormalization(momentum=0.8)(x)
#    x = ReLU()(x)
#    x = Conv2DTranspose(n_filters//16, kernel_size=(5,5), strides=2, padding='same')(x)
#    x = BatchNormalization(momentum=0.8)(x)
#    x = ReLU()(x)
    #out = Conv2D(n_channels, kernel_size=(3,3), padding='same', activation='sigmoid')(x)
    if tanh:
        out = Conv2DTranspose(n_channels, kernel_size=(5,5), strides=2, padding='same', activation='tanh')(x)
    else:
        out = Conv2DTranspose(n_channels, kernel_size=(5,5), strides=2, padding='same', activation='sigmoid')(x)
    model = Model(inputs=[inp, label], outputs=out)
    return model

def get_models(height, width, disc_filters, gen_filters, noise_dim, n_labels, n_channels):
    gen = generator(height, width, gen_filters, noise_dim, n_labels, n_channels)
    disc = discriminator(height, width, disc_filters, n_labels, n_channels)
    return gen, disc
    

def adversarial(gen, disc, opt):
    frozen_disc = Model(inputs = disc.inputs, outputs=disc.outputs)
    frozen_disc.trainable = False
    model = Model(inputs = gen.input, outputs = frozen_disc(gen.output))
    model.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer=opt, metrics=['accuracy'])
    return model

def null_loss():
    def f(y_true, y_pred):
        return 0*y_pred
    return f
    
def bin_accuracy(y_true, y_pred):
    true = y_true>0.5
    pred = y_pred>0.5
    cor= true==pred
    
    return tf.reduce_mean(tf.cast(cor, 'float32'))

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def __init__(self, batch_size, **kwargs):
        super(_Merge, self).__init__(**kwargs)
        self.batch_size=batch_size
        
    def _merge_function(self, inputs):
        alpha = K.random_uniform((self.batch_size, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])
    
def gradient_penalty_loss(y_true, y_pred, averaged_samples):
    """
    Computes gradient penalty based on prediction and weighted real / fake samples
    """
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)
                  
def wasserstein_loss( y_true, y_pred):
    return K.mean(y_true * y_pred)


def res_generator(height, width, n_filters, noise_dim, n_labels, n_channels=3, tanh=False):
    inp = Input((noise_dim,))
    label = Input((n_labels,))
    conc = Concatenate()([inp, label])
    x = Dense(units=(height//8)*(width//8)*n_filters)(conc)
    x = Reshape((height//8, width//8, n_filters))(x)
    
    x = UpSampling2D((2,2))(x) #8
    shortcut=x
    x = BatchNormalization(momentum=0.8)(x)
    x = ReLU()(x)
    x = Conv2D(n_filters, kernel_size=(3,3), strides=1, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = ReLU()(x)
    x = Conv2D(n_filters, kernel_size=(3,3), strides=1, padding='same')(x)
    x = Add()([shortcut, x])
    
    x = UpSampling2D((2,2))(x) #16
    shortcut=x
    x = BatchNormalization(momentum=0.8)(x)
    x = ReLU()(x)
    x = Conv2D(n_filters, kernel_size=(3,3), strides=1, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = ReLU()(x)
    x = Conv2D(n_filters, kernel_size=(3,3), strides=1, padding='same')(x)
    x = Add()([shortcut, x])
    
    x = UpSampling2D((2,2))(x) #32
    shortcut=x
    x = BatchNormalization(momentum=0.8)(x)
    x = ReLU()(x)
    x = Conv2D(n_filters, kernel_size=(3,3), strides=1, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = ReLU()(x)
    x = Conv2D(n_filters, kernel_size=(3,3), strides=1, padding='same')(x)
    x = Add()([shortcut, x])
    
    x = BatchNormalization(momentum=0.8)(x)
    x = ReLU()(x)
    

    if tanh:
        out = Conv2D(n_channels, kernel_size=(3,3), strides=1, padding='same', activation='tanh')(x)
    else:
        out = Conv2D(n_channels, kernel_size=(3,3), strides=1, padding='same', activation='sigmoid')(x)
    model = Model(inputs=[inp, label], outputs=out)
    return model

def res_discriminator(height, width, n_filters=16, n_labels=10, n_channels=3, wgan=False):
    inp = Input((height, width, n_channels))

    shortcut = AveragePooling2D()(inp)
    shortcut = Conv2D(filters=n_filters, kernel_size=(1,1), strides=1, padding='same')(shortcut)
    x = Conv2D(filters=n_filters, kernel_size=(3,3), strides=1, padding='same')(inp)   
    x = ReLU()(x)
    x = AveragePooling2D()(x)
    x = Conv2D(filters=n_filters, kernel_size=(3,3), strides=1, padding='same')(x)
    x = Add()([shortcut, x])
    
    shortcut = AveragePooling2D()(x)
    x = LayerNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=n_filters, kernel_size=(3,3), strides=1, padding='same')(x)   
    x = LayerNormalization()(x)
    x = ReLU()(x)
    x = AveragePooling2D()(x)
    x = Conv2D(filters=n_filters, kernel_size=(3,3), strides=1, padding='same')(x)
    x = Add()([shortcut, x])
    
    shortcut = x
    x = LayerNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=n_filters, kernel_size=(3,3), strides=1, padding='same')(x)
    x = LayerNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=n_filters, kernel_size=(3,3), strides=1, padding='same')(x)
    x = Add()([shortcut, x])
    
    shortcut = x
    x = LayerNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=n_filters, kernel_size=(3,3), strides=1, padding='same')(x)
    x = LayerNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=n_filters, kernel_size=(3,3), strides=1, padding='same')(x)
    x = Add()([shortcut, x])
    
    x = ReLU()(x)
    x = GlobalAveragePooling2D()(x)
    
    if wgan:
        validity = Dense(units=1)(x)
    else:
        validity = Dense(units=1, activation='sigmoid')(x)
    label = Dense(units=n_labels, activation='softmax')(x)
    model = Model(inputs=inp, outputs=[validity, label])
    return model