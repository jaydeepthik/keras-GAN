# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 23:55:11 2018

@author: jaydeep thik
"""

import keras
from keras import layers, models
import numpy as np
import os
from keras.preprocessing import image

latent_dim = 32
channels =3
height =32
width = 32

##generator network
generator_input = keras.Input(shape=(latent_dim, ))

x = layers.Dense(128 * 16 * 16)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((16,16,128))(x)

x = layers.Conv2D(256, 5, padding='same')(x)  # 16*16*256
x = layers.BatchNormalization(momentum=0.8)(x)
x = layers.LeakyReLU()(x)
x = layers.UpSampling2D()(x)
#x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x) # 32*32*256 upsampling
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.BatchNormalization(momentum=0.8)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.BatchNormalization(momentum=0.8)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x) #sample 32*32*3 image

generator = models.Model(generator_input, x)
generator.summary()

#descriminator network
descriminator_input = layers.Input(shape=(height, width, channels))
x = layers.Conv2D(128, 3)(descriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Dropout(0.25)(x)

x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.BatchNormalization(momentum=0.8)(x)
x = layers.LeakyReLU()(x)
x = layers.Dropout(0.25)(x)

x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.BatchNormalization(momentum=0.8)(x)
x = layers.LeakyReLU()(x)
x = layers.Dropout(0.25)(x)

x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)
#x = layers.Dense(128, activation='tanh')(x)
x = layers.Dropout(0.4)(x)

x = layers.Dense(1, activation='sigmoid')(x)
descriminator = models.Model(descriminator_input, x)
descriminator.summary()

#descriminator_optimizer = keras.optimizers.Adam(lr=0.0004,clipvalue=1.0, decay=1e-8)
descriminator_optimizer = keras.optimizers.Adam(lr=0.0002,clipvalue=1.0, decay=1e-8)

descriminator.compile(optimizer=descriminator_optimizer, loss='binary_crossentropy')

descriminator.trainable = False

#gan
gan_input = layers.Input(shape=(latent_dim, ))
gan_output = descriminator(generator(gan_input))
gan = models.Model(gan_input, gan_output)

#gan_optimizer = keras.optimizers.Adam(lr=0.0004,clipvalue=1.0, decay=1e-8)
gan_optimizer = keras.optimizers.Adam(lr=0.0002,clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')


#training the gan
(X_train, y_train), (_,_) =keras.datasets.cifar10.load_data()
X_train  = X_train[y_train.flatten() == 6]
X_train = X_train.reshape((X_train.shape[0], )+(height, width, channels)).astype('float32')/255.

iterations = 10000
batch_size = 32
save_dir = "./gan_generated_1"
start = 0

for step in range(iterations):
    random_latent_vectors = np.random.normal(size = (batch_size, latent_dim)) #sample random points
    generated_images = generator.predict(random_latent_vectors) #output of the generator (decoded fake images)
    
    stop = start + batch_size
    real_images = X_train[start:stop]
    combined_images = np.concatenate([generated_images, real_images])#combine fake and real images
    
    labels = np.concatenate([np.zeros((batch_size, 1)), np.ones((batch_size, 1))])
    #labels += 0.005*np.random.random(size=labels.shape) #add noise to the labels
    
    d_loss = descriminator.train_on_batch(combined_images, labels) #train the descriminator
    
    random_latent_vectors = np.random.normal(size = (batch_size, latent_dim))#sample random vectors
    misleading_targets = np.ones((batch_size, 1)) #labels that lie that they are real images
    
    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets) #train generator model via gan
    
    start += batch_size
    if start > len(X_train)-batch_size:
        start = 0
    
    if step %10 ==0:
        gan.save_weights('gan.h5')
        
        print("descriminator loss ",step," : ", d_loss)
        print("adverserial loss : ",step," : ", a_loss)
        
        img = image.array_to_img(generated_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, "generated_frog_"+str(step)+'.png'))
        img = image.array_to_img(real_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, "real_frog_"+str(step)+'.png'))
        
        