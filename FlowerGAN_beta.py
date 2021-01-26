from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, UpSampling2D, BatchNormalization, Reshape, AveragePooling2D, Flatten, Add, Concatenate, Lambda, Dropout, Input, Activation, LeakyReLU, PReLU
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.constraints import max_norm

import tensorflow as tf

from MinibatchStdev import MinibatchStdev
from PixelNormalization import PixelNormalization
from WeightedSum import WeightedSum
from DataLoader import DataLoader

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


class FlowerGAN:
    def __init__(self, batch_size, data_loader):
        self.batch_size = batch_size
        self.data_loader = data_loader

        self.initialization = RandomNormal(stddev=0.02)
        self.const = max_norm(1.0)

        self.n_blocks = 6
        self.latent_dim = 100

        self.d_models = self.define_discriminator(self.n_blocks)
        self.g_models = self.define_generator(self.latent_dim, self.n_blocks)
        self.gan_models = self.define_composite(self.d_models, self.g_models)

        self.n_batch = [16, 16, 16, 8, 4, 4]
        self.n_epochs = [5, 8, 8, 10, 10, 10]

    def add_discriminator_block(self, old_model, n_input_layers=3):
        in_shape = list(old_model.input.shape)

        input_shape = (in_shape[-2].value * 2, in_shape[-2].value * 2, in_shape[-1].value)
        in_image = Input(shape=input_shape)

        d = Conv2D(128, (1, 1), padding='same', kernel_initializer=self.initialization, kernel_constraint=self.const)(in_image)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(128, (3, 3), padding='same', kernel_initializer=self.initialization, kernel_constraint=self.const)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(128, (3, 3), padding='same', kernel_initializer=self.initialization, kernel_constraint=self.const)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = AveragePooling2D()(d)
        block_new = d
        for i in range(n_input_layers, len(old_model.layers)):
            d = old_model.layers[i](d)
        model1 = Model(in_image, d)
        model1.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

        downsample = AveragePooling2D()(in_image)
        block_old = old_model.layers[1](downsample)
        block_old = old_model.layers[2](block_old)

        d = WeightedSum()([block_old, block_new])

        for i in range(n_input_layers, len(old_model.layers)):
            d = old_model.layers[i](d)

        model2 = Model(in_image, d)

        model2.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
        return [model1, model2]

    def define_discriminator(self, n_blocks, input_shape=(4, 4, 3)):
        model_list = list()

        in_image = Input(shape=input_shape)
        d = Conv2D(128, (1, 1), padding='same', kernel_initializer=self.initialization, kernel_constraint=self.const)(in_image)
        d = LeakyReLU(alpha=0.2)(d)
        d = MinibatchStdev()(d)
        d = Conv2D(128, (3, 3), padding='same', kernel_initializer=self.initialization, kernel_constraint=self.const)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(128, (4, 4), padding='same', kernel_initializer=self.initialization, kernel_constraint=self.const)(d)
        d = LeakyReLU(alpha=0.2)(d)

        out_class = Conv2D(filters=1, kernel_size=3, strides=1, padding='valid', use_bias=False)(d)
        model = Model(in_image, out_class)

        model.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
        model_list.append([model, model])

        for i in range(1, n_blocks):
            old_model = model_list[i - 1][0]
            models = self.add_discriminator_block(old_model)
            model_list.append(models)
        return model_list

    def add_generator_block(self, old_model):
        block_end = old_model.layers[-2].output

        upsampling = UpSampling2D()(block_end)
        g = Conv2D(128, (3, 3), padding='same', kernel_initializer=self.initialization, kernel_constraint=self.const)(upsampling)
        g = PixelNormalization()(g)
        g = LeakyReLU(alpha=0.2)(g)
        g = Conv2D(128, (3, 3), padding='same', kernel_initializer=self.initialization, kernel_constraint=self.const)(g)
        g = PixelNormalization()(g)
        g = LeakyReLU(alpha=0.2)(g)

        out_image = Conv2D(3, (1, 1), padding='same', kernel_initializer=self.initialization, kernel_constraint=self.const)(g)
        model1 = Model(old_model.input, out_image)

        out_old = old_model.layers[-1]
        out_image2 = out_old(upsampling)
        merged = WeightedSum()([out_image2, out_image])
        model2 = Model(old_model.input, merged)
        return [model1, model2]

    def define_generator(self, latent_dim, n_blocks, in_dim=4):
        model_list = list()

        in_latent = Input(shape=(latent_dim,))
        g = Dense(128 * in_dim * in_dim, kernel_initializer=self.initialization, kernel_constraint=self.const)(in_latent)
        g = Reshape((in_dim, in_dim, 128))(g)
        g = Conv2D(128, (3, 3), padding='same', kernel_initializer=self.initialization, kernel_constraint=self.const)(g)
        g = PixelNormalization()(g)
        g = LeakyReLU(alpha=0.2)(g)
        g = Conv2D(128, (3, 3), padding='same', kernel_initializer=self.initialization, kernel_constraint=self.const)(g)
        g = PixelNormalization()(g)
        g = LeakyReLU(alpha=0.2)(g)
        out_image = Conv2D(3, (1, 1), padding='same', kernel_initializer=self.initialization, kernel_constraint=self.const)(g)

        model = Model(in_latent, out_image)

        model_list.append([model, model])

        for i in range(1, n_blocks):
            old_model = model_list[i - 1][0]
            models = self.add_generator_block(old_model)
            model_list.append(models)

        return model_list

    @staticmethod
    def define_composite(discriminators, generators):
        model_list = list()

        for i in range(len(discriminators)):
            g_models, d_models = generators[i], discriminators[i]

            d_models[0].trainable = False
            model1 = Sequential()
            model1.add(g_models[0])
            model1.add(d_models[0])
            model1.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

            d_models[1].trainable = False
            model2 = Sequential()
            model2.add(g_models[1])
            model2.add(d_models[1])
            model2.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

            model_list.append([model1, model2])
        return model_list

    def train(self, epochs):
        for epoch in epochs:
            bat_per_epo = int(dataset.shape[0] / self.n_batch)

            X_real = self.data_loader.load_batch(self.batch_size)

            noise = np.random.normal(0., 1., size=(self.latent_dim * self.n_sa))



data_loader = DataLoader()
gan = FlowerGAN(8, data_loader)
