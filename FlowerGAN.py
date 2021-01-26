import cv2
from functools import partial
from tensorflow.keras.layers import Conv2D, MaxPool2D, ZeroPadding2D, Concatenate, Conv2DTranspose, Input, UpSampling2D, LeakyReLU, \
    PReLU, add, Dropout, BatchNormalization, Lambda, Activation, Dense, Flatten, Layer, Reshape, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import vgg16
import tensorflow.keras.backend as K
from DataLoader import DataLoader
import tensorflow as tf
from tensorflow.keras.losses import Huber, MAE, MSE
from tensorflow.keras.models import load_model
from InstanceNormalization import InstanceNormalization
from RandomWeightedAverage import RandomWeightedAverage
from PixelNormalization import PixelNormalization
from MinibatchStdev import MinibatchStdev

import matplotlib.pyplot as plt
import numpy as np

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def reconstruction_loss(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))


def classification_loss(Y_true, Y_pred):
    return K.categorical_crossentropy(Y_true, Y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = K.square(1 - gradient_l2_norm)
    return K.mean(gradient_penalty)


def set_trainable(m, val):
    m.trainable = val
    for l in m.layers:
        l.trainable = val


class FlowerGAN:
    def __init__(self, data_loader, batch_size):
        self.batch_size = batch_size

        self.data_loader = data_loader
        self.gf = 128
        self.channels = 3
        self.image_shape = (256, 256, 3)
        self.num_labels = 102
        self.latent_dim = 100
        self.optimizer_dis = Adam(lr=0.0001, beta_1=0.5)
        self.optimizer_gen = Adam(lr=0.0001, beta_1=0.5)
        self.init = RandomNormal(mean=0.0, stddev=0.02)
        self.sample_noise = np.random.normal(0., 1., size=(3, self.latent_dim))
        self.sample_labels = np.eye(self.num_labels)[:3]

        self.generator = self.build_generator()
        self.generator.summary()
        plot_model(self.generator, to_file='generator_model.png', show_shapes=True)

        self.discriminator = self.build_discriminator()
        self.discriminator.summary()
        plot_model(self.discriminator, to_file='discriminator_model.png', show_shapes=True)

        set_trainable(self.generator, False)
        input_real = Input(shape=self.image_shape)
        input_label = Input(shape=(self.num_labels, ))
        input_noise = Input(shape=(self.latent_dim, ))
        dis_real, cls_real = self.discriminator(input_real)
        input_fake = self.generator([input_noise, input_label])
        dis_fake, _ = self.discriminator(input_fake)
        rnd_w_avg = RandomWeightedAverage(self.batch_size)
        x_hat = rnd_w_avg([input_real, input_fake])
        dis_hat, _ = self.discriminator(x_hat)
        partial_GP_loss = partial(gradient_penalty_loss, averaged_samples=x_hat)
        partial_GP_loss.__name__ = 'gradient_penalty'
        self.train_D = Model([input_real, input_noise, input_label], [dis_real, cls_real, dis_fake, dis_hat])
        self.train_D.compile(loss=[wasserstein_loss, classification_loss, wasserstein_loss, partial_GP_loss], optimizer=self.optimizer_dis, loss_weights=[1., 1., 1., 10.])
        plot_model(self.train_D, to_file='train_discriminator_model.png', show_shapes=True)

        set_trainable(self.generator, True)
        set_trainable(self.discriminator, False)
        noise_input = Input(shape=(self.latent_dim, ))
        label_input = Input(shape=(self.num_labels, ))
        fake_input = self.generator([noise_input, label_input])
        fake_dis, fake_cls = self.discriminator(fake_input)
        self.adversarial = Model([noise_input, label_input], [fake_dis, fake_cls, fake_input])
        self.adversarial.compile(loss=[wasserstein_loss, classification_loss, reconstruction_loss], optimizer=self.optimizer_gen, loss_weights=[1., 1., 10.])

    def build_discriminator(self):
        inp_img = Input(shape=self.image_shape)

        curr_dim = 64
        x = inp_img
        for i in range(4):
            x = ZeroPadding2D(padding=1)(x)
            x = Conv2D(filters=curr_dim * 2, kernel_size=4, strides=2, padding='valid')(x)
            x = LeakyReLU(0.01)(x)
            if i == 0:
                x = MinibatchStdev()(x)
            curr_dim = curr_dim * 2

        out_src = ZeroPadding2D(padding=1)(x)
        out_src = Conv2D(filters=1, kernel_size=3, strides=1, padding='valid', use_bias=False)(out_src)

        out_cls = Conv2D(filters=self.num_labels, kernel_size=3, strides=1, padding='valid', use_bias=False)(x)
        out_cls = Flatten()(out_cls)
        out_cls = Dense(self.num_labels)(out_cls)
        out_cls = Activation('softmax')(out_cls)

        return Model(inp_img, [out_src, out_cls], name='discriminator')

    def build_generator(self):
        def conv2d_transpose(layer_input, filters=16, strides=1, name=None, f_size=4):
            u = Conv2D(filters, kernel_size=f_size, padding='same', use_bias=False)(layer_input)
            u = UpSampling2D(size=2, interpolation='bilinear')(u)
            # u = InstanceNormalization(axis=-1, name=name + "_bn")(u)
            u = PixelNormalization()(u)
            u = PReLU(shared_axes=[1, 2])(u)
            u = Dropout(0.3)(u)
            return u

        input_noise = Input(shape=(self.latent_dim, ))
        input_labels = Input(shape=(self.num_labels, ))

        image_resize = self.image_shape[0] // 8
        x = concatenate([input_noise, input_labels], axis=1)
        x = Dense(image_resize * image_resize * self.gf)(x)
        x = Reshape((image_resize, image_resize, self.gf))(x)

        d1 = conv2d_transpose(x, filters=self.gf * 4, f_size=3, strides=2, name='g_d1_dc')
        d2 = conv2d_transpose(d1, filters=self.gf * 2, f_size=3, strides=2, name='g_d2_dc')
        d3 = conv2d_transpose(d2, filters=self.gf, f_size=3, strides=2, name='g_d2_dc')

        output_img = Conv2D(self.channels, kernel_size=7, strides=1, padding='same', activation='tanh', kernel_initializer=self.init)(d3)

        return Model(inputs=[input_noise, input_labels], outputs=output_img, name='generator')

    def plot_images(self, epoch):
        res = self.generator.predict([self.sample_noise, self.sample_labels])
        comb = (np.hstack([res[0], res[1], res[2]]) + 1) / 2
        plt.figure(figsize=(15, 7))
        plt.imshow(comb[:, :, ::-1])
        plt.axis('off')
        plt.savefig(f'./results/epoch_{epoch}.jpg')
        plt.close()

    def get_generator(self, epoch=None):
        if epoch is not None:
            self.generator.load_weights(
                f'/media/bonilla/HDD_2TB_basura/models/FlowerGAN/gen_epoch_{epoch}.h5')
        return self.generator

    def fit(self, epochs):
        num_critic = 2
        for epoch in range(epochs):
            real_X, labels = self.data_loader.load_batch(batch_size=self.batch_size)
            fake = np.ones((self.batch_size, 16, 16, 1))
            real = - np.ones((self.batch_size, 16, 16, 1))
            dummy = np.zeros((self.batch_size, 16, 16, 1))

            noise = np.random.normal(0., 1., size=(self.batch_size, self.latent_dim))

            self.optimizer_dis.learning_rate.assign(0.0001 * (0.999 ** epoch))
            self.optimizer_gen.learning_rate.assign(0.0001 * (0.999 ** epoch))

            set_trainable(self.discriminator, True)
            if np.random.rand() <= 0.1:
                real, fake = fake, real
            real_X += np.random.normal(0., 0.5, size=(self.batch_size, self.image_shape[0], self.image_shape[1], self.image_shape[2]))
            d_loss = self.train_D.train_on_batch([real_X, noise, labels], [real, labels, fake, dummy])

            if epoch % num_critic == 0:
                set_trainable(self.discriminator, False)
                set_trainable(self.generator, True)
                g_loss = self.adversarial.train_on_batch([noise, labels], [real, labels, real_X])

                print(f"[Epoch: {epoch}/{epochs}]\t[adv_loss: {g_loss}, d_loss: {d_loss}]")

            if epoch % 25 == 0:
                self.plot_images(epoch)
                if epoch % 100 == 0:
                    self.generator.save_weights(f'/media/bonilla/HDD_2TB_basura/models/FlowerGAN/gen_epoch_{epoch}.h5')


if __name__ == '__main__':
    data_loader = DataLoader(256)
    gan = FlowerGAN(data_loader, 8)
    gan.fit(10_000)
