from __future__ import print_function, division

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from glob import glob
import os
from utils import *
from six.moves import xrange


class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 48
        self.img_cols = 48
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(64 * 8 * 3 * 3, activation="relu", input_dim=self.latent_dim))
        # 转换成3x3x64*8维，作为卷积层的输入
        model.add(Reshape((3, 3, 64 * 8)))
        # 变为6x6x64*8维
        model.add(UpSampling2D())

        model.add(Conv2D(64 * 4, kernel_size=5, padding="same"))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation("relu"))
        # 变为12x12x64*4维
        model.add(UpSampling2D())

        model.add(Conv2D(64 * 2, kernel_size=5, padding="same"))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation("relu"))
        # 变为24x24x64*2维
        model.add(UpSampling2D())

        # 变为24x24x64维
        model.add(Conv2D(64, kernel_size=5, padding="same"))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation("relu"))
        # 变为48x48x64维
        model.add(UpSampling2D())

        # 变为48x48x3维
        model.add(Conv2D(self.channels, kernel_size=5, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        # 生成100维的噪声
        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        # 48x48x3变为24x24x64
        model.add(Conv2D(64, kernel_size=5, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        # 变为12x12x64*2
        model.add(Conv2D(64 * 2, kernel_size=5, strides=2, padding="same"))
        # model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        # 变为6x6x64*4
        model.add(Conv2D(64 * 4, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        # 变为3x3x64*8
        model.add(Conv2D(64 * 8, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        # 3*3*64*8
        model.add(Flatten())
        # 生成一维数据
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        # (X_train, _), (_, _) = mnist.load_data()
        ## 归一化
        # X_train = X_train / 127.5 - 1.  # shape = (60000, 28, 28)
        # X_train = np.expand_dims(X_train, axis=3)  # shape=(60000, 28, 28, 1)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            datas = glob(os.path.join(
                './data', 'faces', '*.jpg'))
            np.random.shuffle(datas)
            batch_idxs = len(datas) // batch_size

            for idx in xrange(0, int(batch_idxs)):
                batch_files = datas[idx * batch_size:(idx + 1) * batch_size]
                X_train = np.array([
                    get_image(batch_file, input_height=96, input_width=96, resize_height=48, resize_width=48, crop=True,
                              grayscale=False) for
                    batch_file in batch_files]).astype(np.float32)

                idx = np.random.randint(0, X_train.shape[0], batch_size)  # 在[0,60000)上，随机生成128维向量
                imgs = X_train[idx]  # shape=(128, 28, 28, 1),随机挑选出128张图片

                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))  # 生成(128,100)维矩阵
                gen_imgs = self.generator.predict(noise)  # 从低维到高维

                d_loss_real = self.discriminator.train_on_batch(imgs, valid)  # 从高维到低维
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)  # 从高维到低维
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                g_loss = self.combined.train_on_batch(noise, valid)

                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

                if epoch % save_interval == 0:
                    self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))  # 生成(25,100)维随机矩阵
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0:3])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/anim_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=4000, batch_size=32, save_interval=50)
