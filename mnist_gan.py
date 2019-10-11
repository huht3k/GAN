# coding: utf-8

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.layers import LeakyReLU
from keras.optimizers import SGD, Adam
from keras.datasets import mnist
import numpy as np
from PIL import Image
import argparse
import math


def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    # model.add(BatchNormalization())
    model.add(Activation('tanh'))
    # model.add(Activation('sigmoid'))
    # model.add(BatchNormalization())
    # model.add(LeakyReLU(alpha=0.050))

    model.add(Dense(128*7*7))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    # model.add(Activation('sigmoid'))
    # model.add(LeakyReLU(alpha=0.050))

    model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    # model.add(BatchNormalization())
    # model.add(Activation('sigmoid'))
    # model.add(BatchNormalization())
    # model.add(LeakyReLU(alpha=0.050))

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    # model.add(Activation('sigmoid'))

    return model


def discriminator_model():
    model = Sequential()
    model.add(
            Conv2D(64, (5, 5),
            padding='same',
            input_shape=(28, 28, 1))
            )
    # model.add(Activation('tanh'))
    model.add(Activation('relu'))
    # model.add(LeakyReLU(alpha=0.050))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5)))
    # model.add(Activation('tanh'))
    model.add(Activation('relu'))
    # model.add(LeakyReLU(alpha=0.050))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    # model.add(Activation('tanh'))
    model.add(Activation('relu'))
    # model.add(LeakyReLU(alpha=0.050))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def stack_gan(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image


def train(BATCH_SIZE):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    # X_train = X_train.astype(np.float32) / 127.5
    X_train = X_train[:, :, :, None]
    d = discriminator_model()
    g = generator_model()
    d_on_g = stack_gan(g, d)   # gan: gen with dis supervising (fixed)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    # g_optim = Adam(lr=5e-5)
    g.compile(loss='binary_crossentropy', optimizer="SGD")          # gen loss
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)   # gan loss
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)        # dis loss


    for epoch in range(100):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))    # noise with dim=100
            # noise = np.random.uniform(0, 1, size=(BATCH_SIZE, 100))  # noise with dim=100
            image_batch = X_train[index*BATCH_SIZE: (index+1)*BATCH_SIZE]
            generated_images = g.predict(noise, verbose=0)              # generating images
            if index % 200 == 0:                # save combined inmages
                image = combine_images(generated_images)
                image = image*127.5 + 127.5         # (-1, 1) from tanh ==> (0, 255)
                # image *= 127.5
                Image.fromarray(image.astype(np.uint8)).save(
                    str(epoch)+"_"+str(index)+".png")
            X = np.concatenate((image_batch, generated_images))     # (real, generated)
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE                 # labelling
            d_loss = d.train_on_batch(X, y)                         # TRAINING DIS
            if index % 10 == 0:
                print("batch %d d_loss : %f" % (index, d_loss))
            noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
            # noise = np.random.uniform(0, 1, (BATCH_SIZE, 100))
            d.trainable = False                                     # DIS freeze
            g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE)     # labelling and TRAINING GEN
            d.trainable = True
            if index % 10 == 0:
                print("batch %d g_loss : %f" % (index, g_loss))

        if epoch % 10 == 9:
            g.save_weights('generator', True)
            d.save_weights('discriminator', True)

    g.save_weights('generator', True)
    d.save_weights('discriminator', True)


# inference
def generate(BATCH_SIZE):
    g = generator_model()
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('generator')

    noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
    # noise = np.random.uniform(0, 1, size=(BATCH_SIZE, 100))
    generated_images = g.predict(noise, verbose=0)
    image = combine_images(generated_images)

    image = image*127.5 + 127.5
    # image *= 127.5
    Image.fromarray(image.astype(np.uint8)).save("./generated_image.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size)
