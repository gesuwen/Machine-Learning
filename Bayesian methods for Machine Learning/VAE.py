# -*- coding: utf-8 -*-
# A Variational Autoencoder trained on the MNIST dataset.

import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Lambda, InputLayer, concatenate
from keras.models import Model, Sequential
from keras import backend as K
from keras.datasets import mnist
from keras.utils import np_utils

# Variational Lower Bound
def vlb_binomial(x, x_decoded_mean, t_mean, t_log_var):
    """Returns the value of Variational Lower Bound
    
    The inputs are tf.Tensor
        x: (batch_size x number_of_pixels) matrix with one image per row with zeros and ones
        x_decoded_mean: (batch_size x number_of_pixels) mean of the distribution p(x | t), real numbers from 0 to 1
        t_mean: (batch_size x latent_dim) mean vector of the (normal) distribution q(t | x)
        t_log_var: (batch_size x latent_dim) logarithm of the variance vector of the (normal) distribution q(t | x)
    
    Returns:
        A tf.Tensor with one element (averaged across the batch), VLB
    """
    klterm=0.5*K.sum(-1-t_log_var+K.square(t_mean)+K.exp(t_log_var),axis=1)#batch_size
    reconst=K.sum(K.binary_crossentropy(x,x_decoded_mean),axis=1)
    return K.mean(klterm+reconst)


def create_encoder(input_dim):
    # Encoder network.
    # We instantiate these layers separately so as to reuse them later
    encoder = Sequential(name='encoder')
    encoder.add(InputLayer([input_dim]))
    encoder.add(Dense(intermediate_dim, activation='relu'))
    encoder.add(Dense(2 * latent_dim))
    return encoder


# Sampling from the distribution 
#     q(t | x) = N(t_mean, exp(t_log_var))
# with reparametrization trick.
def sampling(args):
    """Returns sample from a distribution N(args[0], diag(args[1]))
    
    The sample should be computed with reparametrization trick.
    
    The inputs are tf.Tensor
        args[0]: (batch_size x latent_dim) mean of the desired distribution
        args[1]: (batch_size x latent_dim) logarithm of the variance vector of the desired distribution
    
    Returns:
        A tf.Tensor of size (batch_size x latent_dim), the samples.
    """
    t_mean, t_log_var = args
    output = tf.random_normal(t_mean.get_shape())
    output = output * tf.exp(0.5 * t_log_var) + t_mean
    return output


def create_decoder(input_dim):
    # Decoder network
    # We instantiate these layers separately so as to reuse them later
    decoder = Sequential(name='decoder')
    decoder.add(InputLayer([input_dim]))
    decoder.add(Dense(intermediate_dim, activation='relu'))
    decoder.add(Dense(original_dim, activation='sigmoid'))
    return decoder


if __name__ == '__main__':
    # Start tf session so we can run code.
    sess = tf.InteractiveSession()
    # Connect keras to the created session.
    K.set_session(sess)
    
    batch_size = 100
    original_dim = 784 # Number of pixels in MNIST images.
    latent_dim = 100 # d, dimensionality of the latent code t.
    intermediate_dim = 256 # Size of the hidden layer.
    epochs = 20
    
    x = Input(batch_shape=(batch_size, original_dim))
    
    encoder = create_encoder(original_dim)

    get_t_mean = Lambda(lambda h: h[:, :latent_dim])
    get_t_log_var = Lambda(lambda h: h[:, latent_dim:])
    h = encoder(x)
    t_mean = get_t_mean(h)
    t_log_var = get_t_log_var(h)
    
    t = Lambda(sampling)([t_mean, t_log_var])
    
    decoder = create_decoder(latent_dim)
    x_decoded_mean = decoder(t)
    
    
    
    loss = vlb_binomial(x, x_decoded_mean, t_mean, t_log_var)
    vae = Model(x, x_decoded_mean)
    # Keras will provide input (x) and output (x_decoded_mean) to the function that
    # should construct loss, but since our function also depends on other
    # things (e.g. t_means), it is easier to build the loss in advance and pass
    # a function that always returns it.
    vae.compile(optimizer=keras.optimizers.RMSprop(lr=0.001), loss=lambda x, y: loss)
    
    # Load and prepare the data
    # train the VAE on MNIST digits
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # One hot encoding.
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    
    # Training the model
    hist = vae.fit(x=x_train, y=x_train,
                   shuffle=True,
                   epochs=epochs,
                   batch_size=batch_size,
                   validation_data=(x_test, x_test),
                   verbose=2)
    
    # Visualize reconstructions for train and validation data
    fig = plt.figure(figsize=(10, 10))
    for fid_idx, (data, title) in enumerate(
                zip([x_train, x_test], ['Train', 'Validation'])):
        n = 10  # figure with 10 x 2 digits
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * 2))
        decoded = sess.run(x_decoded_mean, feed_dict={x: data[:batch_size, :]})
        for i in range(10):
            figure[i * digit_size: (i + 1) * digit_size,
                   :digit_size] = data[i, :].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   digit_size:] = decoded[i, :].reshape(digit_size, digit_size)
        ax = fig.add_subplot(1, 2, fid_idx + 1)
        ax.imshow(figure, cmap='Greys_r')
        ax.set_title(title)
        ax.axis('off')
    plt.show()
    
    
    # Hallucinating new data
    # generate new samples of images from your trained VAE
    n_samples = 10  # To pass automatic grading please use at least 2 samples here.
    # sampled_im_mean is a tf.Tensor of size 10 x 784 with 10 random
    # images sampled from the vae model.
    sampled_im_mean = decoder(tf.random_normal((n_samples,latent_dim)))
    
    sampled_im_mean_np = sess.run(sampled_im_mean)
    # Show the sampled images.
    plt.figure()
    for i in range(n_samples):
        ax = plt.subplot(n_samples // 5 + 1, 5, i + 1)
        plt.imshow(sampled_im_mean_np[i, :].reshape(28, 28), cmap='gray')
        ax.axis('off')
    plt.show()
    
    
    # Conditional VAE
    # Implement CVAE model
    # One-hot labels placeholder.
    x = Input(batch_shape=(batch_size, original_dim))
    label = Input(batch_shape=(batch_size, 10))
    cond_encoder = create_encoder(original_dim+10)
    cond_h = cond_encoder(concatenate([x, label]))
    cond_t_mean =  get_t_mean(cond_h) # Mean of the latent code (without label) for cvae model.
    cond_t_log_var = get_t_log_var(cond_h) # Logarithm of the variance of the latent code (without label) for cvae model.
    cond_t = Lambda(sampling)([cond_t_mean, cond_t_log_var])
    cond_decoder = create_decoder(latent_dim+10)
    cond_x_decoded_mean = cond_decoder(concatenate([cond_t, label])) # Final output of the cvae model.
    
    # Define the loss and the model
    conditional_loss = vlb_binomial(x, cond_x_decoded_mean, cond_t_mean, cond_t_log_var)
    cvae = Model([x, label], cond_x_decoded_mean)
    cvae.compile(optimizer=keras.optimizers.RMSprop(lr=0.001), loss=lambda x, y: conditional_loss)
    
    # Train the model
    hist = cvae.fit(x=[x_train, y_train],
                    y=x_train,
                    shuffle=True,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=([x_test, y_test], x_test),
                    verbose=2)
    
    # Visualize reconstructions for train and validation data
    fig = plt.figure(figsize=(10, 10))
    for fid_idx, (x_data, y_data, title) in enumerate(
                zip([x_train, x_test], [y_train, y_test], ['Train', 'Validation'])):
        n = 10  # figure with 10 x 2 digits
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * 2))
        decoded = sess.run(cond_x_decoded_mean,
                           feed_dict={x: x_data[:batch_size, :],
                                      label: y_data[:batch_size, :]})
        for i in range(10):
            figure[i * digit_size: (i + 1) * digit_size,
                   :digit_size] = x_data[i, :].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   digit_size:] = decoded[i, :].reshape(digit_size, digit_size)
        ax = fig.add_subplot(1, 2, fid_idx + 1)
        ax.imshow(figure, cmap='Greys_r')
        ax.set_title(title)
        ax.axis('off')
    plt.show()
    
    # Conditionally hallucinate data
    # Prepare one hot labels of form
    #   0 0 0 0 0 1 1 1 1 1 2 2 2 2 2 ...
    # to sample five zeros, five ones, etc
    curr_labels = np.eye(10)
    curr_labels = np.repeat(curr_labels, 5, axis=0)  # Its shape is 50 x 10.
    # cond_sampled_im_mean is a tf.Tensor of size 50 x 784 with 5 random zeros,
    # then 5 random ones, etc sampled from the cvae model.
    cond_sampled_im_mean = cond_decoder(concatenate([tf.random_normal((50,latent_dim)), tf.convert_to_tensor(curr_labels, dtype=tf.float32)]))
    
    cond_sampled_im_mean_np = sess.run(cond_sampled_im_mean)
    # Show the sampled images.
    plt.figure(figsize=(10, 10))
    global_idx = 0
    for digit in range(10):
        for _ in range(5):
            ax = plt.subplot(10, 5, global_idx + 1)
            plt.imshow(cond_sampled_im_mean_np[global_idx, :].reshape(28, 28), cmap='gray')
            ax.axis('off')
            global_idx += 1
    plt.show()