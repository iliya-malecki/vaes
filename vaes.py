import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.engine import data_adapter
import plotly.express as px
from plotly.subplots import make_subplots
L = keras.layers


class Sampling(L.Layer):

	def call(self, inputs):
		z_mean, z_log_var = inputs
		epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
		return z_mean + tf.exp(0.5 * z_log_var) * epsilon




def decaying(length, initial, final, slope):
    def weight_impl(iterations):
        sigmoid_shape = 1 + tf.exp(
            slope * (iterations - length)
        )
        return initial + (final - initial) / sigmoid_shape

    return weight_impl





class VAE(keras.Model):
    def __init__(
        self,
        shape: tuple[int],
        encoding_body: keras.Sequential,
        decoding_body: keras.Sequential,
        classify=True,
        decay_function=None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        if classify and decay_function is None:
            raise ValueError()
        self.decay_function = decay_function

        input = L.Input(shape)
        encoded = encoding_body(input)
        z_mean = L.Dense(2, name='z_mean')(encoded)
        z_log_var = L.Dense(2, name='z_log_var')(encoded)


        if classify:
            self.classifier = keras.Sequential([
                L.InputLayer((2,)),
                L.Dense(10, activation='leaky_relu', name='classifier1'),
                L.Dense(10, activation='leaky_relu', name='classifier2'),
                L.Dense(10, activation='softmax', name='classifier3'),
            ])
        else:
            self.classifier = None

        self.encoder = keras.Model(input, [z_mean, z_log_var])
        self.decoder = keras.Sequential([
            L.InputLayer((2,)),
            decoding_body
        ])

    def call(self, inputs, training=None, mask=None):
        if training:
            m, v = self.encoder(inputs, training=training)
            sample = Sampling()([m,v])
            return self.decoder(sample, training=training)
        else:
            m, v = self.encoder(inputs, training=training)
            return self.decoder(m, training=training)

    def train_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(x, training=True)
            sampled = Sampling()([z_mean, z_log_var])
            decoded = self.decoder(sampled, training=True)
            reconstruction_loss = tf.reduce_mean((x - decoded)**2) * 28 * 28
            kl_divergence_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.abs(z_mean) - tf.exp(z_log_var))
            if self.classifier is not None:
                weight = self.decay_function(tf.cast(self.optimizer.iterations, 'float32'))
                pred = self.classifier(sampled)
                classification_loss = tf.reduce_mean(
                    keras.losses.sparse_categorical_crossentropy(
                        y,
                        pred
                    )
                )
                loss = reconstruction_loss + kl_divergence_loss + classification_loss * weight
                metrics = {
                    'loss':loss,
                    'reconstruction_loss':reconstruction_loss,
                    'kl_divergence_loss':kl_divergence_loss,
                    'classification_accuracy': tf.reduce_mean(keras.metrics.sparse_categorical_accuracy(y, pred)),
                    'classification_loss':classification_loss,
                    'classification_weight':weight,
                }

            else:
                loss = reconstruction_loss + kl_divergence_loss
                metrics = {
                    'loss':loss,
                    'reconstruction_loss':reconstruction_loss,
                    'kl_divergence_loss':kl_divergence_loss,
                }
            if self.optimizer.iterations < 100:
                loss = reconstruction_loss


        self._validate_target_and_loss(y, loss)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return metrics
