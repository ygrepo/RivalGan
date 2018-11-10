""" Utility methods to build GAN netwworks """
import tensorflow as tf


def create_generator_layers(prior_z, g_h1, g_h2, g_h3, keep_prob=None):
    """ Fully connect the layers
    :param prior_z: input noise
    :param g_h1: hidden layer 1 weights and biases
    :param g_h2: hidden layer 2 weights and biases
    :param g_h3: hidden layer 3 weights and biases
    :return: last layer
    """

    g1 = tf.nn.leaky_relu(tf.matmul(prior_z, g_h1['weights']) + g_h1['biases'])
    if keep_prob is not None:
        drop_out = tf.nn.dropout(g1, keep_prob)  # DROP-OUT here
    g2 = tf.matmul(g1, g_h2['weights']) + g_h2['biases']
    out = tf.nn.tanh(tf.matmul(g2, g_h3['weights']) + g_h3['biases'])
    # Activation changed from sigmoid to tanh
    return out


def create_descriminator_layers(X, d_h1, d_h2, d_h3):
    """ Fully connect the layers
     :param X: input data
     :param d_h1: hidden layer 1 weights and biases
     :param d_h2: hidden layer 2 weights and biases
     :param d_h3: hidden layer 3 weights and biases
     :return: last layer
     """

    d1 = tf.nn.leaky_relu(tf.matmul(X, d_h1['weights']) + d_h1['biases'])
    d2 = tf.matmul(d1, d_h2['weights']) + d_h2['biases']
    out = tf.nn.sigmoid(tf.matmul(d2, d_h3['weights']) + d_h3['biases'])
    return out


def Q(X, h1, h2):
    """ Build Q prior network """
    h = tf.nn.relu(tf.matmul(X, h1['weights']) + h1['biases'])
    z = tf.matmul(h, h2['weights']) + h2['biases']
    return z


def P(prior_z, h1, h2):
    """ Build P prior network """
    h = tf.nn.relu(tf.matmul(prior_z, h1['weights']) + h1['biases'])
    logits = tf.matmul(h, h2['weights']) + h2['biases']
    prob = tf.nn.sigmoid(logits)
    return prob, logits
