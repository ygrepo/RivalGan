""" Implementation of Least Square GAN """

import tensorflow as tf

from rivalgan.base_gan import BaseGAN
from rivalgan.net_layers import create_descriminator_layers, create_generator_layers
from rivalgan.utils import xavier_init


class LeastSquaresGAN(BaseGAN):
    """ Implementation of Least Square GAN """

    def generator(self, h1_nodes=500, h2_nodes=500):
        """ Create generator
        Generator weights/biases
        g_h1: hidden layer 1 weights and biases
        g_h2: hidden layer 2 weights and biases
        g_h3: hidden layer 3 weights and biases
        """
        g_h1 = {'weights': tf.Variable(xavier_init([self.z_dim, h1_nodes], 'g_w1', tf.float32)),
                'biases': tf.Variable(tf.zeros(h1_nodes), name='g_b1', dtype=tf.float32)}
        g_h2 = {'weights': tf.Variable(xavier_init([h1_nodes, h2_nodes], 'g_w2', tf.float32)),
                'biases': tf.Variable(tf.zeros([h2_nodes]), name='g_b2', dtype=tf.float32)}
        g_h3 = {'weights': tf.Variable(xavier_init([h2_nodes, self.X_node], 'g_w3', tf.float32)),
                'biases': tf.Variable(tf.zeros([self.X_node]), name='g_b3', dtype=tf.float32)}

        # Generate fake x's from output layer of generator

        self.gen_X = create_generator_layers(self.prior_z, g_h1, g_h2, g_h3, self.keep_prob)

        # List of 'var_list' for generator trainer to optimise
        self.gen_params = [g_h1['weights'], g_h1['biases'],
                           g_h2['weights'], g_h2['biases'],
                           g_h3['weights'], g_h3['biases']]

    def discriminator(self, h1_nodes=500, h2_nodes=500):
        """ Create discriminator
         Discriminator weights/biases
         d_h1: hidden layer 1 weights and biases
         d_h2: hidden layer 2 weights and biases
         d_h3: hidden layer 3 weights and biases
         """
        d_h1 = {'weights': tf.Variable(xavier_init([self.X_node, h1_nodes], 'd_w1', tf.float32)),
                'biases': tf.Variable(tf.zeros([h1_nodes]), name='d_b1', dtype=tf.float32)}
        d_h2 = {'weights': tf.Variable(xavier_init([h1_nodes, h2_nodes], 'd_w2', tf.float32)),
                'biases': tf.Variable(tf.zeros([h2_nodes]), name='d_b2', dtype=tf.float32)}
        d_h3 = {'weights': tf.Variable(xavier_init([h2_nodes, self.y_node], 'd_w3', tf.float32)),
                'biases': tf.Variable(tf.zeros([self.y_node]), name='d_b3', dtype=tf.float32)}
        # Output shape has 2 features; Shape: [batch(real) + batch(gen.), 2]

        # Real data output
        self.y_data = create_descriminator_layers(self.X, d_h1, d_h2, d_h3)  # 'y_data' == D(x)

        # Generated data output
        self.gen_y = create_descriminator_layers(self.gen_X, d_h1, d_h2, d_h3)  # 'gen_y' == D[G(z)]

        ## List of 'var_list' for discriminator trainer to optimise ##
        self.dis_params = [d_h1['weights'], d_h1['biases'],
                           d_h2['weights'], d_h2['biases'],
                           d_h3['weights'], d_h3['biases']]

    def optimise(self, train_step=0.0001):
        """ Optimizers for discriminator and generator losses """

        self.d_loss = 0.5 * (tf.reduce_mean((self.y_data - 1) ** 2) + tf.reduce_mean(self.gen_y ** 2))
        self.g_loss = 0.5 * tf.reduce_mean((self.gen_y - 1) ** 2)

        # Optimisation Trainers
        self.d_trainer = tf.train.GradientDescentOptimizer(learning_rate=train_step).minimize(self.d_loss,
                                                                                              var_list=self.dis_params)
        self.g_trainer = tf.train.AdamOptimizer(learning_rate=train_step).minimize(self.g_loss,
                                                                                   var_list=self.gen_params)
