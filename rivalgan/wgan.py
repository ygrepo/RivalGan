""" Implementation of the Wasserstein GAN and improved Wasserstein GAN """

import tensorflow as tf

from rivalgan.base_gan import BaseGAN
from rivalgan.net_layers import create_descriminator_layers, create_generator_layers
from rivalgan.utils import xavier_init


class WassersteinGAN(BaseGAN):
    """ Implementation of the Wasserstein GAN and improved Wasserstein GAN """

    def __init__(self, gan_configuration):
        self.d_h1, self.d_h2, self.d_h3 = None, None, None
        self.clip_dis = None
        if gan_configuration._name == 'IWGAN':
            self.improved_wgan = True
        super().__init__(gan_configuration)

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
        self.d_h1 = {'weights': tf.Variable(xavier_init([self.X_node, h1_nodes], 'd_w1', tf.float32)),
                     'biases': tf.Variable(tf.zeros([h1_nodes]), name='d_b1', dtype=tf.float32)}
        self.d_h2 = {'weights': tf.Variable(xavier_init([h1_nodes, h2_nodes], 'd_w2', tf.float32)),
                     'biases': tf.Variable(tf.zeros([h2_nodes]), name='d_b2', dtype=tf.float32)}
        self.d_h3 = {'weights': tf.Variable(xavier_init([h2_nodes, self.y_node], 'd_w3', tf.float32)),
                     'biases': tf.Variable(tf.zeros([self.y_node]), name='d_b3', dtype=tf.float32)}
        # Output shape has 2 features; Shape: [batch(real) + batch(gen.), 2]

        # Real data output
        self.y_data = create_descriminator_layers(self.X, self.d_h1, self.d_h2, self.d_h3)  # 'y_data' == D(x)

        # Generated data output
        self.gen_y = create_descriminator_layers(self.gen_X, self.d_h1, self.d_h2, self.d_h3)  # 'gen_y' == D[G(z)]

        ## List of 'var_list' for discriminator trainer to optimise ##
        self.dis_params = [self.d_h1['weights'], self.d_h1['biases'],
                           self.d_h2['weights'], self.d_h2['biases'],
                           self.d_h3['weights'], self.d_h3['biases']]

        if not self.improved_wgan:
            # Clipping of discriminator
            print("Clipping discriminator weights")
            self.clip_dis = [param.assign(tf.clip_by_value(param, -0.05, 0.05)) for param in self.dis_params]

    def optimise(self, train_step=0.0001):
        """ Optimizers for discriminator and generator losses """

        # Improved GAN (with regularization)
        if self.improved_wgan:
            eps = tf.random_uniform([self.X_node, ], minval=0., maxval=1.)
            X_inter = eps * self.X + (1. - eps) * self.gen_X
            interp = create_descriminator_layers(X_inter, self.d_h1, self.d_h2, self.d_h3)  # Interpolation
            grad = tf.gradients(interp, [X_inter])[0]
            grad_norm = tf.sqrt(tf.reduce_sum((grad) ** 2, axis=1))
            lam = 10
            grad_pen = lam * tf.reduce_mean((grad_norm - 1) ** 2)

            self.d_loss = tf.reduce_mean(self.y_data) - tf.reduce_mean(self.gen_y) + grad_pen
            self.g_loss = - tf.reduce_mean(self.gen_y)

            self.d_trainer = tf.train.GradientDescentOptimizer(learning_rate=train_step).minimize(self.d_loss,
                                                                                                  var_list=self.dis_params)
            self.g_trainer = tf.train.AdamOptimizer(learning_rate=train_step).minimize(self.g_loss,
                                                                                       var_list=self.gen_params)


        else:
            self.d_loss = tf.reduce_mean(self.y_data) - tf.reduce_mean(self.gen_y)
            self.g_loss = - tf.reduce_mean(self.gen_y)

            # Optimisation Trainers
            self.d_trainer = tf.train.GradientDescentOptimizer(learning_rate=train_step).minimize(self.d_loss,
                                                                                                  var_list=self.dis_params)
            self.g_trainer = tf.train.AdamOptimizer(learning_rate=train_step).minimize(self.g_loss,
                                                                                       var_list=self.gen_params)
