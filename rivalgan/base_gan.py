""" Base class for GANs"""

from abc import ABC, abstractmethod

import tensorflow as tf


class BaseGAN(ABC):
    """ Base class for GANs"""

    def __init__(self, gan_configuration):

        self.name = gan_configuration.name

        # Initialise inputs
        self.batch, self.X_node, self.y_node, self.z_dim = gan_configuration.batch_size, gan_configuration.X_nodes, \
                                                           gan_configuration.y_nodes, gan_configuration.z_dims

        # Initialise placeholders
        self.X = tf.placeholder(tf.float32, [None, self.X_node], name=gan_configuration.X_name)
        self.prior_z = tf.placeholder(tf.float32, [None, self.z_dim], name=gan_configuration.z_name)

        # Generator parameters
        self.gen_X, self.gen_params = None, None

        # Discriminator parameters
        self.y_data, self.gen_y, self.dis_params = None, None, None

        # Optimisation parameters
        self.d_loss, self.g_loss, self.d_trainer, self.g_trainer = None, None, None, None

        if gan_configuration.drop_out:
            self.keep_prob = tf.placeholder(tf.float32)
        else:
            self.keep_prob = None

        self.tf_loss_ph, self.tf_loss_summary, self.loss_summaries = None, None, None

        self.tf_loss_ph = tf.placeholder(tf.float32, shape=None, name='loss_summary')
        self.tf_loss_summary = tf.summary.scalar('loss', self.tf_loss_ph)
        self.loss_summaries = tf.summary.merge_all()

        super().__init__()

    @abstractmethod
    def generator(self, h1_nodes=500, h2_nodes=500):
        """ Method to create the generator """
        pass

    @abstractmethod
    def discriminator(self, h1_nodes=500, h2_nodes=500):
        """ Method to create the discriminator """
        pass
