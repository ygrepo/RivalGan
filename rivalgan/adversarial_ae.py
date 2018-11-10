""" Adverserial autoencoder GAN"""

import tensorflow as tf

from rivalgan.net_layers import P, Q
from rivalgan.utils import xavier_init


class AdversarialAE:
    def __init__(self, gan_name, batches=2048, X_nodes=29, y_nodes=1, z_dims=100, X_name='X_data', z_name='z_prior'):
        self.name = gan_name

        # Initialise inputs
        self.batch_, self.X_node_, self.y_node_, self.z_dim_ = batches, X_nodes, y_nodes, z_dims

        # Initialise placeholders
        self.X_ = tf.placeholder(tf.float32, [None, self.X_node_], name=X_name)
        self.prior_z_ = tf.placeholder(tf.float32, [None, self.z_dim_], name=z_name)

        # Generator parameters
        self.gen_X_, self.gen_params_ = None, None

        # Discriminator parameters
        self.y_data_, self.gen_y_, self.dis_params_ = None, None, None

        # Optimisation parameters
        self.d_loss_, self.g_loss_, self.d_trainer_, self.g_trainer_ = None, None, None, None

        self.z_sample_, self.logits_ = None, None

        self.tf_dloss_ph_, self.tf_gloss_ph_, self.loss_summaries_ = None, None, None

        self.d_h1_, self.d_h2_ = None, None

        self.z_sample_, self.logits_, self.recon_loss_, self.AE_trainer_ = None, None, None, None

        Q_h1_nodes = 500
        Q_h2_nodes = 500

        Q_h1 = {'weights': tf.Variable(xavier_init([self.X_node_, Q_h1_nodes], 'Q_w1', tf.float32)),
                'biases': tf.Variable(tf.zeros(Q_h1_nodes), name='Q_b1', dtype=tf.float32)}
        Q_h2 = {'weights': tf.Variable(xavier_init([Q_h1_nodes, z_dims], 'Q_w2', tf.float32)),
                'biases': tf.Variable(tf.zeros([z_dims]), name='Q_b2', dtype=tf.float32)}

        # List of 'var_list' for generator trainer to optimise
        self.Q_params_ = [Q_h1['weights'], Q_h1['biases'],
                          Q_h2['weights'], Q_h2['biases']]

        P_h1_nodes = 500

        P_h1 = {'weights': tf.Variable(xavier_init([self.z_dim_, P_h1_nodes], 'P_w1', tf.float32)),
                'biases': tf.Variable(tf.zeros([P_h1_nodes]), name='P_b1', dtype=tf.float32)}
        P_h2 = {'weights': tf.Variable(xavier_init([P_h1_nodes, self.X_node_], 'P_w2', tf.float32)),
                'biases': tf.Variable(tf.zeros([self.X_node_]), name='P_b2', dtype=tf.float32)}

        self.P_params_ = [P_h1['weights'], P_h1['biases'],
                          P_h2['weights'], P_h2['biases']]

        self.z_sample_ = Q(self.X_, Q_h1, Q_h2)
        _, self.logits_ = P(self.z_sample_, P_h1, P_h2)

        # Sample from random z
        self.gen_X_, _ = P(self.prior_z_, P_h1, P_h2)

        D_h1_nodes = 500
        D_h2_nodes = 500
        D_h1 = {'weights': tf.Variable(xavier_init([self.z_dim_, D_h1_nodes], 'D_w1', tf.float32)),
                'biases': tf.Variable(tf.zeros([D_h1_nodes]), name='D_b1', dtype=tf.float32)}
        D_h2 = {'weights': tf.Variable(xavier_init([D_h1_nodes, 1], 'D_w2', tf.float32)),
                'biases': tf.Variable(tf.zeros([1]), name='D_b2', dtype=tf.float32)}

        self.D_params_ = [D_h1['weights'], D_h1['biases'],
                          D_h2['weights'], D_h2['biases']]

        _, self.y_data_ = P(self.prior_z_, D_h1, D_h2)
        _, self.gen_y_ = P(self.z_sample_, D_h1, D_h2)

        self.keep_prob_ = None

    def optimise(self, train_step=0.0001):
        """ Training """

        # E[log P(X|z)]
        self.recon_loss_ = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits_, labels=self.X_))
        self.AE_trainer_ = tf.train.AdamOptimizer().minimize(self.recon_loss_, var_list=self.Q_params_ + self.P_params_)

        # Adversarial loss to approx. Q(z|X)
        self.d_loss_ = -tf.reduce_mean(tf.log(self.y_data_) + tf.log(1.0 - self.gen_y_))
        self.g_loss_ = -tf.reduce_mean(tf.log(self.gen_y_))

        # Optimisation Trainers
        self.d_trainer_ = tf.train.AdamOptimizer(learning_rate=train_step).minimize(self.d_loss_,
                                                                                    var_list=self.D_params_)
        self.g_trainer_ = tf.train.AdamOptimizer(learning_rate=train_step).minimize(self.g_loss_,
                                                                                    var_list=self.Q_params_)

        self.tf_loss_ph_ = tf.placeholder(tf.float32, shape=None, name='loss_summary')
        self.tf_loss_summary_ = tf.summary.scalar('loss', self.tf_loss_ph_)
        self.loss_summaries_ = tf.summary.merge_all()
