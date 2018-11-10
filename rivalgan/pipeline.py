""" Main pipeline """
import os
import random
import time
from pathlib import Path

from sklearn.model_selection import ShuffleSplit
from tensorflow.contrib.tensorboard.plugins import projector

from rivalgan.configuration import Configuration
from rivalgan.gan import create_gan
from rivalgan.gan_configuration import GANConfiguration
from rivalgan.metrics import precision_score, recall_score, compute_F1
from rivalgan.metrics import report_scores, plot_metrics, accuracy_score, plot_scores
from rivalgan.utils import *
from rivalgan.utils import create_imbalanced_learn_classifier, create_classifier, hms_string
from rivalgan.utils import load_classifier
from rivalgan.utils import plot_decision_function, parse_arguments
from rivalgan.utils import plot_pca, plot_tsne, plot_learning_curve, create_dataset
from rivalgan.utils import preprocess_credit_card_data, compute_number_of_features
from rivalgan.utils import read_data, filename_timeftm
from rivalgan.utils import uniform_draw_feat_class, compute_steady_frame


class Pipeline:
    """ Create pipeline  """

    def __init__(self):
        self.config = Configuration()
        self.rd_data_samples = 1000
        self.rd_data_features = 2
        self.rd_data_classes = 3
        self.train_df = None
        self.X_train_df = None
        self.X_train = None
        self.X_test_df = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.non_corr_column_names = None
        self.num_features = 30
        self.gan_X = None
        self.X_rd, self.y_rd = None, None

    def set_configuration(self, dargs):
        """ Configure the pipeline using a preset configuration """

        self.config.set_configuration(dargs)

    def read_process_data(self):
        """ Read pickled data set """

        data_dir = Path(self.config.get('DATA_FOLDER'))
        data = read_data(data_dir.joinpath(self.config.get('PICKLED_DATA')), True)
        corr_columns = []
        parameters = [self.config.get('SEED'), data, corr_columns, "train_test_split", self.config.get('CLASS_NAME')]
        [self.train_df,
         self.X_train_df,
         self.X_train,
         self.X_test_df,
         self.X_test,
         self.y_train,
         self.y_test] = preprocess_credit_card_data(parameters)
        self.non_corr_column_names, self.num_features = \
            compute_number_of_features(data, self.config.get('FILTERED_COLUMN_NAMES'))
        return data

    def train_classifier(self, classifier):
        """Train the baseline classifier """

        # Measure Start Time
        start_time = time.time()

        # Fit
        classifier.fit(self.X_train, self.y_train.ravel())

        # Measure End Time
        print('Time elapsed to train: ', hms_string(time.time() - start_time))

        classifier_name = self.config.get('CLASSIFIER')
        path_to_file = Path(self.config.get('CACHE_FOLDER') + classifier_name + '_' + self.config.get('APP') + ".pkl")
        print("Saving {} in {}".format(classifier_name, path_to_file))
        pickle.dump(classifier, open(path_to_file, 'wb'))

    def run_train_classifier(self):
        """ Train classifier or sampled classifier """

        if self.config.get('CLASSIFIER') is None and self.config.get('SAMPLER') is None:
            print('No classifier to train')
            return

        classifier_name = self.config.get('CLASSIFIER')
        classifier = create_classifier(classifier_name, None)
        print("Training {} features with classifier {}".format(self.X_train.shape[1], classifier_name))
        self.train_classifier(classifier)

        if self.config.get('SAMPLER') is None:
            print('No sampler to train')
            return

        sampler_name = self.config.get('SAMPLER')
        desired_num_samples = np.sum(self.y_train) + self.config.get('AUGMENTED_DATA_SIZE')
        imbalanced_learn_classifier = create_imbalanced_learn_classifier(classifier, sampler_name,
                                                                         {1: desired_num_samples})
        print("Imbalanced learning classifier {} samples to {} observations".format(sampler_name,
                                                                                    desired_num_samples))
        self.train_classifier(imbalanced_learn_classifier)

    def score_classifier(self, show_graph):
        """ Score the classifier and plot the scores if needed (show_graph=True) """

        classifier_name = self.config.get('CLASSIFIER')
        classifier = load_classifier(classifier_name, self.config.get('CACHE_FOLDER'), self.config.get('APP'))
        print("Predicting {} features".format(self.X_test.shape[1]))
        y_pred = classifier.predict(self.X_test)
        scores = classifier.decision_function(self.X_test)
        parameters = [self.y_test, y_pred, scores, show_graph]
        report_scores(parameters)

    def run_classifier_scores_report(self):
        """ Run scores report for classifier or sampled classifier """

        if self.config.get('CLASSIFIER') is not None and self.config.get('SAMPLER') is None:
            classifier_name = self.config.get('CLASSIFIER')
            print("Baseline classifier {}".format(classifier_name))
            self.score_classifier(False)
            return

        if self.config.get('SAMPLER') is None:
            print('No sampler to use')
            return

        sampler_name = self.config.get('SAMPLER')
        print("Imbalanced learning classifier {}".format(sampler_name))
        self.score_classifier(False)

    def train_gan(self, parameters):
        """ Train GAN """

        # General Parameters
        [gan, minority_class_index, save] = parameters

        seed = self.config.get('SEED')
        random.seed()
        tf.set_random_seed(seed)
        np.random.seed(seed)

        # Measure Start Time
        start_time = time.time()

        ### Training phase ###
        init = tf.global_variables_initializer()

        if save:
            saver = tf.train.Saver()

        batch_size = self.config.get('AUGMENTED_DATA_SIZE')
        with tf.Session() as sess:

            if save:
                writer1 = tf.summary.FileWriter(str(Path(
                    self.config.get('LOG_FOLDER') + filename_timeftm() + '_' + gan.name + '_' + str(
                        batch_size) + '_dlosses')), sess.graph)
                writer2 = tf.summary.FileWriter(str(Path(
                    self.config.get('LOG_FOLDER') + filename_timeftm() + '_' + gan.name + '_' + str(
                        batch_size) + '_glosses')), sess.graph)

            sess.run(init)

            # Collect list of generator/discriminator losses
            losses = []

            # Collect generated
            gan_X = {}

            ## Execute training total_steps ##
            total_steps = self.config.get('TOTAL_TRAINING_STEPS')

            step_incr = self.config.get('TRAINING_INCR_STEP')

            print("Training {} total_steps={}, #generatedData={}".format(gan.name, total_steps, batch_size))
            for step in range(total_steps + 1):

                if step % step_incr == 0:
                    # Feed prior(z)
                    prior_z_dim = self.config.get('Z_DIM')
                    z_sample = np.random.normal(size=[batch_size, prior_z_dim]).astype(np.float32)

                    # Minimise Losses
                    # Feed X mini-batch
                    split_df_dict = uniform_draw_feat_class(self.train_df, self.config.get('CLASS_NAME'), batch_size)
                    X_mb = split_df_dict[minority_class_index].drop(
                        columns=self.config.get('CLASS_COLUMN_NAME')).values.astype(np.float32)

                    # if gan.name == 'AAE':
                    #     _, recon_loss_curr = sess.run([gan.AE_trainer_, gan.recon_loss_], feed_dict={gan.X_: X_mb})

                    _, dloss = sess.run([gan.d_trainer, gan.d_loss], feed_dict={gan.X: X_mb, gan.prior_z: z_sample})

                    if gan.keep_prob is not None:
                        _, gloss = sess.run([gan.g_trainer, gan.g_loss],
                                            feed_dict={gan.X: X_mb, gan.prior_z: z_sample, gan.keep_prob: 0.5})
                    else:
                        _, gloss = sess.run([gan.g_trainer, gan.g_loss],
                                            feed_dict={gan.X: X_mb, gan.prior_z: z_sample})

                    if save:
                        dlosses_sum = sess.run(gan.loss_summaries, feed_dict={gan.tf_loss_ph: dloss})
                        writer1.add_summary(dlosses_sum, step)

                        glosses_sum = sess.run(gan.loss_summaries, feed_dict={gan.tf_loss_ph: gloss})
                        writer2.add_summary(glosses_sum, step)

                    # Store losses per step
                    losses.append((dloss, gloss))

                    if step % 100 == 0:
                        # Print generator, discriminator losses
                        print('Step: {}'.format(step))
                        print('Generator loss: {} | discriminator loss: {}'.format(gloss, dloss), '\n')

                    # Store generated data
                    z_gen = np.random.normal(size=[batch_size, prior_z_dim]).astype(np.float32)
                    gen_data = sess.run(gan.gen_X, feed_dict={gan.prior_z: z_gen})
                    if self.config.get('SAMPLE'):
                        print("Recording generated {} generated for step={}".format(batch_size, step))
                        gan_X[step] = gen_data

            if not self.config.get('SAMPLE'):
                print("Recording generated {} samples for step={}".format(batch_size, total_steps))
                gan_X[total_steps] = gen_data

            if save:
                save_path = saver.save(sess,
                                       str(Path(self.config.get('MODEL_DIR') + gan.name + '_' + str(
                                           batch_size) + '_' + filename_timeftm() + '.ckpt')))
                print("Model saved in path: %s" % save_path)

        # Measure End Time
        print('Time elapsed to train: ', hms_string(time.time() - start_time))

        if not save:
            return gan_X

        ### Saving data ###

        if self.config.get('SAMPLE'):
            # Pickle, save generator/discrimnator losses
            self.save_fake_data(gan_X, True)

        self.save_fake_data(gan_X, False)
        return gan_X

    def run_train_gan(self):
        """ Entry point method to train GAN """

        configuration = GANConfiguration()
        configuration.name = self.config.get('GAN_NAME')
        configuration.batch_size = self.config.get('AUGMENTED_DATA_SIZE')
        configuration.X_nodes = self.num_features
        configuration.y_output = self.config.get('Y_OUTPUT')
        configuration.z_dims = self.config.get('Z_DIM')
        gan = create_gan(configuration)
        parameters = [gan, self.config.get('FAKE'), True]
        self.gan_X = self.train_gan(parameters)

    def augmented_data_model_scores_report(self):
        """ Scores the classifier on augmented data set: REAL + FAKE DATA"""

        if self.config.get('CLASSIFIER') is None:
            print('No classifier to use for predictions, exiting...')
            return
        classifier_name = self.config.get('CLASSIFIER')
        classifier = create_classifier(classifier_name, None)

        total_steps = self.config.get('TOTAL_TRAINING_STEPS')
        # Loading generated data
        print("Loading data for total_count={}".format(total_steps))
        gan_X = self.load_fake_data(False)

        if self.config.get('SAMPLE'):
            # GAN Losses
            gan_loss = self.load_fake_data(True)

            # Extract generator and discriminator losses from 'gan'
            gen_losses = [gen[1] for gen in gan_loss]
            disc_losses_fraud = [disc[0] for disc in gan_loss]
            # Obtain steady step for the GAN
            # Stabilise within 5% of total_steps ran, all total_steps within this frame must have losses fluctuating about 0.75 s.d.
            dloss_sdy = compute_steady_frame(disc_losses_fraud, num_steps_ran=total_steps, sd_fluc_pct=0.75,
                                             scan_frame=int(total_steps * 0.05), stab_fn=np.median)
            gloss_sdy = compute_steady_frame(gen_losses, num_steps_ran=total_steps, sd_fluc_pct=0.75,
                                             scan_frame=int(total_steps * 0.05), stab_fn=np.median)

            if dloss_sdy is not None and gloss_sdy is not None:
                sdy_count = int(max(max(dloss_sdy), max(gloss_sdy)))
            elif dloss_sdy is not None:
                sdy_count = dloss_sdy
            elif gloss_sdy is not None:
                sdy_count = gloss_sdy
            else:
                sdy_count = None

            if sdy_count is not None:
                print('Steady count: {}'.format(sdy_count))

                print('\n',
                      '############################################# STEADY COUNT #############################################')
                steady_frame = pd.DataFrame(gan_X[sdy_count], columns=self.non_corr_column_names)
                steady_frame[self.config.get('CLASS_NAME')] = self.config.get('FAKE')
                gan_data_steady = pd.concat([self.train_df, steady_frame], axis='rows')
                X_gan_steady = gan_data_steady.drop(columns=self.config.get('CLASS_NAME')).values
                y_gan_steady = gan_data_steady[self.config.get('CLASS_NAME')].values

                # Perform classification

                # Fit and obtain predictions
                classifier.fit(X_gan_steady, y_gan_steady)
                y_pred_gan_steady = classifier.predict(self.X_test_df.values)
                pred_score_gan_steady = classifier.score(self.X_test_df.values, self.y_test)

                print('{} Prediction Metrics'.format(self.config.get('GAN_NAME')))
                parameters = [self.y_test, y_pred_gan_steady, pred_score_gan_steady, False]
                report_scores(parameters)

        print('\n',
              '############################################# FINAL COUNT #############################################')
        final_step = pd.DataFrame(gan_X[total_steps], columns=self.non_corr_column_names)
        final_step[self.config.get('CLASS_NAME')] = self.config.get('FAKE')

        gan_data_final = pd.concat([self.train_df, final_step], axis='rows')
        X_gan_final = gan_data_final.drop(columns=self.config.get('CLASS_NAME')).values
        y_gan_final = gan_data_final[self.config.get('CLASS_NAME')].values

        # Perform classification

        # Fit and obtain predictions
        classifier.fit(X_gan_final, y_gan_final)
        y_pred_gan_final = classifier.predict(self.X_test_df.values)
        pred_score_gan_final = classifier.score(self.X_test_df.values, self.y_test)

        print('Final Count {} Prediction Metrics'.format(self.config.get('GAN_NAME')))
        parameters = [self.y_test, y_pred_gan_final, pred_score_gan_final, False]
        report_scores(parameters)

    def gan_prediction_last_step(self, classifier):
        """ Predicts labels y_pred on testing data set X_test using classifier
        trained on (X_Gan, y_gan_final): real + fake data set"""

        gan_X = self.load_fake_data(False)

        last_step = self.config.get('TOTAL_TRAINING_STEPS')
        final_step = pd.DataFrame(gan_X[last_step], columns=self.non_corr_column_names)
        final_step[self.config.get('CLASS_NAME')] = self.config.get('FAKE')

        gan_data_final = pd.concat([self.train_df, final_step], axis='rows')
        X_gan_final = gan_data_final.drop(columns=self.config.get('CLASS_NAME')).values
        y_gan_final = gan_data_final[self.config.get('CLASS_NAME')].values

        # Perform classification
        classifier.fit(X_gan_final, y_gan_final)
        y_pred_gan_final = classifier.predict(self.X_test_df.values)

        return X_gan_final, y_gan_final, y_pred_gan_final, final_step

    def generate_distribution_plots(self):
        """ Plot PCAs or t-sne equivalent of Fraud/Non-Fraud for real and augmented data sets"""

        if self.config.get('CLASSIFIER') is None:
            print('No classifier to use for comparison, exiting...')
            return

        target_names = self.train_df[self.config.get('CLASS_NAME')].map(
            lambda x: 'Non-Fraud' if x == 0 else 'Fraud').unique()
        plt.figure(figsize=(6, 5))
        plt.subplot(2, 1, 1)
        if self.config.get('PCA') is not None:
            parameters = [self.config.get('SEED'), self.X_train, self.y_train.reshape(-1, ),
                          None, None, target_names,
                          'PCA of Fraud/Non-Fraud real data']
            realmin, realmax = plot_pca(parameters)
        else:
            parameters = [self.X_train[0:5000], self.y_train.reshape(-1, ), target_names,
                          'PCA of Fraud/Non-Fraud real data']
            plot_tsne(parameters)

        classifier_name = self.config.get('CLASSIFIER')
        classifier = create_classifier(classifier_name, None)
        if self.config.get('GEN_FILENAME') is None:
            print('No filename specify for comparison, exiting...')
            return

        X_gan_final, y_gan_final, _, _ = self.gan_prediction_last_step(classifier)
        plt.subplot(2, 1, 2)
        if self.config.get('PCA') is not None:
            parameters = [self.config.get('SEED'), X_gan_final, y_gan_final.reshape(-1, ), realmin, realmax,
                          target_names, 'PCA of Fraud/Non-Fraud augmented data']
            plot_pca(parameters)
        else:
            parameters = [X_gan_final[0:5000], y_gan_final.reshape(-1, ), target_names,
                          'PCA of Fraud/Non-Fraud augmented data']
            plot_tsne(parameters)
        plt.show()

    def create_embeddings(self):
        """ Create embeddings """
        X_real_non_fraud = self.train_df[0:self.config.get('AUGMENTED_DATA_SIZE')]
        X_real_fraud = self.train_df[self.train_df.Class == self.config.get('FAKE')]
        X_real_data = pd.concat([X_real_non_fraud, X_real_fraud])
        classifier_name = self.config.get('CLASSIFIER')
        classifier = create_classifier(classifier_name, None)
        if self.config.get('GEN_FILENAME') is None:
            print('No filename specify for comparison, exiting...')
            return
        _, _, _, gen_data_last_step = self.gan_prediction_last_step(classifier)
        X_gen_fraud = gen_data_last_step[gen_data_last_step.Class == self.config.get('FAKE')]
        X_gen_fraud = X_gen_fraud[0:self.config.get('AUGMENTED_DATA_SIZE') // 2]
        """ To visualize differently generated from real data """
        X_gen_fraud['Class'] = 2
        print('X_real_data={}, X_real_fraud={}, X_gen_fraud={}'
              .format(X_real_data.shape, X_real_fraud.shape, X_gen_fraud.shape))

        X_combined = pd.concat([X_real_data, X_gen_fraud])

        tf_data = tf.Variable(X_combined.drop(columns=self.config.get('CLASS_NAME')), name='data')
        with open(Path(self.config.get('EMB_FOLDER') + 'metadata_data.csv'), 'w') as f:
            f.write('Class' + '\t' + 'Name' + '\n')
            for idx, row_df in X_combined.iterrows():
                value = row_df[self.config.get('CLASS_NAME')]
                if value == self.config.get('REAL'):
                    f.write('Normal' + '\t' + 'Normal\n')
                elif value == self.config.get('FAKE'):
                    f.write('Fraud' + '\t' + 'Fraud\n')
                elif value == 2:
                    f.write('Gen' + '\t' + 'Gen\n')
        print("done creating metadata tsv")

        ## Running TensorFlow Session
        print("Generating embeddings")
        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.save(sess, str(Path(self.config.get('EMB_FOLDER') + 'embedding_data.ckpt')))

            config = projector.ProjectorConfig()
            # One can add multiple embeddings.
            embedding = config.embeddings.add()
            embedding.tensor_name = tf_data.name

            # Link this tensor to its metadata(Labels) file
            embedding.metadata_path = 'metadata_data.csv'

            # Saves a config file that TensorBoard will read during startup.
            projector.visualize_embeddings(tf.summary.FileWriter(str(Path(self.config.get('EMB_FOLDER')))), config)

    def compare_classifier_gan_scores(self):
        """ Compare accuracy, precision, recall and F1 scores """

        if self.config.get('GEN_FILENAME') is None:
            print('No filename to use for comparison, exiting...')
            return

        if self.config.get('CLASSIFIER') is None and self.config.get('SAMPLER') is None:
            print('Both classifier and sampler are not specified, exiting...')
            return

        if self.config.get('SAMPLER') is None:
            classifier_name = self.config.get('CLASSIFIER')
        else:
            classifier_name = self.config.get('SAMPLER')

        classifier = load_classifier(classifier_name, self.config.get('CACHE_FOLDER'), self.config.get('APP'))
        scores_baseline = classifier.decision_function(self.X_test)
        y_pred_baseline = classifier.predict(self.X_test)

        classifier_name = self.config.get('CLASSIFIER')
        classifier = create_classifier(classifier_name, None)
        _, _, y_pred_gan, _ = self.gan_prediction_last_step(classifier)
        scores_gan = classifier.decision_function(self.X_test)

        parameters = [self.y_test, y_pred_baseline, scores_baseline, y_pred_gan, scores_gan]
        plot_metrics(parameters)

    def compute_learning_curves(self):
        """
        Cross validation with 100 iterations to get train and test
        score curves, each time with 20% data randomly selected as a validation set.
        """

        if self.config.get('GEN_FILENAME') is None:
            print('No filename to use for computing errors, exiting...')
            return

        if self.config.get('CLASSIFIER') is None:
            print('No classifier to train')
            return

        classifier_name = self.config.get('CLASSIFIER')
        classifier = create_classifier(classifier_name, None)
        X_gan_final, y_gan_final, _, _ = self.gan_prediction_last_step(classifier)

        crossval = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

        my_plt = plot_learning_curve(classifier, 'Initial learning curves', self.X_train_df, self.y_train.ravel(),
                                     ylim=None, cv=crossval,
                                     n_jobs=4)
        my_plt.show()
        my_plt = plot_learning_curve(classifier, 'Augmented data learning curves', X_gan_final, y_gan_final.ravel(),
                                     ylim=None,
                                     cv=crossval, n_jobs=4)
        my_plt.show()

    def plot_augmented_data_learning_curves(self):
        """ Plot learning curves on REAL + FAKE data set"""

        classifier_name = self.config.get('CLASSIFIER')
        classifier = create_classifier(classifier_name, None)

        gan_X = self.load_fake_data(False)

        last_step = self.config.get('TOTAL_TRAINING_STEPS')
        final_step = pd.DataFrame(gan_X[last_step], columns=self.non_corr_column_names)
        final_step[self.config.get('CLASS_NAME')] = self.config.get('FAKE')

        start = self.config.get('AUGMENTED_DATA_SIZE') * 0.1
        stop = self.config.get('AUGMENTED_DATA_SIZE')
        sample_idx = np.linspace(start, stop, self.config.get('NUM_TRAINING_STEPS'))
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        for idx in sample_idx:
            intidx = int(idx)
            print('Generating data for {}'.format(intidx))
            sub_gen_data = final_step[0:intidx]
            gan_data = pd.concat([self.train_df, sub_gen_data], axis='rows')
            X_gan = gan_data.drop(columns=self.config.get('CLASS_NAME')).values
            y_gan = gan_data[self.config.get('CLASS_NAME')].values

            # Perform classification
            classifier.fit(X_gan, y_gan)
            y_pred = classifier.predict(self.X_test_df.values)
            acc = accuracy_score(y_pred, self.y_test)
            accuracy_scores.append(acc)
            precision = precision_score(y_pred, self.y_test)
            precision_scores.append(precision)
            recall = recall_score(y_pred, self.y_test)
            recall_scores.append(recall)
            f1_score = compute_F1(precision, recall)
            f1_scores.append(f1_score)

        plot_scores(sample_idx, accuracy_scores, precision_scores, recall_scores, f1_scores)

    def create_random_data_set(self):
        """ Create random n-classes data set for classification problems """

        print('samples={}, features={}, classes={}'
              .format(self.rd_data_samples, self.rd_data_features, self.rd_data_classes))
        # X, y = create_dataset(n_samples=10000, weights=(0.01, 0.95), n_features=n_features, n_classes=n_classes)
        self.X_rd, self.y_rd = create_dataset(n_samples=self.rd_data_samples, weights=(0.01, 0.05, 0.94),
                                              n_features=self.rd_data_features, n_classes=self.rd_data_classes)
        train_df = pd.DataFrame(self.X_rd)
        train_df[self.config.get('CLASS_NAME')] = self.y_rd
        print(str(train_df.groupby(self.config.get('CLASS_NAME'))[self.config.get('CLASS_NAME')].count()))

    def plot_decision_boundaries_random_dataset(self):
        """
            Plot decision boundaries for SVM ovr, ovo and SVNM trained on augmented data sets.
            This method provides a visual comparison of performance improvements of an SVM classifier
            trained on a balanced data set.
        """

        self.config.set('CLASSIFIER', 'SVC')
        self.config.set('TOTAL_TRAINING_STEPS', 10)
        self.config.set('AUGMENTED_DATA_SIZE', 500)
        self.create_random_data_set()

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))
        svm = create_classifier(self.config.get('CLASSIFIER'), False)
        svm.C = 10
        svm.decision_function_shape = 'ovr'
        clf = svm.fit(self.X_rd, self.y_rd)
        plot_decision_function(self.X_rd, self.y_rd, clf, ax1)
        ax1.set_title('OVR SVM')

        svm = create_classifier(self.config.get('CLASSIFIER'), False)
        svm.C = .5
        svm.decision_function_shape = 'ovo'
        clf = svm.fit(self.X_rd, self.y_rd)
        plot_decision_function(self.X_rd, self.y_rd, clf, ax2)
        ax2.set_title('OVO SVM')

        batch_size = int(self.rd_data_samples / self.rd_data_classes)
        train_df = pd.DataFrame(self.X_rd)
        non_corr_column_names = train_df.columns
        train_df[self.config.get('CLASS_NAME')] = self.y_rd
        self.train_df = train_df

        gan_data = train_df.copy()
        for class_index in range(self.rd_data_classes - 1):
            print('Generating data for class={}'.format(class_index))
            configuration = GANConfiguration()
            configuration.name = self.config.get('GAN_NAME')
            configuration.batch_size = batch_size
            configuration.X_nodes = self.rd_data_features
            configuration.y_output = self.config.get('Y_OUTPUT')
            configuration.z_dims = self.config.get('Z_DIM')
            gan = create_gan(configuration)
            parameters = [gan, class_index, False]
            gan_X = self.train_gan(parameters)
            last_step = self.config.get('TOTAL_TRAINING_STEPS')
            final_step = pd.DataFrame(gan_X[last_step], columns=non_corr_column_names)
            final_step[self.config.get('CLASS_NAME')] = class_index
            gan_data = pd.concat([gan_data, final_step], axis='rows')
            del gan

        print(str(gan_data.groupby(self.config.get('CLASS_NAME'))[self.config.get('CLASS_NAME')].count()))
        X_gan = gan_data.drop(columns=self.config.get('CLASS_NAME')).values
        y_gan = gan_data[self.config.get('CLASS_NAME')].values

        svm = create_classifier(self.config.get('CLASSIFIER'), False)
        svm.C = .5
        svm.decision_function_shape = 'ovo'
        clf = svm.fit(X_gan, y_gan)
        plot_decision_function(X_gan, y_gan, clf, ax3)
        ax3.set_title('OVO SVM + Augmented Data ')
        fig.tight_layout()
        plt.show()

    def last_iter(self):
        """ Convenience method to return last iteration batch of generated data"""

        gen_final = pd.DataFrame(self.gan_X[self.config.get('TOTAL_TRAINING_STEPS')],
                                 columns=self.non_corr_column_names)
        return gen_final

    def retrieve_real_data_generated_data(self):
        """ Convenience method to return real and generated data """

        data_dir = Path(self.config.get('DATA_FOLDER'))
        data = read_data(data_dir.joinpath(self.config.get('PICKLED_DATA')), True)
        gan_X_pt = self.load_fake_data(False)
        gan_X = gan_X_pt
        last_step = self.config.get('TOTAL_TRAINING_STEPS')
        non_corr_column_names, _ = compute_number_of_features(data, self.config.get('FILTERED_COLUMN_NAMES'))
        gen_final = pd.DataFrame(gan_X[last_step], columns=non_corr_column_names)
        return data, gen_final

    def save_fake_data(self, data, loss):
        """ Save generated data or losses """

        if loss:
            pkl_str = '_Loss.pkl'
        else:
            pkl_str = '_X.pkl'

        path_to_file = Path(self.config.get('CACHE_FOLDER') + filename_timeftm() + '_'
                       + str(self.config.get('AUGMENTED_DATA_SIZE')) + '_'
                       + str(self.config.get('TOTAL_TRAINING_STEPS')) + '_'
                       + self.config.get('GAN_NAME') + '_'
                       + self.config.get('APP') + pkl_str)

        if loss:
            print('Saving losses in path {}'.format(path_to_file))
        else:
            print('Saving fake data in path {}'.format(path_to_file))

        pd.to_pickle(data, path_to_file)

    def load_fake_data(self, loss):
        """ Load generated data or losses """

        filename = self.config.get('GEN_FILENAME')
        if filename is None:
            base_path_to_file = self.config.get('CACHE_FOLDER') + filename_timeftm()
        else:
            base_path_to_file = self.config.get('CACHE_FOLDER') + filename

        if loss:
            pkl_str = '_Loss.pkl'
        else:
            pkl_str = '_X.pkl'

        path_to_file = Path(base_path_to_file + '_'
                       + str(self.config.get('AUGMENTED_DATA_SIZE')) + '_'
                       + str(self.config.get('TOTAL_TRAINING_STEPS')) + '_'
                       + self.config.get('GAN_NAME') + '_'
                       + self.config.get('APP') + pkl_str)
        # Loading generated data
        if loss:
            print('Loading losses in path {}'.format(path_to_file))
        else:
            print("Loading data for total_count={} from path {}".format(self.config.get('TOTAL_TRAINING_STEPS'),
                                                                        path_to_file))
        gan_X = pd.read_pickle(path_to_file)
        return gan_X

    def help(self):
        """ Helper method to display options of the pipeline """

        parser = argparse.ArgumentParser()
        parser.parse_args(args=[])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    dargs = vars(args)

    if not dargs:
        print('Nothing to compute, exiting')
        exit(0)

    pipeline = Pipeline()
    pipeline.set_configuration(dargs)
    pipeline.read_process_data()

    if dargs['train_classifier']:
        pipeline.run_train_classifier()
        exit(0)

    if dargs['classifier_scores']:
        pipeline.run_classifier_scores_report()
        exit(0)

    if dargs['generate_data']:
        pipeline.run_train_gan()
        exit(0)

    if dargs['aug_model_scores']:
        pipeline.augmented_data_model_scores_report()
        exit(0)

    if dargs['generate_distribution_plots']:
        pipeline.generate_distribution_plots()
        exit(0)

    if dargs['create_embeddings']:
        pipeline.create_embeddings()
        exit(0)

    if dargs['compare_scores']:
        pipeline.compare_classifier_gan_scores()
        exit(0)

    if dargs['compute_learning_curves']:
        pipeline.compute_learning_curves()
        exit(0)

    if dargs['plot_augmented_data_learning_curves']:
        pipeline.plot_augmented_data_learning_curves()
        exit(0)

    if dargs['random_dataset']:
        pipeline.plot_decision_boundaries_random_dataset()
        exit(0)

    if dargs['retrieve_real_data_generated_data']:
        pipeline.retrieve_real_data_generated_data()
        exit(0)
