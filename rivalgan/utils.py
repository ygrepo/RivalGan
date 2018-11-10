""" Utility methods """

import argparse
import datetime
import pickle
import warnings

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn import pipeline
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import TomekLinks
from sklearn import manifold
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler
from sklearn.svm import LinearSVC, SVC
from xgboost import XGBClassifier


class ScalerWrapper:
    """
        Class wrapper for Data a preprocessing scaler
        :param name of the scalar
        :return a scaler
    """

    def __init__(self, scaler_name):
        self.name = scaler_name
        if scaler_name == 'StandardScaler':
            self.scaler = StandardScaler()
        elif scaler_name == 'MaxAbsScaler':
            self.scaler = MaxAbsScaler()
        elif scaler_name == 'MinMaxScaler':
            self.scaler = MinMaxScaler()
        else:
            raise argparse.ArgumentTypeError('Unexpected scaler {}'.format(scaler_name))

    def fit_transform(self, data):
        """ Wrapper around fit_tranform method of the scaler """
        return self.scaler.fit_transform(data)


def read_data(filename, pickled):
    """

    Load a data set from a file as csv or pickled
    :param filename
    :param pickled
    :return: the data loaded as data frame
    """
    print("Loading data from {}".format(filename))
    if pickled:
        data = pickle.load(open(filename, 'rb'))
    else:
        data = pd.read_csv(filename)
    print("Shape of the data={}".format(data.shape))
    print('Head:', '\n', data.head(3))
    return data


def preprocess_credit_card_data(parameters):
    """

     Preprocess the data and returns training and test data sets
     :param parameters list of parameters
     :return: train_df, X_train_df, X_train, X_test_df, X_test, y_train, y_test
     """
    [seed, data, corr_columns, data_splitter, class_name] = parameters

    data_X = data.drop(labels=[class_name], axis='columns')
    data_y = data[class_name].values

    if data_splitter == "train_test_split":
        # Train-test split data: 75/25
        X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.25, random_state=seed)
    elif data_splitter == "stratified_shuffle_split":
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        for train_index, test_index in sss.split(data_X, data_y):
            print("Train:", train_index, "Test:", test_index)
            X_train, X_test = data_X.iloc[train_index], data_X.iloc[test_index]
            y_train, y_test = data_y.iloc[train_index], data_y.iloc[test_index]

    # Labels
    X_cols = list(data.drop(labels=[class_name], axis='columns').columns)
    y_cols = data[[class_name]].columns

    # Training data as dataframe
    X_train_df = pd.DataFrame(X_train, columns=X_cols)
    X_train_df = X_train_df.reset_index().drop(columns=['index'])
    X_train_df = X_train_df.drop(columns=corr_columns)

    y_train_df = pd.DataFrame(y_train, columns=y_cols)

    # Full dataframe
    train_df = pd.concat([X_train_df, y_train_df], join='outer', axis='columns')

    # X, y as numpy array
    X_train, y_train = X_train_df.values, y_train_df.values

    # Test data as dataframe
    X_test_df = pd.DataFrame(X_test, columns=X_cols)
    X_test_df = X_test_df.reset_index().drop(columns=['index'])
    X_test_df = X_test_df.drop(columns=corr_columns)

    y_test_df = pd.DataFrame(y_test, columns=y_cols)

    # Full dataframe
    # X, y as numpy array
    X_test, y_test = X_test_df.values, y_test_df.values

    # Check number of frauds in both training and testing data
    print('Number of frauds in training data: {} out of {} cases ({:.10f}% fraud)'.format(np.sum(y_train),
                                                                                          y_train.shape[0],
                                                                                          np.mean(y_train) * 100))
    print('Number of frauds in test data: {} out of {} cases ({:.10f}% fraud)'.format(np.sum(y_test), y_test.shape[0],
                                                                                      np.mean(y_test) * 100))
    return train_df, X_train_df, X_train, X_test_df, X_test, y_train, y_test


def uniform_draw_feat_class(data_df, target_name, draw_size):
    """ Sample out 1 particular class out of the full data uniformly in a predefined sample size,
    segregate pandas dataframe into their respective target classes. """

    # Break down full dataset into their respective classes
    target_class_dfs = {}
    n_classes = data_df[target_name].unique()

    for unique_class in n_classes:
        # Store each segregated class dataframes as dictionary
        class_i_df = data_df[data_df[target_name] == unique_class]
        class_i_df = (class_i_df.reset_index()).drop(columns=['index'])
        target_class_dfs[unique_class] = class_i_df

    # Obtain uniform samples of size 'draw_size' and store as dictionary
    uniform_class_sample_draws = {}

    # Pandas df. sampling default None equivalently uniform draws
    for unique_class in n_classes:
        unif_samples = target_class_dfs[unique_class].sample(n=draw_size, replace=True)
        uniform_class_sample_draws[unique_class] = unif_samples

    # Returns dictionary of pandas dataframes, split into respective classes
    return uniform_class_sample_draws


def compute_steady_frame(loss_lst, num_steps_ran, sd_fluc_pct=0.15, scan_frame=225, stab_fn=min):
    """
    FOR THE PURPOSE OF FINDING A FRAME OF STEPS WHERE THE LOSSES HAS STABILISED WITHIN SPECIFIED PERCENTAGE OF
    1SD OF LOSSES FOR ALL STEPS EXECUTED

    DEFAULT FRAME OF STEPS WHERE LOSS IS CONSIDERED STEADY IS 225, FLUCTUATING WITHIN +/- 15% OF 1 S.D. OF THE LOSSES

    [EXAMPLE - 5% OF S.D., 5% OF TOTAL STEPS RAN]
    Obtain minimum of losses and its corresponding 1 s.d. of loss of the given array of losses
    Do for a list of losses
        For some 5% out of all epochs ran
            if that particular 5% of steps have losses each within +/-5% of the minimum loss,
                Step frame chosen as steady step-period
                Else return 'no steady step-periods found'; break
        Return frame of steps found
    """
    # Obtain minimum and s.d. value of the losses for all epochs
    loss_1sd = np.std(loss_lst)
    stab_fn_loss = stab_fn(loss_lst)

    frame_start = 0
    frame_end = frame_start + scan_frame

    # Return nothing in the event no frame of losses are found to have steady losses
    steady_frame = None

    # For loop termination
    exit_loop = True
    while exit_loop:

        # Reset counter when one of the step is not within +/-'sd_fluc_pct'% of minimum losses
        counter_5pct = 0

        for step in range(frame_start, frame_end, 1):
            if (loss_lst[step] > stab_fn_loss and loss_lst[step] <= (stab_fn_loss + sd_fluc_pct * loss_1sd)) \
                    or (loss_lst[step] < stab_fn_loss and loss_lst[step] >= (stab_fn_loss - sd_fluc_pct * loss_1sd)):

                # Increase counter progressively until all (225) steps in 'scan_frame' are within specified tolerance
                counter_5pct += 1

                # When steps are within +/-'sd_fluc_pct'% of minimal loss, flag the final step
                if counter_5pct == scan_frame:
                    print('Steady step frame found at step {} as final'.format(frame_end))

                    # Return numpy array of step numbers within specified range
                    steady_frame = np.linspace(start=(frame_end - scan_frame + 1), stop=frame_end,
                                               num=scan_frame)
                    exit_loop = False
                    break
            else:
                break

        # When a frame of steps within +/-'sd_fluc_pct'% of minimum losses is not found,
        # shift to next frame by 1 step
        frame_start += 1
        frame_end += 1

        if frame_end > num_steps_ran:
            print('No steady step frame found!')
            break

    return steady_frame


def compute_number_of_features(data, column_names_to_filter):
    """
        Compute the number of relevant features given a set of correlated features to remove from
        the dataset
    :param data:
    :param column_names_to_filter:
    :return: the list of non-correlated columns and the number of features
    """
    data_column_names = list(data.columns)
    non_corr_column_names = [x for x in data_column_names if x not in column_names_to_filter]
    num_features = len(non_corr_column_names)
    print("Number of features={}".format(str(num_features)))
    return non_corr_column_names, num_features


def create_classifier(classifier_name, gridSearch):
    """
        Factory methods to create classifiers
    :param classifier_name:
    :param gridSearch: if needed to tune hyper-parameters
    :return: the classifier
    """
    if classifier_name == 'Logit':
        classifier = LogisticRegression()
        params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    elif classifier_name == 'LinearSVC':
        classifier = LinearSVC()
    elif classifier_name == 'SVC':
        classifier = SVC()
    elif classifier_name == 'RandomForest':
        classifier = RandomForestClassifier(n_jobs=-1, random_state=42,
                                            n_estimators=500,
                                            max_features='auto',
                                            min_samples_leaf=2,
                                            criterion='entropy')
    elif classifier_name == 'xgb':
        classifier = XGBClassifier(random_state=42, n_jobs=-1)
    elif classifier_name == 'SGDClassifier':
        classifier = SGDClassifier(penalty='elasticnet', random_state=42, class_weight='balanced', tol=1e-3)

        params = {'loss': ['hinge', 'log', 'modified_huber', 'perceptron'],
                  'penalty': ['l1', 'l2', 'elasticnet'],
                  'alpha': np.logspace(0.0001, 5, 5),
                  'l1_ratio': list(np.linspace(0.0001, 1.0, 20)) + [0.15],
                  'random_state': [42],
                  'n_jobs': [-1],
                  'eta0': [0.0001],
                  'class_weight': ['balanced']}
    if gridSearch is not None:
        if gridSearch == 'grid':
            classifier = GridSearchCV(classifier, params, cv=5, verbose=1, n_jobs=-1)
        elif gridSearch == 'grid':
            classifier = RandomizedSearchCV(LogisticRegression(), params, n_iter=4)

    return classifier


def create_imbalanced_learn_classifier(classifier, sampler_name, ratio):
    """
    Factory methods to create SMOTE sampler with given classifier and ratio of resampling the data set.
    :param classifier:
    :param sampler_name:
    :param ratio:
    :return: the sampled classifier
    """
    if sampler_name == 'SMOTE':
        sampler = SMOTE(ratio=ratio, random_state=42, n_jobs=-1)
    elif sampler_name == 'RandomOverSampler':
        sampler = RandomOverSampler(ratio=ratio, random_state=42)
    elif sampler_name == 'SMOTETomek':
        smt = SMOTE(ratio=ratio, random_state=42, n_jobs=-1)
        tl = TomekLinks(ratio=ratio, random_state=42, n_jobs=-1)
        sampler = SMOTETomek(random_state=42, tomek=tl, smote=smt, n_jobs=-1)
    classifier = pipeline.make_pipeline(sampler, classifier)
    return classifier


def plot_pca(parameters):
    """
    2-component PCA plot.
    PCA is sensitive to dimensional scales, thus the need to standardise matrix X
    :param parameters:
    """

    [seed, X_pca, y_pca, realmin, realmax, target_names, title] = parameters

    pca = PCA(n_components=2, random_state=seed)
    X_r = pca.fit_transform(X_pca)

    # Percentage of variance explained for each components
    print('explained variance ratio (first two components): {}'.format(pca.explained_variance_ratio_))

    colors = ['red', 'aqua']
    lw = 2
    for color, i, target_name in zip(colors, [0, 1], target_names):
        plt.scatter(X_r[y_pca == i, 0], X_r[y_pca == i, 1], color=color, alpha=.6, lw=lw, label=target_name)
    plt.legend(loc='lower right', shadow=False, scatterpoints=1)
    plt.title(title)
    plt.tight_layout()
    plt.margins(0.02)
    if realmin is None:
        realmin = np.nanmin(X_r)
    if realmax is None:
        realmax = np.nanmax(X_r)

    plt.yticks(np.arange(-50, 100, 50))
    plt.xticks(np.arange(realmin, realmax, 50))
    return realmin, realmax


def plot_tsne(parameters):
    """ Plot tsne for GAN distribution """
    [X, y, target_names, title] = parameters
    # tsne = manifold.TSNE(n_components=2)
    iso = manifold.Isomap(n_neighbors=10, n_components=2)
    X_r = iso.fit_transform(X)
    colors = ['brown', 'cyan']
    for color, i, target_name in zip(colors, [0, 1], target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, cmap=plt.get_cmap('spectral'))
    #     plt.scatter(X_r[y_pca == i, 0], X_r[y_pca == i, 1], color=color, alpha=.6, lw=lw, label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title(title)
    plt.tight_layout()
    plt.margins(0.02)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Plot learning curves training and cross-validation scores
    :param estimator:
    :param title:
    :param X:
    :param y:
    :param ylim:
    :param cv:
    :param n_jobs:
    :param train_sizes:
    :return: plot
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Scores")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def load_classifier(classifier_name=None, cache_folder=None, app=None):
    """
    Load a pickled classifier for a specific app or data set
    :param classifier_name:
    :param cache_folder:
    :param app:
    :return: 'unpickled' classifier
    """
    path_to_file = Path(cache_folder + classifier_name + '_' + app + ".pkl")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        print("Loading classifier {} from file {}".format(classifier_name, path_to_file))
        return pickle.load(open(path_to_file, 'rb'))


def plot_decision_function(X, y, clf, ax):
    """
    Decision function for data set (X,y) and classifier clf
    :param X:
    :param y:
    :param clf:
    :param ax:
    """
    plot_step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.8, c=y, edgecolor='k')


def create_dataset(n_samples=1000, weights=(0.01, 0.01, 0.98), n_features=2, n_classes=3,
                   class_sep=0.8, n_clusters=1):
    """
    Proxy method for make_classification which generates random n-class for classification problems
    :param n_samples:
    :param weights:
    :param n_features:
    :param n_classes:
    :param class_sep:
    :param n_clusters:
    :return: (X,y)
    """
    return make_classification(n_samples=n_samples, n_features=n_features,
                               n_informative=n_features, n_redundant=0, n_repeated=0,
                               n_classes=n_classes,
                               n_clusters_per_class=n_clusters,
                               weights=list(weights),
                               class_sep=class_sep, random_state=0)


def xavier_init(size, label, data_type):
    """ For usage in tensorflow neural nets weights initialisation per layer.
    Helps resolve issue of overfitting to data. """

    tf.set_random_seed(42)
    in_dim = size[0]
    out_dim = size[1]
    xavier_stddev = 1. / tf.sqrt((in_dim + out_dim) / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev, name=label, dtype=data_type)


def hms_string(sec_elapsed):
    """
    Format seconds in H/M/S
    :param sec_elapsed:
    :return: Formatted seconds
    """
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


def filename_timeftm():
    """
    Format current time seconds in YYYYMMDDHH
    :return: Formatted current time
    """

    now = datetime.datetime.now()
    fmt_date_time = now.strftime("%Y%m%d%H")
    return fmt_date_time


def str2bool(value):
    """
    Converts a string to a Boolean
    :param value: input string
    :return: boolean
    """
    vlow = value.lower()

    if vlow in ('yes', 'true', 't', 'y', '1'):
        return True

    if vlow in ('no', 'false', 'f', 'n', '0'):
        return False

    raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_arguments(parser):
    """
    Set the options for the command line of the tool
    :param parser:
    :return: parsed arguments
    """
    parser.add_argument('--CLASSIFIER', type=str, default=None,
                        choices=['Logit', 'LinearSVC', 'RandomForest', 'SGDClassifier', 'SVC'],
                        help='Baseline classifier')
    parser.add_argument('--SAMPLER', type=str, default=None, choices=['SMOTE', 'SMOTETomek'],
                        help=' SMOTE sampler')
    parser.add_argument('--AUGMENTED_DATA_SIZE', type=int, default=None,
                        help='The size of the imbalanced class samples to sample. (default value: 5000)')
    parser.add_argument('--TOTAL_TRAINING_STEPS', type=int, default=None,
                        help='Total number of training iterFations. (default value: 6300)')
    parser.add_argument('--GEN_FILENAME', type=str, default=None,
                        help='filename of the generated data file to load')
    parser.add_argument('--GAN_NAME', type=str, default=None,
                        choices=['VGAN', 'WGAN', 'IWGAN', 'LGAN'],
                        help='GAN architecture')
    parser.add_argument('--train_classifier', type=bool, default=False,
                        help='If True the model trains the base classifier)')
    parser.add_argument('--classifier_scores', type=bool, default=False,
                        help='If True use model to predict and log metrics to measure the quality of the predictions')
    parser.add_argument('--generate_data', type=bool, default=False,
                        help='If True the model trains the gan on data set and generate new data')
    parser.add_argument('--compute_learning_curves', type=bool, default=False,
                        help='If True plot learning curves using a specific classifier and dataset')
    parser.add_argument('--aug_model_scores', type=bool, default=False,
                        help='If True use augmented data model to predict and measure the quality of the predictions')
    parser.add_argument('--plot_augmented_data_learning_curves', type=bool, default=False,
                        help='If True plot augmented data score curves')
    parser.add_argument('--generate_distribution_plots', type=bool, default=False,
                        help='Generate scatter plots for real and augmented data')
    parser.add_argument('--create_embeddings', type=bool, default=False,
                        help='Generate embeddings for real and augmented data')
    parser.add_argument('--compare_scores', type=bool, default=False,
                        help='If True generate model scores')
    parser.add_argument('--random_dataset', type=bool, default=False,
                        help='Generate decision boundaries for random classification problem')
    parser.add_argument('--retrieve_real_data_generated_data', type=bool, default=False,
                        help='Retrieve real and augmented data distributions')

    args = parser.parse_args()
    return args
