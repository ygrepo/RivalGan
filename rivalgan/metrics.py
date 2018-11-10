""" Various methods to plot metrics and scores """

import matplotlib.pyplot as plt
import numpy as np
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def report_scores(parameters):
    """
    Report accuracy, precision, recall and F1 scores. Plot confusion matrix and AUC curves
    :param parameters: y_test, y_pred and scores
    :return:
    """
    [y_test, y_pred, scores, show_graph] = parameters
    print('Classification Report:', '\n', classification_report_imbalanced(y_test, y_pred))
    print('Accuracy score: {}'.format(accuracy_score(y_pred, y_test)))
    precision = precision_score(y_pred, y_test)
    print('Precision score: {}'.format(precision))
    recall = recall_score(y_pred, y_test)
    print('Recall score: {}'.format(recall))
    print('F1 score: {}'.format(compute_F1(precision, recall)))
    if show_graph:
        fig = plt.figure(figsize=(6, 5))
        fig.subplots_adjust(hspace=.5)
        plt.subplot(2, 1, 1)
        plot_cm(y_test, y_pred)
        plt.subplot(2, 1, 2)
        plot_aucprc(y_test, scores)
        plt.show()
    else:
        print('Confusion Matrix: ', '\n', confusion_matrix(y_test, y_pred), '\n')


def plot_metrics(parameters):
    """
    Report baseline scores vs scores on real + fake data
    :param parameters: y_test_baseline, y_pred_baseline, scores_baseline,
     y_pred_gan, scores_gan
    """
    [y_test_baseline, y_pred_baseline, scores_baseline,
     y_pred_gan, scores_gan] = parameters
    print('\n',
          '############################################# BASELINE REPORT #############################################')

    print('Classification Report:', '\n', classification_report_imbalanced(y_test_baseline, y_pred_baseline))
    print('Accuracy score: {}'.format(accuracy_score(y_pred_baseline, y_test_baseline)))
    precision = precision_score(y_pred_baseline, y_test_baseline)
    print('Precision score: {}'.format(precision))
    recall = recall_score(y_pred_baseline, y_test_baseline)
    print('Recall score: {}'.format(recall))
    print('F1 score: {}'.format(compute_F1(precision, recall)))

    print('\n',
          '############################################# GAN (DATA AUGMENTATION) REPORT ##############################')
    print('Classification Report:', '\n', classification_report_imbalanced(y_test_baseline, y_pred_gan))
    print('Accuracy score: {}'.format(accuracy_score(y_pred_gan, y_test_baseline)))
    precision = precision_score(y_pred_gan, y_test_baseline)
    print('Precision score: {}'.format(precision))
    recall = recall_score(y_pred_gan, y_test_baseline)
    print('Recall score: {}'.format(recall))
    print('F1 score: {}'.format(compute_F1(precision, recall)))

    fig = plt.figure(figsize=(8, 8))

    fig.subplots_adjust(hspace=.5)

    plt.subplot(2, 2, 1)
    plot_cm(y_test_baseline, y_pred_baseline)
    plt.subplot(2, 2, 2)
    plot_cm(y_test_baseline, y_pred_gan)

    plt.subplot(2, 2, 3)
    plot_aucprc(y_test_baseline, scores_baseline)
    plt.subplot(2, 2, 4)
    plot_aucprc(y_test_baseline, scores_gan)

    plt.show()


def plot_cm(y_test, predictions):
    """ Plot confusion matrix """

    cm = confusion_matrix(y_test, predictions)
    plt.imshow(cm, interpolation='nearest', cmap='tab10')
    class_names = ['Normal', 'Fraud']
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i][j]),
                     horizontalalignment='center', color='White')


def plot_scores(X_axis, accuracy_scores, precision_scores, recall_scores, f1_scores):
    """ Plot scores """
    plt.figure(figsize=(6, 5))
    plt.title("Scores with increasing additional data", fontsize=10)
    plt.xlabel("Additional training data size")
    plt.ylabel("Scores")
    plt.grid()
    # Plot a dotted vertical line at the best score for that scorer marked by x
    plt.plot(X_axis, accuracy_scores, linestyle='-.', color='k', marker='x', markeredgewidth=3, ms=8,
             label="accuracy")
    plt.plot(X_axis, precision_scores, linestyle='-.', color='b', marker='x', markeredgewidth=3, ms=8,
             label="precision")
    plt.plot(X_axis, recall_scores, linestyle='-.', color='g', marker='x', markeredgewidth=3, ms=8,
             label="recall")
    plt.plot(X_axis, f1_scores, linestyle='-.', color='r', marker='x', markeredgewidth=3, ms=8,
             label="f1")

    ax = plt.gca()
    X_axis_values = np.arange(0, 30, 5)
    ax.set_xticklabels(['{}%'.format(x) for x in X_axis_values])
    plt.legend(loc="best")
    plt.tight_layout(pad=0)
    plt.show()


def compute_F1(precision, recall):
    """ Compute F1 score """
    return 2 * recall * precision / (recall + precision)


def plot_aucprc(y_test, scores):
    """ Plot AUC curves """
    precision, recall, _ = precision_recall_curve(y_test, scores, pos_label=0)
    average_precision = average_precision_score(y_test, scores)

    print('Average precision-recall score: {0:0.3f}'.format(
        average_precision))

    plt.plot(recall, precision, label='area = %0.3f' % average_precision, color="green")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall Curve')
    plt.legend(loc="best")
    # plt.show()
