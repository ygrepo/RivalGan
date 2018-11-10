""" Various methods to investigate the data """

import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from pathlib import Path

from rivalgan.configuration import Configuration
from rivalgan.utils import ScalerWrapper, read_data, preprocess_credit_card_data


def standardize_data(filename, class_name, data_folder, pickled_data, scaler_name=None):
    """ Preprocess and standardize the raw Credit Card Fraud data set """

    data = pd.read_csv(filename)
    print(data.shape)
    print(data.columns)
    print(data.head(3))
    print(data.describe())

    data_cols = data.columns
    print(data_cols)
    # print('# of data columns: ', len(data_cols))
    # 284315 normal transactions (class 0)
    # 492 fraud transactions (class 1)

    data.groupby(class_name)[class_name].count()
    # Total nulls in dataset (sum over rows, then over columns)

    data.isnull().sum().sum()
    # Duplicates? Yes

    normal_duplicates = sum(data.loc[data.Class == 0].duplicated())
    fraud_duplicates = sum(data.loc[data.Class == 1].duplicated())
    total_duplicates = normal_duplicates + fraud_duplicates

    print('Normal duplicates', normal_duplicates)
    print('Fraud duplicates', fraud_duplicates)
    print('Total duplicates', total_duplicates)
    print('Fraction duplicated', total_duplicates / len(data))

    # 'Time' is seconds from first transaction in set
    # 48 hours worth of data
    # Let's convert time to time of day, in hours

    print('Last time value: {:.2f}'.format(data['Time'].max() / 3600))

    data['Time'] = (data['Time'].values / 3600) % 24

    scaled_data = None
    if scaler_name is not None:
        scaler = ScalerWrapper(scaler_name)
        # data columns will be all other columns except class
        data_copy = data.copy()
        data_copy = data_copy.drop(columns=class_name)
        data_copy = scaler.fit_transform(data_copy)
        # data columns will be all other columns except class
        filtered_data_cols = list(data.columns[data.columns != class_name])
        scaled_data = pd.DataFrame(data_copy, columns=filtered_data_cols)
        scaled_data[class_name] = data[class_name]

    # Save engineered dataset for use in analysis
    # Save as pickle for faster reload
    if scaled_data is not None:
        print(scaled_data.shape)
        print(scaled_data.columns)
        print(scaled_data.head(3))
        print(scaled_data.describe())

        pickle.dump(scaled_data, open(data_folder + scaler.name + '_' + pickled_data, 'wb'))


def analyze_data(filename, class_name):
    """ Credit card data set analysis """

    data = read_data(filename, True)
    print('No Frauds', round(data[class_name].value_counts()[0] / len(data) * 100, 2), '% of the dataset')
    print('Frauds', round(data[class_name].value_counts()[1] / len(data) * 100, 2), '% of the dataset')

    colors = ["#0101DF", "#DF0101"]
    sns.countplot(class_name, data=data, palette=colors)
    plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)

    parameters = [42, data, [], "train_test_split"]
    [_, X_train_df, _, _, _, y_train, y_test] = preprocess_credit_card_data(parameters)

    # Check number of frauds in both training and testing data
    print('Number of frauds in training data: {} out of {} cases ({:.10f}% fraud)'.format(np.sum(y_train),
                                                                                          y_train.shape[0],
                                                                                          np.mean(y_train) * 100))
    print('Number of frauds in test data: {} out of {} cases ({:.10f}% fraud)'.format(np.sum(y_test), y_test.shape[0],
                                                                                      np.mean(y_test) * 100))

    print(data.head())

    print('Distribution of the Classes in the subsample dataset')
    print(data[class_name].value_counts() / len(data))

    X_cols = list(data.drop(labels=['Time', class_name], axis='columns').columns)
    y_cols = data[[class_name]].columns
    y_train_df = pd.DataFrame(y_train, columns=y_cols)
    extract_relevant_features(X_cols, X_train_df, y_train_df)
    plt.show()


def extract_relevant_features(class_name, X_cols, X_train_df, y_train_df):
    """ Perform Wilcoxon Rank-Sum Test at 5% Level to determine feature relevance """

    # Keep results if needed later
    wilcox_result = dict([])

    for feat in X_cols:
        temp_df = pd.concat([X_train_df[[feat]], y_train_df], axis='columns')

        # Draw 300 samples (only 379 fraud data in training set, attempt to draw close to the hilt)
        random.seed(42)
        fraud_wilcox = random.sample(list(temp_df[temp_df[class_name] == 1][feat]), k=300)
        nonfraud_wilcox = random.sample(list(temp_df[temp_df[class_name] == 0][feat]), k=300)

        rank_sums, p_val_wilcox = stats.wilcoxon(fraud_wilcox, nonfraud_wilcox)

        if p_val_wilcox > 0.05:
            wilcox_result[feat] = ('No_Diff', p_val_wilcox)
            print('Feature "{}" failed to be rejected at 5% level with p-value {:.10f}'.format(feat, p_val_wilcox))
        else:
            wilcox_result[feat] = ('Diff', p_val_wilcox)

    # Rank-Sum Test determined differences in classes within feature
    wilcox_feat = [feat for feat in wilcox_result.keys() if wilcox_result[feat][0] == 'Diff']
    print('\n', 'Wilcoxon Rank-Sum Relevant Features: ', wilcox_feat)
    print('Total number of features selected: {}'.format(len(wilcox_feat)))
    unselected_feats = list(set(X_cols).difference(wilcox_feat))
    print('Unselected features: {}'.format(unselected_feats))
    # Obtain the top 6 most correlated wilcoxon-selected variables' to observe their scatter pair plot relations
    # print(type(unselected_feats))
    # Check for high correlations
    wilcox_corr = X_train_df.drop(columns=unselected_feats).corr().abs()
    # wilcox_corr = X_train_df.drop(columns=['V15', 'Amount', 'V13', 'V26', 'V22', 'V23']).corr().abs()
    wilcox_corr_pdSr = wilcox_corr.unstack()
    wilcox_corr_pdSr_sort = wilcox_corr_pdSr.sort_values(ascending=False).to_frame()

    # Top 6 correlated variable pairs
    # for i in range(7):
    #     print(wilcox_corr_pdSr_sort.iloc[i])
    print("Shape of the data={}".format(wilcox_corr_pdSr_sort.shape))
    print('Columns:', '\n', wilcox_corr_pdSr_sort.columns)
    print('Head:', '\n', wilcox_corr_pdSr_sort.head(3))

    # wilcox_corr_pdSr_sort.rename(index=str, columns={0: 'pearson_rho'})
    # print(wilcox_corr_pdSr_sort.columns)
    # wilcox_corr_pdSr_sort.apply(lambda x: print(x))

    # print(wilcox_corr_pdSr_sort)
    # print(wilcox_corr_pdSr_sort.iloc[23:35:2, :])
    # print(wilcox_corr_pdSr_sort.iloc[23:35:2, :].rename(index=str, columns={0: 'pearson_rho'}))


if __name__ == "__main__":
    # analyze_data()
    config = Configuration()
    standardize_data(Path(config.get('DATA_FOLDER') + config.get('DATA_FILE_NAME')),
                     config.get('CLASS_NAME'),
                     config.get('DATA_FOLDER'),
                     config.get('PICKLED_DATA'),
                     'StandardScaler')
    standardize_data(Path(config.get('DATA_FOLDER') + config.get('DATA_FILE_NAME')),
                     config.get('CLASS_NAME'),
                     config.get('DATA_FOLDER'),
                     config.get('PICKLED_DATA'),
                     'MaxAbsScaler')