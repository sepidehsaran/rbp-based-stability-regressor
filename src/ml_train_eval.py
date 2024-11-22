import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import random
import os
import pickle
import pybedtools
from pybedtools.bedtool import BedTool
import numpy as np
import pandas as pd
from math import sqrt, log, log2
import matplotlib.pyplot as plt
import seaborn as sns
import neptune
import neptune.integrations.sklearn as npt_utils
from neptune.integrations.tensorflow_keras import NeptuneCallback
from neptune.utils import stringify_unsupported
from joblib import dump, load
from datetime import datetime, date
from operator import itemgetter
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, average_precision_score, r2_score
from sklearn.metrics import roc_curve, auc, classification_report, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import MinMaxScaler, label_binarize, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeClassifier
from sklearn.model_selection import RepeatedKFold, KFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
from sklearn.svm import SVR
from xgboost import XGBClassifier, XGBRegressor
from xgboost import plot_importance
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import device_lib


# ----------------------------------------------------------------------------------------------------------------------
#  three main running modes repeated k fold, diff with random seeds, non diff fixed test set with multi random seeds
# ----------------------------------------------------------------------------------------------------------------------

def run_with_repeated_kfold(df_data, ls_features_cols, ls_label_col, run, config):
    """
    # do not give separate test set - let kfold select randomly and repeat with random seeds
    :param df_data:
    :param ls_features_cols:
    :param ls_label_col:
    :param run:
    :param config:
    :return:
    """
    verbose = config.getboolean('general', 'verbose')
    rand_state = int(config.get('general', 'random_seed'))
    k_folds = int(config.get('model_selection', 'k_folds'))
    n_repeats = int(config.get('model_selection', 'n_repeats'))
    res_path = config.get('general', 'res_path') + \
               config.get('input_data_selection', 'rbp_input_for_regressor') + '_' + \
               config.get('input_data_selection', 'transcript_regions') + '/' + 'diff_isoforms_' + \
               config.get('input_data_selection', 'differential_isoforms') + '/' + \
               config.get('input_data_selection', 'ls_prediction_labels').replace(',', '') + '/' + \
               config.get('input_data_selection', 'ls_train_cell_lines').replace(',', '') + '_' + \
               config.get('input_data_selection', 'ls_test_cell_lines').replace(',', '') + '/'
    os.makedirs(res_path, exist_ok=True)
    model_category = config.get('model_selection', 'model_category')
    model_type = config.get('model_selection', 'model_type')
    str_ls_pred_labels = config.get('input_data_selection', 'ls_prediction_labels').lower()
    if verbose: print('in function {0}'.format('run_with_repeated_kfold'))

    ls_mae = []
    ls_mse = []
    ls_r2 = []
    ls_p_cor = []
    ls_pipe_score = []
    ls_sr_feat_imp = []
    ls_mean_perm_imp = []

    # prep data
    df_data.drop_duplicates(inplace=True)
    df_data = df_data.sample(frac=1, random_state=rand_state).reset_index(drop=True)
    df_feat = df_data[ls_features_cols]

    # note: here y_train is dataframe for multiple labels and previously was series for 1 label
    if len(ls_label_col) > 1:
        sr_label = df_data[ls_label_col].to_numpy()
    else:
        sr_label = df_data[ls_label_col[0]]

    # generate kfold repeat splits
    rkf = RepeatedKFold(n_splits=k_folds, n_repeats=n_repeats, random_state=rand_state)  # , shuffle=False)
    i_fold = 0
    for idx_train, idx_test in rkf.split(df_feat):
        i_fold += 1
        x_train = df_feat.iloc[idx_train]
        x_test = df_feat.iloc[idx_test]
        y_train = sr_label.iloc[idx_train]
        y_test = sr_label.iloc[idx_test]

        y_pred, df_unsorted_coefs, perm_imp, pipe_score = fold_train_pred_model(x_train, y_train,
                                                                                x_test, y_test,
                                                                                i_fold, run, config)
        ls_mae.append(mean_absolute_error(y_test, y_pred))
        ls_mse.append(mean_squared_error(y_test, y_pred))

        if len(df_unsorted_coefs) > 0:
            ls_sr_feat_imp.append(df_unsorted_coefs)
        if perm_imp != []:
            ls_mean_perm_imp.append(perm_imp['importances_mean'])
        if model_category == 'regressor':
            ls_r2.append(r2_score(y_test, y_pred))
            ls_p_cor.append(np.corrcoef(y_test, y_pred)[0, 1])
        elif model_category == 'classifier':
            ls_pipe_score.append(pipe_score)

    run['avg_mean_absolute_error'] = np.mean(np.array(ls_mae))
    run['avg_mean_squared_error'] = np.mean(np.array(ls_mse))
    run['avg_mean_perm_imp'] = np.mean(np.array(ls_mean_perm_imp), axis=0)
    run['sd_mean_absolute_error'] = np.std(np.array(ls_mae))
    run['sd_mean_squared_error'] = np.std(np.array(ls_mse))
    run['sd_mean_perm_imp'] = np.std(np.array(ls_mean_perm_imp), axis=0)

    if model_category == 'regressor':
        run['avg_r2_score'] = np.mean(np.array(ls_r2))
        run['avg_pearson_correlation'] = np.mean(np.array(ls_p_cor))
        run['sd_r2_score'] = np.std(np.array(ls_r2))
        run['sd_pearson_correlation'] = np.std(np.array(ls_p_cor))
        if verbose:
            print("average MAE ", np.mean(np.array(ls_mae)))
            print("average MSE ", np.mean(np.array(ls_mse)))
            print("average r2 ", np.mean(np.array(ls_r2)))
            print("average Pearson Correlation ", np.mean(np.array(ls_p_cor)))

    elif model_category == 'classifier':
        run['avg_pipe_score'] = np.mean(np.array(ls_pipe_score))
        if verbose:
            print("average MAE ", np.mean(np.array(ls_mae)))
            print("average MSE ", np.mean(np.array(ls_mse)))
            print("average ACC ", np.mean(np.array(ls_pipe_score)))
    # permutation importance
    df_perm_imp = pd.DataFrame(columns=['features', 'avg_mean_perm_imp', 'sd_mean_perm_imp'])
    df_perm_imp['features'] = x_train.columns.values
    df_perm_imp['avg_mean_perm_imp'] = np.mean(np.array(ls_mean_perm_imp), axis=0)
    df_perm_imp['sd_mean_perm_imp'] = np.std(np.array(ls_mean_perm_imp), axis=0)

    df_perm_imp.to_csv(res_path + model_type + '_' + str_ls_pred_labels + '_repeated_kfold_perm_impt_avg_sd.csv',
                       sep='|')
    run['data/perm_impt_avg_sd'].track_files(res_path + model_type + '_' + str_ls_pred_labels
                                             + '_repeated_kfold_perm_impt_avg_sd.csv')
    if verbose:
        print("perm_impt_avg_sd saved to " + res_path + model_type + '_' + str_ls_pred_labels
              + '_repeated_kfold_perm_impt_avg_sd.csv')
    # built-in feature importance
    if len(ls_sr_feat_imp) > 0:
        df_all_feat_imp = pd.concat(ls_sr_feat_imp, axis=1)
        av_feat_imp = pd.DataFrame(df_all_feat_imp.mean(axis=1)).sort_values(by=0, axis=0)
        sd_feat_imp = pd.DataFrame(df_all_feat_imp.std(axis=1))
        av_feat_imp.rename(columns={0: 'avg_coef'}, inplace=True)
        sd_feat_imp.rename(columns={0: 'sd_coef'}, inplace=True)
        df_feat_imp = pd.merge(av_feat_imp, sd_feat_imp, how='left',
                               left_on=av_feat_imp.index, right_on=sd_feat_imp.index)
        fig4 = av_feat_imp[-10:].plot(kind='barh', legend=False).figure
        fig5 = av_feat_imp[:10].plot(kind='barh', legend=False).figure
        run["avg_feature_importance_top"].upload(neptune.types.File.as_image(fig4))
        run["avg_feature_importance_bot"].upload(neptune.types.File.as_image(fig5))
        df_all_feat_imp.rename(columns={'Unnamed: 0': 'features'}, inplace=True)
        df_all_feat_imp.to_csv(res_path + model_type + '_' + str_ls_pred_labels + '_repeated_kfold_all_feat_impt.csv',
                               sep='|')
        run['data/all_feat_impt'].track_files(res_path + model_type + '_' + str_ls_pred_labels
                                              + '_repeated_kfold_all_feat_impt.csv')
        if verbose:
            print("feat_impt_avg_sd saved to " + res_path + model_type + '_' + str_ls_pred_labels
                  + '_repeated_kfold_all_feat_impt.csv')
        df_feat_imp.to_csv(res_path + model_type + '_' + str_ls_pred_labels + '_repeated_kfold_feat_impt_avg_sd.csv',
                           sep='|')
        run['data/feat_impt_avg_sd'].track_files(res_path + model_type + '_' + str_ls_pred_labels
                                                 + '_repeated_kfold_feat_impt_avg_sd.csv')
        if verbose:
            print("feat_impt_avg_sd saved to " + res_path + model_type + '_' + str_ls_pred_labels
                  + '_repeated_kfold_feat_impt_avg_sd.csv')

    return


def run_with_multiple_rand_seeds(df_train_data, df_test_data, ls_features_cols, ls_label_col, run, config):
    """
    # give separate test set - repeat with rand seed only without kfold

    :param df_train_data:
    :param df_test_data:
    :param ls_features_cols:
    :param label_col:
    :param name:
    :param run:
    :param config:
    :return:
    """
    verbose = config.getboolean('general', 'verbose')
    version = config.get('general', 'version')
    rand_state = int(config.get('general', 'random_seed'))
    model_category = config.get('model_selection', 'model_category')
    model_type = config.get('model_selection', 'model_type')
    str_ls_pred_labels = config.get('input_data_selection', 'ls_prediction_labels').lower()
    ls_random_seeds = [int(s.strip()) for s in config.get('general', 'ls_random_seeds').split(',')]
    res_path = config.get('general', 'res_path') + \
               config.get('input_data_selection', 'rbp_input_for_regressor') + '_' + \
               config.get('input_data_selection', 'transcript_regions') + '/' + 'diff_isoforms_' + \
               config.get('input_data_selection', 'differential_isoforms') + '/' + \
               config.get('input_data_selection', 'ls_prediction_labels').replace(',', '') + '/' + \
               config.get('input_data_selection', 'ls_train_cell_lines').replace(',', '') + '_' + \
               config.get('input_data_selection', 'ls_test_cell_lines').replace(',', '') + '/'
    os.makedirs(res_path, exist_ok=True)
    if verbose: print('in function {0}'.format('run_with_multiple_rand_seeds'))

    df_train_data.to_pickle(res_path + 'version_' + version + '_trainset.pickle')
    df_test_data.to_pickle(res_path + 'version_' + version + '_testset.pickle')

    x_train = df_train_data[ls_features_cols]
    x_test = df_test_data[ls_features_cols]
    # note: here y_train is dataframe for multiple labels and previously was series for 1 label
    if len(ls_label_col) > 1:
        y_train = df_train_data[ls_label_col].to_numpy()
        y_test = df_test_data[ls_label_col].to_numpy()
    else:
        y_train = df_train_data[ls_label_col[0]]
        y_test = df_test_data[ls_label_col[0]]
    # for saving results later
    df_test_data_pred = df_test_data.copy()
    # avg preds for all seeds
    df_test_data_pred['avg_pred'] = 0.0
    ls_mae = []
    ls_mse = []
    ls_r2 = []
    ls_p_cor = []
    ls_pipe_score = []
    ls_sr_feat_imp = []
    ls_mean_perm_imp = []

    for rand_i in ls_random_seeds:
        tf.random.set_seed(rand_i)
        np.random.seed(rand_i)
        random.seed(rand_i)
        y_pred, df_unsorted_coefs, perm_imp, pipe_score = fold_train_pred_model(x_train, y_train,
                                                                                x_test, y_test,
                                                                                rand_i, run, config)
        ls_mae.append(mean_absolute_error(y_test, y_pred))
        ls_mse.append(mean_squared_error(y_test, y_pred))
        if len(df_unsorted_coefs) > 0:
            ls_sr_feat_imp.append(df_unsorted_coefs)
        if perm_imp != []:
            ls_mean_perm_imp.append(perm_imp['importances_mean'])
        if model_category == 'regressor':
            ls_r2.append(r2_score(y_test, y_pred))
            ls_p_cor.append(np.corrcoef(y_test, y_pred)[0, 1])
            df_test_data_pred['pred_' + str(rand_i)] = 0.0
            df_test_data_pred['avg_pred'] = df_test_data_pred['avg_pred'] + y_pred
            df_test_data_pred['pred_' + str(rand_i)] = y_pred
        elif model_category == 'classifier':
            ls_pipe_score.append(pipe_score)

    run['avg_mean_absolute_error'] = np.mean(np.array(ls_mae))
    run['avg_mean_squared_error'] = np.mean(np.array(ls_mse))
    run['avg_mean_perm_imp'] = np.mean(np.array(ls_mean_perm_imp), axis=0)
    run['sd_mean_absolute_error'] = np.std(np.array(ls_mae))
    run['sd_mean_squared_error'] = np.std(np.array(ls_mse))
    run['sd_mean_perm_imp'] = np.std(np.array(ls_mean_perm_imp), axis=0)

    if model_category == 'regressor':
        run['avg_r2_score'] = np.mean(np.array(ls_r2))
        run['avg_pearson_correlation'] = np.mean(np.array(ls_p_cor))
        run['sd_r2_score'] = np.std(np.array(ls_r2))
        run['sd_pearson_correlation'] = np.std(np.array(ls_p_cor))
        if verbose:
            print("average MAE ", np.mean(np.array(ls_mae)))
            print("average MSE ", np.mean(np.array(ls_mse)))
            print("average r2 ", np.mean(np.array(ls_r2)))
            print("average Pearson Correlation ", np.mean(np.array(ls_p_cor)))

        df_test_data_pred['avg_pred'] = df_test_data_pred['avg_pred'] / len(ls_random_seeds)
        df_test_data_pred.to_csv(res_path + 'version_' + version + '_'+ model_type +'_test_dataset_pred.csv', sep='|')
        if verbose:
            print("test results saved to " + res_path + 'version_' + version + '_'+ model_type + '_test_dataset_pred.csv')
        run['data/test_dataset_pred'].track_files(res_path + 'version_' + version + '_'+ model_type + '_test_dataset_pred.csv')
    elif model_category == 'classifier':
        run['avg_pipe_score'] = np.mean(np.array(ls_pipe_score))
        if verbose:
            print("average MAE ", np.mean(np.array(ls_mae)))
            print("average MSE ", np.mean(np.array(ls_mse)))
            print("average ACC ", np.mean(np.array(ls_pipe_score)))

    # permutation importance
    df_perm_imp = pd.DataFrame(columns=['features', 'avg_mean_perm_imp', 'sd_mean_perm_imp'])
    df_perm_imp['features'] = x_train.columns.values
    df_perm_imp['avg_mean_perm_imp'] = np.mean(np.array(ls_mean_perm_imp), axis=0)
    df_perm_imp['sd_mean_perm_imp'] = np.std(np.array(ls_mean_perm_imp), axis=0)

    df_perm_imp.to_csv(res_path + model_type + '_' + str_ls_pred_labels + '_multi_randseed_perm_impt_avg_sd.csv',
                       sep='|')
    run['data/perm_impt_avg_sd'].track_files(res_path + model_type + '_' + str_ls_pred_labels
                                             + '_multi_randseed_perm_impt_avg_sd.csv')
    if verbose:
        print("perm_impt_avg_sd saved to " + res_path + model_type + '_' + str_ls_pred_labels
              + '_multi_randseed_perm_impt_avg_sd.csv')

    # built-in feature importance
    if len(ls_sr_feat_imp) > 0:
        df_all_feat_imp = pd.concat(ls_sr_feat_imp, axis=1)
        av_feat_imp = pd.DataFrame(df_all_feat_imp.mean(axis=1)).sort_values(by=0, axis=0)
        sd_feat_imp = pd.DataFrame(df_all_feat_imp.std(axis=1))
        av_feat_imp.rename(columns={0: 'avg_coef'}, inplace=True)
        sd_feat_imp.rename(columns={0: 'sd_coef'}, inplace=True)
        df_feat_imp = pd.merge(av_feat_imp, sd_feat_imp, how='left',
                               left_on=av_feat_imp.index, right_on=sd_feat_imp.index)
        fig4 = av_feat_imp[-10:].plot(kind='barh', legend=False).figure
        fig5 = av_feat_imp[:10].plot(kind='barh', legend=False).figure
        run["avg_feature_importance_top"].upload(neptune.types.File.as_image(fig4))
        run["avg_feature_importance_bot"].upload(neptune.types.File.as_image(fig5))
        df_all_feat_imp.rename(columns={'Unnamed: 0': 'features'}, inplace=True)
        df_all_feat_imp.to_csv(res_path + model_type + '_' + str_ls_pred_labels + '_multi_randseed_all_feat_impt.csv',
                               sep='|')
        run['data/all_feat_impt'].track_files(res_path + model_type + '_' + str_ls_pred_labels
                                              + '_multi_randseed_all_feat_impt.csv')
        if verbose:
            print("feat_impt_avg_sd saved to " + res_path + model_type + '_' + str_ls_pred_labels
                  + '_multi_randseed_all_feat_impt.csv')

        df_feat_imp.to_csv(res_path + model_type + '_' + str_ls_pred_labels +
                           '_multi_randseed_feat_impt_avg_sd.csv', sep='|')
        run['data/feat_impt_avg_sd'].track_files(res_path + model_type + '_' + str_ls_pred_labels +
                                                 '_multi_randseed_feat_impt_avg_sd.csv')
        if verbose:
            print("feat_impt_avg_sd saved to " + res_path + model_type + '_' + str_ls_pred_labels +
                  '_multi_randseed_feat_impt_avg_sd.csv')

    return


def run_with_presplitted_data_with_multiple_rand_seeds(df_train_data, df_test_data, ls_features_cols, ls_label_col, run,
                                                       config):
    """
    # give separate test set - repeat with rand seed only without kfold

    :param df_train_data:
    :param df_test_data:
    :param ls_features_cols:
    :param label_col:
    :param name:
    :param run:
    :param config:
    :return:
    """
    verbose = config.getboolean('general', 'verbose')
    version = config.get('general', 'version')
    rand_state = int(config.get('general', 'random_seed'))
    model_category = config.get('model_selection', 'model_category')
    model_type = config.get('model_selection', 'model_type')
    str_ls_pred_labels = config.get('input_data_selection', 'ls_prediction_labels').lower()
    ls_prediction_labels = [s.strip() for s in config.get('input_data_selection', 'ls_prediction_labels').split(',')]
    ls_random_seeds = [int(s.strip()) for s in config.get('general', 'ls_random_seeds').split(',')]
    res_path = config.get('general', 'res_path') + \
               config.get('input_data_selection', 'rbp_input_for_regressor') + '_' + \
               config.get('input_data_selection', 'transcript_regions') + '/' + 'diff_isoforms_' + \
               config.get('input_data_selection', 'differential_isoforms') + '/' + \
               config.get('input_data_selection', 'ls_prediction_labels').replace(',', '') + '/' + \
               config.get('input_data_selection', 'ls_train_cell_lines').replace(',', '') + '_' + \
               config.get('input_data_selection', 'ls_test_cell_lines').replace(',', '') + '/'
    os.makedirs(res_path, exist_ok=True)
    if verbose:
        print('in function {0}'.format('run_with_presplitted_data_with_multiple_rand_seeds'))

    df_train_data.to_pickle(res_path + 'version_' + version + '_rand_' + str(rand_state) + '_trainset.pickle')
    df_test_data.to_pickle(res_path + 'version_' + version + '_rand_' + str(rand_state) + '_testset.pickle')

    x_train = df_train_data[ls_features_cols]
    x_test = df_test_data[ls_features_cols]
    # note: here y_train is dataframe for multiple labels and previously was series for 1 label
    if len(ls_label_col) > 1:
        y_train = df_train_data[ls_label_col].to_numpy()
        y_test = df_test_data[ls_label_col].to_numpy()
    else:
        y_train = df_train_data[ls_label_col[0]]
        y_test = df_test_data[ls_label_col[0]]

    # for saving results later
    df_test_data_pred = df_test_data.copy()
    # avg preds for all seeds
    dict_res_labels = dict()
    ls_sr_feat_imp = []
    ls_mean_perm_imp = []
    for pred_label in ls_prediction_labels:
        df_test_data_pred['avg_pred_' + pred_label.lower()] = 0.0
        dict_res_labels['ls_mae_' + pred_label.lower()] = []
        dict_res_labels['ls_mse_' + pred_label.lower()] = []
        dict_res_labels['ls_r2_' + pred_label.lower()] = []
        dict_res_labels['ls_p_cor_' + pred_label.lower()] = []
        dict_res_labels['ls_pipe_score_' + pred_label.lower()] = []

    for rand_i in ls_random_seeds:
        tf.random.set_seed(rand_i)
        np.random.seed(rand_i)
        random.seed(rand_i)
        y_pred, df_unsorted_coefs, perm_imp, pipe_score = fold_train_pred_model(x_train, y_train,
                                                                                x_test, y_test,
                                                                                rand_i, run, config)

        if len(ls_prediction_labels) == 1:
            pred_label = ls_prediction_labels[0]
        dict_res_labels['ls_mae_' + pred_label.lower()].append(mean_absolute_error(y_test, y_pred))
        dict_res_labels['ls_mse_' + pred_label.lower()].append(mean_squared_error(y_test, y_pred))
        if len(df_unsorted_coefs) > 0:
            ls_sr_feat_imp.append(df_unsorted_coefs)
        if perm_imp != []:
            ls_mean_perm_imp.append(perm_imp['importances_mean'])
        if model_category == 'regressor':
            dict_res_labels['ls_r2_' + pred_label.lower()].append(r2_score(y_test, y_pred))
            dict_res_labels['ls_p_cor_' + pred_label.lower()].append(np.corrcoef(y_test, y_pred)[0, 1])
            df_test_data_pred['pred_' + str(rand_i)] = 0.0
            df_test_data_pred['avg_pred_' + pred_label.lower()] = df_test_data_pred[
                                                                      'avg_pred_' + pred_label.lower()] + y_pred
            df_test_data_pred['pred_' + str(rand_i)] = y_pred
        elif model_category == 'classifier':
            dict_res_labels['ls_pipe_score_' + pred_label.lower()].append(pipe_score)

    run['avg_mean_absolute_error'] = np.mean(np.array(dict_res_labels['ls_mae_' + pred_label.lower()]))
    run['avg_mean_squared_error'] = np.mean(np.array(dict_res_labels['ls_mse_' + pred_label.lower()]))
    run['avg_mean_perm_imp'] = np.mean(np.array(ls_mean_perm_imp), axis=0)
    run['sd_mean_absolute_error'] = np.std(np.array(dict_res_labels['ls_mae_' + pred_label.lower()]))
    run['sd_mean_squared_error'] = np.std(np.array(dict_res_labels['ls_mse_' + pred_label.lower()]))
    run['sd_mean_perm_imp'] = np.std(np.array(ls_mean_perm_imp), axis=0)
    if model_category == 'regressor':
        run['avg_r2_score'] = np.mean(np.array(dict_res_labels['ls_r2_' + pred_label.lower()]))
        run['avg_pearson_correlation'] = np.mean(np.array(dict_res_labels['ls_p_cor_' + pred_label.lower()]))
        run['sd_r2_score'] = np.std(np.array(dict_res_labels['ls_r2_' + pred_label.lower()]))
        run['sd_pearson_correlation'] = np.std(np.array(dict_res_labels['ls_p_cor_' + pred_label.lower()]))
        if verbose:
            print("average MAE ", np.mean(np.array(dict_res_labels['ls_mae_' + pred_label.lower()])))
            print("average MSE ", np.mean(np.array(dict_res_labels['ls_mse_' + pred_label.lower()])))
            print("average r2 ", np.mean(np.array(dict_res_labels['ls_r2_' + pred_label.lower()])))
            print("average Pearson Correlation ", np.mean(np.array(dict_res_labels['ls_p_cor_' + pred_label.lower()])))

        df_test_data_pred['avg_pred_' + pred_label.lower()] = df_test_data_pred['avg_pred_' + pred_label.lower()] / len(
            ls_random_seeds)
        df_test_data_pred.to_csv(res_path + 'version_' + version + '_rand_' + str(rand_state)  + '_'+ model_type +
                                 '_test_dataset_pred.csv', sep='|')
        if verbose:
            print("test results saved to " + res_path + 'version_' + version + '_rand_' + str(rand_state)  + '_'+ model_type +
                  '_test_dataset_pred.csv')
        run['data/test_dataset_pred'].track_files(res_path + 'version_' + version + '_rand_' + str(rand_state) + '_'+ model_type +
                                                  '_test_dataset_pred.csv')
    elif model_category == 'classifier':
        run['avg_pipe_score'] = np.mean(np.array(dict_res_labels['ls_pipe_score_' + pred_label.lower()]))
        if verbose:
            print("average MAE ", np.mean(np.array(dict_res_labels['ls_mae_' + pred_label.lower()])))
            print("average MSE ", np.mean(np.array(dict_res_labels['ls_mse_' + pred_label.lower()])))
            print("average ACC ", np.mean(np.array(dict_res_labels['ls_pipe_score_' + pred_label.lower()])))
    # permutation importance
    df_perm_imp = pd.DataFrame(columns=['features', 'avg_mean_perm_imp', 'sd_mean_perm_imp'])
    df_perm_imp['features'] = x_train.columns.values
    df_perm_imp['avg_mean_perm_imp'] = np.mean(np.array(ls_mean_perm_imp), axis=0)
    df_perm_imp['sd_mean_perm_imp'] = np.std(np.array(ls_mean_perm_imp), axis=0)
    df_perm_imp.to_csv(
        res_path + model_type + '_' + str_ls_pred_labels + '_preslpitted_multi_randseed_perm_impt_avg_sd.csv',
        sep='|')
    run['data/perm_impt_avg_sd'].track_files(res_path + model_type + '_' + str_ls_pred_labels
                                             + '_preslpitted_multi_randseed_perm_impt_avg_sd.csv')
    if verbose:
        print("perm_impt_avg_sd saved to " + res_path + model_type + '_' + str_ls_pred_labels
              + '_preslpitted_multi_randseed_perm_impt_avg_sd.csv')
    # built-in feature importance
    if len(ls_sr_feat_imp) > 0:
        df_all_feat_imp = pd.concat(ls_sr_feat_imp, axis=1)
        av_feat_imp = pd.DataFrame(df_all_feat_imp.mean(axis=1)).sort_values(by=0, axis=0)
        sd_feat_imp = pd.DataFrame(df_all_feat_imp.std(axis=1))
        av_feat_imp.rename(columns={0: 'avg_coef'}, inplace=True)
        sd_feat_imp.rename(columns={0: 'sd_coef'}, inplace=True)
        df_feat_imp = pd.merge(av_feat_imp, sd_feat_imp, how='left',
                               left_on=av_feat_imp.index, right_on=sd_feat_imp.index)
        fig4 = av_feat_imp[-10:].plot(kind='barh', legend=False).figure
        fig5 = av_feat_imp[:10].plot(kind='barh', legend=False).figure
        run["avg_feature_importance_top"].upload(neptune.types.File.as_image(fig4))
        run["avg_feature_importance_bot"].upload(neptune.types.File.as_image(fig5))
        df_all_feat_imp.rename(columns={'Unnamed: 0': 'features'}, inplace=True)
        df_all_feat_imp.to_csv(res_path + model_type + '_' + str_ls_pred_labels +
                               '_preslpitted_multi_randseedall_feat_impt.csv', sep='|')
        run['data/all_feat_impt'].track_files(res_path + model_type + '_' + str_ls_pred_labels
                                              + '_preslpitted_multi_randseed_all_feat_impt.csv')
        if verbose:
            print("feat_impt_avg_sd saved to " + res_path + model_type + '_' + str_ls_pred_labels
                  + '_preslpitted_multi_randseed_all_feat_impt.csv')

        df_feat_imp.to_csv(res_path + model_type + '_' + str_ls_pred_labels +
                           '_preslpitted_multi_randseed_feat_impt_avg_sd.csv', sep='|')
        run['data/feat_impt_avg_sd'].track_files(res_path + model_type + '_' + str_ls_pred_labels +
                                                 '_preslpitted_multi_randseed_feat_impt_avg_sd.csv')
        if verbose:
            print("feat_impt_avg_sd saved to " + res_path + model_type + '_' + str_ls_pred_labels +
                  '_preslpitted_multi_randseed_feat_impt_avg_sd.csv')

    return


####### for each fold / random seed

def fold_train_pred_model(x_train, y_train, x_test, y_test, i_fold, run, config):
    verbose = config.getboolean('general', 'verbose')
    model_category = config.get('model_selection', 'model_category')
    model_type = config.get('model_selection', 'model_type')
    ls_prediction_labels = [s.strip() for s in config.get('input_data_selection', 'ls_prediction_labels').split(',')]
    perm_imp = []
    # run regression model
    if model_category == 'regressor':
        if model_type in ['linear_regression', 'lasso_regression',
                          'ridge_regression', 'gradient_boosted_tree_regressor']:
            y_pred, df_unsorted_coefs, perm_imp, pipe_score = train_pred_linear_gbr_model(x_train, y_train,
                                                                                          x_test, y_test,
                                                                                          i_fold, run, config)
        elif model_type in ['multi_layer_perceptron', 'multi_legged_nn']:
            y_pred, df_unsorted_coefs, pipe_score = train_pred_nn_model(x_train, y_train,
                                                                        x_test, y_test,
                                                                        i_fold, run, config)
        if verbose:
            print('type of y_test is ', type(y_test))
            print('type of y_pred is ', type(y_pred))
            print('shape of y_test is ', y_test.shape)
            print('shape of y_pred is ', y_pred.shape)

        # store results and print
        run['fold_seed_' + str(i_fold) + '_mean_absolute_error'] = mean_absolute_error(y_test, y_pred)
        run['fold_seed_' + str(i_fold) + '_mean_squared_error'] = mean_squared_error(y_test, y_pred)
        run['fold_seed_' + str(i_fold) + '_r2_score'] = r2_score(y_test, y_pred)
        run['fold_seed_' + str(i_fold) + '_pearson_correlation'] = np.corrcoef(y_test, y_pred)[0, 1]
        if verbose:
            print(20 * '==')
            print('in fold / random seed ', i_fold)
            print('train shapes', x_train.shape, y_train.shape)
            print('test shapes', x_test.shape, y_test.shape)
            print('# training samples: ', x_train.shape[0], "\n",
                  "MAE: ", mean_absolute_error(y_test, y_pred), "\n",
                  "MSE: ", mean_squared_error(y_test, y_pred), "\n",
                  "r2: ", r2_score(y_test, y_pred), "\n",
                  "pipline score: ", pipe_score, "\n",
                  "Pearson correlation: ", np.corrcoef(y_test, y_pred)[0, 1], "\n")

        if len(ls_prediction_labels) > 1:
            for i in range(len(ls_prediction_labels)):
                # store results and print
                run['fold_seed_' + str(i_fold) + '_mean_absolute_error_' + ls_prediction_labels[
                    i].lower()] = mean_absolute_error(y_test[:, i], y_pred[:, i])
                run['fold_seed_' + str(i_fold) + '_mean_squared_error_' + ls_prediction_labels[
                    i].lower()] = mean_squared_error(y_test[:, i], y_pred[:, i])
                run['fold_seed_' + str(i_fold) + '_r2_score_' + ls_prediction_labels[i].lower()] = r2_score(
                    y_test[:, i], y_pred[:, i])
                run['fold_seed_' + str(i_fold) + '_pearson_correlation_' + ls_prediction_labels[i].lower()] = \
                    np.corrcoef(y_test[:, i], y_pred[:, i])[0, 1]
                if verbose:
                    print('multi-task setting - for each label separately - ' + ls_prediction_labels[i] + ' :')
                    print("MAE: ", mean_absolute_error(y_test[:, i], y_pred[:, i]), "\n",
                          "MSE: ", mean_squared_error(y_test[:, i], y_pred[:, i]), "\n",
                          "r2: ", r2_score(y_test[:, i], y_pred[:, i]), "\n",
                          "Pearson correlation: ", np.corrcoef(y_test[:, i], y_pred[:, i])[0, 1], "\n")

                fig1 = plt.figure()
                sns.scatterplot(y_test[:, i], y_pred[:, i], alpha=0.2)
                run['visuals/fold_seed_' + str(i_fold) + "_test_scatter_" + ls_prediction_labels[i].lower()].upload(
                    neptune.types.File.as_image(fig1))

    elif model_category == 'classifier':
        plt.clf()
        fig_class_imb_train = sns.distplot(y_train).figure
        plt.clf()
        fig_class_imb_test = sns.distplot(y_test).figure
        run['fold_seed_' + str(i_fold) + "train_class_imbalance"].upload(
            neptune.types.File.as_image(fig_class_imb_train))
        run['fold_seed_' + str(i_fold) + "test_class_imbalance"].upload(neptune.types.File.as_image(fig_class_imb_test))

        if model_type in ['logistic_regression', 'ridge_classifier', 'gradient_boosted_tree_classifier']:
            y_pred, df_unsorted_coefs, perm_imp, pipe_score = train_pred_linear_gbr_model(x_train, y_train,
                                                                                          x_test, y_test,
                                                                                          i_fold, run, config)
        elif model_type in ['multi_layer_perceptron', 'multi_legged_nn']:
            y_pred, df_unsorted_coefs, pipe_score = train_pred_nn_model(x_train, y_train,
                                                                        x_test, y_test,
                                                                        i_fold, run, config)
        # store results and print
        run['fold_seed_' + str(i_fold) + '_mean_absolute_error'] = mean_absolute_error(y_test, y_pred)
        run['fold_seed_' + str(i_fold) + '_mean_squared_error'] = mean_squared_error(y_test, y_pred)
        # run['fold_' + str(i_fold) + '_acc_score'] = acc(y_test, y_pred) or  model.score(x_test, y_test)
        if verbose:
            print(20 * '==')
            print('in fold / random seed ', i_fold)
            print('train shapes', x_train.shape, y_train.shape)
            print('test shapes', x_test.shape, y_test.shape)
            print('# training samples: ', x_train.shape[0], "\n",
                  "MAE: ", mean_absolute_error(y_test, y_pred), "\n",
                  "MSE: ", mean_squared_error(y_test, y_pred), "\n",
                  "pipline score: ", pipe_score, "\n")
            print("classification report: ")
            print(classification_report(y_test, y_pred))

    if len(df_unsorted_coefs) > 0:
        df_sorted_feat_impt = pd.DataFrame(df_unsorted_coefs).sort_values(by='coefficients', axis=0)
        fig2 = df_sorted_feat_impt[-10:].plot(kind='barh', legend=False).figure
        fig3 = df_sorted_feat_impt[:10].plot(kind='barh', legend=False).figure
        run['visuals/fold_seed_' + str(i_fold) + "_feature_importance_top"].upload(neptune.types.File.as_image(fig2))
        run['visuals/fold_seed_' + str(i_fold) + "_feature_importance_bot"].upload(neptune.types.File.as_image(fig3))
        plt.show()
        df_unsorted_coefs.rename(columns={'coefficients': 'coefficients_' + str(i_fold)}, inplace=True)
    return y_pred, df_unsorted_coefs, perm_imp, pipe_score


def train_pred_linear_gbr_model(x_train, y_train, x_test, y_test, i_fold, run, config):
    verbose = config.getboolean('general', 'verbose')
    version = config.get('general', 'version')
    rand_state = int(config.get('general', 'random_seed'))
    rbp_input_for_regressor = config.get('input_data_selection', 'rbp_input_for_regressor')
    raw_agg = config.get('input_data_selection', 'raw_clip_agg')
    res_path = config.get('general', 'res_path') + \
               config.get('input_data_selection', 'rbp_input_for_regressor') + '_' + \
               config.get('input_data_selection', 'transcript_regions') + '/' + 'diff_isoforms_' + \
               config.get('input_data_selection', 'differential_isoforms') + '/' + \
               config.get('input_data_selection', 'ls_prediction_labels').replace(',', '') + '/' + \
               config.get('input_data_selection', 'ls_train_cell_lines').replace(',', '') + '_' + \
               config.get('input_data_selection', 'ls_test_cell_lines').replace(',', '') + '/'
    os.makedirs(res_path, exist_ok=True)
    model_category = config.get('model_selection', 'model_category')
    model_type = config.get('model_selection', 'model_type')
    pipeline_scaling_method = config.get(model_type, 'scaling_method')
    perm_repeat = int(config.get('general', 'perm_repeat'))
    # scaling
    if pipeline_scaling_method == 'minmax':
        scale_prep = MinMaxScaler()
    elif pipeline_scaling_method == 'standard':
        scale_prep = StandardScaler()
    elif pipeline_scaling_method == 'noscaling':
        scale_prep = ''
    else:
        print('invalid config value for pipeline_scaling_method, rolling back to default minmax')
        scale_prep = MinMaxScaler()
    # model
    if model_type == 'linear_regression':
        model = LinearRegression()
    elif model_type == 'lasso_regression':
        alpha = float(config.get('lasso_regression', 'alpha'))
        model = Lasso(alpha=alpha)
    elif model_type == 'ridge_regression':
        model = Ridge()
    elif model_type == 'gradient_boosted_tree_regressor':
        model = GradientBoostingRegressor(
            n_estimators=int(config.get('gradient_boosted_tree_regressor', 'n_estimators')),
            max_depth=int(config.get('gradient_boosted_tree_regressor', 'max_depth')),
            n_iter_no_change=int(config.get('gradient_boosted_tree_regressor', 'patience')),
            subsample=float(config.get('gradient_boosted_tree_regressor', 'subsample'))
            # ,random_state=rand_state
        )
    elif model_type == 'logistic_regression':
        # Note: since we are using One Vs Rest algorithm we must use 'liblinear' solver with it.
        model = LogisticRegression(multi_class='ovr', solver='liblinear')
    elif model_type == 'ridge_classifier':
        # Note: since we are using One Vs Rest algorithm we must use 'liblinear' solver with it.
        model = RidgeClassifier()
    elif model_type == 'gradient_boosted_tree_classifier':
        model = GradientBoostingClassifier(
            n_estimators=int(config.get('gradient_boosted_tree_regressor', 'n_estimators')),
            max_depth=int(config.get('gradient_boosted_tree_regressor', 'max_depth')),
            n_iter_no_change=int(config.get('gradient_boosted_tree_regressor', 'patience')),
            subsample=float(config.get('gradient_boosted_tree_regressor', 'subsample'))
            # ,random_state=rand_state
        )
    else:
        print('invalid config value for model_type, rolling back to default ridge_regression')
        model = Ridge()
    # make pipeline
    if scale_prep != '':
        pipe = Pipeline([('scale_prep', scale_prep),
                         ('model', model)])
    else:
        pipe = Pipeline([('model', model)])
    # train and predict
    pipe.fit(x_train, y_train)
    dump(pipe, res_path + 'fitted_model_' + model_type + '_rs_' + config.get('general', 'random_seed') + '.joblib')
    with open(res_path + 'fitted_model_' + model_type + '_rs_' + config.get('general', 'random_seed') +'.pkl','wb') as f:
        pickle.dump(pipe, f)
    y_pred = pipe.predict(x_test)
    # Note: we return unscaled unsorted coefficients
    if model_type in ['gradient_boosted_tree_regressor', 'gradient_boosted_tree_classifier']:
        df_coefs = pd.DataFrame(pipe.named_steps['model'].feature_importances_, columns=['coefficients'],
                                index=x_train.columns)
    else:
        df_coefs = pd.DataFrame(pipe.named_steps['model'].coef_, columns=['coefficients'],
                                index=x_train.columns)
    perm_imp = permutation_importance(model, x_test, y_test, n_repeats=perm_repeat, random_state=rand_state)


    if model_category == 'regressor':
        run['fold_seed_' + str(i_fold) + "model_summary"] = stringify_unsupported(npt_utils.create_regressor_summary(
            pipe, x_train, x_test, y_train, y_test))
        run['fold_seed_' + str(i_fold) + "estimator_test_scores"] = npt_utils.get_scores(pipe, x_test, y_test)
        run['visuals/fold_seed_' + str(i_fold) + "prediction_error"] = npt_utils.create_prediction_error_chart(pipe,
                                                                                                               x_train,
                                                                                                               x_test,
                                                                                                               y_train,
                                                                                                               y_test)
    elif model_category == 'classifier':
        run['fold_seed_' + str(i_fold) + "model_summary"] = stringify_unsupported(npt_utils.create_classifier_summary(
            pipe, x_train, x_test, y_train, y_test))
        run['fold_seed_' + str(i_fold) + "estimator_test_scores"] = npt_utils.get_scores(pipe, x_test, y_test)
        run['visuals/fold_seed_' + str(
            i_fold) + 'classification_report'] = npt_utils.create_classification_report_chart(
            pipe, x_train, x_test, y_train, y_test)
        run['fold_seed_' + str(i_fold) + '_acc_score'] = pipe.score(x_test, y_test)

    return y_pred, df_coefs, perm_imp, pipe.score(x_test, y_test)


def train_pred_nn_model(x_train, y_train, x_test, y_test, i_fold, run, config):
    verbose = config.getboolean('general', 'verbose')
    version = config.get('general', 'version')
    rand_state = int(config.get('general', 'random_seed'))
    rbp_input_for_regressor = config.get('input_data_selection', 'rbp_input_for_regressor')
    raw_agg = config.get('input_data_selection', 'raw_clip_agg')

    res_path = config.get('general', 'res_path') + \
               config.get('input_data_selection', 'rbp_input_for_regressor') + '_' + \
               config.get('input_data_selection', 'transcript_regions') + '/' + 'diff_isoforms_' + \
               config.get('input_data_selection', 'differential_isoforms') + '/' + \
               config.get('input_data_selection', 'ls_prediction_labels').replace(',', '') + '/' + \
               config.get('input_data_selection', 'ls_train_cell_lines').replace(',', '') + '_' + \
               config.get('input_data_selection', 'ls_test_cell_lines').replace(',', '') + '/'
    os.makedirs(res_path, exist_ok=True)
    model_category = config.get('model_selection', 'model_category')
    ls_prediction_labels = [s.strip() for s in config.get('input_data_selection', 'ls_prediction_labels').split(',')]
    n_tasks = len(ls_prediction_labels)
    model_type = config.get('model_selection', 'model_type')

    if model_type == 'multi_layer_perceptron':
        epochs = int(config.get('multi_layer_perceptron', 'epochs'))
        patience = int(config.get('multi_layer_perceptron', 'patience'))
        pipeline_scaling_method = config.get('multi_layer_perceptron', 'scaling_method')

        # scaling
        if pipeline_scaling_method == 'minmax':
            scale_prep = MinMaxScaler()
        elif pipeline_scaling_method == 'standard':
            scale_prep = StandardScaler()
        elif pipeline_scaling_method == 'noscaling':
            scale_prep = ''
        else:
            print('invalid config value for pipeline_scaling_method, rolling back to default minmax')
            scale_prep = MinMaxScaler()
        model = MLPRegressor(n_iter_no_change=patience,
                             learning_rate='adaptive',
                             activation='relu',
                             solver='adam',
                             max_iter=epochs,
                             early_stopping=True
                             # , random_state=rand_state
                             )
        # make pipeline
        if scale_prep != '':
            pipe = Pipeline([('scale_prep', scale_prep),
                             ('model', model)])
        else:
            pipe = Pipeline([('model', model)])
        # train and predict
        pipe.fit(x_train, y_train)
        dump(pipe, res_path + 'fitted_model_' + model_type + '_rs_' + config.get('general', 'random_seed') + '.joblib')
        import pickle
        with open(res_path + 'fitted_model_' + model_type + '_rs_' + config.get('general', 'random_seed') +'.pkl','wb') as f:
            pickle.dump(pipe, f)
        y_pred = pipe.predict(x_test)
        score_val = pipe.score(x_test, y_test)

    elif model_type == 'multi_legged_nn':

        valid_frac = float(config.get('multi_legged_nn', 'valid_frac'))
        epochs = int(config.get('multi_legged_nn', 'epochs'))
        batch_size = int(config.get('multi_legged_nn', 'batch_size'))
        patience = int(config.get('multi_legged_nn', 'patience'))
        exon_density_included = config.getboolean('input_data_selection', 'exon_density_included')
        verbose_for_nn = 0
        ls_features_cols = [i for i in x_train.columns.values]
        ls_features_cols_deepripe = [i for i in ls_features_cols if str(i).split('_')[-1] == 'deepripe']
        ls_features_cols_rawclip = [i for i in ls_features_cols if str(i).split('_')[-1] == 'raw']
        x_train_deepripe = x_train[ls_features_cols_deepripe]
        x_train_rawclip = x_train[ls_features_cols_rawclip]
        x_test_deepripe = x_test[ls_features_cols_deepripe]
        x_test_rawclip = x_test[ls_features_cols_rawclip]

        my_callbacks = [keras.callbacks.EarlyStopping(patience=patience)
            , NeptuneCallback(run=run, base_namespace='metrics')]

        if exon_density_included:
            x_train_exon_density = x_train[['exon_density']]
            x_test_exon_density = x_test[['exon_density']]

            nn_model = create_three_legged_nn(x_train_deepripe, x_train_rawclip, x_train_exon_density, n_tasks)
            hist = nn_model.fit([x_train_deepripe, x_train_rawclip, x_train_exon_density], y_train,
                                validation_split=valid_frac,
                                epochs=epochs,
                                verbose=verbose_for_nn,
                                batch_size=batch_size,
                                shuffle=True, callbacks=my_callbacks)
            y_pred = nn_model.predict([x_test_deepripe, x_test_rawclip, x_test_exon_density]).flatten()

        else:
            nn_model = create_two_legged_nn(x_train_deepripe, x_train_rawclip, n_tasks)
            hist = nn_model.fit([x_train_deepripe, x_train_rawclip], y_train,
                                validation_split=valid_frac,
                                epochs=epochs,
                                verbose=verbose_for_nn,
                                batch_size=batch_size,
                                shuffle=True, callbacks=my_callbacks)
            y_pred = nn_model.predict([x_test_deepripe, x_test_rawclip]).flatten()
        nn_model.save(res_path + 'fitted_model_' + model_type + '_rs_' + config.get('general', 'random_seed'))
        fig1 = plt.figure()
        sns.scatterplot(y_test, y_pred, alpha=0.2)
        run['visuals/fold_seed_' + str(i_fold) + "_test_scatter"].upload(neptune.types.File.as_image(fig1))
        plt.show()
        # plotting accuracy
        plt.clf()
        fig2 = plt.figure()
        plt.plot(hist.history['mse'])
        plt.plot(hist.history['val_mse'])
        plt.title('Model MSE')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Valid'], loc='upper left')
        run['visuals/fold_seed_' + str(i_fold) + "_acc_plot"].upload(neptune.types.File.as_image(fig2))
        plt.show()
        # Plot training & validation loss values
        plt.clf()
        fig3 = plt.figure()
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Valid'], loc='upper left')
        run['visuals/fold_seed_' + str(i_fold) + "_loss_plot"].upload(neptune.types.File.as_image(fig3))
        plt.show()

        score_val = 0.0

    return y_pred, [], score_val


####### create NN models

def create_two_legged_nn(x_train_1, x_train_2, n_tasks):
    """

    :param x_train_1:
    :param x_train_2:
    :return:
    """
    # DeepRipe features
    input_1 = keras.Input(shape=(x_train_1.shape[1],), dtype='float32')
    x1 = keras.layers.Dense(100, activation='relu')(input_1)
    # x1 = keras.layers.BatchNormalization()(x1)
    # x1 = keras.layers.Dropout(0.25)(x1)
    x1 = keras.layers.Dense(20, activation='relu')(x1)

    # Raw clip features
    input_2 = keras.Input(shape=(x_train_2.shape[1],), dtype='float32')
    x2 = keras.layers.Dense(100, activation='relu')(input_2)
    # x2 = keras.layers.BatchNormalization()(x2)
    # x2 = keras.layers.Dropout(0.25)(x2)
    x2 = keras.layers.Dense(20, activation='relu')(x2)
    # merge
    merged = keras.layers.Concatenate()([x1, x2])
    # merged = keras.layers.Flatten()(merged)
    # merged = keras.layers.BatchNormalization()(merged)
    # merged = keras.layers.Dropout(0.25)(merged)
    merged = keras.layers.Dense(40, activation='relu')(merged)
    # merged = keras.layers.BatchNormalization()(merged)
    # merged = keras.layers.Dropout(0.25)(merged)
    preds = keras.layers.Dense(n_tasks, activation=None)(merged)

    model = keras.models.Model(inputs=[input_1, input_2], outputs=preds)
    model.compile(loss='mean_squared_error', optimizer='nadam', metrics=['mse'])

    return model


def create_three_legged_nn(x_train_1, x_train_2, x_train_3, n_tasks):
    """

    :param x_train_1:
    :param x_train_2:
    :param x_train_3:
    :return:
    """

    # DeepRipe features
    input_1 = keras.Input(shape=(x_train_1.shape[1],), dtype='float32')
    x1 = keras.layers.Dense(100, activation='relu')(input_1)
    # x1 = keras.layers.BatchNormalization()(x1)
    # x1 = keras.layers.Dropout(0.25)(x1)
    x1 = keras.layers.Dense(20, activation='relu')(x1)

    # Raw clip features
    input_2 = keras.Input(shape=(x_train_2.shape[1],), dtype='float32')
    x2 = keras.layers.Dense(100, activation='relu')(input_2)
    # x2 = keras.layers.BatchNormalization()(x2)
    # x2 = keras.layers.Dropout(0.25)(x2)
    x2 = keras.layers.Dense(20, activation='relu')(x2)

    # Exon density features
    input_3 = keras.Input(shape=(x_train_3.shape[1],), dtype='float32')
    x3 = keras.layers.Dense(1, activation='relu')(input_3)
    # x3 = keras.layers.BatchNormalization()(x3)
    # x3 = keras.layers.Dropout(0.25)(x3)
    # x3 = keras.layers.Dense(20, activation='relu')(x3)

    # merge
    merged = keras.layers.Concatenate()([x1, x2, x3])
    # merged = keras.layers.Flatten()(merged)
    # merged = keras.layers.BatchNormalization()(merged)
    # merged = keras.layers.Dropout(0.25)(merged)
    merged = keras.layers.Dense(41, activation='relu')(merged)
    # merged = keras.layers.BatchNormalization()(merged)
    # merged = keras.layers.Dropout(0.25)(merged)
    preds = keras.layers.Dense(n_tasks, activation=None)(merged)

    model = keras.models.Model(inputs=[input_1, input_2, input_3], outputs=preds)
    model.compile(loss='mean_squared_error', optimizer='nadam', metrics=['mse'])

    return model
