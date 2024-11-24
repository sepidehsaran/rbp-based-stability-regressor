import warnings
import neptune
import os
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
sns.set_theme(style='ticks', font_scale=1.5)


# config and run setting
def assign_run_variables(rbp_input_for_regressor, run, config):
    # read relevant config and set run params
    verbose = config.getboolean('general', 'verbose')
    if verbose: print('in function {0}'.format('assign_run_variables'))
    run['version'] = int(config.get('general', 'version'))
    run['ls_deepripe_pretrained_cell_lines'] = [s.strip() for s in
                                                config.get('input_data_selection',
                                                           'deepripe_pretrained_cell_lines').split(
                                                    ',')]
    run['pretrained_deepripes_agg'] = config.get('input_data_selection', 'pretrained_deepripes_agg')
    run['deepripe_selected_rbps_only'] = config.getboolean('input_data_selection', 'deepripe_selected_rbps_only')
    run['deepripe_inference'] = config.get('input_data_selection', 'deepripe_inference')
    run['model_category'] = config.get('model_selection', 'model_category')
    run['model_type'] = config.get('model_selection', 'model_type')
    run['ls_prediction_labels'] = config.get('input_data_selection', 'ls_prediction_labels')
    run['random_seed'] = int(config.get('general', 'random_seed'))
    run['raw_clip_agg'] = config.get('input_data_selection', 'raw_clip_agg')
    run['replace_rawclip_with_deepripe_rbps'] = config.getboolean('input_data_selection',
                                                                  'replace_rawclip_with_deepripe_rbps')
    run['exon_density_included'] = config.getboolean('input_data_selection', 'exon_density_included')
    run['log2_exon_density'] = config.getboolean('input_data_selection', 'log2_exon_density')
    run['non_diff_with_kfold'] = config.getboolean('input_data_selection', 'non_diff_with_kfold')
    run['transcript_regions'] = config.get('input_data_selection', 'transcript_regions')
    run['ls_train_cell_lines'] = [s.strip() for s in
                                  config.get('input_data_selection', 'ls_train_cell_lines').split(',')]
    run['ls_test_cell_lines'] = [s.strip() for s in
                                 config.get('input_data_selection', 'ls_test_cell_lines').split(',')]

    diff_isoforms = config.getboolean('input_data_selection', 'differential_isoforms')
    run['diff_isoforms'] = diff_isoforms
    diff_genes = config.getboolean('input_data_selection', 'differential_genes')
    run['diff_genes'] = diff_genes
    run['fill_missing_raw_clip'] = config.getboolean('input_data_selection', 'fill_missing_raw_clip')
    run['pipeline_scaling_method'] = config.get('linear_regression', 'scaling_method')

    # later overwrite those not used in special cases here
    if rbp_input_for_regressor == 'deepripe_only':
        run['raw_clip_agg'] = ''
        run['replace_rawclip_with_deepripe_rbps'] = ''

    elif rbp_input_for_regressor == 'deepripe_rawclip':
        pass

    elif rbp_input_for_regressor == 'rawclip_only':
        run['ls_deepripe_pretrained_cell_lines'] = ''
        run['pretrained_deepripes_agg'] = ''
        run['deepripe_selected_rbps_only'] = ''
        run['deepripe_inference'] = ''
        run['replace_rawclip_with_deepripe_rbps'] = ''

    elif rbp_input_for_regressor == 'baseline_deepripe_only':
        run['pretrained_deepripes_agg'] = ''
        run['deepripe_selected_rbps_only'] = ''
        run['deepripe_inference'] = ''
        run['replace_rawclip_with_deepripe_rbps'] = ''
    else:
        raise ValueError("Value selected for rbp_input_for_regressor is not supported.")

    return


# evaluation metric functions
def precision(y_true, y_pred):
    true_positives = keras.sum(keras.round(keras.clip(y_true * y_pred, 0, 1)))
    predicted_positives = keras.sum(keras.round(keras.clip(y_pred, 0, 1)))
    y_precision = true_positives / (predicted_positives + keras.epsilon())
    return y_precision


def recall(y_true, y_pred):
    true_positives = keras.sum(keras.round(keras.clip(y_true * y_pred, 0, 1)))
    # TPs=K.sum(K.round(K.clip(y_ture * y_pred , 0, 1)))
    possible_positives = keras.sum(keras.round(keras.clip(y_true, 0, 1)))
    y_recall = true_positives / (possible_positives + keras.epsilon())
    return y_recall


# visualization functions
def show_class_eval_plots_onehotlabels(run, y_true, y_pred, n_classes, fold=''):
    """
    Code credit to Mahsa Ghanbari - DeepRiPe
    :param run:
    :param y_true:
    :param y_pred:
    :param n_classes:
    :param fold:
    :return:
    """

    if n_classes == 3:
        task_names = ["low", "mid", "high"]
    elif n_classes == 4:
        task_names = ["low", "mid-low", "mid-high", "high"]
    else:
        raise ValueError("Invalid number of classes")

    precision = [None] * n_classes
    recall = [None] * n_classes
    average_precision = [None] * n_classes
    fpr = [None] * n_classes
    tpr = [None] * n_classes
    roc_auc = [None] * n_classes
    pr_auc = [None] * n_classes

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], y_pred[:, i])
        pr_auc[i] = auc(recall[i], precision[i])
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    #### define colors
    colors = [(39, 64, 139), (0, 128, 128), (31, 119, 180), (44, 160, 44), (152, 223, 138), (174, 199, 232),
              (255, 127, 14), (255, 187, 120), (214, 39, 40), (255, 152, 150), (148, 103, 189), (197, 176, 213),
              (122, 55, 139), (247, 182, 210), (140, 86, 75), (139, 34, 82)]

    for i in range(len(colors)):
        r, g, b = colors[i]
        colors[i] = (r / 255., g / 255., b / 255.)

    # modifying parameters for plot
    # used for size
    golden_mean = (sqrt(5) - 1.0) / 2.0
    # fig width in inches
    fig_width = 4.5
    # fig height in inches
    fig_height = fig_width * golden_mean

    # plot
    fig_pr = plt.figure()
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label='{0} (AUROC = {1:0.2f} , AP= {2:0.2f})' ''.format(task_names[i], roc_auc[i],
                                                                         average_precision[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.tick_params(axis='x', which='both', top='off')
    plt.tick_params(axis='y', which='both', right='off')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curves')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc=2, ncol=1, bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()
    plt.show()
    run['fold_' + str(fold) + "_test_per_rec"].upload(neptune.types.File.as_image(fig_pr))

    fig_roc = plt.figure()
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='{0} (AUROC = {1:0.2f} , AP= {2:0.2f})' ''.format(task_names[i], roc_auc[i],
                                                                         average_precision[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.tick_params(axis='x', which='both', top='off')
    plt.tick_params(axis='y', which='both', right='off')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc=2, ncol=1, bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()
    plt.show()
    run['fold_' + str(fold) + "_test_roc"].upload(neptune.types.File.as_image(fig_roc))
    return


def read_importance(datapath, rbp, region, test_cl, train_cl, model, label,
                    importance_kind, diff_isoforms=False, absolute=False):
    """
    Function to read out feature importance for new results paths.
    By default, reads permutation importance.
    Importance can be:
    :param datapath:
    :param rbp:
    :param region:
    :param test_cl:
    :param train_cl:
    :param model:
    :param label:
    :param importance_kind: selection between feat or perm
    :param diff_isoforms:
    :param absolute: if True, for lr, make negative importance positive (for plotting)
    :return:
    """

    sufx = '_repeated_kfold_' + importance_kind + '_impt_avg_sd.csv'
    modelpath = os.path.join(datapath,
                             rbp + '_' + region,
                             'diff_isoforms' + '_' + str(diff_isoforms),
                             label,
                             train_cl + '_' + test_cl,
                             model + '_' + label.lower() + sufx
                             )
    if not os.path.exists(modelpath):
        raise ValueError("path %s does not exist" % modelpath)
    # print(f'reading from ... {modelpath}')

    if importance_kind == 'perm':
        df = pd.read_csv(modelpath, sep="|", index_col=0).set_index('features')
    else:
        df = pd.read_csv(modelpath, sep="|", index_col=0).rename(columns={'key_0': 'features'}).set_index('features')

    df.columns = ['imp', 'sd']

    # convert to absolute importance
    if absolute == True:
        df.imp = np.abs(df.imp)

    return df.sort_values('imp', ascending=True)


# plot top n coefficients form one model with sd
def plot_coef(datapath, rbp, region, test_cl, train_cl, model, label,
              importance_kind, absolute=False, diff_isoforms=False, ax=None, n=10):
    """
    plot importance for a single case - average coefficients and SDs
    :param datapath:
    :param rbp:
    :param region:
    :param test_cl:
    :param train_cl:
    :param model:
    :param label:
    :param importance_kind:
    :param absolute:
    :param diff_isoforms:
    :param ax:
    :param n: number of top RBPs and bottom RBPs to plot (default 10)
    :return:
    """
    # read feature importance
    toplt = (read_importance(datapath=datapath
                             , rbp=rbp
                             , region=region
                             , diff_isoforms=diff_isoforms
                             , test_cl=test_cl
                             , train_cl=train_cl
                             , model=model
                             , label=label
                             , importance_kind=importance_kind)
             .reset_index()
             .sort_values('imp', ascending=True)
             )

    # plot split head/tail if we have + and - coefs (lr)
    if model in ['ridge_regression', 'lasso_regression', 'linear_regression'] and importance_kind == 'feat':
        data = toplt.head(n).append(toplt.tail(n))
    else:
        data = toplt.tail(2 * n)[::-1]

    if ax == None:
        f, ax = plt.subplots()
    sns.pointplot(data=data, x='imp', y='features', join=False, palette='mako', ax=ax)
    ax.errorbar(data=data, x='imp', y='features', xerr='sd', ls='none',
                ecolor=sns.color_palette("mako", n_colors=2 * n))

    dict_rename_labels = {'Deg': 'Degradation',
                          'logT': 'Translation',
                          'logTE': 'TE',
                          'logTPM': 'mRNA abundance'}

    dict_rename_models = {'multi_legged_nn': 'multilegged NN',
                          'multi_layer_perceptron': 'multilayer perceptron',
                          'gradient_boosted_tree_regressor': 'gradient boosted tree regression',
                          'ridge_regression': 'ridge regression',
                          'linear_regression': 'linear regression',
                          'lasso_regression': 'lasso regression'}

    ax.set_title(dict_rename_labels[label] + "\n " + dict_rename_models[model])
    ax.set_ylabel('')
    ax.set_xlabel('Feature importance')
    return ax


## function to plot importances against each other
def plot_imp_2D(datapath,
                    rbp,
                    region,
                    test_cl1,test_cl2,
                    train_cl1,train_cl2,
                    model1, model2,
                    label1, label2,
                   importance_kind1,importance_kind2,
                ax=None, absolute=False,diff_isoforms=False,
               q=0.1):
    """
    plot importance for two models for non-differential case
    :param datapath:
    :param rbp:
    :param region:
    :param test_cl1:
    :param test_cl2:
    :param train_cl1:
    :param train_cl2:
    :param model1:
    :param model2:
    :param label1:
    :param label2:
    :param importance_kind1:
    :param importance_kind2:
    :param ax:
    :param absolute:
    :param diff_isoforms:
    :param q:
    :return:
    """

    # dict for nice labels
    nicelabel = {'Deg': 'Degradation', 'logT': 'Translation', 'logTE': 'Translation efficiency',
                 'logTPM': 'mRNA abundance'}

    i1=read_importance(datapath, rbp, region, diff_isoforms, test_cl1, train_cl1, model1, label1,
                       importance_kind1, absolute=absolute)
    i2=read_importance(datapath, rbp, region, diff_isoforms, test_cl2, train_cl2, model2, label2, importance_kind2,
                       absolute=absolute)
    print(i1.shape)
    print(i2.shape)
    m = i1.merge(i2,left_index=True, right_index=True,how='inner')
    print(m.shape)
    print(m.head(5))
    print(m.tail(5))
    #cutoffs on quantile
    x1=np.quantile(m.imp_x,q)
    x2=np.quantile(m.imp_x,(1-q))
    y1=np.quantile(m.imp_y,q)
    y2=np.quantile(m.imp_y,(1-q))
    #plot
    if ax==None:
        f,ax=plt.subplots()
    #f,ax=plt.subplots(figsize=(5, 5))
    #plt.figure(figsize=(8, 8))
    sns.scatterplot(data=m, x='imp_x',y='imp_y',ax=ax)
    ax.errorbar(data=m, x='imp_x',y='imp_y',
                 xerr='sd_x',yerr='sd_y',
                ls='none')
    for i,rbp in enumerate(m.index):
        #note: make sure np.max takes an array
        alpha=np.max((abs(m.iloc[i]['imp_x'])/np.max(abs(m.imp_x)),
                     abs(m.iloc[i]['imp_y'])/np.max(abs(m.imp_y))))**4
        ax.text(m.iloc[i]['imp_x'],y=m.iloc[i]['imp_y'],
                 s=rbp, fontsize=12,alpha=alpha)
    ax.set_xlabel(model1+'\n'+train_cl1 + ' ' + nicelabel[label1] +  ' ' + importance_kind1)
    ax.set_ylabel(model2+'\n'+train_cl2 + ' ' + nicelabel[label2]+  ' ' + importance_kind2)
    ax.set_title(train_cl2)
    #ax.set_title()
    #plt.show()
    return(ax)


def retrieve_results_from_neptune(config):
    res_path = config.get('general', 'res_path')
    version = config.get('general', 'version')
    # download res pf all runs for a specific version
    project = neptune.init_project(project=config.get('general', 'neptune_project'),
                                   api_token=config.get('general', 'neptune_token'),
                                   mode="read-only")
    # Get dashboard with runs contributed by "jackie" and tagged "cycleLR"
    df_runs_table = project.fetch_runs_table().to_pandas()
    print('columns of run results table are:')
    print(i for i in df_runs_table.columns.values)
    df_runs_table.head()

    res_neptune_path = res_path + 'all_neptune_res.csv'
    df_runs_table.to_csv(res_neptune_path)

    return df_runs_table
