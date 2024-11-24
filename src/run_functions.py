import matplotlib.pyplot as plt
from data_prep import *
from ml_train_eval import *


def regression_deepripe_only(config, run):
    """

    :param config:
    :param run:
    :return:
    """

    # read relevant config and set run params
    verbose = config.getboolean('general', 'verbose')
    if verbose: print('in function {0}'.format('regression_deepripe_only'))
    # note that df_dataset_test is empty for non-differntial case
    df_dataset_train, df_dataset_test, ls_all_rbps = prep_dataset_regression_deepripe_only(config, run)
    run_ml_model(df_dataset_train, df_dataset_test, ls_all_rbps, config, run)

    return


def regression_rawclip_only(config, run):
    """

    :param config:
    :param run:
    :return:
    """
    # read relevant config and set run params
    verbose = config.getboolean('general', 'verbose')
    if verbose: print('in function {0}'.format('regression_rawclip_only'))
    # note that df_dataset_test is empty for non-differntial case
    df_dataset_train, df_dataset_test, ls_all_rbps = prep_dataset_regression_rawclip_only(config, run)
    run_ml_model(df_dataset_train, df_dataset_test, ls_all_rbps, config, run)

    return


def regression_deepripe_rawclip(config, run):
    """

    :param config:
    :param run:
    :return:
    """
    # read relevant config and set run params
    verbose = config.getboolean('general', 'verbose')
    if verbose: print('in function {0}'.format('regression_deepripe_rawclip'))
    # note that df_dataset_test is empty for non-differntial case
    df_dataset_train, df_dataset_test, ls_all_rbps = prep_dataset_regression_deepripe_rawclip(config, run)
    run_ml_model(df_dataset_train, df_dataset_test, ls_all_rbps, config, run)


def baseline_n_bins(config, run):
    """
        TODO: this function is deprecated and needs to be updated
        running baseline models with only number of bins as feature - removing all RBP info
        goal is to see if nr_bins as a proxy of length of transcript is already a good predictor without RBP info
        :param config:
        :param run:
        :return:
        """
    # read relevant config and set run params
    verbose = config.getboolean('general', 'verbose')
    if verbose: print('in function {0}'.format('baseline_n_bins'))
    # update some run params - here we just need the nr_bins representing the transcript length
    ls_deepripe_pretrained_cell_lines = ['hek293']
    run['ls_deepripe_pretrained_cell_lines'] = ls_deepripe_pretrained_cell_lines
    test_cell_line = 'hek293'
    run['test_cell_line'] = test_cell_line
    deepripe_inference = config.get('input_data_selection', 'deepripe_inference')
    prediction_label = config.get('input_data_selection', 'prediction_label')
    diff_isoforms = config.getboolean('input_data_selection', 'differential_isoforms')
    # TODO: implement diff genes for this baseline
    diff_genes = config.getboolean('input_data_selection', 'differential_genes')
    deepripe_selected_rbps_only = config.getboolean('input_data_selection', 'deepripe_selected_rbps_only')
    ls_train_cell_lines = [s.strip() for s in config.get('input_data_selection', 'ls_train_cell_lines').split(',')]
    ls_test_cell_lines = [s.strip() for s in config.get('input_data_selection', 'ls_test_cell_lines').split(',')]

    # get deepripe output - rbp binding info
    if deepripe_inference == 'run':
        if verbose: print('running deepripe inference')
        df_res_deepripe = run_deepripe(config)
    elif deepripe_inference == 'load':
        # TODO: improve this part
        if verbose: print('loading precalculated deepripe outputs')
        print(ls_deepripe_pretrained_cell_lines)
        df_res_deepripe = load_deepripe(config, ls_deepripe_pretrained_cell_lines, deepripe_selected_rbps_only,
                                        include_nr_bins=True)
    else:
        print('config value set for deepripe_inference is not valid. please choose run or load.')
    print('columns of the deepripe dataset are')
    print([i for i in df_res_deepripe.columns])
    run['count_deepripe_output_rbps'] = len(df_res_deepripe.columns) - 1

    # get regression labels and join with rbp info
    df_parclip_labels = gene_level_labels_for_transcripts(config, prediction_label, ls_train_cell_lines)
    df_parclip_labels = df_parclip_labels[['tx_id', prediction_label]]
    df_dataset_parclip = pd.merge(df_res_deepripe, df_parclip_labels, how='inner', on='tx_id')
    df_dataset_parclip.drop_duplicates(inplace=True)
    df_dataset_parclip = df_dataset_parclip.sample(frac=1).reset_index(drop=True)

    if 'nr_bins' in df_dataset_parclip.columns:
        df_dataset_parclip = df_dataset_parclip[['tx_id', 'nr_bins', prediction_label]]
    elif 'nr_bins_x' in df_dataset_parclip.columns:
        df_dataset_parclip['nr_bins'] = df_dataset_parclip['nr_bins_x']
        df_dataset_parclip = df_dataset_parclip[['tx_id', 'nr_bins', prediction_label]]
    else:
        print('nr_bins info not found in parclip dataset')
        return
    # log transform
    df_dataset_parclip['nr_bins'] = np.log2(df_dataset_parclip['nr_bins'])

    if verbose:
        print('shape of dataset', df_dataset_parclip.shape)
        print('columns of dataset', [i for i in df_dataset_parclip.columns])

    run['total_rbp_features'] = 0

    if diff_isoforms:
        # test set would be transcripts from K562 that don't appear in Hek
        df_test_labels = gene_level_labels_for_transcripts(config, prediction_label, ls_test_cell_lines)
        df_test_labels = df_test_labels[['tx_id', prediction_label]]
        df_dataset_test = pd.merge(df_res_deepripe, df_test_labels, how='inner', on='tx_id')
        df_dataset_test.drop_duplicates(inplace=True)

        if 'nr_bins' in df_dataset_test.columns:
            df_dataset_test = df_dataset_test[['tx_id', 'nr_bins', prediction_label]]
        elif 'nr_bins_x' in df_dataset_test.columns:
            df_dataset_test['nr_bins'] = df_dataset_test['nr_bins_x']
            df_dataset_test = df_dataset_test[['tx_id', 'nr_bins', prediction_label]]
        else:
            print('nr_bins info not found in test dataset')
            return

        print('test columns')
        print(20 * '=')
        print([i for i in df_dataset_test.columns])
        # TODO: in future avoid hardcoding
        test_cell_line = 'K562'
        run['test_cell_line'] = test_cell_line

    run_ml_model(df_dataset_parclip, df_dataset_test, ['nr_bins'], config, run)

    return


def prep_dataset_regression_deepripe_only(config, run):
    """

    :param config:
    :param run:
    :return:
    """

    # read config setting
    verbose = config.getboolean('general', 'verbose')
    rand_state = int(config.get('general', 'random_seed'))
    if verbose: print('in function {0}'.format('prep_dataset_regression_deepripe_only'))
    # prediction_label = config.get('input_data_selection', 'prediction_label')
    ls_prediction_labels = [s.strip() for s in config.get('input_data_selection', 'ls_prediction_labels').split(',')]
    diff_isoforms = config.getboolean('input_data_selection', 'differential_isoforms')
    diff_genes = config.getboolean('input_data_selection', 'differential_genes')
    exon_density_included = config.getboolean('input_data_selection', 'exon_density_included')
    ls_train_cell_lines = [s.strip() for s in config.get('input_data_selection', 'ls_train_cell_lines').split(',')]
    ls_test_cell_lines = [s.strip() for s in config.get('input_data_selection', 'ls_test_cell_lines').split(',')]

    # sanity check train and test cell lines
    ls_train_cell_lines, ls_test_cell_lines = sanity_check_cell_lines(ls_train_cell_lines, ls_test_cell_lines)

    # read deepripe run results
    df_res_deepripe_renamed = get_deepripe_output(config)
    run['count_deepripe_output_rbps'] = len(df_res_deepripe_renamed.columns) - 1

    # add exon density feature if required
    if exon_density_included:
        df_exon_density = get_exon_density(config)
        df_res_deepripe_renamed = pd.merge(df_res_deepripe_renamed, df_exon_density, how='inner', on='tx_id')

    if 'gene_id' in df_res_deepripe_renamed.columns:
        del df_res_deepripe_renamed['gene_id']

    ls_remove_cols = ['tx_id', 'tx_id_x', 'tx_id_y', 'nr_bins_x', 'nr_bins', 'gene_id',
                      'gene_id_x', 'gene_id_y', 'cell_line', 'top_iso', 'nr_bins',
                      'gene_name', 'nr_bins_y']
    ls_remove_cols.extend(ls_prediction_labels)
    ls_all_rbps = [i for i in df_res_deepripe_renamed.columns if i not in ls_remove_cols]

    # TODO: add check to make sure ls_test and ls_train cell_lines don't overlap
    # get training and test labels
    df_train_cell_lines_labels = gene_level_labels_for_transcripts(config, ls_prediction_labels, ls_train_cell_lines)
    df_test_cell_lines_labels = gene_level_labels_for_transcripts(config, ls_prediction_labels, ls_test_cell_lines)
    if diff_isoforms:
        df_train_labels, df_test_labels = find_differntial_isoforms(config, ls_prediction_labels,
                                                                    df_train_cell_lines_labels,
                                                                    df_test_cell_lines_labels)
    elif diff_genes:
        df_train_labels, df_test_labels = find_differntial_genes(df_train_cell_lines_labels,
                                                                 df_test_cell_lines_labels)
    else:
        print("Note! runing with non-differntial setting!")
        df_train_labels = df_train_cell_lines_labels.copy()
        df_test_labels = pd.DataFrame()

    df_train_set = pd.merge(df_res_deepripe_renamed, df_train_labels, how='inner', on='tx_id')
    df_train_set = df_train_set.sample(frac=1, random_state=rand_state).reset_index(drop=True)
    if diff_genes or diff_isoforms:
        df_test_set = pd.merge(df_res_deepripe_renamed, df_test_labels, how='inner', on='tx_id')
    else:
        df_test_set = pd.DataFrame()
    run['total_rbp_features'] = len(ls_all_rbps)
    if verbose:
        print('number of RBP features: ', len(ls_all_rbps))
        print('list of RBP features:')
        print([i for i in ls_all_rbps])
        print(20 * '-')

    return df_train_set, df_test_set, ls_all_rbps


def prep_dataset_regression_rawclip_only(config, run):
    """

    :param config:
    :param run:
    :return:
    """
    # read config setting
    verbose = config.getboolean('general', 'verbose')
    rand_state = int(config.get('general', 'random_seed'))
    if verbose: print('in function {0}'.format('prep_dataset_regression_rawclip_only'))
    # prediction_label = config.get('input_data_selection', 'prediction_label')
    ls_prediction_labels = [s.strip() for s in config.get('input_data_selection', 'ls_prediction_labels').split(',')]
    diff_isoforms = config.getboolean('input_data_selection', 'differential_isoforms')
    diff_genes = config.getboolean('input_data_selection', 'differential_genes')
    exon_density_included = config.getboolean('input_data_selection', 'exon_density_included')
    ls_train_cell_lines = [s.strip() for s in config.get('input_data_selection', 'ls_train_cell_lines').split(',')]
    ls_test_cell_lines = [s.strip() for s in config.get('input_data_selection', 'ls_test_cell_lines').split(',')]

    # sanity check train and test cell lines
    ls_train_cell_lines, ls_test_cell_lines = sanity_check_cell_lines(ls_train_cell_lines, ls_test_cell_lines)

    # read raw clip data
    df_agg_raw_renamed = get_raw_clip(config)

    # add exon density feature if required
    if exon_density_included:
        df_exon_density = get_exon_density(config)
        # print('shape of df_agg_raw_renamed before merge with exon density ', df_agg_raw_renamed.shape)
        df_agg_raw_renamed = pd.merge(df_agg_raw_renamed, df_exon_density, how='inner', on='tx_id')
        # print('shape of df_agg_raw_renamed after merge with exon density ', df_agg_raw_renamed.shape)

    if 'gene_id' in df_agg_raw_renamed.columns:
        del df_agg_raw_renamed['gene_id']

    ls_remove_cols = ['tx_id', 'tx_id_x', 'tx_id_y', 'nr_bins_x', 'nr_bins', 'gene_id',
                      'gene_id_x', 'gene_id_y', 'cell_line', 'top_iso', 'nr_bins',
                      'gene_name', 'nr_bins_y']
    ls_remove_cols.extend(ls_prediction_labels)
    ls_all_rbps = [i for i in df_agg_raw_renamed.columns if i not in ls_remove_cols]

    # TODO: add check to make sure ls_test and ls_train cell_lines don't overlap
    # get training and test labels
    df_train_cell_lines_labels = gene_level_labels_for_transcripts(config, ls_prediction_labels, ls_train_cell_lines)
    df_test_cell_lines_labels = gene_level_labels_for_transcripts(config, ls_prediction_labels, ls_test_cell_lines)
    if diff_isoforms:
        df_train_labels, df_test_labels = find_differntial_isoforms(config, ls_prediction_labels,
                                                                    df_train_cell_lines_labels,
                                                                    df_test_cell_lines_labels)
    elif diff_genes:
        df_train_labels, df_test_labels = find_differntial_genes(df_train_cell_lines_labels,
                                                                 df_test_cell_lines_labels)
    else:
        print("Note! runing with non-differntial setting!")
        df_train_labels = df_train_cell_lines_labels.copy()
        df_test_labels = pd.DataFrame()

    df_train_set = pd.merge(df_agg_raw_renamed, df_train_labels, how='inner', on='tx_id')
    df_train_set = df_train_set.sample(frac=1, random_state=rand_state).reset_index(drop=True)
    if diff_genes or diff_isoforms:
        df_test_set = pd.merge(df_agg_raw_renamed, df_test_labels, how='inner', on='tx_id')
    else:
        df_test_set = pd.DataFrame()
    run['total_rbp_features'] = len(ls_all_rbps)
    if verbose:
        print('number of RBP features: ', len(ls_all_rbps))
        print('list of RBP features:')
        print([i for i in ls_all_rbps])
        print(20 * '-')

    return df_train_set, df_test_set, ls_all_rbps


def prep_dataset_regression_deepripe_rawclip(config, run):
    """

    :param config:
    :param run:
    :return:
    """

    # read config setting
    verbose = config.getboolean('general', 'verbose')
    rand_state = int(config.get('general', 'random_seed'))
    if verbose: print('in function {0}'.format('prep_dataset_regression_deepripe_rawclip'))
    # prediction_label = config.get('input_data_selection', 'prediction_label')
    ls_prediction_labels = [s.strip() for s in config.get('input_data_selection', 'ls_prediction_labels').split(',')]
    diff_isoforms = config.getboolean('input_data_selection', 'differential_isoforms')
    diff_genes = config.getboolean('input_data_selection', 'differential_genes')
    exon_density_included = config.getboolean('input_data_selection', 'exon_density_included')
    fill_missing_raw_clip = config.getboolean('input_data_selection', 'fill_missing_raw_clip')
    replace_rawclip_with_deepripe_rbps = config.getboolean('input_data_selection', 'replace_rawclip_with_deepripe_rbps')
    ls_train_cell_lines = [s.strip() for s in config.get('input_data_selection', 'ls_train_cell_lines').split(',')]
    ls_test_cell_lines = [s.strip() for s in config.get('input_data_selection', 'ls_test_cell_lines').split(',')]

    # sanity check train and test cell lines
    ls_train_cell_lines, ls_test_cell_lines = sanity_check_cell_lines(ls_train_cell_lines, ls_test_cell_lines)

    # read deepripe run results
    df_res_deepripe_renamed = get_deepripe_output(config)
    run['count_deepripe_output_rbps'] = len(df_res_deepripe_renamed.columns) - 1

    # read raw clip data
    df_agg_raw_renamed = get_raw_clip(config)

    # TODO: consider removing this option altogether!
    # if deepripe is running with selected RBPs only and
    # we want to replace good deepripe rbps over rawclip ones
    if replace_rawclip_with_deepripe_rbps:
        # TODO: replace hard coded part - this is only available for hek293
        ls_selected_rbps = [s.strip() for s in
                            config.get('deepripe', 'selected_rbps_hek293').split(',')]
        for rbp in ls_selected_rbps:
            if rbp + "_raw" in df_agg_raw_renamed.columns:
                del df_agg_raw_renamed[rbp + "_raw"]

    # merge deepripe and raw clip
    if fill_missing_raw_clip:
        df_deepripe_raw = pd.merge(df_res_deepripe_renamed, df_agg_raw_renamed, how='left', on='tx_id')
        df_deepripe_raw.fillna(0, inplace=True)
    else:

        df_deepripe_raw = pd.merge(df_res_deepripe_renamed, df_agg_raw_renamed, how='inner', on='tx_id')
    df_deepripe_raw.drop_duplicates(inplace=True)

    # add exon density feature if required
    if exon_density_included:
        df_exon_density = get_exon_density(config)
        # print('shape of df_deepripe_raw before merge with exon density ', df_deepripe_raw.shape)
        df_deepripe_raw = pd.merge(df_deepripe_raw, df_exon_density, how='inner', on='tx_id')
        # print('shape of df_deepripe_raw after merge with exon density ', df_deepripe_raw.shape)

    if 'gene_id' in df_deepripe_raw.columns:
        del df_deepripe_raw['gene_id']

    ls_remove_cols = ['tx_id', 'tx_id_x', 'tx_id_y', 'nr_bins_x', 'nr_bins', 'gene_id',
                      'gene_id_x', 'gene_id_y', 'cell_line', 'top_iso', 'nr_bins',
                      'gene_name', 'nr_bins_y']
    ls_remove_cols.extend(ls_prediction_labels)

    ls_all_rbps = [i for i in df_deepripe_raw.columns if i not in ls_remove_cols]

    # TODO: add check to make sure ls_test and ls_train cell_lines don't overlap
    # get training and test labels
    df_train_cell_lines_labels = gene_level_labels_for_transcripts(config, ls_prediction_labels, ls_train_cell_lines)
    df_test_cell_lines_labels = gene_level_labels_for_transcripts(config, ls_prediction_labels, ls_test_cell_lines)
    if diff_isoforms:
        df_train_labels, df_test_labels = find_differntial_isoforms(config, ls_prediction_labels,
                                                                    df_train_cell_lines_labels,
                                                                    df_test_cell_lines_labels)
    elif diff_genes:
        df_train_labels, df_test_labels = find_differntial_genes(df_train_cell_lines_labels,
                                                                 df_test_cell_lines_labels)
    else:
        print("Note! runing with non-differntial setting!")
        df_train_labels = df_train_cell_lines_labels.copy()
        df_test_labels = pd.DataFrame()

    df_train_set = pd.merge(df_deepripe_raw, df_train_labels, how='inner', on='tx_id')
    df_train_set = df_train_set.sample(frac=1, random_state=rand_state).reset_index(drop=True)
    if diff_genes or diff_isoforms:
        df_test_set = pd.merge(df_deepripe_raw, df_test_labels, how='inner', on='tx_id')
    else:
        df_test_set = pd.DataFrame()
    run['total_rbp_features'] = len(ls_all_rbps)
    if verbose:
        print('number of RBP features: ', len(ls_all_rbps))
        print('list of RBP features:')
        print([i for i in ls_all_rbps])
        print(20 * '-')

    return df_train_set, df_test_set, ls_all_rbps


def get_deepripe_output(config):
    verbose = config.getboolean('general', 'verbose')
    if verbose: print('in function {0}'.format('get_deepripe_data'))
    ls_deepripe_pretrained_cell_lines = [s.strip() for s in
                                         config.get('input_data_selection', 'deepripe_pretrained_cell_lines').split(
                                             ',')]
    deepripe_inference = config.get('input_data_selection', 'deepripe_inference')

    deepripe_selected_rbps_only = config.getboolean('input_data_selection', 'deepripe_selected_rbps_only')
    # get deepripe output - rbp binding info
    if deepripe_inference == 'run':
        if verbose: print('running deepripe inference')
        df_res_deepripe = run_deepripe(config)
    elif deepripe_inference == 'load':
        # TODO: improve this part
        if verbose:
            print('loading precalculated deepripe predictions - trained on cell lines:')
            print(ls_deepripe_pretrained_cell_lines)
        df_res_deepripe = load_deepripe(config, ls_deepripe_pretrained_cell_lines, deepripe_selected_rbps_only)
    else:
        print('config value set for deepripe_inference is not valid. please choose run or load.')
    # rename columns to be able to differentiate the source in feature importance
    df_res_deepripe_renamed = df_res_deepripe.copy()

    ls_deepripe_rbps = [col for col in df_res_deepripe_renamed.columns.values if col != 'tx_id']
    for rbp in ls_deepripe_rbps:
        df_res_deepripe_renamed.rename(columns={rbp: rbp + "_deepripe"}, inplace=True)

    df_res_deepripe_renamed = df_res_deepripe_renamed.drop_duplicates().reset_index(drop=True)
    return df_res_deepripe_renamed


def sanity_check_cell_lines(ls_train_cell_lines, ls_test_cell_lines):
    if len(ls_train_cell_lines) > 1:
        print("you have selected multiple cell lines in your training set")
        print("program will continue to run but you might have undesirable results if genes/isoforms overlaps")

    ls_overlapping_test_set = [i for i in ls_train_cell_lines if i in ls_test_cell_lines]

    if len(ls_overlapping_test_set) > 0:
        print("you have overlapping cell lines in your train and test set which will be removed from test set: ")
        print([i for i in ls_overlapping_test_set])
        ls_test_cell_lines = [i for i in ls_test_cell_lines if i not in ls_train_cell_lines]
        print('new test set is:')
        print([i for i in ls_test_cell_lines])

    if len(ls_train_cell_lines) < 1 or len(ls_test_cell_lines) < 1:
        print(
            "you have not specified either training or test cell line, rolling back to defaults with train on hek293 and test on k562.")
        ls_train_cell_lines = ['hek293']
        ls_test_cell_lines = ['k562']

    return ls_train_cell_lines, ls_test_cell_lines


def run_ml_model(df_dataset_train, df_dataset_test, ls_all_rbps, config, run):
    """

    :param df_dataset_train:
    :param df_dataset_test:
    :param ls_all_rbps:
    :param config:
    :param run:
    :return:
    """

    # read relevant config and set run params
    verbose = config.getboolean('general', 'verbose')
    model_category = config.get('model_selection', 'model_category')
    model_type = config.get('model_selection', 'model_type')
    random_seed = int(config.get('general', 'random_seed'))
    ls_prediction_labels = [s.strip() for s in config.get('input_data_selection', 'ls_prediction_labels').split(',')]
    diff_isoforms = config.getboolean('input_data_selection', 'differential_isoforms')
    diff_genes = config.getboolean('input_data_selection', 'differential_genes')
    non_diff_with_kfold = config.getboolean('input_data_selection', 'non_diff_with_kfold')
    if verbose: print('in function {0}'.format('run_ml_model'))

    # close all previous plots
    plt.close()

    # if classification - first change labels to classes
    if model_category == 'classifier':
        for prediction_label in ls_prediction_labels:
            sr_describe = df_dataset_train[prediction_label].describe()
            df_dataset_train[prediction_label] = df_dataset_train[prediction_label].map(
                lambda x: regression_label_to_class_label(x, sr_describe)).astype(int)
            plt.clf()
            fig_class_imb_train = sns.distplot(df_dataset_train[prediction_label]).figure
            run["train_class_imbalance_" + prediction_label].upload(neptune.types.File.as_image(fig_class_imb_train))

            if df_dataset_test.shape[0] > 0:
                sr_describe = df_dataset_test[prediction_label].describe()
                df_dataset_test[prediction_label] = df_dataset_test[prediction_label].map(
                    lambda x: regression_label_to_class_label(x, sr_describe)).astype(int)
                plt.clf()
                fig_class_imb_test = sns.distplot(df_dataset_test[prediction_label]).figure
                run["test_class_imbalance_" + prediction_label].upload(neptune.types.File.as_image(fig_class_imb_test))

    # three running modes:
    # for differential cases, train and testset will be separated based on cell line
    # it can only run with run_with_multiple_rand_seeds function
    if diff_isoforms or diff_genes:
        if verbose:
            print('running differential case with multiple random seeds')
            # print('shape of df_train_set', df_dataset_train.shape)
            # print('shape of df_test_set', df_dataset_test.shape)
        run_with_multiple_rand_seeds(df_dataset_train,
                                     df_dataset_test,
                                     ls_all_rbps,
                                     ls_prediction_labels,
                                     run, config)

    # for non-differential cases, train and testset will be from the same (set of) cell line(s)
    # it can only run with run_with_repeated_kfold function where split is random
    # or it can run with run_with_presplitted_data_with_multiple_rand_seeds where
    # data split is kept the same (config random_seed) and training is repeated for a list of different random seeds

    else:
        if non_diff_with_kfold:
            if verbose: print('running non-differential case with repeated k-fold')
            run_with_repeated_kfold(df_dataset_train,
                                    ls_all_rbps,
                                    ls_prediction_labels,
                                    run,
                                    config)

        else:
            if verbose:
                print('running non-differential case with fixed test set and multiple random seeds')
            # TODO: replace with chromosome based split
            test_frac = float(config.get(model_type, 'test_frac'))
            df_dataset_train, df_dataset_test = train_test_split(df_dataset_train,
                                                                 test_size=test_frac,
                                                                 random_state=random_seed)

            run_with_presplitted_data_with_multiple_rand_seeds(df_dataset_train,
                                                               df_dataset_test,
                                                               ls_all_rbps,
                                                               ls_prediction_labels,
                                                               run, config)

    return
