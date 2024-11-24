from os import listdir
from os.path import isfile, join
import random
from datetime import datetime
from tensorflow.keras.models import load_model
from util_functions import *


def seq_to_1hot(seq, randomsel=True):
    """
    function copied from DeepReg - code by Mahsa Ghanbari
    converts the sequence to one-hot encoding

    :param seq:
    :param randomsel:
    :return:
    """
    seq_len = len(seq)
    seq = seq.upper()
    seq_code = np.zeros((4, seq_len), dtype='int')
    for i in range(seq_len):
        nt = seq[i]
        if nt == 'A':
            seq_code[0, i] = 1
        elif nt == 'C':
            seq_code[1, i] = 1
        elif nt == 'G':
            seq_code[2, i] = 1
        elif nt == 'T':
            seq_code[3, i] = 1
        elif randomsel:
            rn = random.randint(0, 3)
            seq_code[rn, i] = 1
    return seq_code


def read_genecode_transcripts(config):
    """
    code by S. Lebedeva
    # Get transcript sequences directly from Gencode. They only have coding transcripts and lncRNAs.
    # Use this if we don't need any padding.
    :param config: config object
    :return:
    """
    verbose = config.getboolean('general', 'verbose')
    if verbose: print('in function {0}'.format('read_genecode_transcripts'))
    transcript_regions = config.get('input_data_selection', 'transcript_regions')

    # Get protein coding fasta file
    pc_path = config.get('general', 'data_path') + config.get('data_files', 'pro_coding_fasta_file')
    df_pc = pd.read_csv(pc_path, header=None, sep=">")
    # Get non-coding fasta file
    nc_path = config.get('general', 'data_path') + config.get('data_files', 'non_coding_fasta_file')
    df_nc = pd.read_csv(nc_path, header=None, sep=">")
    # Concatenate both into one dataframe
    df_seq = pd.concat(
        [pd.concat([df_pc[[1]].dropna().reset_index(drop=True),
                    df_pc[[0]].dropna().reset_index(drop=True)], axis=1),
         pd.concat([df_nc[[1]].dropna().reset_index(drop=True),
                    df_nc[[0]].dropna().reset_index(drop=True)], axis=1)], axis=0).reset_index(drop=True)
    df_seq.rename(columns={0: "sequence", 1: "long_id"}, inplace=True)

    # split long id into separate ones (not sure why is there a last empty value)
    df_seq[['tx_id', 'gene_id', 'gene_id2', 'tx_id2', 'tx_name', 'gene_name', 'length', 'UTR5_coord', 'CDS_coord',
            'UTR3_coord', 'None']] = df_seq["long_id"].str.split("|", expand=True)
    # note that we are removing versioning of transcripts and genes whenever we read data files
    df_seq['tx_id'] = df_seq['tx_id'].map(lambda x: str(x).split('.')[0])
    df_seq['gene_id='] = df_seq['gene_id'].map(lambda x: str(x).split('.')[0])
    df_seq['length'] = df_seq['length'].astype(int)

    if transcript_regions == 'all':
        print('all transcipt regions are considered')
    elif transcript_regions == '3utr':
        print('only 3UTR is considered')
        # Get 3'UTR only coding fasta file - filter for both None values and empty string
        df_coding = df_seq.dropna(subset=['UTR3_coord']).pipe(lambda x: x[x.UTR3_coord != ""])
        # because we already have 3UTR coordinates we can just replace the sequence
        # take care of 0-1 based coordinates! starts should be changed from 1 to 0
        # to accomodate python but stops are fine because python substring is half-open
        starts = [(int(i[0]) - 1) for i in df_coding.UTR3_coord.str.replace("UTR3:", "").str.split("-")]
        stops = [int(i[1]) for i in df_coding.UTR3_coord.str.replace("UTR3:", "").str.split("-")]
        df_coding['sequence'] = [seq[start:stop] for seq, start, stop in zip(df_coding['sequence'], starts, stops)]
        df_coding['length'] = df_coding['sequence'].map(len)
        # filter on seq length
        df_seq = df_coding[df_coding['length'] > 100]

    if verbose:
        print("number of unique transcripts is", len(df_seq.tx_id.unique()))
        print("number of unique genes is", len(df_seq.gene_id.unique()))
    return df_seq


def read_bs_centers(config):
    """
    Get the center of binding sites for PARCLIP data - mapped centers of RBP binding sites by a separate R script
    :param config:
    :return:
    """
    verbose = config.getboolean('general', 'verbose')
    if verbose: print('in function {0}'.format('read_bs_centers'))
    mid_bs_from_r = pd.read_csv(config.get('general', 'data_path') + config.get('data_files', 'parclip_bs_centers'),
                                sep='|')
    if verbose:
        print("number of unique samples in mapped centers of RBP binding sites", len(mid_bs_from_r.RBP.unique()))
        print("number of unique transcripts in mapped centers of RBP binding sites",
              len(mid_bs_from_r.seqnames.unique()))

    return mid_bs_from_r


def read_top_isoforms_per_cellline(config):
    verbose = config.getboolean('general', 'verbose')
    if verbose: print('in function {0}'.format('read_top_isoforms_per_cellline'))
    file_path = config.get('general', 'data_path') + config.get('data_files', 'top_isoforms_per_cellline')
    df_top_iso = pd.read_csv(file_path)
    df_top_iso['gene_id'] = df_top_iso['gene_id'].map(lambda x: str(x).split('.')[0])
    df_top_iso['top_iso'] = df_top_iso['top_iso'].map(lambda x: str(x).split('.')[0])
    df_top_iso['cell_line'] = df_top_iso['cell_line'].map(lambda x: str(x).lower())
    return df_top_iso


def read_parclip_clusterbeds(config):
    """
    Note: this old parclip dataset is missing gene_id which is required for later joins
    this function is used only with combination of the following raw clip aggregation modes that are now depricated:
    sum_counts, max_counts, sum_score, max_scores
    :param config:
    :return:
    """
    verbose = config.getboolean('general', 'verbose')
    if verbose: print('in function {0}'.format('read_parclip_clusterbeds'))

    all_rbp_bed_path = config.get('data_files', 'data_path') + config.get('data_files', 'all_rbps_bed_folder_name')
    ls_bed_files = [f for f in listdir(all_rbp_bed_path) if isfile(join(all_rbp_bed_path, f))]
    ls_rbp_dfs = []
    for file_name in ls_bed_files:
        df = pd.read_csv(all_rbp_bed_path + file_name, sep="\s+|,|\t", engine='python', header=None)
        df.columns = ['chr', 'start', 'end', 'ClusterID', 'gene_type', 'gene_name', 'ClusterID2', 'ClusterSequence',
                      'ReadCount', 'ModeLocation', 'ModeScore', 'ConversionLocationCount', 'ConversionEventCount',
                      'NonConversionEventCount', 'FilterType', 'strand']
        df["RBP"] = file_name.split('.')[0]
        ls_rbp_dfs.append(df.copy())
    df_clip_raw = pd.concat(ls_rbp_dfs)
    return df_clip_raw


def read_raw_parclip_bins(config):
    # read raw binding site info for parclip
    transcript_regions = config.get('input_data_selection', 'transcript_regions')
    if transcript_regions == 'all':
        parclip_bins_path = config.get('general', 'data_path') + config.get('data_files', 'parclip_selected_tx_bins')
    elif transcript_regions == '3utr':
        parclip_bins_path = config.get('general', 'data_path') + config.get('data_files', 'parclip_selected_3utr_bins')
    df_rbp_bins = pd.read_csv(parclip_bins_path, sep='\t', header=None,
                              names=['tx_id', 'start_middle', 'end_middle', 'rbp_cellline', 'gene_id', 'strand'])
    df_rbp_bins['tx_id'] = df_rbp_bins['tx_id'].map(lambda x: str(x).split('.')[0])
    df_rbp_bins['gene_id'] = df_rbp_bins['gene_id'].map(lambda x: str(x).split('.')[0])
    df_rbp_bins['rbp'] = df_rbp_bins['rbp_cellline'].map(lambda x: str(x).split('_')[0])
    # read seq_length if needed - right now remove
    # df_seq = read_genecode_transcripts(config)
    # df_seq_filtered = filter_transcripts(config, df_seq, ls_by_cell_lines=['hek293'])
    # df_seq_filtered = df_seq_filtered[['tx_id', 'length', 'cell_line']]
    bin_size = int(config.get('deepripe', 'hek293_bin_size'))
    df_rbp_bins['bin_nr_rbpbs'] = df_rbp_bins['start_middle'].map(lambda x: int(x) // bin_size)
    df_rbp_bins = df_rbp_bins[['tx_id', 'gene_id', 'rbp', 'bin_nr_rbpbs']].drop_duplicates().reset_index(drop=True)
    # count how many bins have a binding site for the given RBP (equal to sum of binary bins in config setting)
    df_agg = df_rbp_bins.groupby(['tx_id', 'gene_id', 'rbp']).aggregate("count")
    df_agg = df_agg.reset_index().pivot(index=['tx_id', 'gene_id'], columns='rbp', values="bin_nr_rbpbs").fillna(
        0).reset_index()
    for col in df_agg.columns:
        if col not in ['tx_id', 'gene_id']:
            df_agg[col] = df_agg[col].astype('int32')

    return df_agg


def read_raw_eclip_bins(config):
    transcript_regions = config.get('input_data_selection', 'transcript_regions')
    if transcript_regions == 'all':
        eclip_bins_path = config.get('general', 'data_path') + config.get('data_files', 'eclip_selected_tx_bins')
    elif transcript_regions == '3utr':
        eclip_bins_path = config.get('general', 'data_path') + config.get('data_files', 'eclip_selected_3utr_bins')
    df_rbp_bins = pd.read_csv(eclip_bins_path, sep='\t', header=None,
                              names=['tx_id', 'start_middle', 'end_middle', 'rbp_cellline', 'gene_id', 'strand'])
    df_rbp_bins['tx_id'] = df_rbp_bins['tx_id'].map(lambda x: str(x).split('.')[0])
    df_rbp_bins['gene_id'] = df_rbp_bins['gene_id'].map(lambda x: str(x).split('.')[0])
    df_rbp_bins['rbp'] = df_rbp_bins['rbp_cellline'].map(lambda x: str(x).split('_')[0])
    df_rbp_bins['cellline'] = df_rbp_bins['rbp_cellline'].map(lambda x: str(x).split('_')[1])
    # df_rbp_bins = df_rbp_bins[df_rbp_bins["cellline"] == cellline.upper()]

    bin_size = int(config.get('deepripe', 'every_eclip_bin_size'))
    df_rbp_bins['bin_nr_rbpbs'] = df_rbp_bins['start_middle'].map(lambda x: int(x) // bin_size)
    df_rbp_bins = df_rbp_bins[['tx_id', 'gene_id', 'rbp', 'bin_nr_rbpbs']].drop_duplicates().reset_index(drop=True)
    # count how many bins have a binding site for the given RBP (equal to sum of binary bins in config setting)
    df_agg = df_rbp_bins.groupby(['tx_id', 'gene_id', 'rbp']).aggregate("count")
    df_agg = df_agg.reset_index().pivot(index=['tx_id', 'gene_id'], columns='rbp', values="bin_nr_rbpbs").fillna(
        0).reset_index()
    for col in df_agg.columns:
        if col not in ['tx_id', 'gene_id']:
            df_agg[col] = df_agg[col].astype('int32')

    return df_agg


def read_gene_to_name(config):
    """

    :param config:
    :return:
    """
    verbose = config.getboolean('general', 'verbose')
    if verbose: print('in function {0}'.format('read_gene_to_name'))
    gene_to_name_path = config.get('general', 'data_path') + config.get('data_files', 'gene_to_name')
    df_t2g = pd.read_csv(gene_to_name_path)
    df_t2g["tx_id"] = df_t2g.tx_id.str.replace("\\.[0-9_]+?$", "", 1)
    df_t2g["gene_id"] = df_t2g["gene_id"].map(lambda x: str(x).split('.')[0])
    return df_t2g


def read_labels_for_ls_cell_lines(config, ls_labels, ls_cell_line):
    ls_dfs = []
    selected_cols = ['gene_id', 'cell_line']
    selected_cols.extend(ls_labels)
    for cell_line in ls_cell_line:
        str_label_file_path = config.get('general', 'data_path') + \
                              config.get('data_files', cell_line.lower() + '_gene_level_label_file')
        df = pd.read_csv(str_label_file_path, sep='\t')
        df['gene_id'] = df['Gene'].map(lambda x: str(x).split('.')[0])
        df['cell_line'] = cell_line.lower()
        # print(df.head())
        ls_dfs.append(df[selected_cols])
    agg_df = pd.concat(ls_dfs, axis=0)
    return agg_df


def gene_level_labels_for_transcripts(config, ls_labels, ls_cell_line):
    df_top_iso = read_top_isoforms_per_cellline(config)
    df_top_iso = df_top_iso[df_top_iso['cell_line'].isin(ls_cell_line)]
    df_label_info = read_labels_for_ls_cell_lines(config, ls_labels, ls_cell_line)
    # TODO: @Sveti: should we include cell_line in the merge condition to differntiate btw overlapping gene_ids?
    df_iso_label = pd.merge(df_top_iso[['gene_id', 'top_iso']],
                            df_label_info, how='inner', on='gene_id')
    df_iso_label['tx_id'] = df_iso_label['top_iso']
    df_iso_label.drop_duplicates(inplace=True)
    selected_cols = ['tx_id', 'gene_id']
    selected_cols.extend(ls_labels)
    return df_iso_label[selected_cols]


def find_differntial_isoforms(config, ls_labels, df_train_labels, df_test_labels):
    df_temp = df_train_labels[['gene_id', 'tx_id']].rename(columns={'tx_id': 'tx_id_train'})
    df_test_labels = pd.merge(df_test_labels, df_temp, how='inner', on='gene_id')
    df_test_labels['is_diff_iso'] = np.where(
        df_test_labels['tx_id'] != df_test_labels['tx_id_train'], 1, 0)
    selected_cols = ['tx_id', 'gene_id']
    selected_cols.extend(ls_labels)
    df_test_labels = df_test_labels[df_test_labels['is_diff_iso'] == 1][selected_cols]
    remove_training_iso = config.getboolean('input_data_selection', 'remove_training_iso')
    if remove_training_iso:
        ls_test_gene_id = [i for i in df_test_labels.gene_id.unique()]
        df_train_labels = df_train_labels[~df_train_labels['gene_id'].isin(ls_test_gene_id)]
    return df_train_labels, df_test_labels


def find_differntial_genes(df_train_labels, df_test_labels):
    ls_train_genes = [i for i in df_train_labels.gene_id.unique()]
    df_test_labels = df_test_labels[~df_test_labels['gene_id'].isin(ls_train_genes)]
    return df_train_labels, df_test_labels


def filter_transcripts(config, df_seq, ls_by_cell_lines):
    """

    :param config:
    :param df_seq:
    :param ls_by_cell_lines: filter for those from cell_lines in the list
    :return:
    """
    verbose = config.getboolean('general', 'verbose')
    if verbose: print('in function {0}'.format('filter_transcripts'))
    if len(ls_by_cell_lines) == 0:
        if verbose: print('no cell_line was selected for filtering, all data  ({0} transcripts) will be used'.format(
            len(df_seq.tx_id.unique())))
        return df_seq
    else:
        df_top_iso = read_top_isoforms_per_cellline(config)
        ls_by_cell_lines = [i for i in ls_by_cell_lines]
        df_top_iso = df_top_iso[df_top_iso['cell_line'].isin(ls_by_cell_lines)]
        df_seq_filtered = pd.merge(df_seq, df_top_iso, how='inner', left_on='tx_id', right_on='top_iso')
        if verbose:
            print('cell_line selected for filtering {0}'.format(str(ls_by_cell_lines)))
            print('# transcripts before filtering {0} and after filtering {1} '.format(
                len(df_seq.tx_id.unique()), len(df_seq_filtered.tx_id.unique())))

    return df_seq_filtered


def generate_bins_for_deepripe(config, df_seq, bin_size):
    """

    :param config:
    :param df_seq:
    :param bin_size:
    :return:
    """
    verbose = config.getboolean('general', 'verbose')
    if verbose: print('in function {0}'.format('generate_bins_for_deepripe'))
    # filter dataset for transcripts w.r.t. bin size
    df_ds = df_seq[df_seq['length'] >= bin_size].copy()
    df_ds = df_ds[['tx_id', 'sequence', 'length']].drop_duplicates(ignore_index=True)
    if verbose:
        print("number of unique transcripts with length higher than {0}, is {1}".format(
            bin_size, len(df_ds.tx_id.unique())))
        print("start creating bins at {0}".format(datetime.now()))
    ls_ds = []
    for idx, row in df_ds.iterrows():
        if verbose:
            if idx % 10000 == 0: print("currently in row {0} at {1}".format(idx, datetime.now()))
        tx_id = row['tx_id']
        len_tx = int(row['length'])
        seq_tx = str(row['sequence'])
        nr_bins = 0
        ls_bins = []
        end_tx_not_reached = True
        start_bin_pos = 0
        stride_size = 50
        while (end_tx_not_reached):
            if start_bin_pos + bin_size <= len_tx:
                seq_bin_feat = seq_to_1hot(seq_tx[start_bin_pos:start_bin_pos + bin_size])
                seq_bin_feat = np.transpose(seq_bin_feat, axes=(1, 0))
                ls_bins.append(seq_bin_feat)
                start_bin_pos = start_bin_pos + stride_size
                nr_bins += 1
            else:
                seq_bin_feat = seq_to_1hot(seq_tx[len_tx - bin_size:len_tx])
                seq_bin_feat = np.transpose(seq_bin_feat, axes=(1, 0))
                ls_bins.append(seq_bin_feat)
                nr_bins += 1
                end_tx_not_reached = False

        ls_ds.append([tx_id, nr_bins, ls_bins])

    if verbose: print("Done with generating bins dataset with bin size {0} at {1}".format(bin_size, datetime.now()))
    return ls_ds


def prep_dataset_run_deepripe(config, ls_pretrained_cell_line):
    """

    :param ls_pretrained_cell_line:
    :param config:
    :return: dict_bin_size_ds: dictionary of key = bin_sizes , values = lists containing dataset for running deepripe
    """
    verbose = config.getboolean('general', 'verbose')
    if verbose: print('in function {0}'.format('prep_dataset_run_deepripe'))

    df_seq = read_genecode_transcripts(config)
    df_seq_filtered = filter_transcripts(config, df_seq, ls_by_cell_lines=['hek293', 'k562'])
    # dict: keys are bin_sizes , values are lists containing dataset for running deepripe
    dict_bin_size_ds = dict()
    for cell_line in ls_pretrained_cell_line:
        bin_size = int(config.get('deepripe', cell_line + '_bin_size'))
        if bin_size not in dict_bin_size_ds.keys():
            ls_ds = generate_bins_for_deepripe(config, df_seq_filtered, bin_size)
            dict_bin_size_ds[bin_size] = ls_ds

    return dict_bin_size_ds


def run_deepripe(config, ls_pretrained_cell_line=['hek293', 'k562', 'hepg2']):
    """
    TODO: make sure runs with a single cell line as well as combination of all - tested for all cell-lines
    :param config:
    :param ls_pretrained_cell_line:
    :return:
    """

    verbose = config.getboolean('general', 'verbose')
    if verbose: print('in function {0}'.format('run_deepripe'))
    # selected_rbps_only = config.getboolean('input_data_selection', 'deepripe_selected_rbps_only')
    dict_bin_size_ds = prep_dataset_run_deepripe(config, ls_pretrained_cell_line)
    transcript_regions = config.get('input_data_selection', 'transcript_regions')
    dict_cell_line_res = dict()
    for cell_line in ls_pretrained_cell_line:
        if verbose: print('start running deepripe inference for model pretrained on {0} at {1}'.format(
            cell_line, datetime.now()))
        bin_size = int(config.get('deepripe', cell_line + '_bin_size'))
        model_type = config.get('deepripe', cell_line + '_model_type')
        ls_ds = dict_bin_size_ds[bin_size]
        ls_rbp_list_names = [s.strip() for s in config.get('deepripe', 'rbp_' + cell_line + '_list_names').split(',')]
        ls_model_list_names = [s.strip() for s in
                               config.get('deepripe', 'model_' + cell_line + '_list_names').split(',')]
        ls_rbp_groups = []
        for rbp_group in ls_rbp_list_names:
            ls_rbp_groups.extend([s.strip() for s in config.get('deepripe', rbp_group).split(',')])
        if verbose:
            print('list of RBPs for pretrianed deepripe {0} is '.format(cell_line))
            print(ls_rbp_groups)

        # set region if needed
        models_path = config.get('deepripe', cell_line + '_models_path')
        ls_preds = []
        dict_models = dict()
        for model_name in ls_model_list_names:
            dict_models[model_name] = load_model(models_path + model_name,
                                                 custom_objects={'precision': precision, 'recall': recall})

        ls_columns = ['tx_id', 'nr_bins']
        ls_columns.extend(ls_rbp_groups)
        ls_dataset = []
        for tx in ls_ds:
            arr_all_tx_bins = np.array(tx[2])
            if model_type == 'sequence_region':
                dataset_reg_coded = np.full((arr_all_tx_bins.shape[0], 250, 4), 0)
                model_input = [arr_all_tx_bins, dataset_reg_coded]
            elif model_type == 'sequence_only':
                model_input = [arr_all_tx_bins]
            else:
                print('config value set for model_type is not valid. please choose sequence_region or sequence_only.')
                return pd.DataFrame()
            ls_temp_output = []
            for model_name in dict_models.keys():
                pred = dict_models[model_name].predict(model_input)
                ls_temp_output.append(pred)
            tx_concated_res = np.concatenate(ls_temp_output, axis=1)
            tx_concated_res = np.amax(tx_concated_res, axis=0)
            ls_tx = [tx[0], tx[1]]
            ls_tx.extend(list(tx_concated_res))
            ls_dataset.append(ls_tx)

        df_cell_outputs = pd.DataFrame(ls_dataset, columns=ls_columns)

        del df_cell_outputs['nr_bins']
        if config.getboolean('general', 'save_intermediate_res'):
            df_cell_outputs.to_pickle(config.get('general', 'res_path') + cell_line + '_' + transcript_regions +
                                      '_deepripe_output.pickle')

        dict_cell_line_res[cell_line] = df_cell_outputs

    # merge all deepripe results together
    df_merged_res = pd.DataFrame()
    if config.get('input_data_selection', 'pretrained_deepripes_agg') == 'max':
        df_merged_res = dict_cell_line_res[ls_pretrained_cell_line[0]]
        if len(ls_pretrained_cell_line) > 1:
            for i in range(1, len(ls_pretrained_cell_line)):

                df_temp = pd.merge(df_merged_res, dict_cell_line_res[ls_pretrained_cell_line[i]],
                                   how='inner', left_on='tx_id', right_on='tx_id')
                # take max of overlapping RBPs
                ls_overlapping_rbps = [rbp for rbp in df_merged_res.columns
                                       if rbp in dict_cell_line_res[ls_pretrained_cell_line[i]].columns
                                       and rbp not in ['tx_id', 'nr_bins']]
                print("ls_overlapping_rbps", ls_overlapping_rbps)
                for rbp in ls_overlapping_rbps:
                    df_temp[rbp] = df_temp[[rbp + "_x", rbp + "_y"]].max(axis=1)
                    del df_temp[rbp + "_x"]
                    del df_temp[rbp + "_y"]
                df_merged_res = df_temp.copy()
    else:
        print('config value set for pretrained_deepripes_agg is not valid. please use max.')

    if config.getboolean('general', 'save_intermediate_res'):
        df_merged_res.to_pickle(
            config.get('general', 'res_path') + '_'.join(ls_pretrained_cell_line) + '_' + transcript_regions +
            "_merged_deepripe_output.pickle")
    return df_merged_res


def load_deepripe(config, ls_pretrained_cell_line=['hek293', 'k562', 'hepg2'], selected_rbps_only=False,
                  include_nr_bins=False):
    verbose = config.getboolean('general', 'verbose')
    transcript_regions = config.get('input_data_selection', 'transcript_regions')
    if verbose: print('in function {0}'.format('load_deepripe'))
    # selected_rbps_only = config.getboolean('input_data_selection', 'deepripe_selected_rbps_only')
    if len(ls_pretrained_cell_line) > 1:
        df_res_deepripe = pd.read_pickle(config.get('general', 'res_path') + '_'.join(
            ls_pretrained_cell_line) + "_merged_deepripe_output.pickle")
    else:
        cell_line = ls_pretrained_cell_line[0]
        df_res_deepripe = pd.read_pickle(config.get('general', 'res_path') + cell_line + '_' +
                                         transcript_regions + '_deepripe_output.pickle')
        # Note that the following option of running only with  selected RBPs" , currently only works when
        # using deepripe hek293 model only. Because we have selected RBP list only for this cell-line
        # so if in the future we get selected RBP lists for other cell lines, we need to select them
        # before merging deepripe results, currently run_deepripe returns a merged result based on max cross cell-lines
        if selected_rbps_only and ls_pretrained_cell_line[0] == 'hek293':
            ls_selected_rbps = [s.strip() for s in
                                config.get('deepripe', 'selected_rbps_' + cell_line).split(',')]
            if verbose:
                print('only selected RBPs for DeepRipe will be used for cell-line {0}, lis of RBPs: '.format(cell_line))
                print(ls_selected_rbps)

            ls_selected_columns = ['tx_id', 'nr_bins']
            ls_selected_columns.extend(ls_selected_rbps)
            df_res_deepripe = df_res_deepripe[ls_selected_columns]
        elif selected_rbps_only:
            if verbose: print('selected RBP mode is only available for hek293 cell-line')
        else:
            if verbose: print('all RBPs are used in DeepRiPe')
    if not include_nr_bins:
        if 'nr_bins' in df_res_deepripe.columns:
            del df_res_deepripe['nr_bins']

    return df_res_deepripe


def get_raw_clip(config, log2_counts=True):
    # new implementation - we combine all raw cell lines eclip and parclip
    # idea is that if the RBP is expressed for a transcript in a cellline, we assume it's expressed in another
    agg_mode = config.get('input_data_selection', 'raw_clip_agg')
    df_raw_1 = prep_dataset_raw_parclip(config, agg_mode, log2_counts)
    df_raw_2 = prep_dataset_raw_eclip(config, agg_mode)
    ls_common_rbps = [i for i in df_raw_1.columns if i in df_raw_2.columns and i not in ['tx_id', 'gene_id']]
    df_raw = pd.merge(df_raw_1, df_raw_2, how='inner', on=['tx_id', 'gene_id'])
    for rbp in ls_common_rbps:
        df_raw[rbp] = df_raw[[rbp + '_x', rbp + '_y']].max(axis=1)
        del df_raw[rbp + '_x']
        del df_raw[rbp + '_y']

    # rename columns to be able to differentiate the source in feature importance
    ls_raw_rbps = [col for col in df_raw.columns.values
                   if col not in ['tx_id', 'gene_id', 'gene_id_x', 'gene_id_y', 'gene_name']]
    print('len ls_rawclip_rbps', len(ls_raw_rbps))
    print(ls_raw_rbps)
    for rbp in ls_raw_rbps:
        df_raw.rename(columns={rbp: rbp + "_raw"}, inplace=True)

    df_raw = df_raw.drop_duplicates().reset_index(drop=True)
    return df_raw


def prep_dataset_raw_parclip(config, agg_mode, log2_counts=True):
    """

    please use sum_bins_binary, or log_sum_bins_binary, bins_binary
    as old parclip data is missing gene_id, therefore the following agg_modes are effected and will run to error:
    max_scores / sum_scores / max_counts / sum_counts
    the reason is the new merge strategie with eCLIP and eCLIP data is not supporting these agg_modes

    :param config:
    :param agg_mode: sum or max
    :param log2_counts: used for sum_counts and max_counts

    :return:
    """
    verbose = config.getboolean('general', 'verbose')
    if verbose: print('in function {0}'.format('prep_dataset_raw_parclip'))

    if agg_mode == 'max_scores':
        # based on maximum of Mode score - not read counts - ex. max TC %
        df_clip_raw = read_parclip_clusterbeds(config)
        df_agg_raw = df_clip_raw[["gene_name", "ModeScore", "RBP"]].groupby(["gene_name", "RBP"]).aggregate("max")
        df_agg_raw = df_agg_raw.reset_index().pivot(index="gene_name",
                                                    columns="RBP",
                                                    values="ModeScore").fillna(0)
        df_agg_raw.reset_index(inplace=True)

    elif agg_mode == 'sum_scores':
        # un-normalized sum of Mode scores
        df_clip_raw = read_parclip_clusterbeds(config)
        df_agg_raw = df_clip_raw[["gene_name", "ModeScore", "RBP"]].groupby(["gene_name", "RBP"]).aggregate("sum")
        df_agg_raw = df_agg_raw.reset_index().pivot(index="gene_name",
                                                    columns="RBP",
                                                    values="ModeScore").fillna(0)
        df_agg_raw.reset_index(inplace=True)

    elif agg_mode == 'max_counts':
        df_clip_raw = read_parclip_clusterbeds(config)
        df_agg_raw = df_clip_raw[["gene_name", "ReadCount", "RBP"]].groupby(["gene_name", "RBP"]).aggregate("max")
        if log2_counts:
            df_agg_raw["log_cnt"] = np.log2(df_agg_raw["ReadCount"])
            df_agg_raw = df_agg_raw.drop("ReadCount", axis=1).reset_index().pivot(index="gene_name",
                                                                                  columns="RBP",
                                                                                  values="log_cnt").fillna(0)
        else:
            df_agg_raw = df_agg_raw.reset_index().pivot(index="gene_name",
                                                        columns="RBP",
                                                        values="ReadCount").fillna(0)
        df_agg_raw.reset_index(inplace=True)

    elif agg_mode == 'sum_counts':
        df_clip_raw = read_parclip_clusterbeds(config)
        df_agg_raw = df_clip_raw[["gene_name", "ReadCount", "RBP"]].groupby(["gene_name", "RBP"]).aggregate("sum")
        if log2_counts:
            df_agg_raw["log_cnt"] = np.log2(df_agg_raw["ReadCount"])
            df_agg_raw = df_agg_raw.drop("ReadCount", axis=1).reset_index().pivot(index="gene_name",
                                                                                  columns="RBP",
                                                                                  values="log_cnt").fillna(0)
        else:
            df_agg_raw = df_agg_raw.reset_index().pivot(index="gene_name",
                                                        columns="RBP",
                                                        values="ReadCount").fillna(0)

        df_agg_raw.reset_index(inplace=True)

    elif agg_mode == 'sum_bins_binary':
        df_agg_raw = read_raw_parclip_bins(config)

    elif agg_mode == 'log_sum_bins_binary':
        df_agg_raw = read_raw_parclip_bins(config)
        for col in df_agg_raw.columns:
            if col != 'tx_id':
                df_agg_raw[col] = np.where(df_agg_raw[col] == 0, 0, np.log2(df_agg_raw[col]))

    elif agg_mode == 'bins_binary':
        df_agg_raw = read_raw_parclip_bins(config)
        for col in df_agg_raw.columns:
            if col != 'tx_id':
                df_agg_raw[col] = np.where(df_agg_raw[col] == 0, 0, 1)

    else:
        print('config value set for agg_mode is not valid.')

    return df_agg_raw


def prep_dataset_raw_eclip(config, agg_mode):
    """

    :param config:
    :param agg_mode: max_scores, sum_scores, max_counts, sum_counts, sum_bins_binary, log_sum_bins_binary or bins_binary
    :return:
    """
    verbose = config.getboolean('general', 'verbose')

    if verbose: print('in function {0}'.format('prep_dataset_raw_eclip'))
    if agg_mode in ['max_scores', 'sum_scores', 'max_counts', 'sum_counts']:
        print("not supported agg_mode for eclip for now ")

    elif agg_mode == 'sum_bins_binary':
        df_agg_raw = read_raw_eclip_bins(config)

    elif agg_mode == 'log_sum_bins_binary':
        df_agg_raw = read_raw_eclip_bins(config)
        for col in df_agg_raw.columns:
            if col != 'tx_id':
                df_agg_raw[col] = np.where(df_agg_raw[col] == 0, 0, np.log2(df_agg_raw[col]))

    elif agg_mode == 'bins_binary':
        df_agg_raw = read_raw_eclip_bins(config)
        for col in df_agg_raw.columns:
            if col != 'tx_id':
                df_agg_raw[col] = np.where(df_agg_raw[col] == 0, 0, 1)
    else:
        print('config value set for agg_mode is not valid.')
    return df_agg_raw


def get_exon_density(config):
    """

    :param config:
    :return: df_exon_density
    """
    verbose = config.getboolean('general', 'verbose')
    log2_exon_density = config.getboolean('input_data_selection', 'log2_exon_density')

    if verbose: print('in function {0}'.format('get_exon_density'))

    file_path = config.get('general', 'data_path') + config.get('data_files', 'exon_density_file')
    df_exon_density = pd.read_csv(file_path, sep=',')
    df_exon_density['tx_id'] = df_exon_density['rn'].map(lambda x: str(x).split('.')[0])
    df_exon_density.rename(columns={'ex_density': 'exon_density'}, inplace=True)
    if log2_exon_density:
        df_exon_density['exon_density'] = df_exon_density['exon_density'].map(lambda x: np.log2(x))
    return df_exon_density[['tx_id', 'exon_density']].drop_duplicates().reset_index(drop=True)


def regression_label_to_class_label(x, sr_describe, n_classes=3):
    if n_classes == 3:
        if x > sr_describe['75%']:
            # high deg rate - above 75%
            return 0
        elif x > sr_describe['25%']:
            # mid range deg rate - between 25% and 75%
            return 1
        else:
            # low deg rate - below 25%
            return 2
    elif n_classes == 4:
        if x > sr_describe['75%']:
            # high deg rate - above 75%
            return 0
        elif x > sr_describe['50%']:
            # between 50% and 75%
            return 1
        elif x > sr_describe['25%']:
            # between 25% and 50%
            return 2
        else:
            # low deg rate - below 25%
            return 3
    else:
        raise ValueError('Invalid value set for n_classes. please choose between 3 or 4.')

    return 0
