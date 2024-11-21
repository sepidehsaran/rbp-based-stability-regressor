import neptune
from run_functions import *
from util_functions import *
from evaluation_figures import *

def main():
    # read basic config
    config = configparser.ConfigParser()
    config.read(['../config/config.ini'])

    verbose = config.getboolean('general', 'verbose')
    if verbose:
        print("Starting the script at {0}".format(datetime.now()))
        print(30 * "-")
    str_date_ver = str(date.today()) + "_" + config.get('general', 'version')
    run_mode = config.get('general', 'run_mode')

    if run_mode == 'train_eval':

        # read run setting
        rbp_input_for_regressor = config.get('input_data_selection', 'rbp_input_for_regressor')

        if rbp_input_for_regressor == 'deepripe_only':
            regression_deepripe_only(config)
        elif rbp_input_for_regressor == 'deepripe_rawclip':
            regression_deepripe_rawclip(config)
        elif rbp_input_for_regressor == 'rawclip_only':
            regression_rawclip_only(config)
        elif rbp_input_for_regressor == 'baseline_deepripe_only':
            baseline_n_bins(config)
        else:
            print('value selected for rbp_input_for_regressor is not supported. Terminating run.')

    elif run_mode == 'gen_figure':
        retrieve_results_from_neptune(config)
    else:
        print('value selected for run_mode is not supported.Terminating run.')

    return


def several_runs(config, ls_labels, ls_models, ls_input_settings, ls_transcript_regions, ls_random_seeds):
    verbose = config.getboolean('general', 'verbose')
    # loop over datasets, random seeds and model types
    for label in ls_labels:
        for model in ls_models:
            for setting in ls_input_settings:
                for region in ls_transcript_regions:
                    for rand in ls_random_seeds:
                        # start for each setting
                        config['input_data_selection']['ls_test_cell_lines'] = 'hek293'
                        config['input_data_selection']['ls_prediction_labels'] = label
                        config['model_selection']['model_type'] = model
                        config['input_data_selection']['rbp_input_for_regressor'] = setting
                        config['input_data_selection']['transcript_regions'] = region
                        config['general']['random_seed'] = str(rand)
                        if verbose:
                            print("Starting {0} with random seed {1} for model {2}, for label {3}, with test cell line {4} at {5}".format(
                                  config.get('input_data_selection', 'rbp_input_for_regressor'),
                                  config.get('general', 'random_seed'),
                                  config.get('model_selection', 'model_type'),
                                  config.get('input_data_selection', 'ls_prediction_labels'),
                                  config.get('input_data_selection', 'ls_test_cell_lines'),
                                  datetime.now()))


                        verbose = config.getboolean('general', 'verbose')
                        if verbose:
                            print("Starting the script at {0}".format(datetime.now()))
                        str_date_ver = str(date.today()) + "_" + config.get('general', 'version')
                        # read run setting
                        rbp_input_for_regressor = config.get('input_data_selection', 'rbp_input_for_regressor')

                        assign_run_variables(rbp_input_for_regressor, config)

                        if rbp_input_for_regressor == 'deepripe_only':
                            regression_deepripe_only(config)
                        elif rbp_input_for_regressor == 'deepripe_rawclip':
                            regression_deepripe_rawclip(config)
                        elif rbp_input_for_regressor == 'rawclip_only':
                            regression_rawclip_only(config)
                        elif rbp_input_for_regressor == 'baseline_deepripe_only':
                            baseline_n_bins(config)
                        else:
                            print('value selected for rbp_input_for_regressor is not supported. Terminating script.')


if __name__ == "__main__":
    main()
