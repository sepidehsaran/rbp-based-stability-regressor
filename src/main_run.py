import neptune
import configparser
from datetime import date
from run_functions import *
from util_functions import *


def main():
    # read config
    config = configparser.ConfigParser()
    config.read(['../config/config.ini'])

    verbose = config.getboolean('general', 'verbose')
    if verbose:
        print("Starting the script at {0}".format(datetime.now()))
        print(30 * "-")
    str_date_ver = str(date.today()) + "_" + config.get('general', 'version')
    run_mode = config.get('general', 'run_mode')

    if run_mode == 'train_eval':
        # start neptune session
        run = neptune.init_run(
            project=config.get('general', 'neptune_project'),
            api_token=config.get('general', 'neptune_token')
        )
        run['run_date'] = str_date_ver

        # read run setting
        rbp_input_for_regressor = config.get('input_data_selection', 'rbp_input_for_regressor')
        run['rbp_input_for_regressor'] = rbp_input_for_regressor

        assign_run_variables(rbp_input_for_regressor, run, config)

        if rbp_input_for_regressor == 'deepripe_only':
            regression_deepripe_only(config, run)
        elif rbp_input_for_regressor == 'deepripe_rawclip':
            regression_deepripe_rawclip(config, run)
        elif rbp_input_for_regressor == 'rawclip_only':
            regression_rawclip_only(config, run)
        elif rbp_input_for_regressor == 'baseline_deepripe_only':
            baseline_n_bins(config, run)
        else:
            print('value selected for rbp_input_for_regressor is not supported. Terminating run.')

        run.stop()

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
                            print(40 * '=')
                            print(40 * '=')
                            print(
                                "Starting {0} with random seed {1} for model {2}, for label {3}, with test cell line {4} at {5}".format(
                                    config.get('input_data_selection', 'rbp_input_for_regressor'),
                                    config.get('general', 'random_seed'),
                                    config.get('model_selection', 'model_type'),
                                    config.get('input_data_selection', 'ls_prediction_labels'),
                                    config.get('input_data_selection', 'ls_test_cell_lines'),
                                    datetime.now()))
                        print(30 * "=")
                        # start neptune session
                        run = neptune.init_run(
                            project=config.get('general', 'neptune_project'),
                            api_token=config.get('general', 'neptune_token'))

                        verbose = config.getboolean('general', 'verbose')
                        if verbose:
                            print("Starting the script at {0}".format(datetime.now()))
                            print(30 * "-")
                        str_date_ver = str(date.today()) + "_" + config.get('general', 'version')
                        run['run_date'] = str_date_ver
                        # read run setting
                        rbp_input_for_regressor = config.get('input_data_selection', 'rbp_input_for_regressor')
                        run['rbp_input_for_regressor'] = rbp_input_for_regressor

                        assign_run_variables(rbp_input_for_regressor, run, config)

                        if rbp_input_for_regressor == 'deepripe_only':
                            regression_deepripe_only(config, run)
                        elif rbp_input_for_regressor == 'deepripe_rawclip':
                            regression_deepripe_rawclip(config, run)
                        elif rbp_input_for_regressor == 'rawclip_only':
                            regression_rawclip_only(config, run)
                        elif rbp_input_for_regressor == 'baseline_deepripe_only':
                            baseline_n_bins(config, run)
                        else:
                            print('value selected for rbp_input_for_regressor is not supported. Terminating script.')

                        run.stop()


if __name__ == "__main__":
    main()
