# rbp-based-stability-regressor
Code for the "cell-type specific prediction of RNA stability from RNA-protein interactions" manuscript. Preprint available at [bioRxiv](https://doi.org/10.1101/2024.11.19.624283).



# To run the code
Before running the code, please go through the parameters in `config/config.ini` and set up your run. These parameters are the knobs that enable you to run different experiments by choosing from a selection of pre-processing decisions and ML models. 

We use [Neptune.ai](neptune.ai) to compare our experiments. To run the current code, you need to create an account at [neptune.ai](neptune.ai) (free) and create a project with an access token. You can then fill in the project name and token in the config file. 

Additionally, you need to specify the folder that contains the data files and the folder path where the results would be stored.  This information is necessary for running the code and needs to be set in the general section in `config/config.ini` file. If you use pre-trained DeepRiPe models, you need to specify the path to those models in the deepripe section in config as well. 
