import src.constants as const
from os import listdir
from os.path import isfile, join
import pandas as pd
from sklearn import preprocessing
import numpy as np
import torch
import torch.utils.data as data_utils
import src.autoencoder as autoencoder
from src.evaluation import Evaluation
from config import config as conf
import math

import argparse
import random
import time
import optuna
import pymysql

pymysql.install_as_MySQLdb()

def combine_plotly_figs_to_html(plotly_figs, html_fname, include_plotlyjs="cdn"):
    with open(html_fname, "w") as f:
        f.write(plotly_figs[0].to_html(include_plotlyjs=include_plotlyjs))
        for fig in plotly_figs[1:]:
            f.write(fig.to_html(full_html=False, include_plotlyjs=False))

def plot_optuna_default_graphs(optuna_study):
    history_plot = optuna.visualization.plot_optimization_history(optuna_study)
    parallel_plot = optuna.visualization.plot_parallel_coordinate(optuna_study)
    slice_plot = optuna.visualization.plot_slice(optuna_study)
    plot_list = [history_plot, parallel_plot, slice_plot]
    return plot_list

def main(config, trial_number="best"):
    # Pre-requisites
    min_max_scaler = preprocessing.MinMaxScaler()

    # setting seed for reproducibility
    torch.manual_seed(conf.SEED)
    np.random.seed(conf.SEED)

    dataset_path = const.PROPRIETARY_DATASET_LOCATION
    # Read normal data
    normal_path = join(dataset_path,'normal_data/')
    # As the first step, combine the csvs inside normal_data folder
    normal_data_files = [f for f in listdir(normal_path) if isfile(join(normal_path, f))]
    normal_df_list = [pd.read_csv(normal_path + normal_data_file) for normal_data_file in normal_data_files]
    # Next, drop the datetime column
    normal_df_list_without_datetime = [normal_df.drop(columns=['datetime']) for normal_df in normal_df_list]
    # Finally merge those dataframes
    normal_df = pd.concat(normal_df_list_without_datetime)
    normal_df = normal_df.astype(float)

    # Read anomaly data
    anomaly_path = join(dataset_path,'anomaly_data/')
    # As the first step, combine the csvs inside anomaly_data folder
    anomaly_data_files = [f for f in listdir(anomaly_path) if isfile(join(anomaly_path, f))]
    anomaly_df_list = [pd.read_csv(anomaly_path + anomaly_data_file) for anomaly_data_file in anomaly_data_files]
    # Next, drop the datetime column
    anomaly_df_list_without_datetime = [anomaly_df.drop(columns=['datetime']) for anomaly_df in anomaly_df_list]
    # Finally merge those dataframes
    anomaly_df = pd.concat(anomaly_df_list_without_datetime)
    # Separate out the is_anomaly labels before normalisation/standardization
    anomaly_df_labels = anomaly_df['is_anomaly']
    anomaly_df = anomaly_df.drop(['is_anomaly'], axis=1)
    anomaly_df = anomaly_df.astype(float)

    # Normalise/ standardize the normal and anomaly dataframe
    full_df = pd.concat([normal_df, anomaly_df])
    min_max_scaler.fit(full_df)

    normal_df_values = normal_df.values
    normal_df_values_scaled = min_max_scaler.transform(normal_df_values)
    normal_df_scaled = pd.DataFrame(normal_df_values_scaled)

    # Normalise/ standardize the anomaly dataframe
    anomaly_df_values = anomaly_df.values
    anomaly_df_values_scaled = min_max_scaler.transform(anomaly_df_values)
    anomaly_df_scaled = pd.DataFrame(anomaly_df_values_scaled)

    # Preparing the datasets for training and testing using AutoEncoder
    windows_normal = normal_df_scaled.values[np.arange(config["WINDOW_SIZE"])[None, :] + np.arange(normal_df_scaled.shape[0] - config["WINDOW_SIZE"])[:, None]]
    windows_anomaly = anomaly_df_scaled.values[np.arange(config["WINDOW_SIZE"])[None, :] + np.arange(anomaly_df_scaled.shape[0] - config["WINDOW_SIZE"])[:, None]]

    w_size = windows_normal.shape[1] * windows_normal.shape[2] # w_size is the input window size
    z_size = windows_normal.shape[1] * config["HIDDEN_SIZE"] # z_size is the latent size

    windows_normal_train = windows_normal[:int(np.floor(.8 * windows_normal.shape[0]))]
    windows_normal_val = windows_normal[int(np.floor(.8 * windows_normal.shape[0])):]

    # Create batches of training and testing data
    train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_normal_train).float().view(([windows_normal_train.shape[0], w_size]))
    ), batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=0)
    val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_normal_val).float().view(([windows_normal_val.shape[0], w_size]))
    ), batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_anomaly).float().view(([windows_anomaly.shape[0], w_size]))
    ), batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=0)

    # Initialise the AutoEncoder model
    autoencoder_model = autoencoder.AutoEncoder(in_size=w_size, latent_size=z_size, num_layers=config["NUM_LAYERS"])
    # Start training and save the best model, i.e. the model with the least validation loss
    model_path = const.MODEL_LOCATION
    model_name = join(model_path, "ae_model_{}.pth".format(trial_number))  # parameterize the run number
    autoencoder.training(conf.N_EPOCHS, autoencoder_model, train_loader, val_loader, config["LEARNING_RATE"], model_name)

    # Load the model
    checkpoint = torch.load(model_name)
    autoencoder_model.encoder.load_state_dict(checkpoint['encoder'])
    autoencoder_model.decoder.load_state_dict(checkpoint['decoder'])

    # Use the trained model to obtain predictions for the test set
    results = autoencoder.testing(autoencoder_model, test_loader)
    y_pred_for_test_set = np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(), results[-1].flatten().detach().cpu().numpy()])

    # Process the actual labels
    windows_labels = []
    for i in range(len(anomaly_df_labels) - config["WINDOW_SIZE"]):
        windows_labels.append(list(np.int_(anomaly_df_labels[i:i + config["WINDOW_SIZE"]])))

    processed_test_labels = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels]

    thresholding_percentile = 100 - (((processed_test_labels.count(1.0)) / (len(processed_test_labels))) * 100)

    # Obtain threshold based on pth percentile of the mean squared error
    threshold = np.percentile(y_pred_for_test_set, [thresholding_percentile])[0]  # 90th percentile

    # Map the predictions to anomaly labels after applying the threshold
    predicted_labels = []
    for val in y_pred_for_test_set:
        if val > threshold:
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)

    # Evaluate the predicted_labels against the actual labels
    test_eval = Evaluation(processed_test_labels, predicted_labels)
    test_eval.print()

    return test_eval.auc

def objective(trial):
    params = dict()
    params["NUM_LAYERS"] = trial.suggest_int("NUM_LAYERS", 2, 10)
    window_size_limit = math.ceil((2 ** (params["NUM_LAYERS"] - 1)) / conf.n_features)
    params["WINDOW_SIZE"] = trial.suggest_int("WINDOW_SIZE", window_size_limit, 100)
    hidden_size_limit = int((params["WINDOW_SIZE"]*conf.n_features)/(2**(params["NUM_LAYERS"]-1)))
    params["HIDDEN_SIZE"] = trial.suggest_int("HIDDEN_SIZE", 1, max(2,hidden_size_limit))
    params["BATCH_SIZE"] = trial.suggest_int("BATCH_SIZE", 20, 1000)
    params["LEARNING_RATE"] = trial.suggest_float("LEARNING_RATE", 1e-5, 1e-1, log=True)

    print(f"Initiating Run {trial.number} with params : {trial.params}")

    loss = main(params, trial.number)
    return loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--optuna-db", type=str, help="Path to the Optuna Database file",
                        default="sqlite:///optuna.db")
    parser.add_argument("-n", "--optuna-study-name", type=str, help="Name of the optuna study",
                        default="duneesha_autoencoder_run_1")
    args = parser.parse_args()

    # wait for some time to avoid overlapping run ids when running parallel
    wait_time = random.randint(0, 10) * 3
    print(f"Waiting for {wait_time} seconds before starting")
    time.sleep(wait_time)

    study = optuna.create_study(direction="maximize",
                                study_name=args.optuna_study_name,
                                storage=args.optuna_db,
                                load_if_exists=True,
                                )
    study.optimize(objective, n_trials=1)  # When running locally, set n_trials as the no.of trial required

    # print best study
    best_trial = study.best_trial
    print(best_trial.params)

    plots = plot_optuna_default_graphs(study)

    combine_plotly_figs_to_html(plotly_figs=plots, html_fname="optimization_trial_plots/autoencoder_hpo.html")

# Best trials
# Run 17 with params : {'WINDOW_SIZE': 7, 'NUM_LAYERS': 5, 'HIDDEN_SIZE': 1, 'BATCH_SIZE': 59, 'LEARNING_RATE': 5.729089814295318e-05}
# Run 51 with params : {'WINDOW_SIZE': 6, 'NUM_LAYERS': 4, 'HIDDEN_SIZE': 2, 'BATCH_SIZE': 265, 'LEARNING_RATE': 1.1522142669867557e-05}