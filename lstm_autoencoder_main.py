import src.constants as const
from os import listdir
from os.path import isfile, join
import pandas as pd
from sklearn import preprocessing
import numpy as np
import torch
import torch.utils.data as data_utils
import src.lstm_autoencoder as lstm_autoencoder
from src.evaluation import Evaluation
from config import config as conf

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
    plot_list = [history_plot]
    return plot_list

def main(config, trial_number="best"):
    # Pre-requisites
    min_max_scaler = preprocessing.MinMaxScaler()

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
    # Normalise/ standardize the normal dataframe
    normal_df_values = normal_df.values
    normal_df_values_scaled = min_max_scaler.fit_transform(normal_df_values)
    normal_df_scaled = pd.DataFrame(normal_df_values_scaled) # shape = (9652, 9)

    # Read anomaly data
    anomaly_path = join(dataset_path,'anomaly_data/')
    # As the first step, combine the csvs inside anomaly_data folder
    anomaly_data_files = [f for f in listdir(anomaly_path) if isfile(join(anomaly_path, f))]
    anomaly_df_list = [pd.read_csv(anomaly_path + anomaly_data_file) for anomaly_data_file in anomaly_data_files]
    # Next, drop the datetime column
    anomaly_df_list_without_datetime = [anomaly_df.drop(columns=['datetime']) for anomaly_df in anomaly_df_list]
    # Finally merge those dataframes
    anomaly_df = pd.concat(anomaly_df_list_without_datetime)
    anomaly_df = anomaly_df.astype(float)
    # Separate out the is_anomaly labels before normalisation/standardization
    anomaly_df_labels = anomaly_df['is_anomaly']
    anomaly_df = anomaly_df.drop(['is_anomaly'], axis=1)
    # Normalise/ standardize the anomaly dataframe
    anomaly_df_values = anomaly_df.values
    anomaly_df_values_scaled = min_max_scaler.transform(anomaly_df_values)
    anomaly_df_scaled = pd.DataFrame(anomaly_df_values_scaled) # shape = (3635, 9)

    # Preparing the datasets for training and testing using AutoEncoder
    windows_normal = normal_df_scaled.values[np.arange(config["WINDOW_SIZE"])[None, :] + np.arange(normal_df_scaled.shape[0] - config["WINDOW_SIZE"])[:, None]] # shape = (9647, 5, 9)
    windows_anomaly = anomaly_df_scaled.values[np.arange(config["WINDOW_SIZE"])[None, :] + np.arange(anomaly_df_scaled.shape[0] - config["WINDOW_SIZE"])[:, None]] # shape = (3630, 5, 9)

    # Create batches of training and testing data
    train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_normal).float()
    ), batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
        torch.from_numpy(windows_anomaly).float()
    ), batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=0)

    # Initialise the LSTMAutoEncoder model
    lstm_autoencoder_model = lstm_autoencoder.LstmAutoencoder(seq_len=config["WINDOW_SIZE"], n_features=windows_normal.shape[2], num_layers=config["NUM_LAYERS"])
    # Start training
    lstm_autoencoder.training(conf.N_EPOCHS, lstm_autoencoder_model, train_loader, config["LEARNING_RATE"])

    # Save the model and load the model
    model_path = const.MODEL_LOCATION
    model_name = join(model_path,"lstm_ae_model_{}.pth".format(trial_number))
    torch.save({
        'encoder': lstm_autoencoder_model.encoder.state_dict(),
        'decoder': lstm_autoencoder_model.decoder.state_dict()
    }, model_name)
    checkpoint = torch.load(model_name)
    lstm_autoencoder_model.encoder.load_state_dict(checkpoint['encoder'])
    lstm_autoencoder_model.decoder.load_state_dict(checkpoint['decoder'])

    # Use the trained model to obtain predictions for the test set
    results = lstm_autoencoder.testing(lstm_autoencoder_model, test_loader)
    y_pred_for_test_set = np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(), results[-1].flatten().detach().cpu().numpy()])

    # Obtain threshold based on pth percentile of the mean squared error
    threshold = np.percentile(y_pred_for_test_set, [81.61])[0]  # 90th percentile
    # Map the predictions to anomaly labels after applying the threshold
    predicted_labels = []
    for val in y_pred_for_test_set:
        if val > threshold:
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)

    # Evaluate the predicted_labels against the actual labels
    test_eval = Evaluation(anomaly_df_labels[config["WINDOW_SIZE"]:], predicted_labels)
    test_eval.print()

    return test_eval.auc

def objective(trial):
    params = dict()
    params["WINDOW_SIZE"] = trial.suggest_int("WINDOW_SIZE", 6, 100)
    params["NUM_LAYERS"] = trial.suggest_int("NUM_LAYERS", 2, 4)
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
                        default="duneesha_lstm_autoencoder_run_2")
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
    study.optimize(objective, n_trials=1) # When running locally, set n_trials as the no.of trials required

    # print best study
    best_trial = study.best_trial
    print(best_trial.params)

    plots = plot_optuna_default_graphs(study)

    combine_plotly_figs_to_html(plotly_figs=plots, html_fname="optimization_trial_plots/lstm_autoencoder_hpo.html")
