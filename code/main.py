import os
import argparse
import time
import pickle
import time
# 3rd party libraries
import numpy as np
import pandas as pd
import datetime as datetime
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime
from sklearn.tree import DecisionTreeClassifier
import utils
import sklearn.metrics
from glob import glob
from neural_net import NeuralNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question

    if question == "2":
        # Data set
        X_training_filenames = glob(os.path.join('..','..','train','X','X_*.csv'))
        X_training_dataframes = [pd.read_csv(f) for f in X_training_filenames]

        y_training_filenames = glob(os.path.join('..','..','train','y','y_*.csv'))
        y_training_dataframes = [pd.read_csv(f) for f in y_training_filenames]

        # X_validate_filenames = glob(os.path.join('..','val','X','X_*.csv'))
        # X_validate_dataframes = [pd.read_csv(f) for f in X_validate_filenames]

        # y_validate_filenames = glob(os.path.join('..','val','y','y_*.csv'))
        # y_validate_dataframes = [pd.read_csv(f) for f in y_validate_filenames]

        X_test_filenames = glob(os.path.join('..','..','test','X','X_*.csv'))
        X_test_dataframes = [pd.read_csv(f) for f in X_test_filenames]

        print("Finished reading and formatting data!")
        
        agent_position = 0
        X_x = np.zeros((len(X_training_dataframes), 33))
        X_y = np.zeros((len(X_training_dataframes), 33))
        X_test_x = np.zeros((len(X_test_dataframes), 33))
        X_test_y = np.zeros((len(X_test_dataframes), 33))
        y_x = np.zeros((len(X_training_dataframes), 30))
        y_y = np.zeros((len(X_training_dataframes), 30))

        for x in range(len(X_training_dataframes)):
            result_x_positions = y_training_dataframes[x].loc[:, ' x'].to_numpy()
            result_y_positions = y_training_dataframes[x].loc[:, ' y'].to_numpy()
            result_x_positions = np.resize(result_x_positions, (30,))
            result_y_positions = np.resize(result_y_positions, (30,))

            y_x[x] = result_x_positions
            y_y[x] = result_y_positions

            other_car_limit_train = 2
            other_car_limit_test = 2

            x_positions = np.array([])
            y_positions = np.array([])

            x_positions_test = np.array([])
            y_positions_test = np.array([])

            agent_found_flag = False
            agent_found_flag_test = False

            for i in range(10):
                
                if X_training_dataframes[x].iloc[0, 6 * i + 2] == " agent":
                    temp_x = X_training_dataframes[x].iloc[:, 6 * i + 2 + 2].to_numpy()
                    temp_y = X_training_dataframes[x].iloc[:, 6 * i + 2 + 3].to_numpy()
                    x_positions = np.append(temp_x, x_positions)
                    y_positions = np.append(temp_y, y_positions)
                    agent_found_flag = True

                # select 2 non-agent vehicles
                elif other_car_limit_train > 0:
                    x_positions = np.append(x_positions, X_training_dataframes[x].iloc[:, 6 * i + 2 + 2].to_numpy())
                    y_positions = np.append(y_positions, X_training_dataframes[x].iloc[:, 6 * i + 2 + 3].to_numpy())

                    other_car_limit_train -= 1
                
                elif other_car_limit_train == 0 and agent_found_flag == True:
                    X_x[x] = x_positions
                    X_y[x] = y_positions

                if x < len(X_test_dataframes) and X_test_dataframes[x].iloc[0, 6 * i + 2] == " agent":
                    temp_x = X_test_dataframes[x].iloc[:, 6 * i + 2 + 2].to_numpy()
                    temp_y = X_test_dataframes[x].iloc[:, 6 * i + 2 + 3].to_numpy()
                    x_positions_test = np.append(temp_x, x_positions_test)
                    y_positions_test = np.append(temp_y, y_positions_test)
                    agent_found_flag_test = True
                
                # select 2 non-agent vehicles
                elif x < len(X_test_dataframes) and other_car_limit_test > 0:
                    x_positions_test = np.append(x_positions_test, X_test_dataframes[x].iloc[:, 6 * i + 2 + 2].to_numpy())
                    y_positions_test = np.append(y_positions_test, X_test_dataframes[x].iloc[:, 6 * i + 2 + 3].to_numpy())

                    other_car_limit_test -= 1
                
                elif x < len(X_test_dataframes) and other_car_limit_test == 0 and agent_found_flag_test == True:
                    X_test_x[x] = x_positions_test
                    X_test_y[x] = y_positions_test


        model1 = NeuralNet([50], max_iter=10000)
        model1.fit(X_x, y_x)

        y_hat_x = model1.predict(X_test_x).flatten()

        model2 = NeuralNet([30], max_iter=10000)
        model2.fit(X_y, y_y)

        y_hat_y = model2.predict(X_test_y).flatten()

        y_hat = np.insert(y_hat_y, np.arange(len(y_hat_x)), y_hat_x)
        pd.DataFrame(y_hat).to_csv("output.csv")

