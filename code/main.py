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
import linear_model
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

        # X_train = pd.concat(X_training_dataframes, axis=1)
        # y_train = pd.concat(y_training_dataframes, axis=1)
        # X_validate = pd.concat(X_validate_dataframes, axis=1)
        # y_validate = pd.concat(y_validate_dataframes, axis=1)
        # X_test = pd.concat(X_test_dataframes, axis=1)

        print("Finished reading and formatting data!")
        
        # X = X_train.loc[:,[' x0', ' y0', ' x1', ' y1', ' x2', ' y2', ' x3', ' y3', ' x4', ' y4', ' x5', ' y5', ' x6', ' y6', 
        #                     ' x7', ' y7', ' x8', ' y8', ' x9', ' y9']]
        agent_position = 0
        X_x = np.zeros((len(X_training_dataframes), 11))
        X_y = np.zeros((len(X_training_dataframes), 11))
        X_test_x = np.zeros((len(X_test_dataframes), 11))
        X_test_y = np.zeros((len(X_test_dataframes), 11))
        y_x = np.zeros((len(X_training_dataframes), 30))
        y_y = np.zeros((len(X_training_dataframes), 30))

        for x in range(len(X_training_dataframes)):
            result_x_positions = y_training_dataframes[x].loc[:, ' x'].to_numpy()
            result_y_positions = y_training_dataframes[x].loc[:, ' y'].to_numpy()
            result_x_positions = np.resize(result_x_positions, (30,))
            result_y_positions = np.resize(result_y_positions, (30,))

            y_x[x] = result_x_positions
            y_y[x] = result_y_positions

            for i in range(10):
                if X_training_dataframes[x].iloc[0, 6 * i + 2] == " agent":
                    x_positions = X_training_dataframes[x].iloc[:, 6 * i + 2 + 2].to_numpy()
                    y_positions = X_training_dataframes[x].iloc[:, 6 * i + 2 + 3].to_numpy()
                    X_x[x] = x_positions
                    X_y[x] = y_positions

                if x < len(X_test_dataframes) and X_test_dataframes[x].iloc[0, 6 * i + 2] == " agent":
                    x_positions = X_test_dataframes[x].iloc[:, 6 * i + 2 + 2].to_numpy()
                    y_positions = X_test_dataframes[x].iloc[:, 6 * i + 2 + 3].to_numpy()
                    X_test_x[x] = x_positions
                    X_test_y[x] = y_positions


        model1 = NeuralNet([30], max_iter=10000)
        model1.fit(X_x, y_x)

        y_hat_x = model1.predict(X_test_x).flatten()

        model2 = NeuralNet([30], max_iter=10000)
        model2.fit(X_y, y_y)

        y_hat_y = model2.predict(X_test_y).flatten()
        print(y_hat_x)
        print(y_hat_y)


        y_hat = np.insert(y_hat_y, np.arange(len(y_hat_x)), y_hat_x)
        pd.DataFrame(y_hat).to_csv("output.csv")

