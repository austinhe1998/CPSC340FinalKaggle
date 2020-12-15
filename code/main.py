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
        X_training_filenames = glob(os.path.join('..','train','X','X_*.csv'))
        X_training_dataframes = [pd.read_csv(f) for f in X_training_filenames]

        y_training_filenames = glob(os.path.join('..','train','y','y_*.csv'))
        y_training_dataframes = [pd.read_csv(f) for f in y_training_filenames]

        # X_validate_filenames = glob(os.path.join('..','val','X','X_*.csv'))
        # X_validate_dataframes = [pd.read_csv(f) for f in X_validate_filenames]

        # y_validate_filenames = glob(os.path.join('..','val','y','y_*.csv'))
        # y_validate_dataframes = [pd.read_csv(f) for f in y_validate_filenames]

        X_test_filenames = glob(os.path.join('..','test','X','X_*.csv'))
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
        X = np.zeros((len(X_training_dataframes), 22))
        X_test = np.zeros((len(X_test_dataframes), 22))
        y = np.zeros((len(X_training_dataframes), 60))

        for x in range(len(X_training_dataframes)):
            result_x_positions = y_training_dataframes[x].loc[:, ' x'].to_numpy()
            result_y_positions = y_training_dataframes[x].loc[:, ' y'].to_numpy()
            result_positions = np.concatenate((result_x_positions, result_y_positions))
            result_positions = np.resize(result_positions, (60,))
            # print(result_positions.shape)
            y[x] = result_positions

            for i in range(10):
                if X_training_dataframes[x].iloc[0, 6 * i + 2] == " agent":
                    x_positions = X_training_dataframes[x].iloc[:, 6 * i + 2 + 2].to_numpy()
                    y_positions = X_training_dataframes[x].iloc[:, 6 * i + 2 + 3].to_numpy()
                    positions = np.concatenate((x_positions, y_positions))
                    X[x] = positions

                if x < len(X_test_dataframes) and X_test_dataframes[x].iloc[0, 6 * i + 2] == " agent":
                    x_positions = X_test_dataframes[x].iloc[:, 6 * i + 2 + 2].to_numpy()
                    y_positions = X_test_dataframes[x].iloc[:, 6 * i + 2 + 3].to_numpy()
                    positions = np.concatenate((x_positions, y_positions))
                    X_test[x] = positions

        model = NeuralNet([5], max_iter=10000)
        model.fit(X, y)

        y_hat = model.predict(X_test)
        print(y_hat.shape)
        pd.DataFrame(y_hat.flatten()).to_csv("output.csv")

        # X = X_train.loc[:, [' x0', ]]

        ''' Features to choose from '''
        # X = data.loc[:,['country_id', 'deaths',
        #                             #   'cases',
        #                               'cases_14_100k',
        #                               'cases_100k'
        #                               ]]

        ''' Choose from Countries for training '''
        #X = X[(X['country_id']=='CA')|(X['country_id']=='SE')]

        ''' Choose K '''
        # K = 35
        # mintest = 1000
        # ans= np.array([9504,9530,9541,9557,9585,9585,9585,9627,9654,9664,9699])
        #for k in range(K):
            # Fit weighted least-squares estimator
        # model = linear_model.MultiFeaturesAutoRegressor(K)
        # model.fit(X)

        #    currtest = np.sqrt(sklearn.metrics.mean_squared_error(model.predict(X[X['country_id']=='CA'],11), ans))
        #    print(k)
        #    if currtest<=mintest:
        #        mintest = currtest
        #        print(mintest)
        #print(np.sqrt(sklearn.metrics.mean_squared_error(r, ans)))

        # r = model.predict(X[X['country_id']=='CA'],5)
        # print(r)
