# -*- coding: utf-8 -*-
# @Time    : 3/9/2023 6:57 AM
# @Author  : Gang Qu
# @FileName: main_linear_model.py

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse

def main(args):
    # Set the random seed for reproducibility
    np.random.seed(42)

    # Get data
    X = np.random.randn(100, 2)
    y = 2 * X[:, 0] + 3 * X[:, 1] + 1 + 0.1 * np.random.randn(100)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.train_val_test[-1], random_state=args.seed)

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create a LinearRegression object with L2 regularization
    model = LinearRegression(fit_intercept=True, alpha=0.1)

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Compute the root mean squared error
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    # Print the model RMSE
    print("RMSE:", rmse)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PNC multi-regression')
    parser.add_argument('--L2', default=1e-6, help='L2 regularization')
    parser.add_argument('--seed', default=100, type=int, help='random seed')
    parser.add_argument('--config', default='GGT_PNCmultiregression', help='config file name')
    parser.add_argument('--train_val_test', default=[0.7, 0.1, 0.2], help='train, val, test split')
    parser.add_argument('--dataset', default='dglPNC', help='dataset name')
    parser.add_argument('--cnb_scores', default='wrat',
                        choices=[
                            'wrat', 'pvrt', 'pmat', 'all'
                        ],
                        help='type of cnb scores')
    parser.add_argument('--paradigms', default='rest_pnc',
                        choices=[
                            'emoid_pnc', 'nback_pnc', 'rest_pnc',
                        ],
                        help='fmri paradigms')
    args = parser.parse_args()
    main()

