
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import datasets, linear_model
from scipy import stats
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv('Eeg classifiers accuracy vs architecture relationship.csv')
    x = df.drop('Accuracy', axis=1)
    y = df[['Accuracy']]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                  random_state=1)
    reg = LinearRegression()
    
    # multiple regression
    result = reg.fit(X_train[['Iterations','Hidden_nodes']], y_train)
    y_predicted = reg.predict(X_train[['Iterations','Hidden_nodes']])

    # print results
    print("Mean squared error: %.2f" % mean_squared_error(y_train, y_predicted))
    print('R²: %.2f' % r2_score(y_train, y_predicted))
    #print(result.summary())

    # plot
    fig, ax = plt.subplots()
    ax.scatter(y_train, y_predicted)
    ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)
    ax.set_xlabel('measured')
    ax.set_ylabel('predicted')
    plt.show()