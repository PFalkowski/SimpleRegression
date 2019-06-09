import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import datasets, linear_model
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    df = pd.read_csv('Eeg classifiers accuracy vs architecture relationship - FFT2.csv')
    x = df.drop(['Accuracy', 'Precision', 'Recall', 'F-Score', 'AUC', 'Condition', 'LossFunction', 'AverageLogLoss', 'TrainingLogLoss', 'SplitPerSession' ], axis=1)
    y = df[['Accuracy']]

    poly = PolynomialFeatures(degree=2)
    poly_variables = poly.fit_transform(x,y)

    poly_var_train, poly_var_test, res_train, res_test = train_test_split(poly_variables, y, test_size = 0.3)
    
    regression = linear_model.LinearRegression()
    model = regression.fit(poly_var_train, res_train)
    score = model.score(poly_var_test, res_test)
    
    plt.plot(x,y,"b.")
    plt.plot(poly_variables[:,12],model.predict(poly_variables),'-r')
    plt.show()