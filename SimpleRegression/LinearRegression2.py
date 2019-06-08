import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import datasets, linear_model
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    df = pd.read_csv('Eeg classifiers accuracy vs architecture relationship - FFT2.csv')
    x = df.drop(['Accuracy', 'Precision', 'Recall', 'F-Score', 'AUC', 'Condition', 'LossFunction', 'AverageLogLoss', 'TrainingLogLoss', 'SplitPerSession', 'Features' ], axis=1)
    y = df[['Accuracy']]

    X2 = sm.add_constant(x)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    
    print(est2.summary())
    sns.regplot(x='HiddenNodes', y='Accuracy', data=df)
    plt.show()
    sns.regplot(x='Iterations', y='Accuracy', data=df)
    plt.show()