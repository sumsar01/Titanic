import pandas as pd

#Read the data
X_train = pd.read_csv('./Data/train.csv')
X_test = pd.read_csv('./Data/test.csv')
y_train = X_train.Survived
X_train.drop(['Survived'], axis=1, inplace=True)


