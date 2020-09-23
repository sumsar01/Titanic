import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams



#Read the data
X_train = pd.read_csv('./Data/train.csv')
X_test = pd.read_csv('./Data/test.csv')

#fusing data to make a larger data set
data = pd.concat([X_train, X_test], axis=0, sort=True)


#Preparing data for use
data.drop(['Cabin'], axis=1)
data.Embarked.fillna(method = 'backfill')
data.Fare.dropna()

#Imputing missing Age
data['Title'] = data['Name'].str.extract('([A-Za-z]+)\.', expand=True)
data['Title'].value_counts()
#Replacing rare titles
mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr',
           'Don': 'Mr', 'Mme': 'Mrs', 'Jonkheer': 'Mr', 'Lady': 'Mrs',
           'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}
data.replace({'Title': mapping}, inplace=True)

#median age of title holders
title_median_age = dict(data.groupby('Title')['Age'].median())
#average ages
data['age_med'] = data['Title'].apply(lambda x: title_median_age[x])

#replacing missing ages
data['Age'].fillna(data['age_med'], inplace=True)
del data['age_med']


print(data['Title'].value_counts())
