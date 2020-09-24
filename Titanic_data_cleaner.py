import pandas as pd
from sklearn.preprocessing import LabelEncoder

def Titanic_data_cleaner(data):
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
    
#encoding coulumns
    
    label_encoder = LabelEncoder()
    data.Sex = label_encoder.fit_transform(data.Sex)
    
    
    return data