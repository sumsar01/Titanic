import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
from  Titanic_data_cleaner import Titanic_data_cleaner
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rcParams['figure.figsize'] = 10,8
sns.set(style='whitegrid', palette='bright',
        rc={'figure.figsize': (15,10)})

#Read the data
X_train = pd.read_csv('./Data/train.csv')
X_test = pd.read_csv('./Data/test.csv')

#fusing data to make a larger data set
data = pd.concat([X_train, X_test], axis=0, sort=True)

"""
#Plotting
plt.figure()
ax = sns.swarmplot(x="Pclass", y="Fare", hue='Survived', data=data)
ax.figure.savefig("Swarmplot.pdf")
plt.figure()
ax = sns.countplot(x="Pclass", hue="Sex", data=data)
ax.figure.savefig("Count_by_class_plot.pdf")
plt.figure()
ax = sns.countplot(x="Survived", hue="Sex", data=data)
ax.figure.savefig("Survived_by_sex_plot.pdf")
"""


#Preparing data for use
data = Titanic_data_cleaner(data)

#Splitting data up again
train_data = data[pd.notnull(data['Survived'])]
X_test = data[pd.isnull(data['Survived'])].drop(['Survived'], axis=1)

#splitting training and validation data
X_train, X_val, y_train, y_val = train_test_split(
    train_data.drop(['Survived'], axis=1),
    train_data['Survived'],
    test_size=0.2, random_state=42)

"""
#testing shape of new data
for i in [X_train, X_val, X_test]:
    print(i.shape)
"""

#First model
model = RandomForestClassifier(random_state=0)

#fitting to model
model.fit(X_train, y_train)

#making predictions on validation data
preds = model.predict(X_val)



print(accuracy_score(y_val, preds))






















