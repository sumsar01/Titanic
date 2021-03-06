import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
from  Titanic_data_cleaner import Titanic_data_cleaner
from sklearn.model_selection import train_test_split
from RandomForest_model_trainer import RandomForest_model_trainer
from Neural_Network_trainer import NN_model
rcParams['figure.figsize'] = 10,8
sns.set(style='whitegrid', palette='bright',
        rc={'figure.figsize': (15,10)})

#Read the data
X_train = pd.read_csv('./Data/train.csv')
test = pd.read_csv('./Data/test.csv')

#fusing data to make a larger data set
data = pd.concat([X_train, test], axis=0, sort=True)

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

#Preparing data for use
data = Titanic_data_cleaner(data)

#making neural network model and predictions
model, X_test = NN_model(data)
test['Survived'] = model.predict(X_test)
test['Survived'] = test['Survived'].apply(lambda x: round(x,0)).astype('int')
predictions = test[['PassengerId', 'Survived']]

#Save predictions as csv
predictions.to_csv("Neural_Network_predictions.csv", index=False)

#RandomForestClassifier
#Splitting data up again
train_data = data[pd.notnull(data['Survived'])]
X_test = data[pd.isnull(data['Survived'])].drop(['Survived'], axis=1)

#splitting training and validation data
X_train, X_val, y_train, y_val = train_test_split(
    train_data.drop(['Survived'], axis=1),
    train_data['Survived'],
    test_size=0.2, random_state=42)

#Time to improve the model
X_train = pd.concat([X_train, X_val])
y_train = pd.concat([y_train, y_val])

#Training model
model = RandomForest_model_trainer(X_train, y_train)

test['Survived'] = model.predict(X_test)
predictions = test[['PassengerId', 'Survived']]
predictions['Survived'] = predictions['Survived'].apply(int)

#Save predictions as csv
predictions.to_csv("Random_Forest_predictions.csv", index=False)






