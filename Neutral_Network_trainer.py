import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
from  Titanic_data_cleaner import Titanic_data_cleaner
from sklearn.model_selection import train_test_split
from RandomForest_model_trainer import RandomForest_model_trainer
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