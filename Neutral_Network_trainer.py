import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
from  Titanic_data_cleaner import Titanic_data_cleaner
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from numpy.random import seed
import tensorflow

rcParams['figure.figsize'] = 10,8
sns.set(style='whitegrid', palette='bright',
        rc={'figure.figsize': (15,10)})

#Read the data
X_train = pd.read_csv('./Data/train.csv')
test = pd.read_csv('./Data/test.csv')

#fusing data to make a larger data set
data = pd.concat([X_train, test], axis=0, sort=True)


#Preparing data for use
data = Titanic_data_cleaner(data)

#scaling "continuous" data features for NN model use
continuous = ['Age', 'Fare', 'Parch', 'Pclass', 'SibSp', 'Family_Size']
scaler = StandardScaler()

for var in continuous:
    data[var] = data[var].astype('float64')
    data[var] = scaler.fit_transform(data[var].values.reshape(-1,1))

#Splitting training and testing data
X_train = data[pd.notnull(data['Survived'])].drop(['Survived'], axis=1)
y_train = data[pd.notnull(data['Survived'])]['Survived']
X_test = data[pd.isnull(data['Survived'])].drop(['Survived'], axis=1)

#Creating NN model
def create_model(lyrs=[8], act='linear', opt='Adam', dr=0.0):
    
    #setting seed
    seed(1)
    tensorflow.random.set_seed(1)
    
    model = Sequential()
    
    #create first hidden layer
    model.add(Dense(lyrs[0], input_dim=X_train.shape[1], activation=act))

    #create more layers
    for i in range(1, len(lyrs)):
        model.add(Dense(lyrs[i], activation=act))
        
    #add dropout
    model.add(Dropout(dr))
    
    #create output layer
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model


#We now want to optimize our model with grid search
#creating model
model = KerasClassifier(build_fn=create_model, verbose=0)

#define grid search parameters
batch_size = [16, 32, 64]
epochs = [50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)


#perform grid search
grid = GridSearchCV(estimator=model,
                    param_grid=param_grid,
                    cv=3,
                    verbose=2)

grid_result = grid.fit(X_train, y_train)

#Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))







