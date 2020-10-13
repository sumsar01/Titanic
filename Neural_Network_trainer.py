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
"""
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
"""
def NN_model(data):
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
    def create_model(lyrs=[8], act='linear', opt='RMSprop', dr=0.0):
    
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

    #create final model
    model = create_model(lyrs=[8], dr=0.0)
    print(model.summary())

    #training model with 80/20 CV split
    training = model.fit(X_train, y_train, epochs=50, batch_size=32,
                         validation_split=0.2, verbose=0)

    #evaluate model
    scores = model.evaluate(X_train, y_train)
    print(print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)))

    # summarize history for accuracy
    plt.figure()
    plt.plot(training.history['accuracy'])
    plt.plot(training.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.show()


    return model, X_test


"""
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

#Finding best optimization algorithm
model = KerasClassifier(build_fn=create_model, epochs=50, batch_size=32, verbose=0)

#defining grid search params
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Nadam']
param_grid = dict(opt=optimizer)

#searching grid
grid = GridSearchCV(estimator=model, param_grid=param_grid, verbose=2)
grid_result = grid.fit(X_train, y_train)

#Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


#optimizing number of dropout

# create model
model = KerasClassifier(build_fn=create_model, 
                        epochs=50, batch_size=32, verbose=0)

# define the grid search parameters
drops = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
param_grid = dict(dr=drops)

# search the grid
grid = GridSearchCV(estimator=model, param_grid=param_grid, verbose=2)
grid_result = grid.fit(X_train, y_train)

#Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
"""
"""
# create model
model = KerasClassifier(build_fn=create_model, 
                        epochs=50, batch_size=32, verbose=0)

# define the grid search parameters
layers = [[8], [10]]
param_grid = dict(lyrs=layers)

# search the grid
grid = GridSearchCV(estimator=model, param_grid=param_grid, verbose=2)
grid_result = grid.fit(X_train, y_train)
"""

























