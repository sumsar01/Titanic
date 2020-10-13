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

model = create_model()
print(model.summary())

#Train model with 80/20 CV split
training = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, 
                     verbose=0)

val_acc = np.mean(training.history['val_accuracy'])
print("\n%s: %.2f%%" % ('val_acc', val_acc*100))

#plotting training acc as a function of epoch
plt.plot(training.history['accuracy'])
plt.plot(training.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower right')
plt.show()


















