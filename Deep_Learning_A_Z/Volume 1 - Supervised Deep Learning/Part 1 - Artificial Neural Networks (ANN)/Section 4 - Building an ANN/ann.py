# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
onehotencoder = ColumnTransformer([('one_hot_encoder',OneHotEncoder(),[1])],remainder = 'passthrough')

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:,1]) #encode the country column
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:,2]) #encode the gender field
X = onehotencoder.fit_transform(X)

X = X[:,1:] # remove dummy variable trap

# Splitting the dataset into the Training set and the Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Encoding the Dependend Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(Y)

# import the Keras Libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# initializing the Artificial Neural Network (ANN)
classifier = Sequential()

# Adding the input and first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# add the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# add the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', \
                   metrics = ['accuracy'])# use stochastic gradient descent using adam
    
# fit the ANN to the Training Set
classifier.fit(X_train, Y_train, batch_size = 50, epochs = 100)

# Predicting the test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

# display accuracy
accuracy = (cm[0,0] + cm[1,1]) / sum(sum(cm)) * 100
print(f'This method resulted in an accuracy of {accuracy}%')

# predict if a new customer will leave
# Geography : France
# Credit Score: 600
# Gender : Male
# Age : 40
# Tenure : 3
# Balance : 60000
# # of Products : 2
# Active Credit Card : Yes
# Active member : Yes
# Estimated Salary : 50000
newCustomer = sc.transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))
newCustomerPrediction = classifier.predict(newCustomer)
newCustomerPrediction = (newCustomerPrediction > 0.5)
if newCustomerPrediction:
    print('This new customer is predicted to leave the bank')
else:
    print('This new customer is predicted to stay with the bank')


# Improve and Tune the ANN using K-Fold Cross Validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    # initializing the Artificial Neural Network (ANN)
    classifier = Sequential()
    
    # Adding the input and first hidden layer
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    
    # add the second hidden layer
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    
    # add the output layer
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    
    # compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])# use stochastic gradient descent using adam
    
    return classifier

kFoldClassifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = kFoldClassifier, X = X_train, y = Y_train, cv = 10, n_jobs = -1)
