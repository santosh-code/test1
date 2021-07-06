# Artificial Neural Network

# Step 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#Encoding tge Country
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
#Encoding the Gender
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

#Applying One Hot Encoder on Countries which is the first Column 
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
#Avoiding Dummy Variable Trap
X = X[:, 1:]               

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



# Step 2 Making The ANN

# Importing the Keras libraries and packages
#sequential model is intialize the ANN
#dense model is used to add layers to the Ann
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
#We don not add anything here as we will be adding layers to our classifier(Neural Network)
classifier = Sequential()

# Adding the input layer and the first hidden layer
#here the input layer units is 11 and one unit in the output; so taking average
#the input_dim tells that this first hidden layer will be taking 11 inputs, 
#After specifying the input_dim at the first layer we do not need to specify again at the second layer
#as it can know from the output dimention of the first layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dropout(p=0.1))
# Adding the second hidden layer
#notice we are using recifier function as the activation function for hidden layers
#sigmoid function will be used in the final output layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(p=0.1))
# Adding the output layer
#sigmoid functions gives the probablity, so in this case we have two classes
#i.e whether the user will leave or stay
#when more classes we should use softmax function
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

#Step-3  Compiling the ANN
#Optimizer can be assumed as a gradient descent method
#In this case adam is a kind of Stochastic Gradient Descent method
#loss function is like the sum of squared errors
# here we are using logarithmic loss function 
#incase of two outcomes this function is called by name binary_crossentropy
#incase of multiple outcomes this function is called crossvalidation_crossentropy
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
#Converting the result into true and false form as only that is accepted by the confusion matrix
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Testing the model on single training data
X_sample=np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1,50000]])
sc.transform(X_sample)
classifier.predict(X_sample)



#################################           MODEL EVALUATION              ##############################
## To Perform Model Evaluation ,Using K-Fold CrossValidation Technique
# To Run This Part Perfrom Step 1 of Data Preprocessing and then this i.e Skipping Part -2 Making of Ann and Part -3 Making Predictions


# Evaluating the ANN
import tensorflow
import keras
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
#from joblib import Parallel, delayed
#from joblib import load, dump
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

    classifier = KerasClassifier(build_fn = build_classifier, batch_size = 100, epochs = 100)
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 10)
    mean = accuracies.mean()
    variance = accuracies.std()
    print("\nfinished")
    print("Mean: ", mean)
    print("Variance : ",variance)
    
    
    
    
###########################     Model Evaluation using GRID SEARCH BY SKLEARN ###############################
def build_classifier_all_optimization(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer =optimizer , loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier


model = KerasClassifier(build_fn = build_classifier_all_optimization, batch_size = 100, epochs = 100)
  
# define the grid search parameters
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
param_grid = dict(optimizer=optimizer)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(X, y)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))    
    
    
    
    
    
    