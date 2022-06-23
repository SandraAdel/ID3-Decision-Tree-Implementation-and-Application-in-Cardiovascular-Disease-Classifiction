
################################################################      Cardiovascular Disease Dataset        ##############################################################

############################################################################     RUN CODE     ############################################################################
##################   FOR TRAINING TIME, TESTING TIME AND ACCURACY METRICS COMPARISON OF MY IMPLEMENTATION AND SCIKIT-LEARN'S OF ID3 DECISION TREE   ######################

# NOTE: DOCUMENTATION PROVIDED IN COMMENTS


# Imports
import math
import time
import copy
import random
import pandas as pd
from sklearn.metrics import accuracy_score
from ID3_DecisionTree_Implementation import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Preprocessing Functions: (Specific to Dataset)

# 1) 'ap_hi' and 'ap_lo' Features Discretization
def BloodPressureDiscretization(dataset):

    # Systolic Blood Pressure Discretization: 1--> normal, 2--> above normal, 3--> well above normal
    dataset.loc[dataset.ap_hi < 120, 'ap_hi'] = 1
    dataset.loc[(dataset.ap_hi >= 120) & (dataset.ap_hi <= 139), 'ap_hi'] = 2
    dataset.loc[dataset.ap_hi >= 140, 'ap_hi'] = 3

    # Diastolic Blood Pressure Discretization: 1--> normal, 2--> above normal, 3--> well above normal
    dataset.loc[dataset.ap_lo < 80, 'ap_lo'] = 1
    dataset.loc[(dataset.ap_lo >= 80) & (dataset.ap_lo <= 89), 'ap_lo'] = 2
    dataset.loc[dataset.ap_lo >= 90, 'ap_lo'] = 3

    return dataset

# 2) Replacing 'weight' and 'height' Features with 'bmi' and discretizing it
def BMICalculationAndDiscretization(dataset):

    # Replacing 'weight' and 'height' with 'bmi'
    dataset.height = round(dataset['weight'] / (dataset['height']/100)**2, 1)
    dataset.rename(columns = {'height':'bmi'}, inplace = True)
    del dataset['weight']

    # 'bmi' Discretization: 1--> underweight, 2--> normal, 3--> overweight, 4--> obese
    dataset.loc[dataset.bmi < 18.5, 'bmi'] = int(1)
    dataset.loc[(dataset.bmi >= 18.5) & (dataset.bmi <= 24.9), 'bmi'] = int(2)
    dataset.loc[(dataset.bmi >= 25) & (dataset.bmi <= 29.9), 'bmi'] = int(3)
    dataset.loc[dataset.bmi >= 30, 'bmi'] = int(4)

    return dataset

# 3) Converting 'age' from days to years, discretizing it and deleting 'id' column
def IDDeletionAndAgeDiscretization(dataset):

    # Replacing 'age' in days with 'age' in years
    dataset['age'] = round(dataset['age'] /365)
    
    # 'age' Discretization: 1--> young adult, 2--> middle-aged adult, 3--> old adult
    dataset.loc[(dataset.age >= 30) & (dataset.age <= 39), 'age'] = 1
    dataset.loc[(dataset.age >= 40) & (dataset.age <= 59), 'age'] = 2
    dataset.loc[dataset.age >= 60, 'age'] = 3

    # Deleting 'id' column
    del dataset['id']

    return dataset


if __name__ == '__main__':

    dataset = pd.read_csv('Cardiovascular Disease Dataset.csv', sep=';')

    # Dataset Preprocessing Pipeline
    dataset = BloodPressureDiscretization(dataset)
    dataset = BMICalculationAndDiscretization(dataset)
    dataset = IDDeletionAndAgeDiscretization(dataset)

    # Splitting dataset into Training & Testing sub-datasets
    features = dataset.columns[:10].tolist()
    X = dataset[features]
    y = dataset['cardio']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    ########## Scikit-Learn ID3 Decision Tree Classifier ##########

    print('Scikit-Learn ID3 Decision Tree Classifier:')
    print('')

    sklearn_tree = DecisionTreeClassifier(criterion="entropy")
    
    start = time.time()
    sklearn_tree.fit(X_train, y_train)
    stop = time.time()
    print(f"Training time: {round(stop - start, 5)} sec")
    
    start = time.time()
    sklearn_predictions = sklearn_tree.predict(X_test)
    stop = time.time()
    print(f"Testing time: {round(stop - start, 5)} sec")

    sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)
    print(f"Accuracy: {round(sklearn_accuracy * 100, 2)}%")
    
    print('')

    ########## My Implementation ID3 Decision Tree Classifier ##########

    print('My Implementation ID3 Decision Tree Classifier:')
    print('')

    tree = ID3DecisionTree()
    
    start = time.time()
    tree.fit(X_train, y_train)
    stop = time.time()
    print(f"Training time: {round(stop - start, 5)} sec")

    start = time.time()
    my_predictions = tree.predict(X_test.to_numpy().tolist())
    stop = time.time()
    print(f"Testing time: {round(stop - start, 5)} sec")

    my_accuracy = accuracy_score(y_test, my_predictions)
    print(f"Accuracy: {round(my_accuracy * 100, 2)}%")

    print('')
