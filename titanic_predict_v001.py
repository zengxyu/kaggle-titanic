import tkinter
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,BaggingClassifier
from sklearn.preprocessing import LabelEncoder
matplotlib.use('TkAgg')

def pre_processing_data(df):
    encoded_df = pd.DataFrame()
    label = LabelEncoder()
    df = df.fillna(0)
    # df = df.drop(['Cabin','Ticket'],axis=1)
    for c in df.columns:
        if df[c].dtype == 'object': # if the data is in string type, encode it
            df[c] = df[c].astype(str)
            encoded_df[c] = label.fit_transform(df[c])
        else:
            encoded_df[c] = df[c]
    
    return encoded_df

def read_training_data(file_name):
    data = pd.read_csv(file_name)
    df = pd.DataFrame(data)
    excluded_col = ['PassengerId','Name']
    x_cols = [i for i in data.columns if i not in excluded_col]
    df = df[x_cols]
    # df = df.drop('PassengerId',axis=1)
    return df

def read_testing_data(file_name):
    data = pd.read_csv(file_name)
    df = pd.DataFrame(data)
    # df = df.drop('PassengerId', axis=1)
    print("df----------------\n",df)
    return df


def training(training_df):
    print("df:\n", training_df)
    y = training_df['Survived']
    x = training_df.drop('Survived', axis=1)
    # clf = tree.DecisionTreeClassifier()
    # clf = clf.fit(x, y)
    # clf = RandomForestClassifier(max_depth=3, random_state=0)
    # clf = clf.fit(x,y)
    clf = BaggingClassifier(tree.DecisionTreeClassifier(),max_samples = 0.5, max_features = 0.5).fit(x,y)
    # clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth = 1, random_state = 0).fit(x, y)
    return clf

def predict(clf,testing_df):
    excluded_col = ['PassengerId', 'Name']
    x_cols = [i for i in testing_df.columns if i not in excluded_col]
    print("testing_df['PassengerId'],\n",testing_df['PassengerId'],)
    predicted_y = clf.predict(testing_df[x_cols])
    result_df = pd.DataFrame({'PassengerId':testing_df['PassengerId'],'Survived':predicted_y})
    result_df.to_csv("data/gender_submission.csv",sep=',', index = False, header=True)
    print(result_df)

# def write_result_2_file(file_name,result_df):





training_file_name = "data/train.csv"
testing_file_name = "data/test.csv"
training_df = read_training_data(training_file_name)
testing_df = read_testing_data(testing_file_name)
training_df = pre_processing_data(training_df)
testing_df = pre_processing_data(testing_df)
clf = training(training_df)
predict(clf, testing_df)
