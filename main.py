#Author: Salil Shenoy

"""
    Dataset: http://archive.ics.uci.edu/ml/datasets/Adult
"""
#Imports
import pandas as pd
import numpy as np
import sklearn as skl
import sklearn.preprocessing as preprocessing
import sklearn.linear_model as linear_model
import sklearn.cross_validation as cross_validation
import sklearn.metrics as metrics
import seaborn as sns
import matplotlib.pyplot as plt
import math
sns.set(style="ticks", color_codes=True)

'''
VERSION 2: Updated the code as some calls were failing because of update in 
sklearn module after submission of Version 1
'''

'''
    Plots a histogram for the original data
'''
def plotData(original_data):
    fig = plt.figure(figsize=(20,15))
    cols = 5
    rows = math.ceil(float(original_data.shape[1]) / cols)
    for i, column in enumerate(original_data.columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)
        if original_data.dtypes[column] == np.object:
            original_data[column].value_counts().plot(kind='bar', axes=ax)
        else:
            original_data[column].hist(axes=ax)
        plt.xticks(rotation='vertical')
    plt.subplots_adjust(hspace=0.7, wspace=0.2)
    plt.show()


'''
    Print the country wise percentage of data
'''
def sepDataCountrywise(original_data):
    print (original_data["Country"].value_counts() / original_data.shape[0]).head()
    
'''
    Encoding the data
'''    
def DataEncoder(data):
    result = data.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            encoders[column] = preprocessing.LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders


'''
    The data type of different columns in the dataset
'''
def datatype(df):
    print df.dtypes
    return


'''
    Function to clean the dataset
'''
def CleanDataset(data):
     data = data.replace(r"\s*[?]\s*", np.NAN, regex=True)
     data = data.dropna()
     return data
     
     
'''
    Plots the Correlation Matrix
'''
def HeatMap(data):
    sns.heatmap(data.corr(),annot=True, fmt=".2g", square=True)
    plt.yticks(rotation=0)
    plt.xticks(rotation=80)
    plt.show()
    return    
    
    
'''
    The function prints null values in the dataset. But it does not account 
    for any fillers 
'''
def PrintMissingData():
    df = ReadDataset()
    null_data = df[df.isnull().any(axis=1)]
    print null_data
    
    
'''
    Read the data into a pandas dataframe. 
'''
def ReadDataset():
    df = pd.read_csv('data/adult.txt') 
    return df
    
    
'''
    This function uses the encoded data dummy for classification.
'''
def SplitScaleFeatures(data):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            data[data.columns.difference(['Target'])], 
            data['Target'], 
            train_size=0.70)
    scaler = preprocessing.StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train.astype("float64")),
                            columns=X_train.columns)
    X_test = scaler.transform(X_test.astype("float64"))
    
    cls = linear_model.LogisticRegression()
    cls.fit(X_train, y_train)
    y_pred = cls.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12,12))
    plt.subplot(2,1,1)
    sns.heatmap(cm, annot=True, fmt='.2g', xticklabels=data['Target'].values, 
                                            yticklabels=data['Target'].values)
    plt.ylabel('Real value')
    plt.xlabel('Predicted value')
    print "F1 score: %f" % skl.metrics.f1_score(y_test, y_pred)
    print 'Precision:  %f' % skl.metrics.precision_score(y_test, y_pred)
    print 'Recall:  %f' % skl.metrics.recall_score(y_test, y_pred)
    coefs = pd.Series(cls.coef_[0], index=X_train.columns)
    coefs.sort_values()
    plt.subplot(2,1,2)
    coefs.plot(kind='bar')
    plt.show()
    
    
'''
    This function tranforms the data into binary using the dummy function and 
    then this transformed data is used for classification.
'''
def Classify(original_data, encode_data):
    binary_data = pd.get_dummies(original_data)
    binary_data['Target'] = binary_data['Target_ >50K']
    del binary_data['Target_ <=50K']
    del binary_data['Target_ >50K']
    #plt.subplots(figsize=(20,20))
    
    #This heatmap does not get printed clearly as there are too many classes
    #HeatMap(binary_data) 
    
    '''
    This works in the previous version of Pandas
        X_train, X_test, y_train, y_test = cross_validation.train_test_split
        (binary_data[binary_data.columns - ['Target']], 
        binary_data['Target'], train_size=0.70)
    '''
    '''
        HAD TO CHANGE THIS LINE TO USE THE difference() FUNCTION WHICH WAS 
        INTRODUCED IN THE NEW VERSION OF PANDAS   
    '''
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
         binary_data[binary_data.columns.difference(['Target'])], 
         binary_data['Target'], 
         train_size=0.70)
    scaler = preprocessing.StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = scaler.transform(X_test)

    cls = linear_model.LogisticRegression()

    cls.fit(X_train, y_train)
    y_pred = cls.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(20,20))
    plt.subplot(2,1,1)
    #sns.heatmap(cm, annot=True, fmt=".2g", xticklabels=encode_data['Target'].classes_, yticklabels=encode_data['Target'].classes_)
    sns.heatmap(cm, annot=True, fmt=".2g", 
                xticklabels=encode_data['Target'].values, 
                yticklabels=encode_data['Target'].values)
    plt.ylabel('Real value')
    plt.xlabel('Predicted value')
    print 'F1 score: %f' % skl.metrics.f1_score(y_test, y_pred)
    print 'Precision:  %f' % skl.metrics.precision_score(y_test, y_pred)
    print 'Recall:  %f' % skl.metrics.recall_score(y_test, y_pred)
    coefs = pd.Series(cls.coef_[0], index=X_train.columns)
    coefs.sort_values()
    plt.subplot(2,1,2)
    coefs.plot(kind="bar")
    plt.show()

 
def main():
    #Assign Column Names
    columnNames=[
        'Age', 'Workclass', 'fnlwgt', 'Education', 'Education-Num', 
        'Martial Status','Occupation', 'Relationship', 'Race', 'Sex', 
        'Capital Gain', 'Capital Loss',
        'Hours per week', 'Country', 'Target']
        
    #Read the data
    print 'Calling ReadDataSet...'
    data = ReadDataset()
    
    #Tried to print out missing data
    print 'Now, checking for missing rows...'
    PrintMissingData()
         
    #Add the column names to the dataframe
    data.columns = columnNames
    
    #Cleans the data of missing values
    print 'Cleaning the dataset...'
    data = CleanDataset(data)
    
    #Plot for individual feature historgram
    plotData(data)
    
    #Correlation Matrix
    if False:
        HeatMap(data)
        
    '''
        After observing the heat map the conclusion is we can delete the 
        education column
    '''
    del data['Education']
    
    '''
        Now, we encode the data
    '''
    encode_data, encoder = DataEncoder(data)
    
    #Prints the country wise ditribution of the dataset after cleaning
    print 'Printing the data seperated Country wise...'
    sepDataCountrywise(data)
       
    #Histogram for the encoded data
    plotData(encode_data)
    
    bClassifyBinary = False
    
    #Observe the F Scores, Precision, Recall Scores in this
    if bClassifyBinary:
        print 'Scaling Features'
        SplitScaleFeatures(encode_data)
    else:
        print 'Binary Data'
        Classify(data, encode_data)
    
if __name__ == '__main__':
    main()