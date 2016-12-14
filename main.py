#Author: Salil Shenoy

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def DataVisualization():
    data = ReadDataset()
    g = sns.heatmap(data,annot=True, fmt="d", linewidths=.5)
    plt.show()
    return
    
    
def PrintMissingData():
    df = ReadDataset()
    null_data = df[df.isnull().any(axis=1)]
    print null_data
    
    
def ReadDataset():
    df = pd.read_csv('data/adult.txt', header = None) 
    return df
    
    
def main():
    DataVisualization()
    #PrintMissingData()
    #ReadDataset()
    return
    
if __name__ == '__main__':
    main()