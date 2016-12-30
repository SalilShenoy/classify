two types of classification used

F1 Score calculated on encoded data = 0.57
After the data was transformed into binary using dummy values the F1 scored 
improved to F1 score: 0.659046

You Can find the Plots in the Visuals Folder
Plots
Histogram of orignal data, encoded data
Heatmap

Toggle the bClassifyBinary flag to go between methods

*******************************************************************************
*******************************************************************************
********************************SUBMISSION 2***********************************
*******************************************************************************
*******************************************************************************
THE ERROR IN THE PREVIOUS SUBMISSION WAS DUE TO SKLEARN UPDATE. 

The latest submission has 
F1 Score
Precision
Recall

The heatmap plotted after the data is encoded has too many classes and hence the
plot is not clear. I have commented that plot.
The remaining plot are present in the visuals folder. 

OUTPUT:
Index: []
This line is printed as there is no missing data (no nans)(Fnc: PrintMissingData)
The data has '?' in place of missin values. The (Fnc: CleanMissingData) takes
care of such data by skipping such rows.
The (Fnc: sepDataCountrywise) prints the how the data is split country wise

Binary Data
F1 score: 0.658718
Precision:  0.726575
Recall:  0.602453

Scaling Features
F1 score: 0.559501
Precision:  0.705882
Recall:  0.463404

STEPS TO RUN
Download the zip file Assignment12fullpipeline.zip
Unzip the Assignment12fullpipeline.zip
Run the main.py 

Folder Structure 
Assignment 12 full pipeline
---> main.py
---> data -> adult.txt
---> visuals - { results }