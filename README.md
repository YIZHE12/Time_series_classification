# ML and DL help robots to navigate

## Background
The goal of this project is to apply machine learning and deep learning methods to predict the surface typo that a robot is on based on time series data recorded by the robot's Inertial Measurement Units (IMU sensors).

## Data
Data can be downloaded in https://www.kaggle.com/c/career-con-2019/data.

The input data has 10 sensor channels of and 128 measurements per time series plus three ID columns:

-row_id: The ID for this row.

-series_id: ID number for the measurement series. Foreign key to y_train/sample_submission.

-measurement_number: Measurement number within the series.

<img src = images/raw_data.png height = 300>

For one example, the data looks like this:

<img src = images/exp.png height = 300>


## Solutions:
This is a time series classification problem. I investigated in two solutions: a machine leanring one and a deep learning one.

For the machine learning solution, first, I started by aggregating the time-series data and do feature engineering using the python package [tsfresh](https://tsfresh.readthedocs.io/en/latest/). It automatically generate a list of time series features, such as the following:

<img src = images/features.png height = 700>

For the complete list of avaliable features, please visit: https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html.

I then used a StratifiedKFold to seperate the data to train and validation set. The folds are made by preserving the percentage of samples for each class. I developed an automatic piplline to test different machine learning models, including knn, cart, svm, bayes, random forest, extra trees and gradient boosting models. 

<img src = images/models.png height = 400>

I eventually chose to use ExtraTreesClassifier based on accuracy and speed trade-off. I then used Grid Search Cross Validation for hyperparameters tuning.

<img src = images/tune.png height = 200>



Here, I demonstrated how to use tsfresh for time series data feature extraction, and use different machine learning methods based on the extracted features. 

I also used a 1D CNN + LSTM on the raw time series data without additonal steps in feature extracton. 

Both methods have similar performance. 
