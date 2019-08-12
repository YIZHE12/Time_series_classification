# ML and DL help robots to navigate

## Background
he goal of this project is to apply machine learning and deep learning methods to predict the surface typo that a robot is on based on time series data recorded by the robot's Inertial Measurement Units (IMU sensors).

## Data
Data can be downloaded in https://www.kaggle.com/c/career-con-2019/data.

The input data has 10 sensor channels of and 128 measurements per time series plus three ID columns:

-row_id: The ID for this row.

-series_id: ID number for the measurement series. Foreign key to y_train/sample_submission.

-measurement_number: Measurement number within the series.

## EDA

## Solutions:





This is a time series classification problem.
Here, I demonstrated how to use tsfresh for time series data feature extraction, and use different machine learning methods based on the extracted features. 

I also used a 1D CNN + LSTM on the raw time series data without additonal steps in feature extracton. 

Both methods have similar performance. 
