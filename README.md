# 255_project_group_6


Motor Vehicle Collision Analysis and Prediction
This project analyzes motor vehicle collision data to identify trends, contributing factors, and predict collision fatalities.

Table of Contents
Description
Setup
Usage
Data
Sample Data
Analysis Details
Model Details
Results
Description
This project performs an exploratory data analysis (EDA) and builds predictive models on a dataset of motor vehicle collisions. Key analyses include identifying top accident locations (zip codes, boroughs), analyzing accident severity by number of injured persons, determining common contributing factors, and exploring time-based trends. Finally, it compares several regression models for predicting the number of persons killed in a collision.

Setup
To run this project, you will need Python and the following libraries:

pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
You can install these libraries using pip:

Bash

pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm
The provided code snippets assume a Google Colab environment for data loading directly from Google Drive. If you are running this locally, you will need to adjust the data loading path (file_path) to point to the location of your CSV file on your local machine.

For Google Colab:
Ensure your Google Drive is mounted and the CSV file is located at the specified file_path:

Python

from google.colab import drive
drive.mount('/content/drive')
file_path = '/content/drive/My Drive/Motor_Vehicle_Collisions_-_Crashes_20250503.csv'
For Local Execution:
Change the file_path to your local file path:

Python

file_path = 'path/to/your/Motor_Vehicle_Collisions_-_Crashes_20250503.csv'
Usage
The project is structured as a series of Python code snippets. To run the full analysis and modeling process, you would typically execute these snippets sequentially in a Python environment (like a Jupyter Notebook, Google Colab, or a Python script).

Data Loading: Run the initial code to load the dataset into a pandas DataFrame named df.
Exploratory Data Analysis (EDA): Execute the snippets for:
Top 5 Accident Zip Codes
Top 5 Accident Boroughs
Accidents by Number of Persons Injured
Analysis of Top Contributing Factors
Time Series Analysis
Model Comparison: Run the snippet for the Fatality Prediction Model comparison. This will train and evaluate the specified regression models.
Results Visualization: Run the snippet to plot the RMSE scores of the models.
Data
The primary dataset used in this project is the Motor Vehicle Collisions - Crashes dataset. The code assumes a file named Motor_Vehicle_Collisions_-_Crashes_20250503.csv. This dataset is typically sourced 1  from open data portals like NYC Open Data.   
 1. 
github.com
github.com

Full Dataset Source (Example - verify the exact source and date for your data):
NYC Open Data: https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95

Sample Data
Below is a small sample of the dataset structure, including the columns used in the analysis and modeling. This sample is synthetic and provided for illustrative purposes and basic testing of the code structure. It does not represent actual collision data.

Code snippet

CRASH DATE,CRASH TIME,BOROUGH,ZIP CODE,LATITUDE,LONGITUDE,ON STREET NAME,CROSS STREET NAME,OFF STREET NAME,NUMBER OF PERSONS INJURED,NUMBER OF PERSONS KILLED,NUMBER OF PEDESTRIANS INJURED,NUMBER OF CYCLIST INJURED,NUMBER OF MOTORIST INJURED,NUMBER OF PEDESTRIANS KILLED,NUMBER OF CYCLIST KILLED,NUMBER OF MOTORIST KILLED,CONTRIBUTING FACTOR VEHICLE 1,CONTRIBUTING FACTOR VEHICLE 2,CONTRIBUTING FACTOR VEHICLE 3,CONTRIBUTING FACTOR VEHICLE 4,CONTRIBUTING FACTOR VEHICLE 5,VEHICLE TYPE CODE 1,VEHICLE TYPE CODE 2,VEHICLE TYPE CODE 3,VEHICLE TYPE CODE 4,VEHICLE TYPE CODE 5
01/01/2023,12:00,BROOKLYN,11201,40.7,-74.0,ATLANTIC AVE,FLATBUSH AVE,,1,0,1,0,0,0,0,0,Driver Inattention/Distraction,Unspecified,,,,PASSENGER VEHICLE,SEDAN,,,,
01/01/2023,14:30,MANHATTAN,10001,40.75,-73.99,8TH AVE,W 34 ST,,0,0,0,0,0,0,0,0,Following Too Closely,Unspecified,,,,PASSENGER VEHICLE,TAXI,,,,
01/02/2023,08:15,QUEENS,11101,40.76,-73.93,QUEENS BLVD,31 ST,,2,0,0,0,2,0,0,0,Failure to Yield Right-of-Way,Unspecified,,,,SUV,PASSENGER VEHICLE,,,,
01/02/2023,18:00,BRONX,10451,40.82,-73.92,GRAND CONCOURSE,E 161 ST,,0,1,0,0,0,0,0,1,Driver Inattention/Distraction,Unspecified,,,,PASSENGER VEHICLE,UNKNOWN,,,,
01/03/2023,09:00,STATEN ISLAND,10301,40.64,-74.08,VICTORY BLVD,BAY ST,,1,0,1,0,0,0,0,0,Pedestrian/Bicyclist/Other Right-of-Way,Unspecified,,,,PASSENGER VEHICLE,,,,
01/03/2023,22:00,BROOKLYN,11215,40.66,-73.98,PROSPECT PARK WEST,UNION ST,,0,0,0,0,0,0,0,0,Passing or Lane Usage Improper,Unspecified,,,,BICYCLE,PASSENGER VEHICLE,,,,
01/04/2023,11:30,MANHATTAN,10019,40.76,-73.98,BROADWAY,W 57 ST,,1,0,0,1,0,0,0,0,Unsafe Speed,Unspecified,,,,MOTORCYCLE,TAXI,,,,
01/04/2023,16:45,QUEENS,11377,40.75,-73.89,NORTHERN BLVD,82 ST,,0,0,0,0,0,0,0,0,Backing Unsafely,Unspecified,,,,VAN,PASSENGER VEHICLE,,,,
01/05/2023,07:00,BRONX,10458,40.86,-73.90,JEROME AVE,E 183 ST,,1,0,0,0,1,0,0,0,Driver Inattention/Distraction,Unspecified,,,,BUS,PASSENGER VEHICLE,,,,
01/05/2023,19:00,STATEN ISLAND,10314,40.59,-74.15,RICHMOND AVE,VICTORY BLVD,,0,0,0,0,0,0,0,0,Following Too Closely,Unspecified,,,,PASSENGER VEHICLE,SUV,,,,
Analysis Details
The project includes the following analyses:

Top 5 Accident Zip Codes: Identifies zip codes with the highest collision counts.
Top 5 Accident Boroughs: Identifies boroughs with the highest collision counts.
Accidents by Number of Persons Injured: Visualizes the frequency of accidents based on the number of injured persons (0-6).
Analysis of Top Contributing Factors: Determines the most frequent contributing factors involved in collisions, excluding generic categories like 'Unspecified' or 'Unknown'.
Time Series Analysis: Explores collision trends over years, months, and weeks, and by day of the week.
Model Details
A regression task is performed to predict the NUMBER OF PERSONS KILLED. The following machine learning models are compared:

XGBoost Regressor (using hist tree method and attempting GPU acceleration)
LightGBM Regressor (attempting GPU acceleration)
HistGradientBoosting Regressor
Linear Regression
Ridge Regression
The models are trained and evaluated using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R 
2) metrics.

Preprocessing Steps:

Numerical features are imputed with the median and scaled using StandardScaler.
Categorical features are imputed with a constant value ('missing') and one-hot encoded using OneHotEncoder. sparse_output=True is used for memory efficiency.
Results
The results of the model comparison, including RMSE scores and training times, are printed to the console. A bar plot is also generated to visually compare the RMSE scores of the successfully trained models. The model with the lowest RMSE is generally considered the best performing model for this prediction task.
