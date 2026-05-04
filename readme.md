# House Price Prediction using Machine Learning
## Project Overview
This project predict the house price using machine learning based on various features of the residental properties

The objective of this project is to build a regression model that can predict the "saleprice" of the house using the historical dataset

This project demonstrates a complete machine learning pipeline including :
-Data preprocessing
-Exploratory Data Analysis (EDA)
-Feature Engineering
-Model Training
-Model Evaluation
-Model Deployment (Streamlit App)

## Project Structure
HOUSE_PRICE_PREDICTION/
│
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── processed_data/
│   │   └── model_data.pkl
│   │
│   └── raw_data/
│       ├── data_description.txt
│       ├── sample_submission.csv
│       ├── test.csv
│       └── train.csv
│
├── images/
│   ├── BsmtFinSF1 vs SalePrice.png
│   ├── Correlation_heatmap.png
│   ├── GarageArea vs SalePrice.png
│   ├── GrLivArea vs SalePrice.png
│   ├── Pairplot.png
│   ├── SalePrice distribution.png
│   └── TotalBsmtSF vs SalePrice.png
│
├── models/
│   ├── LassoRegressionModel.pkl
│   ├── LinearRegressionModel.pkl
│   └── RidgeRegressionModel.pkl
│
├── src/
│   ├── datapreprocessing.py
│   ├── EDA.py
│   └── model_training.py
│
├── requirements.txt
└── README.md


## Dataset

This project uses Ames dataset available in the Kaggle competition;
*HousePrice : Advaced Regression Techniques*

The dataset contains detailed information about the house in Ames , Iwoa , including the structural and location-based features

The dataset includes the housing attributes such as :

-Lot Area
-Overall Quality
-Year Built
-Number of Rooms
-Garage Area
-Neighborhood
-Basement Area
-Sale Price

The target variable is:

-SalePrice

## Exploratory Data Analysis

Exploratory Data Analysis is performed to understand the data distribution ,recognise the pattern in the data and identify the relationships between the features.

Key steps include:

-Checking missing values
-Distribution analysis
-Correlation analysis
-Outlier detection
-Feature relationships

Libraries used for visualization:

-pandas
-matplotlib
-seaborn


## Data Preprocessing
The following preprocessing techniques were applied:

-Handling missing values
-Encoding categorical variables
-Feature scaling
-Removing irrelevant features

Tools used:

-StandardScaler
-OneHotEncoder
-ColumnTransformer

## Model Training

Several regression algorithms were used in this project to Predict house SalePrice.
The models were trained using the processed dataset to learn the patterns between input features and target variable.

Steps involved:

-Train-test split
-Feature transformation
-Model training
-Model evaluation

Libraries used:

-scikit-learn
-numpy
-pandas

## Model Evaluation

The model performance was evaluated using regression metrics such as:

-Mean Absolute Error (MAE)
-Mean Squared Error (MSE)
-Root Mean Squared Error (RMSE)
-R² Score

These metrics help measure how accurately the model predicts house prices.
The model with high accuracy is selected

## Streamlit Deployment

The final trained model was deployed using streamlit to create an interactive web applications where users can input house features and recieve predicted house prices.