# House Price Prediction using Machine Learning

## Project Overview

This project predicts house prices using machine learning based on various features of residential properties.

The objective of this project is to build a regression model that can predict the **SalePrice** of a house using historical housing data.

This project demonstrates a complete machine learning pipeline including:

- Data Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training
- Model Evaluation
- Model Deployment using Streamlit

---

## Project Structure
HOUSE_PRICE_PREDICTION/
│
├── app/
│ └── streamlit_app.py
│
├── data/
│ ├── processed_data/
│ │ └── model_data.pkl
│ │
│ └── raw_data/
│ ├── data_description.txt
│ ├── sample_submission.csv
│ ├── test.csv
│ └── train.csv
│
├── images/
│ ├── BsmtFinSF1 vs SalePrice.png
│ ├── Correlation_heatmap.png
│ ├── GarageArea vs SalePrice.png
│ ├── GrLivArea vs SalePrice.png
│ ├── Pairplot.png
│ ├── SalePrice distribution.png
│ └── TotalBsmtSF vs SalePrice.png
│
├── models/
│ ├── LassoRegressionModel.pkl
│ ├── LinearRegressionModel.pkl
│ └── RidgeRegressionModel.pkl
│
├── src/
│ ├── datapreprocessing.py
│ ├── EDA.py
│ └── model_training.py
│
├── requirements.txt
└── README.md

---

## Dataset

This project uses the **Ames Housing Dataset** available through the Kaggle competition:

**House Prices: Advanced Regression Techniques**

The dataset contains detailed information about residential houses in **Ames, Iowa**, including structural and location-based features.

The dataset includes housing attributes such as:

- Lot Area
- Overall Quality
- Year Built
- Number of Rooms
- Garage Area
- Neighborhood
- Basement Area

The target variable used for prediction is:

- **SalePrice**

---

## Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed to understand the data distribution, identify patterns, and analyze relationships between features.

Key steps include:

- Checking missing values
- Distribution analysis
- Correlation analysis
- Outlier detection
- Feature relationship analysis

Libraries used for visualization:

- pandas
- matplotlib
- seaborn

Example visualizations include:

- SalePrice distribution
- Correlation heatmap
- Scatter plots between features and SalePrice
- Pairplot of important variables

---

## Data Preprocessing

The following preprocessing techniques were applied:

- Handling missing values
- Encoding categorical variables
- Feature scaling
- Feature engineering
- Removing irrelevant features

Tools used:

- StandardScaler
- OneHotEncoder
- OrdinalEncoder
- ColumnTransformer
- Pipeline

---

## Feature Engineering

Additional features were created to improve model performance, including:

- HouseAge
- GarageAge
- RemodAge
- TotalBath
- TotalBsmtBath
- TotalHouseSF
- TotalPorchSF

These engineered features help capture more meaningful information about the houses.

---

## Model Training

Several regression algorithms were used in this project to predict the **SalePrice** of houses.

The models were trained using the processed dataset to learn patterns between input features and the target variable.

Models used:

- Linear Regression
- Ridge Regression
- Lasso Regression

Steps involved:

- Train-test split
- Feature transformation using pipelines
- Model training
- Model comparison

Libraries used:

- scikit-learn
- numpy
- pandas

---

## Model Evaluation

The model performance was evaluated using regression metrics such as:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

These metrics help measure how accurately the model predicts house prices.

The model with the best performance was selected for deployment.

---

## Model Deployment

The final trained model was deployed using **Streamlit** to create an interactive web application.

Users can input house features and receive predicted house prices from the trained machine learning model.
