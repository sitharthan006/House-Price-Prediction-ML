import pandas as pd 
import numpy as np 
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import pickle
from sklearn.model_selection import cross_val_score

df=pickle.load(open(r"data\processed_data\model_data.pkl","rb"))


#Feature Engineering
df["HouseAge"]=df["YrSold"]-df["YearBuilt"]
df["TotalBath"]=df["FullBath"]+0.5*df["HalfBath"]+df["BsmtFullBath"]+0.5*df["BsmtHalfBath"]

features = [
"OverallQual",
"GrLivArea",
"TotalBsmtSF",
"YearBuilt",
"TotalBath",
"HouseAge",
"LotArea",
"KitchenQual",
"Neighborhood"
]

x=df[features]
y=df["SalePrice"]

num_cols = [
"OverallQual",
"GrLivArea",
"TotalBsmtSF",
"YearBuilt",
"TotalBath",
"HouseAge",
"LotArea"
]

cat_cols=[
    "KitchenQual",
    "Neighborhood"
]

preprocessor=ColumnTransformer(
    transformers=[
        ("num",StandardScaler(),num_cols),
        ("cat",OneHotEncoder(handle_unknown="ignore"),cat_cols)
    ]
)

ridge_ui=Pipeline(steps=[
    ("preprocessor",preprocessor),
    ("model",Ridge(alpha=1.0))
])

lasso_ui=Pipeline(steps=[
    ("preprocessor",preprocessor),
    ("model",Lasso(alpha=0.01))
])

linear_ui=Pipeline(steps=[
    ("preprocessor",preprocessor),
    ("model",LinearRegression())
])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

ridge_ui.fit(x_train,y_train)

ridge_pred=ridge_ui.predict(x_test)

lasso_ui.fit(x_train,y_train)

lasso_pred=lasso_ui.predict(x_test)


linear_ui.fit(x_train,y_train)

linear_pred=linear_ui.predict(x_test)

pickle.dump(ridge_ui,open(r"models/ui_models/ridge_ui_model.pkl","wb"))
pickle.dump(lasso_ui,open(r"models/ui_models/lasso_ui_model.pkl","wb"))
pickle.dump(linear_ui,open(r"models/ui_models/linear_ui_model.pkl","wb"))

def metrics(y_test,y_pred,model_name):
    mse=mean_squared_error(y_test,y_pred)
    mae=mean_absolute_error(y_test,y_pred)
    r2=r2_score(y_test,y_pred)
    accuracy_score=r2_score(y_test,y_pred)
    print(f"{model_name} Metrics:")
    print(f"Mean Squared Error:{mse}")
    print(f"Mean Absolute Error:{mae}")
    print(f"R2 score:{r2}")
    print("Root Mean Squared Error",np.sqrt(mse))
    print(f"Accuracy Score : {accuracy_score}")

metrics(y_test,ridge_pred,"Ridge Regression")
metrics(y_test,lasso_pred,"Lasso Regression")
metrics(y_test,linear_pred,"Linear Regression")

def cross_val(model,x,y,name=None):
    scores=cross_val_score(model,x,y,cv=10,scoring="r2")
    print(f"cross_validation_score for {name}",scores.mean())

cross_val(model=ridge_ui,x=x_train,y=y_train,name="RidgeRegression")
cross_val(model=lasso_ui,x=x_train,y=y_train,name='LassoRegression')
cross_val(model=linear_ui,x=x_train,y=y_train,name="LinearRegression")