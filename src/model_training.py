import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder
import pickle
from sklearn.linear_model import LinearRegression,Ridge,Lasso 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import joblib
from sklearn.model_selection import cross_val_score

df=pd.read_pickle('model_data.pkl')

x=df.drop(columns=["SalePrice"])
y=df["SalePrice"]

# split the data for training and testing 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#columns_name based on data types
num_cols=df.select_dtypes(include=["int64","float64"]).columns.tolist()
cat_cols=df.select_dtypes(include=["object"]).columns.tolist()

#remove target variable from num_cols
num_cols.remove("SalePrice")

#sperate nominal and ordinal categorical columns
nom_cat_cols=['MSZoning', 'Street', 'Alley','Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd','MasVnrType','Foundation','Heating','Electrical','GarageType','SaleType','SaleCondition','MiscFeature']
ord_cat_cols=[col for col in cat_cols if col not in nom_cat_cols]

#values of ordinal categorical columns
ord_cols_val = [
['IR3','IR2','IR1','Reg'],                      # LotShape
['Low','HLS','Bnk','Lvl'],                      # LandContour
['NoSeWa','AllPub'],                            # Utilities
['Inside','Corner','FR2','FR3','CulDSac'],      # LotConfig
['Sev','Mod','Gtl'],                            # LandSlope

['Po','Fa','TA','Gd','Ex'],                     # ExterQual
['Po','Fa','TA','Gd','Ex'],                     # ExterCond

['None','Po','Fa','TA','Gd','Ex'],              # BsmtQual
['None','Po','Fa','TA','Gd','Ex'],              # BsmtCond
['None','No','Mn','Av','Gd'],                   # BsmtExposure

['None','Unf','LwQ','Rec','BLQ','ALQ','GLQ'],   # BsmtFinType1
['None','Unf','LwQ','Rec','BLQ','ALQ','GLQ'],   # BsmtFinType2

['Po','Fa','TA','Gd','Ex'],                     # HeatingQC
['N','Y'],                                      # CentralAir

['Po','Fa','TA','Gd','Ex'],                     # KitchenQual
['Sal','Sev','Maj2','Maj1','Mod','Min2','Min1','Typ'],  # Functional

['None','Po','Fa','TA','Gd','Ex'],              # FireplaceQu
['None','Unf','RFn','Fin'],                     # GarageFinish

['None','Po','Fa','TA','Gd','Ex'],              # GarageQual
['None','Po','Fa','TA','Gd','Ex'],              # GarageCond

['N','P','Y'],                                  # PavedDrive
['None','Fa','TA','Gd','Ex'],                   # PoolQC
['None','MnWw','GdWo','MnPrv','GdPrv']          # Fence
]

preprocessor=ColumnTransformer(transformers=[
    ("standard_scaler",StandardScaler(),num_cols),
    ("Nominal_encoder",OneHotEncoder(sparse_output=False,handle_unknown="ignore"),nom_cat_cols),
    ("Ordinal_encoder",OrdinalEncoder(categories=ord_cols_val,handle_unknown="use_encoded_value",unknown_value=-1),ord_cat_cols)
],remainder="passthrough")

lr=LinearRegression()
las=Lasso(alpha=0.01)
rid=Ridge(alpha=0.1)

lin_reg=Pipeline(steps=[
    ("preprocessor",preprocessor),
    ("model",lr)
])

lin_reg.fit(x_train,y_train)

lin_pred=lin_reg.predict(x_test)

las_reg=Pipeline(steps=[
    ("preprocessor",preprocessor),
    ("model",las)
])

las_reg.fit(x_train,y_train)

las_pred=las_reg.predict(x_test)

rid_reg=Pipeline(steps=[
    ("preprocessor",preprocessor),
    ("model",rid)
])

rid_reg.fit(x_train,y_train)

rid_pred=rid_reg.predict(x_test)



def metrics(model_name,y_test,y_pred):
    print(f"Metrics for {model_name}:")
    print("R2_score:",r2_score(y_test,y_pred))
    print("Mean Absolute Error:",mean_absolute_error(y_test,y_pred))
    print("Mean Squared Error:",mean_squared_error(y_test,y_pred))
    print("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,y_pred)))

metrics("Linear Regression",y_test,lin_pred)
metrics("Lasso Regression",y_test,las_pred)
metrics("Ridge Regression",y_test,rid_pred)


joblib.dump(lin_reg,"models/LinearRegressionModel.pkl")
joblib.dump(las_reg,"models/LassoRegressionModel.pkl")
joblib.dump(rid_reg,"models/RidgeRegressionModel.pkl")

#cross validation score

def cross_val(model,x,y,name=None):
    scores=cross_val_score(model,x,y,cv=10,scoring="r2")
    print(f"cross_validation_score for {name}",scores.mean())


cross_val(model=lin_reg,x=x_train,y=y_train,name="LinearRegression")
cross_val(model=las_reg,x=x_train,y=y_train,name='LassoRegression')
cross_val(model=rid_reg,x=x_train,y=y_train,name="RidgeRegression")