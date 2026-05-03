import pandas as pd
import numpy as np

import pickle



#  data preprocessing
df=pd.read_csv('data/train.csv')


# 1. Handle missing values
df.drop(columns=['Id'],inplace=True)
df["PoolQC"]=df["PoolQC"].fillna("None")
df["MiscFeature"]=df['MiscFeature'].fillna("None")
df["Alley"]=df["Alley"].fillna("None")
df["Fence"]=df["Fence"].fillna("None")
df["MasVnrType"]=df["MasVnrType"].fillna("None")
df["FireplaceQu"]=df["FireplaceQu"].fillna("None")
df["LotFrontage"]=df['LotFrontage'].fillna(df["LotFrontage"].median())
df['GarageType']=df["GarageType"].fillna("None")
df['GarageFinish']=df["GarageFinish"].fillna("None")
df['GarageQual']=df["GarageQual"].fillna("None")
df['GarageCond']=df["GarageCond"].fillna("None")
df['BsmtExposure']=df["BsmtExposure"].fillna("None")
df['BsmtFinType2']=df["BsmtFinType2"].fillna("None")
df['BsmtFinType1']=df["BsmtFinType1"].fillna("None")
df['BsmtCond']=df["BsmtCond"].fillna("None")
df['BsmtQual']=df["BsmtQual"].fillna("None")
df["GarageYrBlt"]=df["GarageYrBlt"].fillna(0)
df['MasVnrArea']=df["MasVnrArea"].fillna(0)
df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])

#Feature construction
df["HouseAge"]=df["YrSold"]-df["YearBuilt"]
df["TotalPorchSF"] = df[["WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch"]].sum(axis=1)
df["GarageAge"]=df["YrSold"]-df["GarageYrBlt"]
df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
df["TotalBsmtBath"] = df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
df["TotalBath"] = (df["FullBath"]+ 0.5 * df["HalfBath"]+ df["BsmtFullBath"]+ 0.5 * df["BsmtHalfBath"])
df["TotalBsmtFinSF"] = df["BsmtFinSF1"] + df["BsmtFinSF2"]
df["TotalHouseSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]



#remove outliers
df=df[~(((df['GrLivArea']>4000)&(df['SalePrice']<300000))|((df['GrLivArea']<5000)&(df['SalePrice']>600000)))]


df.to_pickle("model_data.pkl")