import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(r"house_price_prediction\data\raw data\train.csv")

df.drop(columns=["Id"],inplace=True)

df_num=df.select_dtypes(include=['int64',"float64"])
df_cat=df.select_dtypes(include=['object'])

#Correlation Heatmap of Numerical Features
plt.figure(figsize=(12,8))
sns.heatmap(df_num.corr(),cmap="coolwarm")
plt.title("Correlation Heatmap of Numerical Features")
plt.savefig("images/Correlation_heatmap.png")

#Distribution of SalePrice before and after Log Transformation
fig,ax=plt.subplots(1,2,figsize=(12,5))
sns.histplot(df['SalePrice'],ax=ax[0],kde=True)
ax[0].set_title("Distribution of SalePrice before Transformation")
sns.histplot(np.log1p(df['SalePrice']),ax=ax[1],kde=True)
ax[1].set_title("Distribution of SalePrice after Log Transformation")
plt.tight_layout()
plt.savefig("images/SalePrice_distribution.png")


#GrLivArea vs SalePrice
sns.scatterplot(x=df['GrLivArea'],y=df['SalePrice'])
plt.title("GrLivArea vs SalePrice")
plt.savefig("images/GrLivArea vs SalePrice.png")

#BsmtFinSF1 vs SalePrice
sns.scatterplot(x=df['BsmtFinSF1'],y=df['SalePrice'])
plt.title("BsmtFinSF1 vs SalePrice")
plt.savefig("images/BsmtFinSF1 vs SalePrice.png")

#TotalBsmtSF vs SalePrice
sns.scatterplot(x=df['TotalBsmtSF'],y=df['SalePrice'])
plt.title("TotalBsmtSF vs SalePrice")
plt.savefig("images/TotalBsmtSF vs SalePrice.png")

#GarageArea vs SalePrice
sns.scatterplot(x=df['GarageArea'],y=df['SalePrice'])
plt.title("GarageArea vs SalePrice")
plt.savefig("images/GarageArea vs SalePrice.png")

#pairplot of selected nunerical features
sns.pairplot(df[["SalePrice","GrLivArea","TotalBsmtSF","OverallQual"]])
plt.title("Pairplot of selected numerical features")
plt.savefig("images/Pairplot.png")