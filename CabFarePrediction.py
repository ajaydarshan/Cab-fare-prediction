# coding: utf-8
import os
import sys
if(len(sys.argv) != 3):
    print("Usage : python CabFarePrediction.py path\\to\\train_cab.csv path\\to\\test.csv")
    sys.exit(1)
train_file = sys.argv[1]
test_file = sys.argv[2]
os.chdir(os.curdir)
os.getcwd()

import numpy as np
import pandas as pd
from datetime import datetime
from fancyimpute import KNN
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv(train_file)
#Load test set and apply same feature engineering techniques parallely
test = pd.read_csv(test_file)
#save pickup_datetime of test for results
test_pickup_datetime = test["pickup_datetime"]
df.head()
test.head()
test.dtypes
df.shape
df.info()
df.describe()
df.dtypes
df.isnull().sum()
# # Exploratory Analysis
#Convert fare_amount from object to numeric
df["fare_amount"] = pd.to_numeric(df["fare_amount"],errors = "coerce")
df.describe()
#remove rows having fractional passenger count and 0 count and greater than 10 passengers
df = df[df["passenger_count"] > 0]
df = df[df["passenger_count"] <= 10]
df = df[df["passenger_count"] % 1 == 0]
df.shape
df.isnull().sum()
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"],errors = "coerce")
#Apply same for test set
test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"],errors = "coerce")
df.head()
df.dtypes
df["pickup_hour"] = pd.Categorical(df["pickup_datetime"].dt.strftime("%H"))
df["pickup_month"] = pd.Categorical(df["pickup_datetime"].dt.strftime("%m"))
df["pickup_weekday"] = pd.Categorical(df["pickup_datetime"].dt.strftime("%u"))
df["pickup_year"] = pd.Categorical(df["pickup_datetime"].dt.strftime("%Y"))
#Add same features to test
test["pickup_hour"] = pd.Categorical(test["pickup_datetime"].dt.strftime("%H"))
test["pickup_month"] = pd.Categorical(test["pickup_datetime"].dt.strftime("%m"))
test["pickup_weekday"] = pd.Categorical(test["pickup_datetime"].dt.strftime("%u"))
test["pickup_year"] = pd.Categorical(test["pickup_datetime"].dt.strftime("%Y"))
df.head()
df.describe()
#drop pickup_datetime
df.drop(["pickup_datetime"],axis = 1,inplace = True)
#drop from test set
test.drop(["pickup_datetime"],axis = 1,inplace = True)
test.isnull().sum()
df.isnull().sum()
df = df.reset_index(drop=True)


# # Missing value analysis

missing_values = pd.DataFrame(df.isnull().sum()).reset_index()
missing_values
missing_values = missing_values.rename(columns = {"index":"Variables",0:"Missing_values"})
missing_values = missing_values.sort_values("Missing_values",ascending = False)
missing_values
#df['fare_amount'][0] = np.nan
#Impute with median
#df['fare_amount'] = df['fare_amount'].fillna(df['fare_amount'].median())
#df['fare_amount'][0]  #4.5
df.describe()
#replace NaT with NA to impute
for i in range(0, df.shape[1]):
    df.iloc[:,i] = df.iloc[:,i].replace("NaT", np.nan) 
#Apply KNN imputation algorithm
df = pd.DataFrame(KNN(k = 2).fit_transform(df), columns = df.columns)
#remove rows having fare amount -ve
df = df[df["fare_amount"] > 0.0]
df.head()
df.isnull().sum()
df.describe()


# # Feature engineering

def haversine(long1,lat1,long2,lat2):
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    delphi = np.deg2rad(lat2 - lat1)
    dellamda = np.deg2rad(long2 - long1)
  
    a = np.sin(delphi/2) * np.sin(delphi/2) + np.cos(phi1) * np.cos(phi2) * np.sin(dellamda/2) * np.sin(dellamda/2)
  
    c = 2 * np.arctan2(np.sqrt(a),np.sqrt(1-a))
    R = 6371
    return R * c
df["dist"] = haversine(df["pickup_longitude"],df["pickup_latitude"],df["dropoff_longitude"],df["dropoff_latitude"])
#Add for test set
test["dist"] = haversine(test["pickup_longitude"],test["pickup_latitude"],test["dropoff_longitude"],test["dropoff_latitude"])
df.head()
df.drop(["pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"],axis = 1,inplace = True)
#drop from test set
test.drop(["pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"],axis = 1,inplace = True)
df.head()
df = df.iloc[:,[1,2,3,4,5,6,0]]
df.head()
df.describe()
#remove rows having distance 0
df = df[df["dist"] != 0]
df.shape
df.describe()


# # Outlier analysis

cnames = ["dist","fare_amount"]
q25 = np.percentile(df["dist"],25)
fig,ax = plt.subplots(2,figsize = (10,10))
for i in range(0,len(cnames)):
    print(sns.boxplot(y=cnames[i],x="passenger_count",data=df,ax=ax[i]))
for i in cnames:
    q75 = np.percentile(df[i],75)
    q25 = np.percentile(df[i],25)
    iqr = q75 - q25
    maximum = q75 + (1.5 * iqr)
    minimum = q25 - (1.5 * iqr)
    
    df.loc[df[i] > maximum,i] = np.NaN
    df.loc[df[i] < minimum,i] = np.NAN
#missing values
df.isnull().sum()
df = pd.DataFrame(KNN(k=2).fit_transform(df),columns = df.columns)
print(plt.hist(df.fare_amount))


# # Feature selection

df_corr = df.loc[:,cnames]
#Set the width and height of the plot
f, ax = plt.subplots(figsize=(12, 7))
#Generate correlation matrix
corr = df_corr.corr()
#Plot using seaborn library
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap='coolwarm',square = True,linewidths = 1,ax=ax,annot =True)

#ANOVA
import statsmodels.api as sm
from statsmodels.formula.api import ols
 
model = ols('fare_amount ~ C(pickup_hour) + C(pickup_weekday) + C(pickup_month) + C(pickup_year) + C(passenger_count)',
                data=df).fit()
                
aov_table = sm.stats.anova_lm(model)
aov_table
#aov_table.summary()
#remove pickup_weekday
df.drop(["pickup_weekday"],axis = 1,inplace = True)
#drop from test set
test.drop(["pickup_weekday"],axis = 1,inplace = True)
test.head()

# # Sampling

from sklearn.model_selection import train_test_split
X = df.values[:,:-1]
Y = df.values[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,Y,train_size = 0.8,random_state = 1000)
# # Model Selection

# import Error metric
from sklearn import metrics
def mape(y_test,pred):
    return np.mean(np.abs((y_test-pred)/y_test))
def rms(ytrue,yhat):
    mse = metrics.mean_squared_error(ytrue,yhat)
    return np.sqrt(mse)
	

# ## Linear regression

from sklearn.linear_model import LinearRegression
model_lm = LinearRegression()
model_lm.fit(X_train,y_train)
pred_lm = model_lm.predict(X_test)
plt.scatter(y_test,pred_lm)
sns.distplot(y_test-pred_lm)
mse = metrics.mean_squared_error(y_test,pred_lm)
rms_lm = np.sqrt(mse)
rms_lm
mape_lm = mape(y_test,pred_lm)
mape_lm
# ## Random forest

from sklearn.ensemble import RandomForestRegressor
model_rf = RandomForestRegressor(n_estimators = 500)
model_rf.fit(X_train,y_train)
pred_rf = model_rf.predict(X_test)
plt.scatter(y_test,pred_rf)
sns.distplot((y_test - pred_rf))
mse = metrics.mean_squared_error(y_test,pred_rf)
rms_rf = np.sqrt(mse)
rms_rf
mape_rf = mape(y_test,pred_rf)
mape_rf


# ## Boosting
# ### Gradient boosting

from sklearn.ensemble import GradientBoostingRegressor
model_gbm = GradientBoostingRegressor(n_estimators = 120)
model_gbm.fit(X_train,y_train)
pred_gbm = model_gbm.predict(X_test)
plt.scatter(y_test,pred_gbm)
sns.distplot(y_test-pred_gbm)
mse = metrics.mean_squared_error(y_test,pred_gbm)
rms_gbm = np.sqrt(mse)
rms_gbm
mape_gbm = mape(y_test,pred_gbm)
mape_gbm


# ### XGBoost

from xgboost import XGBRegressor
model_xgb = XGBRegressor()
model_xgb.fit(X_train,y_train)
pred_xgb = model_xgb.predict(X_test)
plt.scatter(y_test,pred_xgb)
sns.distplot(y_test-pred_xgb)
rms_xgb = rms(y_test,pred_xgb)
rms_xgb
mape_xgb = mape(y_test,pred_xgb)
mape_xgb
rms_xgb,rms_gbm,rms_rf,rms_lm
mape_xgb,mape_gbm,mape_rf


# ## Prediction on test set
# As GBM and XGBoost produces better results than other models. We will train whole train data and predict on test data
# ## Apply XGBoost

model_xgb2 = XGBRegressor()
model_xgb2.fit(X,Y)
pred_xgb2 = model_xgb2.predict(test.values)
pred_results_xgb2 = pd.DataFrame({"pickup_datetime":test_pickup_datetime,"prediction" : pred_xgb2})
pred_results_xgb2.to_csv("predictions_xgboost.csv",index=False)
# ## Apply GBM

model_gbm2 = GradientBoostingRegressor(n_estimators = 120)
model_gbm2.fit(X,Y)
pred_gbm2 = model_gbm2.predict(test.values)
pred_results_gbm2 = pd.DataFrame({"pickup_datetime":test_pickup_datetime,"prediction" : pred_gbm2})
pred_results_gbm2.to_csv("predictions_gbm.csv",index = False)
