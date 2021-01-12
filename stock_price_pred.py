import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")
import datetime
import smogn

train = pd.read_csv('Train.csv')
print('Shape of df :',train.shape)
train.head()
print('Missing values : ','\n', train.isnull().sum())

# Convert to Datetime format
train.InvoiceDate = pd.to_datetime(train.InvoiceDate)

# Creating Backup
train2 = train.copy()

# Extract date & time of transaction
train2['year'] = train2.InvoiceDate.apply(lambda x : x.year)
train2['month'] = train2.InvoiceDate.apply(lambda x : x.month)
train2['day'] = train2.InvoiceDate.apply(lambda x : x.day)
train2['hour'] = train2.InvoiceDate.apply(lambda x : x.hour)
train2['minute'] = train2.InvoiceDate.apply(lambda x : x.minute)

# Drop existing date column
train2.drop('InvoiceDate', axis = 1, inplace = True)

# converting to circular coordinates 
train3 = train2.copy()
train3['month_sin'] = np.sin((train3.month-1)*(2.*np.pi/12))
train3['month_cos'] = np.cos((train3.month-1)*(2.*np.pi/12))

train3['day_sin'] = np.sin((train3.day-1)*(2.*np.pi/12))
train3['day_cos'] = np.cos((train3.day-1)*(2.*np.pi/12))

train3['hour_sin'] = np.sin((train3.hour-1)*(2.*np.pi/12))
train3['hour_cos'] = np.cos((train3.hour-1)*(2.*np.pi/12))

train3['minute_sin'] = np.sin((train3.minute-1)*(2.*np.pi/12))
train3['minute_cos'] = np.cos((train3.minute-1)*(2.*np.pi/12))

train3.drop(['month', 'hour', 'minute', 'day'], axis = 1, inplace = True)

# Split features and target
X1 = train3.drop(['UnitPrice'], axis = 1)
y1 = train3.UnitPrice.copy()

# standardize the features
ss= StandardScaler()
X1 = ss.fit_transform(X1)

#Cross validation
lr = LinearRegression()
rf = RandomForestRegressor(n_jobs=-1)
xgb = XGBRegressor()
ri = Ridge()
la = Lasso()

print('Cross validation mean RMSE score')
lr_cv = cross_val_score(lr,X1,y1,cv = 10, scoring = 'neg_root_mean_squared_error').mean()
print('For Linear Regression : ',round(-1*lr_cv,3))

lr_rf = cross_val_score(rf,X1,y1,cv = 8, scoring = 'neg_root_mean_squared_error').mean()
print('For Random Forest : ',round(-1*lr_rf,3))

lr_xgb = cross_val_score(xgb,X1,y1,cv = 8, scoring = 'neg_root_mean_squared_error').mean()
print('For XGBoost : ',round(-1*lr_xgb,3))

ri_cv = cross_val_score(ri,X1,y1,cv = 8, scoring = 'neg_root_mean_squared_error').mean()
print('For Ridge : ',round(-1*ri_cv,3))

la_cv = cross_val_score(la,X1,y1,cv = 8, scoring = 'neg_root_mean_squared_error').mean()
print('For Lasso : ',round(-1*la_cv,3))

# oversampling
# separateing minority & majority values
train3_maj = train3[train3.UnitPrice < train3.UnitPrice.quantile(0.75)]
train3_min= train3[train3.UnitPrice > train3.UnitPrice.quantile(0.75)]

# oversample minority values
train3_min_oversampled = train3_min.sample(train3_maj.shape[0], replace = True, random_state = 10)

# combining majority values with over-sampled minority values
balanced_df = pd.concat([train3_maj, train3_min_oversampled], ignore_index = True)

# Shuffle the over-sampled DataFrame
balanced_df = balanced_df.sample(frac = 1)
balanced_df.head()

# Split oversampled data
Xnew = balanced_df.drop(['UnitPrice'], axis = 1)
ynew = balanced_df.UnitPrice.copy()

# standardize the features
ss1 = StandardScaler()
Xnew = ss1.fit_transform(Xnew)

#Cross validation after over-sampling
lr = LinearRegression()
rf = RandomForestRegressor(n_jobs=-1)
xgb = XGBRegressor()
ri = Ridge()
la = Lasso()

print('Cross validation mean RMSE score')
lr_cv = cross_val_score(lr,Xnew,ynew,cv =5, scoring = 'neg_root_mean_squared_error').mean()
print('For Linear Regression : ',round(-1*lr_cv,3))

lr_rf = cross_val_score(rf,Xnew,ynew,cv = 5, scoring = 'neg_root_mean_squared_error').mean()
print('For RandomForest : ',round(-1*lr_rf,3))

lr_xgb = cross_val_score(xgb,Xnew,ynew,cv = 5, scoring = 'neg_root_mean_squared_error').mean()
print('For XGBoost : ',round(-1*lr_xgb,3))

ri_cv = cross_val_score(ri,Xnew,ynew,cv = 5, scoring = 'neg_root_mean_squared_error').mean()
print('For Ridge : ',round(-1*ri_cv,3))

la_cv = cross_val_score(la,Xnew,ynew,cv = 5, scoring = 'neg_root_mean_squared_error').mean()
print('For Lasso : ',round(-1*la_cv,3))

# XGB Hyperparameter Tuning as XGBoost has the best cross-validation result
xgb_new = XGBRegressor()

# Paramete Grid
grid = {
    'max_depth' : [12],
    'min_child_weight' : [1],
    'n_estimators' : [450],
    'gamma' : [0],
    'subsample':[1],'colsample_bytree':[1],
    'reg_alpha':[0.01],
    'learning_rate' : [0.3]
}

# Grid Search
gsnew = GridSearchCV(xgb_new, grid, n_jobs=-1, verbose=2,scoring='neg_root_mean_squared_error')
gsnew.fit(Xnew,ynew)

print('Best Hyper-parameter tuning parameters : ',gsnew.best_params_)
print('Best Score obtained from GridSearchCV : ',-1*gsnew.best_score_)

# best model
xgb=XGBRegressor(max_depth = 12, min_child_weight = 1, n_estimators = 450, gamma = 0, subsample = 1,
                 colsample_bytree = 1, reg_alpha = 0.01, learning_rate = 0.3)
xgb.fit(Xnew,ynew)

# Most important Features
feature_imp = pd.DataFrame()
feature_imp['feature'] = balanced_df.drop(['UnitPrice'], axis = 1).columns
feature_imp['importance'] = xgb.feature_importances_
feature_imp = feature_imp.sort_values(by = 'importance', ascending = False)


plt.figure(figsize = (14,5))
sns.barplot(x = feature_imp.importance , y = feature_imp.feature, palette = 'YlGnBu_r')
plt.title('Feature Importance',fontsize = 15)
plt.show()

# Model evaluation
xtrain,xtest,ytrain,ytest = train_test_split(Xnew, ynew, random_state = 5, test_size = 0.2)

xgb_test = xgb
test_pred = xgb_test.fit(xtrain,ytrain).predict(xtest)
rmse_test = np.sqrt(mean_squared_error(ytest, test_pred))
r2_test = r2_score(ytest, test_pred)
print('RMSE = ', round(rmse_test,2), '& r2 = ', round(r2_test,2))

# Import testing data
test = pd.read_csv('Test.csv')

# Extract date time in required format
test.InvoiceDate = pd.to_datetime(test.InvoiceDate)
test['year'] = test.InvoiceDate.apply(lambda x : x.year)
test['month'] = test.InvoiceDate.apply(lambda x : x.month)
test['day'] = test.InvoiceDate.apply(lambda x : x.day)
test['hour'] = test.InvoiceDate.apply(lambda x : x.hour)
test['minute'] = test.InvoiceDate.apply(lambda x : x.minute)
test.drop('InvoiceDate', axis = 1, inplace = True)

test['month_sin'] = np.sin((test.month-1)*(2.*np.pi/12))
test['month_cos'] = np.cos((test.month-1)*(2.*np.pi/12))

test['day_sin'] = np.sin((test.day-1)*(2.*np.pi/12))
test['day_cos'] = np.cos((test.day-1)*(2.*np.pi/12))

test['hour_sin'] = np.sin((test.hour-1)*(2.*np.pi/12))
test['hour_cos'] = np.cos((test.hour-1)*(2.*np.pi/12))

test['minute_sin'] = np.sin((test.minute-1)*(2.*np.pi/12))
test['minute_cos'] = np.cos((test.minute-1)*(2.*np.pi/12))

test.drop(['month', 'hour', 'minute', 'day'], axis = 1, inplace = True)

# standardize the values
test = ss.fit_transform(test)


# make prediction for testing data
result = xgb.predict(test)


# Export predictions
sub = pd.DataFrame({'UnitPrice':result})
#sub.to_csv('sub.csv', index = False)















