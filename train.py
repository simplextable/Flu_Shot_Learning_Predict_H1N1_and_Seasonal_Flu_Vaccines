import numpy as np
import xgboost as xgb
# data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.display import display, HTML

# collection of machine learning algorithms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Common Model Helpers
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn import model_selection
import pylab as pl
from sklearn.metrics import roc_curve
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
pd.set_option("display.max_rows",222200)
pd.set_option("display.max_columns",222200)

train = pd.read_csv('training_set_features.csv')
y_train = pd.read_csv('training_set_labels.csv')
test = pd.read_csv('test.csv')
df= pd.concat([train,y_train],axis=1)

print(train.isnull().sum())
print(train.info())
 
plt.subplots(figsize=(16,16))
sns.heatmap(df.corr(), annot=True, fmt=".2f")
plt.show()



#Binary Verilerin Eksik Verilerini doldurma 
imputer= SimpleImputer( missing_values=np.nan, strategy = 'most_frequent')   
 

binary_train2 = train[["behavioral_antiviral_meds","behavioral_avoidance","behavioral_face_mask",
"behavioral_wash_hands","behavioral_large_gatherings","behavioral_outside_home","behavioral_touch_face",
"doctor_recc_h1n1","doctor_recc_seasonal","chronic_med_condition","child_under_6_months","health_worker",
"health_insurance","sex","marital_status","rent_or_own"]]

imputer = imputer.fit(binary_train2)
binary_train2 = imputer.transform(binary_train2)
binary_train = pd.DataFrame(data =binary_train2 , index=range(26707), columns=["behavioral_antiviral_meds","behavioral_avoidance","behavioral_face_mask",
"behavioral_wash_hands","behavioral_large_gatherings","behavioral_outside_home","behavioral_touch_face",
"doctor_recc_h1n1","doctor_recc_seasonal","chronic_med_condition","child_under_6_months","health_worker",
"health_insurance","sex","marital_status","rent_or_own"])
print(binary_train.isnull().sum())



#Kategorik Verileri eksik veri bulma
CategorikBasliklar =  train[['h1n1_concern','h1n1_knowledge','opinion_h1n1_vacc_effective','opinion_h1n1_risk',
'opinion_h1n1_sick_from_vacc','opinion_seas_vacc_effective','opinion_seas_risk','opinion_seas_sick_from_vacc',
'age_group','education','race','income_poverty','hhs_geo_region','census_msa','household_adults','household_children',
'employment_industry','employment_occupation','employment_status']]

imputer2= SimpleImputer( missing_values=np.nan, strategy = 'most_frequent') 
imputer2 = imputer.fit(CategorikBasliklar)
CategorikBasliklar = imputer.transform(CategorikBasliklar)
CategorikBasliklar = pd.DataFrame(data =CategorikBasliklar , index=range(26707), columns=['h1n1_concern','h1n1_knowledge','opinion_h1n1_vacc_effective','opinion_h1n1_risk',
'opinion_h1n1_sick_from_vacc','opinion_seas_vacc_effective','opinion_seas_risk','opinion_seas_sick_from_vacc',
'age_group','education','race','income_poverty','hhs_geo_region','census_msa','household_adults','household_children',
'employment_industry','employment_occupation','employment_status'])




#Kategorik verileri one hot encedera çevirme işlemi

ready_cat_data =  pd.get_dummies(CategorikBasliklar , columns =['h1n1_concern','h1n1_knowledge','opinion_h1n1_vacc_effective','opinion_h1n1_risk',
'opinion_h1n1_sick_from_vacc','opinion_seas_vacc_effective','opinion_seas_risk','opinion_seas_sick_from_vacc',
'age_group','education','race','income_poverty','hhs_geo_region','census_msa','household_adults','household_children',
'employment_industry','employment_occupation','employment_status'])


final_df= pd.concat([ready_cat_data,binary_train],axis=1)
y_train.drop(['respondent_id'],axis=1,inplace=True)


###########    TİP DÖNÜŞÜMÜ #############
from sklearn.preprocessing import LabelEncoder
Lb = LabelEncoder()

final_df['sex']= Lb.fit_transform(final_df['sex'])
final_df['marital_status']= Lb.fit_transform(final_df['marital_status'])
final_df['rent_or_own']= Lb.fit_transform(final_df['rent_or_own'])

test['sex']= Lb.fit_transform(test['sex'])
test['marital_status']= Lb.fit_transform(test['marital_status'])
test['rent_or_own']= Lb.fit_transform(test['rent_or_own'])


obj = ["behavioral_antiviral_meds", "behavioral_avoidance", "behavioral_face_mask", "behavioral_wash_hands", 
       "behavioral_large_gatherings", "behavioral_outside_home", "behavioral_touch_face", "doctor_recc_h1n1", "doctor_recc_seasonal", 
       "chronic_med_condition", "child_under_6_months", "health_worker", "health_insurance" ]

for i in obj:
    final_df[i] = final_df[i].astype('float')


for i in obj:
    test[i] = test[i].astype('float')
#################################################





##################################################
Y1_train = y_train.iloc[:,0:1]
Y2_train = y_train.iloc[:,1:2]

#from sklearn.ensemble import RandomForestClassifier
#rfc = RandomForestClassifier(n_estimators = 10, criterion = "entropy")
#rfc.fit(x_train, y_train)
#y_pred = rfc.predict(x_test)
#
#cm = confusion_matrix(y_test,y_pred)
#print("Random Forest")
#print(cm)    

import re


regex = re.compile(r"\[|\]|<", re.IGNORECASE)
final_df.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in final_df.columns.values]
test.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in test.columns.values]



from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


y_train = Y1_train
train = final_df

n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))








#//////////// NORMAL STACK MODEL /////////////


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   



averaged_models = AveragingModels(models = (ENet,  KRR, lasso, model_xgb,model_lgb))

score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



averaged_models.fit(train.values, y_train)
train_pred = averaged_models.predict(train.values)

stacked_pred = np.expm1(averaged_models.predict(test.values))#tEST MODELİ PREDİCT ETTİRDİK

y_pred_17= pd.DataFrame(data = stacked_pred, index = range(26708) , columns = ["h1n1_vaccine"])
idler = pd.read_csv("submission_format.csv")
idler = idler[["respondent_id"]]

nihai=pd.concat([idler,y_pred_17], axis=1)

#nihai.to_csv('17042020.csv',index=False)



#////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////
#///////////////////////////////////////////////////////////



y_train = Y1_train
train = final_df


#from xgboost.sklearn import XGBRegressor
#from sklearn.model_selection import GridSearchCV
#xg_reg = XGBRegressor()
#xgparam_grid= {'learning_rate' : [0.01,0.05],
#               'n_estimators':[2000, 4000],
#               'max_depth':[3,5], 
#               'min_child_weight':[3,5],
#               'colsample_bytree':[0.5,0.9],
#               'reg_alpha':[0.46,0.1],
#               'reg_lambda':[0.8,0.01]}
#
#xg_grid=GridSearchCV(xg_reg, param_grid=xgparam_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
#xg_grid.fit(x_train,y_train)
#print(xg_grid.best_estimator_)
#print(xg_grid.best_score_)
















n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)




model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))








#//////////// NORMAL STACK MODEL /////////////


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   



averaged_models2 = AveragingModels(models = (ENet,  KRR, lasso, model_xgb,model_lgb))

score = rmsle_cv(averaged_models2)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



averaged_models2.fit(train.values, y_train)
train_pred = averaged_models2.predict(train.values)

stacked_pred2 = np.expm1(averaged_models2.predict(test.values))#tEST MODELİ PREDİCT ETTİRDİK

y_pred_18= pd.DataFrame(data = stacked_pred2, index = range(26708) , columns = ["seasonal_vaccine"])



nihai2=pd.concat([nihai,y_pred_18], axis=1)

nihai2.to_csv('19052020.csv',index=False)


































#from xgboost import XGBClassifier
#classifier = XGBClassifier(base_score=0.55, booster='gbtree', colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=1, gamma=0,
#              learning_rate=0.2, max_delta_step=0, max_dept=2, max_depth=3,
#              min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
#              nthread=None, objective='binary:logistic', random_state=0,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#              silent=None, subsample=1, verbosity=1)
#classifier.fit(x_train , y_train)
#y_pred0 = classifier.predict(x_test)
#cm = confusion_matrix(y_test,y_pred0 )
#print(cm)    
#y_pred_niahi_1 = classifier.predict(test)
#
#
#x_train, x_test ,y_train, y_test = train_test_split(final_df ,Y2_train , test_size = 0.33 , random_state = 0)
#from xgboost import XGBClassifier
#import xgboost as xgb
#classifier2 = XGBClassifier(base_score=0.55, booster='gbtree', colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=1, gamma=0,
#              learning_rate=0.2, max_delta_step=0, max_dept=2, max_depth=3,
#              min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
#              nthread=None, objective='binary:logistic', random_state=0,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#              silent=None, subsample=1, verbosity=1)
#classifier2.fit(x_train , y_train)
#y_pred1 = classifier2.predict(x_test)
#cm = confusion_matrix(y_test,y_pred1 )
#print(cm)    
#y_pred_niahi_2 = classifier2.predict(test)




#submission = pd.read_csv("submission_format.csv")
#submission2=submission[["respondent_id"]]
#h1n1= pd.DataFrame(data =y_pred_niahi_1 , index=range(26708), columns=["h1n1_vaccine"])
#ses= pd.DataFrame(data =y_pred_niahi_2 , index=range(26708), columns=["seasonal_vaccine"])
#sonuc= pd.concat([submission2,h1n1,ses],axis=1)
#sonuc.to_csv('Ha-Eg.csv',index=False)