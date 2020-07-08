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



lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0001, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.00064255, l1_ratio=0.7, random_state=3))
KRR = KernelRidge(alpha=0.7, kernel='polynomial')

GBoost = GradientBoostingRegressor(n_estimators=250, learning_rate=0.1,
                                   subsample=0.8,
                                   max_depth=3, max_features=11,
                                   min_samples_leaf=4, min_samples_split=500, 
                                   )

model_xgb = xgb.XGBRegressor(colsample_bytree=0.3, gamma=0.03, 
                             learning_rate=0.07, max_depth=3, 
                             min_child_weight=1.1, n_estimators=250,
                             reg_alpha=0.43, reg_lambda=1,
                             subsample=1, silent=None,
                             random_state =0, nthread = 0)



model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=31, colsample_bytree= 0.6,
                              learning_rate=0.1, n_estimators=50, max_depth = -1, 
                              max_bin = 55, bagging_fraction = 0.8, reg_alpha = 1.5, reg_lambda = 0, min_split_gain=0,subsample=0.2, subsample_freq=0,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

#######################3
              
     

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





















