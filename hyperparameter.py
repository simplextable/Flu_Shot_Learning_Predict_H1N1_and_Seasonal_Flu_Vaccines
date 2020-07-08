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


sayac=0    
for i in range(136):
    re.sub("h", "b", final_df.columns[sayac])    
    sayac+=1
    print(sayac)
    print(final_df.columns[sayac])




import re

regex = re.compile(r"\[|\]|<|>|\.|_|,|\$|\+|-|=", re.IGNORECASE)
final_df.columns = [regex.sub("", col) if any(x in str(col) for x in set(('[',']','<','>','.','_',',','$','+','-','='))) else col for col in final_df.columns.values]
test.columns = [regex.sub("", col) if any(x in str(col) for x in set(('[',']','<','>','.','_'',','$','+','-','='))) else col for col in test.columns.values]

final_df.isnull().sum()
final_df.info()



from sklearn.model_selection  import train_test_split
x_train, x_test ,y_train, y_test = train_test_split(final_df ,Y1_train , test_size = 0.33 , random_state = 0)

#from sklearn.model_selection  import train_test_split
#x_train, x_test ,y_train, y_test = train_test_split(final_df ,Y2_train , test_size = 0.33 , random_state = 0)




from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBRegressor

xg_reg = XGBRegressor()
xgparam_grid= {
               'colsample_bytree':[0.3],
               "n_estimators":[250],
               "gamma":[0.03],
               "learning_rate":[0.07],        
               "min_child_weight":[1.1],    
               "reg_alpha":[0.43],        
               "reg_lambda":[1],
               "subsample":[1],
               "silent":[None],
               "random_state":[0],
               "nthread":[0]
}

xg_grid=GridSearchCV(xg_reg, param_grid=xgparam_grid, cv=5, scoring='roc_auc', n_jobs=-1)
xg_grid.fit(x_train,y_train)
print(xg_grid.best_estimator_)
print(xg_grid.best_score_)

#########################################3


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

gb = GradientBoostingRegressor()
gb_param_grid= {
               'alpha': [0.9],
               "learning_rate":[0.1],

            
               'subsample': [0.8], 
               'max_features':[11],
#               'min_samples_split':range(1000,2100,500), 
              "min_samples_leaf":[4],
               'max_depth':[3], 
               'min_samples_split':[500],
               'n_estimators':[250]
               }
        
gb_grid=GridSearchCV(gb, param_grid=gb_param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
gb_grid.fit(x_train,y_train)
print(gb_grid.best_estimator_)
print(gb_grid.best_score_)         













#############################################

from lightgbm import LGBMRegressor

lgbm = LGBMRegressor()
lgbm_param_grid = {# LightGBM
        
        #n_estimators, max_depth, num_leaves, sub_sample, colsample_bytree
        
        'n_estimators': [50],
        'learning_rate': [0.1],
        'colsample_bytree': [0.6],
        'max_depth': [-1],
        'num_leaves': [31],
        'reg_alpha': [1.5],
        'reg_lambda': [0],
        'min_split_gain': [0],
        'subsample': [0.2],
        'subsample_freq': [0]
        }


lgbm_grid=GridSearchCV(lgbm, param_grid=lgbm_param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
lgbm_grid.fit(x_train,y_train)
print(lgbm_grid.best_estimator_)
print(lgbm_grid.best_score_)



model_xgb = xgb.XGBRegressor(colsample_bytree=0.3, gamma=0.03, 
                             learning_rate=0.07, max_depth=3, 
                             min_child_weight=1.1, n_estimators=250,
                             reg_alpha=0.43, reg_lambda=1,
                             subsample=1, silent=None,
                             random_state =0, nthread = 0)





model_xgb.fit(x_train, y_train)
y_pred = model_xgb.predict(test)

model_xgb.fit(x_train, y_train)
y_pred2 = model_xgb.predict(test)




y_pred= pd.DataFrame(data = y_pred, index = range(26708) , columns = ["h1n1_vaccine"])
y_pred2= pd.DataFrame(data = y_pred2, index = range(26708) , columns = ["seasonal_vaccine"])

submission = pd.read_csv("submission_format.csv")
submission2=submission[["respondent_id"]]


nihai=pd.concat([submission2,y_pred, y_pred2], axis=1)

nihai.to_csv('20052020.csv',index=False)









