from itertools import combinations 
from sklearn.preprocessing import StandardScaler
import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
#import tensorflow as tf
from sklearn.svm import SVC
#import keras
#from keras.models import Sequential
#from keras.layers import Dense
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()



t = pd.read_csv('training_set_features.csv')
y_train = pd.read_csv('training_set_labels.csv')
train = pd.read_csv('test_set_features.csv')


print(train.isnull().sum())
print(train.info())
 
plt.subplots(figsize=(16,16))

plt.show()



#Binary Verilerin Eksik Verilerini doldurma 
imputer= SimpleImputer( missing_values=np.nan, strategy = 'most_frequent')   
 

binary_train2 = train[["behavioral_antiviral_meds","behavioral_avoidance","behavioral_face_mask",
"behavioral_wash_hands","behavioral_large_gatherings","behavioral_outside_home","behavioral_touch_face",
"doctor_recc_h1n1","doctor_recc_seasonal","chronic_med_condition","child_under_6_months","health_worker",
"health_insurance","sex","marital_status","rent_or_own"]]

imputer = imputer.fit(binary_train2)
binary_train2 = imputer.transform(binary_train2)
binary_train = pd.DataFrame(data =binary_train2 , index=range(26708), columns=["behavioral_antiviral_meds","behavioral_avoidance","behavioral_face_mask",
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
CategorikBasliklar = pd.DataFrame(data =CategorikBasliklar , index=range(26708), columns=['h1n1_concern','h1n1_knowledge','opinion_h1n1_vacc_effective','opinion_h1n1_risk',
'opinion_h1n1_sick_from_vacc','opinion_seas_vacc_effective','opinion_seas_risk','opinion_seas_sick_from_vacc',
'age_group','education','race','income_poverty','hhs_geo_region','census_msa','household_adults','household_children',
'employment_industry','employment_occupation','employment_status'])




#Kategorik verileri one hot encedera çevirme işlemi

ready_cat_data =  pd.get_dummies(CategorikBasliklar , columns =['h1n1_concern','h1n1_knowledge','opinion_h1n1_vacc_effective','opinion_h1n1_risk',
'opinion_h1n1_sick_from_vacc','opinion_seas_vacc_effective','opinion_seas_risk','opinion_seas_sick_from_vacc',
'age_group','education','race','income_poverty','hhs_geo_region','census_msa','household_adults','household_children',
'employment_industry','employment_occupation','employment_status'])

print(binary_train.isnull().sum())
final_df1= pd.concat([ready_cat_data,binary_train],axis=1)
y_train.drop(['respondent_id'],axis=1,inplace=True)



final_df1.to_csv('test.csv',index=False)