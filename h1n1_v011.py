import pandas as pd
import numpy as np
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
"doctor_recc_h1n1", "chronic_med_condition","child_under_6_months","health_worker",
"health_insurance","sex","marital_status","rent_or_own"]]

imputer = imputer.fit(binary_train2)
binary_train2 = imputer.transform(binary_train2)
binary_train = pd.DataFrame(data =binary_train2 , index=range(26707), columns=["behavioral_antiviral_meds","behavioral_avoidance","behavioral_face_mask",
"behavioral_wash_hands","behavioral_large_gatherings","behavioral_outside_home","behavioral_touch_face",
"doctor_recc_h1n1","chronic_med_condition","child_under_6_months","health_worker",
"health_insurance","sex","marital_status","rent_or_own"])
print(binary_train.isnull().sum())


################################################
#doctor_recc_seasonal sütununu çekip eksik verileri fill edip binary_train dosyasına concatlıyacağız.

doctor_recc_seasonal = train[["doctor_recc_seasonal"]]

#doctor_recc_seasonal2 sütununu seasonal_vaccine ile birleştiriyor  
doctor_recc_seasonal2 = pd.concat([doctor_recc_seasonal[['doctor_recc_seasonal']],y_train[["seasonal_vaccine"]]], axis=1)


for i in range(26707):
    if(np.isnan(doctor_recc_seasonal2["doctor_recc_seasonal"][i])):
#        print(i)
         doctor_recc_seasonal2["doctor_recc_seasonal"][i] = doctor_recc_seasonal2["seasonal_vaccine"][i]               

doctor_recc_seasonal.isnull().sum()

binary_train = pd.concat([doctor_recc_seasonal2[['doctor_recc_seasonal']],binary_train], axis=1)

################################################

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
#bu çalışmada seasonal_vaccine ile doctor_recc_seasonal arasında yüksek korelasyon nedeniyle doctor_recc_seasonal'in missing valuelarını seasonal_vaccine'e göre alacağız 
#1-binary_train2 verisinden doctor_recc_seasonal ifadesini sildik.
#2-doctor_recc_seasonal ifadesi doctor_recc_h1n1'den sonra geliyordu. önce verileri binary_train2'ye çekerken sonra da bunları imputer'dan sonra
#    tekrar dataframe hale getirirken bu bu şekildeydi


##################################################
Y1_train = y_train.iloc[:,0:1]
Y2_train = y_train.iloc[:,1:2]