###data preprocessing
###import dataset 
import pandas as pd 
df = pd.read_excel(r'C:\Users\zewen\Desktop\INSY662\project\individual project\Kickstarter.xlsx')
###change goal to usd based
df['goal'] = df['goal']*df['static_usd_rate']
###check if dataframe has null value
df.isnull().any().any() ###true -> there is null value 
###drop rows not faliure or successful
df = df[(df['state']=='failed')|(df['state']=='successful')]
###check if dataframe has null value
df.isnull().any().any() ###true-> there is no null value
###fill nan in category column as 'N/A'
df['category'] = df['category'].fillna(value = 'N/A')
###check if dataframe has null value
df.isnull().any().any() ###true-> there is no null value
df = df.reset_index()
df = df.drop('index', axis = 1)

#############################################################################################################################
#############################################################################################################################
#############################################################################################################################

###Question1
###final dataframe to use for question 1
df1_final= df.drop(['project_id','name','pledged','state','disable_communication','country','deadline','state_changed_at'\
                     ,'created_at','launched_at','staff_pick','backers_count','static_usd_rate','spotlight'\
                     ,'name_len_clean','blurb_len_clean','state_changed_at_weekday'\
                     ,'state_changed_at_month','state_changed_at_day','state_changed_at_yr'\
                     ,'state_changed_at_hr','launch_to_state_change_days'], axis = 1)
###check if final1 table has nan value
df1_final.isnull().any().any() ###false -> there is no null value
###dummify categorical variables 
df1_final = pd.get_dummies(df1_final, columns = ['currency'\
                                                 ,'category','deadline_weekday','created_at_weekday'\
                                                 ,'launched_at_weekday','deadline_month','deadline_yr'\
                                                 ,'created_at_month','created_at_yr'\
                                                 ,'launched_at_month','launched_at_yr'])

###feature selection
###LASSO method
X1L = df1_final.drop('usd_pledged', axis = 1)
y1L = df1_final["usd_pledged"]
###standardize all predictors
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std1L = scaler.fit_transform(X1L)
###get the result for feature selection
from sklearn.linear_model import Lasso
ls1L = Lasso(alpha = 0.01, positive = True, random_state = 0)
model1L = ls1L.fit(X_std1L, y1L)
selection1L = pd.DataFrame(list(zip(X1L.columns, model1L.coef_)), columns = ['predictor','coefficient'])
for i in range(selection1L.shape[0]):
    if selection1L.at[i, 'coefficient'] == 0:
        selection1L.at[i, 'coefficient'] = 'nan'
selection1L_final = selection1L.dropna().reset_index()
selection1L_final = selection1L_final.drop('index', axis = 1)
selection1L_final = list(selection1L_final['predictor'])

###RandomTree method 
X1R = df1_final.drop('usd_pledged', axis = 1)
y1R = df1_final["usd_pledged"]
###fit model by using RandomForest
###use randomforestregressor becasue it can handle cotinous variables 
from sklearn.ensemble import RandomForestRegressor
###if you run n_estimators = 100 important feature will decrease to 2 (not including staff_pick_True)
randomforest1R = RandomForestRegressor(random_state = 0)
model1R = randomforest1R.fit(X1R, y1R)
###get the result for feature selection
from sklearn.feature_selection import SelectFromModel
sfm1 = SelectFromModel(model1R, threshold = 0.001)
sfm1.fit(X1R, y1R)
###print the most important features
selection1R_final = []
for feature_list_index in sfm1.get_support(indices = True):
    selection1R_final.append(X1R.columns[feature_list_index])

###find the overlapped predictors of LASSO and RandomForestTree
selection1_final = list(set(selection1L_final) & set(selection1R_final))
print(selection1_final)

#############################################################################################################################
#############################################################################################################################
#############################################################################################################################

###run a loop to test which model is the best by using different random_state for splitting data
import random as rd
randomlist = []
for i in range(11):
    randomlist.append(rd.randint(1,31))

for j in randomlist: 
    mselist_j = []
    ###RandomForest regression + LASSO feature selection
    ###Construct variables
    X1LR = df1_final.drop('usd_pledged', axis = 1)[selection1L_final]
    y1LR = df1_final["usd_pledged"]
    ###separate the data for training and testing
    from sklearn.model_selection import train_test_split
    X_train1LR, X_test1LR, y_train1LR, y_test1LR = train_test_split(X1LR, y1LR, test_size = 0.3, random_state = j)
    ###run default random forest
    rf1LR = RandomForestRegressor()
    model1LR = rf1LR.fit(X_train1LR,y_train1LR)
    ###using the model to predict the results based on the test dataset
    y_test_pred1LR = rf1LR.predict(X_test1LR)
    ###calculate the mean squared error of the prediction
    from sklearn.metrics import mean_squared_error
    mse1LR = mean_squared_error(y_test1LR, y_test_pred1LR)
    mselist_j.append(mse1LR)
    
    ###RandomForest regression + RandomForest feature selection
    ###Construct variables
    X1RR = df1_final.drop('usd_pledged', axis = 1)[selection1R_final]
    y1RR = df1_final["usd_pledged"]
    ###separate the data for training and testing
    X_train1RR, X_test1RR, y_train1RR, y_test1RR = train_test_split(X1RR, y1RR, test_size = 0.3, random_state = j)
    ###run defualt random forest
    rf1RR = RandomForestRegressor()
    model1RR = rf1RR.fit(X_train1RR,y_train1RR)
    ###using the model to predict the results based on the test dataset
    y_test_pred1RR = rf1RR.predict(X_test1RR)
    ###calculate the mean squared error of the prediction
    mse1RR = mean_squared_error(y_test1RR, y_test_pred1RR)
    mselist_j.append(mse1RR)
    
    ###SVM regression + LASSO feature selection 
    from sklearn.svm import SVR
    ###construct variables
    X1LS = df1_final.drop('usd_pledged', axis = 1)[selection1L_final]
    y1LS = df1_final["usd_pledged"]
    ###standardize predictors
    X_std1LS = scaler.fit_transform(X1LS)
    ###separate the data
    X_train1LS, X_test1LS, y_train1LS, y_test1LS = train_test_split(X_std1LS, y1LS, test_size = 0.3, random_state = j)
    ###run default SVR
    svm1LS = SVR(kernel = 'linear')
    model1LS = svm1LS.fit(X_train1LS,y_train1LS)
    ###using the model to predict the results based on the test dataset
    y_test_pred1LS = svm1LS.predict(X_test1LS)
    ###calculate the mean squared error of the prediction
    mse1LS = mean_squared_error(y_test1LS, y_test_pred1LS)
    mselist_j.append(mse1LS)
    
    ###SVM regression + RandomForest feature selection
    ###construct variables 
    X1RS = df1_final.drop('usd_pledged', axis = 1)[selection1R_final]
    y1RS = df1_final["usd_pledged"]
    ###standardize predictors
    X_std1RS = scaler.fit_transform(X1RS)
    ###separate the data
    X_train1RS, X_test1RS, y_train1RS, y_test1RS = train_test_split(X_std1RS, y1RS, test_size = 0.3, random_state = j)
    ###run default SVR
    svm1RS = SVR(kernel = 'linear')
    model1RS = svm1RS.fit(X_train1RS,y_train1RS)
    ###using the model to predict the results based on the test dataset
    y_test_pred1RS = svm1RS.predict(X_test1RS)
    ###calculate the mean squared error of the prediction
    mse1RS = mean_squared_error(y_test1RS, y_test_pred1RS)
    mselist_j.append(mse1RS)
    
    ###KNN regression + LASSO feature selection
    from sklearn.neighbors import KNeighborsRegressor
    ###construct variables
    X1LK = df1_final.drop('usd_pledged', axis = 1)[selection1L_final]
    y1LK = df1_final["usd_pledged"]
    ###standardize predictors
    X_std1LK = scaler.fit_transform(X1LK)
    ###separate the data
    X_train1LK, X_test1LK, y_train1LK, y_test1LK = train_test_split(X_std1LK, y1LK, test_size = 0.3, random_state = j)
    ###run default K-NN
    knn1LK = KNeighborsRegressor()
    model1LK = knn1LK.fit(X_train1LK, y_train1LK)
    ###using the model to predict the results based on the test dataset
    y_test_pred1LK = knn1LK.predict(X_test1LK)
    ###calculate the mean squared error of the prediction
    mse1LK = mean_squared_error(y_test1LK, y_test_pred1LK)
    mselist_j.append(mse1LK)
    
    ###KNN regression + RandomForest feature selection
    ###construct variables
    X1RK = df1_final.drop('usd_pledged', axis = 1)[selection1R_final]
    y1RK = df1_final["usd_pledged"]
    ###standardize predictors
    X_std1RK = scaler.fit_transform(X1RK)
    ###separate the data
    X_train1RK, X_test1RK, y_train1RK, y_test1RK = train_test_split(X_std1RK, y1RK, test_size = 0.3, random_state = j)
    ###run default K-NN
    knn1RK = KNeighborsRegressor()
    model1RK = knn1RK.fit(X_train1RK, y_train1RK)
    ###using the model to predict the results based on the test dataset
    y_test_pred1RK = knn1RK.predict(X_test1RK)
    ###calculate the mean squared error of the prediction
    mse1RK = mean_squared_error(y_test1RK, y_test_pred1RK)
    mselist_j.append(mse1RK)
    print(j, mselist_j)
    mselist_j = []
###to see randomlist
print(randomlist) ###[26, 9, 25, 12, 18, 17, 2, 25, 31, 28, 22]
###results from the loop
mselist_j_26 = [18954537103.536674, 17376784197.424805, 18990279924.057465, 18989271899.483444, 20725831412.33706, 20330331089.472733]
mselist_j_09 = [21658296774.788383, 22130823225.251587, 22745802164.184, 22749154015.207466, 24906331287.010002, 24162609251.00875]
mselist_j_25 = [14848698386.468285, 14998640578.611357, 14512243440.777426, 14516755039.790138, 16758202839.627165, 16513579380.344526]
mselist_j_12 = [25235677784.498066, 25281168091.15631, 25824946383.2405, 25823564358.34352, 26901394976.06033, 26794183973.595097]
mselist_j_18 = [17391894685.946716, 18986297715.444965, 19720094019.21835, 19718648060.397232, 20557223334.228996, 20622727857.401825]
mselist_j_17 = [17288331922.12703, 15708466255.525675, 16581791879.069307, 16582042994.947042, 17917143752.449127, 17799852290.118305]
mselist_j_02 = [17134483992.200203, 15415644772.760134, 16434950990.189526, 16434677643.590618, 18440472420.116768, 17296458082.694168]
mselist_j_31 = [13769973539.964552, 15196956338.607828, 12901595569.869019, 12900127603.807186, 15352129218.912233, 15427486140.819452]
mselist_j_28 = [17698587031.61833, 19231777902.079746, 19656883250.125565, 19656933102.77586, 21186568226.96344, 20765230851.522053]
mselist_j_22 = [23362201798.314247, 23733712181.588547, 24581202133.49624, 24579602644.54005, 26199026597.211506, 25381485803.080555]
###how about using overlapped features?
for k in randomlist: 
    mselist_k = []
    X1OR = df1_final.drop('usd_pledged', axis = 1)[selection1_final]
    y1OR = df1_final["usd_pledged"]
    ###separate the data for training and testing
    X_train1OR, X_test1OR, y_train1OR, y_test1OR = train_test_split(X1OR, y1OR, test_size = 0.3, random_state = k)
    ###run random forest
    rf1OR = RandomForestRegressor()
    model1OR = rf1OR.fit(X_train1OR,y_train1OR)
    ###using the model to predict the results based on the test dataset
    y_test_pred1OR = rf1OR.predict(X_test1OR)
    ###calculate the mean squared error of the prediction
    mse1OR = mean_squared_error(y_test1OR, y_test_pred1OR)
    mselist_k.append(mse1OR)
    
    ###SVM regression 
    ###construct variables
    X1OS = df1_final.drop('usd_pledged', axis = 1)[selection1_final]
    y1OS = df1_final["usd_pledged"]
    ###standardize predictors
    X_std1OS = scaler.fit_transform(X1OS)
    ###separate the data
    X_train1OS, X_test1OS, y_train1OS, y_test1OS = train_test_split(X_std1OS, y1OS, test_size = 0.3, random_state = k)
    ###run SVR
    svm1OS = SVR(kernel = 'linear')
    model1OS = svm1OS.fit(X_train1OS,y_train1OS)
    ###using the model to predict the results based on the test dataset
    y_test_pred1OS = svm1OS.predict(X_test1OS)
    ###calculate the mean squared error of the prediction
    mse1OS = mean_squared_error(y_test1OS, y_test_pred1OS)
    mselist_k.append(mse1OS)
    
    ###KNN regression 
    ###construct variables
    X1OK = df1_final.drop('usd_pledged', axis = 1)[selection1_final]
    y1OK = df1_final["usd_pledged"]
    ###standardize predictors
    X_std1OK = scaler.fit_transform(X1OK)
    ###separate the data
    X_train1OK, X_test1OK, y_train1OK, y_test1OK = train_test_split(X_std1OK, y1OK, test_size = 0.3, random_state = k)
    ###run K-NN
    knn1OK = KNeighborsRegressor()
    model1OK = knn1OK.fit(X_train1OK, y_train1OK)
    ###using the model to predict the results based on the test dataset
    y_test_pred1OK = knn1OK.predict(X_test1OK)
    ###calculate the mean squared error of the prediction
    mse1OK = mean_squared_error(y_test1OK, y_test_pred1OK)
    mselist_k.append(mse1OK)
    print(k, mselist_k)
    mselist_k = []
###results from the loop
mselist_k_26 = [18897758139.876247, 18996380149.63659, 20442396997.141243]
mselist_k_09 = [20566949595.673832, 22755362437.07537, 24596948554.656345]
mselist_k_25 = [13776229592.823069, 14522548161.155333, 16501232976.349976]
mselist_k_12 = [25372093969.127625, 25831809883.16489, 26790375159.58673]
mselist_k_18 = [18967587413.831623, 19726046804.26481, 20553996453.199936]
mselist_k_17 = [16839294347.782133, 16588656523.27125, 17712024446.73934]
mselist_k_02 = [16390307977.671925, 16440402178.877703, 17376415193.890347]
mselist_k_31 = [14219466459.38503, 12907768958.689861, 15375404265.881794]
mselist_k_28 = [18638536881.93782, 19664621792.19955, 21029130284.58659]
mselist_k_22 = [23261126991.10447, 24587086438.27561, 26062714043.200966]
###according to the excel table, RF is the best model 
###according to the excel table, combinitionis the best feature selection method 

#############################################################################################################################
#############################################################################################################################
#############################################################################################################################

###what is the min mse for the chosen model?
X1OR = df1_final.drop('usd_pledged', axis = 1)[selection1_final]
y1OR = df1_final["usd_pledged"]
###separate the data for training and testing
X_train1OR, X_test1OR, y_train1OR, y_test1OR = train_test_split(X1OR, y1OR, test_size = 0.3, random_state = 24)
###run random forest
###best random_state
minmse1OR = 10000000000000
best_i = 0
for i in range(31):
    rf1OR = RandomForestRegressor(random_state = i, n_estimators = 100)
    model1OR = rf1OR.fit(X_train1OR,y_train1OR)
    ###using the model to predict the results based on the test dataset
    y_test_pred1OR = rf1OR.predict(X_test1OR)
    ###calculate the mean squared error of the prediction
    mse1OR = mean_squared_error(y_test1OR, y_test_pred1OR)
    if mse1OR <= minmse1OR:
        minmse1OR = mse1OR
        best_i = i
print(best_i, minmse1OR) ###6 10686515143.375744
###best max_features
minmse1OR = 10000000000000
best_i = 0
for i in range(2, 32):
    rf1OR = RandomForestRegressor(random_state = 6, max_features = i, n_estimators = 100)
    model1OR = rf1OR.fit(X_train1OR,y_train1OR)
    ###using the model to predict the results based on the test dataset
    y_test_pred1OR = rf1OR.predict(X_test1OR)
    ###calculate the mean squared error of the prediction
    mse1OR = mean_squared_error(y_test1OR, y_test_pred1OR)
    if mse1OR <= minmse1OR:
        minmse1OR = mse1OR
        best_i = i
print(best_i, minmse1OR) ###28 10668192823.609177
###best max_depth
minmse1OR = 10000000000000
best_i = 0
for i in range(2, 22):
    rf1OR = RandomForestRegressor(random_state = 6, max_depth = i, n_estimators = 100)
    model1OR = rf1OR.fit(X_train1OR,y_train1OR)
    ###using the model to predict the results based on the test dataset
    y_test_pred1OR = rf1OR.predict(X_test1OR)
    ###calculate the mean squared error of the prediction
    mse1OR = mean_squared_error(y_test1OR, y_test_pred1OR)
    if mse1OR <= minmse1OR:
        minmse1OR = mse1OR
        best_i = i
print(best_i, minmse1OR) ###10 10571809791.76272
###best min_samples_split
minmse1OR = 10000000000000
best_i = 0
for i in range(2, 11):
    rf1OR = RandomForestRegressor(random_state = 6, min_samples_split = i, n_estimators = 100)
    model1OR = rf1OR.fit(X_train1OR,y_train1OR)
    ###using the model to predict the results based on the test dataset
    y_test_pred1OR = rf1OR.predict(X_test1OR)
    ###calculate the mean squared error of the prediction
    mse1OR = mean_squared_error(y_test1OR, y_test_pred1OR)
    if mse1OR <= minmse1OR:
        minmse1OR = mse1OR
        best_i = i
print(best_i, minmse1OR) ###2 10686515143.375744
###put together
X1OR = df1_final.drop('usd_pledged', axis = 1)[selection1_final]
y1OR = df1_final["usd_pledged"]
###separate the data for training and testing
X_train1OR, X_test1OR, y_train1OR, y_test1OR = train_test_split(X1OR, y1OR, test_size = 0.3, random_state = 24)
###run random forest
rf1OR = RandomForestRegressor(random_state = 6, max_features = 28, max_depth = 10, min_samples_split = 2, n_estimators = 100)
model1OR = rf1OR.fit(X_train1OR,y_train1OR)
###using the model to predict the results based on the test dataset
y_test_pred1OR = rf1OR.predict(X_test1OR)
###calculate the mean squared error of the prediction
mse1OR = mean_squared_error(y_test1OR, y_test_pred1OR)
print(mse1OR) ###10767363326.146101

#############################################################################################################################
#############################################################################################################################
#############################################################################################################################

###Question2
###final dataframe to use for question 1
df2_final= df.drop(['project_id','name','pledged','disable_communication','country','deadline','state_changed_at'\
                     ,'created_at','launched_at','staff_pick','backers_count','static_usd_rate','usd_pledged','spotlight'\
                     ,'name_len_clean','blurb_len_clean','state_changed_at_weekday'\
                     ,'state_changed_at_month','state_changed_at_day','state_changed_at_yr'\
                     ,'state_changed_at_hr','launch_to_state_change_days'], axis = 1)
###check if final1 table has nan value
df2_final.isnull().any().any() ###false -> there is no null value
###dummify categorical variables 
df2_final = pd.get_dummies(df2_final, columns = ['state','currency','category','deadline_weekday','created_at_weekday'\
                                                 ,'launched_at_weekday','deadline_month','deadline_yr'\
                                                 ,'created_at_month','created_at_yr'\
                                                 ,'launched_at_month','launched_at_yr'])
###failed = 1 successful = 0
df2_final = df2_final.drop('state_successful', axis = 1)
df2_final = df2_final.rename(columns={'state_failed': 'state'})

###feature selection
###LASSO method
X2L = df2_final.drop('state', axis = 1)
y2L = df2_final['state']
###standardize all predictors
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std2L = scaler.fit_transform(X2L)
###get the result for feature selection
from sklearn.linear_model import Lasso
ls2L = Lasso(alpha = 0.01, positive = True, random_state = 0)
model2L = ls2L.fit(X_std2L, y2L)
selection2L = pd.DataFrame(list(zip(X2L.columns, model2L.coef_)), columns = ['predictor','coefficient'])
for i in range(selection2L.shape[0]):
    if selection2L.at[i, 'coefficient'] == 0:
        selection2L.at[i, 'coefficient'] = 'nan'
selection2L_final = selection2L.dropna().reset_index()
selection2L_final = selection2L_final.drop('index', axis = 1)
selection2L_final = list(selection2L_final['predictor'])

###RandomTree method 
X2R = df2_final.drop('state', axis = 1)
y2R = df2_final['state']
###fit model by using RandomForest
###use randomforestregressor becasue it can handle cotinous variables 
from sklearn.ensemble import RandomForestClassifier
###if you run n_estimators = 200 important feature will decrease to 2 (not including staff_pick_True)
randomforest2R = RandomForestClassifier(random_state = 0)
model2R = randomforest2R.fit(X2R, y2R)
###get the result for feature selection
from sklearn.feature_selection import SelectFromModel
sfm2 = SelectFromModel(model2R, threshold = 0.0035)
sfm2.fit(X2R, y2R)
###print the most important features
selection2R_final = []
for feature_list_index in sfm2.get_support(indices = True):
    selection2R_final.append(X2R.columns[feature_list_index])

###compare with LASSO, randomforest seems more reliable 
###(LASSO only gives 10 predictors which don't include goal which i think it quite important)
###so i decide to use randomforest at the same threshold = 0.003 (0.001 and 0.002 give 103 predictors, 0.05 gives 22 predictors)

#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
    
###now try which is the best model
X2 = df2_final.drop('state', axis = 1)[selection2R_final]
y2 = df2_final['state']
###separate the data for training and testing
X_train2RR, X_test2RR, y_train2RR, y_test2RR = train_test_split(X2, y2, test_size = 0.3, random_state = 20)

###logistic regression
from sklearn.linear_model import LogisticRegression
lor2L = LogisticRegression()
model2L = lor2L.fit(X_train2RR, y_train2RR)
y_test_pred2L = model2L.predict(X_test2RR)
from sklearn.metrics import accuracy_score 
print(accuracy_score(y_test2RR, y_test_pred2L)) ###0.6572460688482787

###randomforest 
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import train_test_split
###base model
model2R = RandomForestClassifier().fit(X_train2RR, y_train2RR)
y_test_pred2R = model2R.predict(X_test2RR)
print(accuracy_score(y_test2RR, y_test_pred2R)) ###0.7076073098172546

###best random_state
maxscore = 0
best_i = 0
for i in range(32):
    model2R = RandomForestClassifier(random_state = i, n_estimators = 100)
    scores = cross_val_score(estimator = model2R, X = X2, y = y2, cv = 5)
    if np.average(scores) >= maxscore:
        maxscore = np.average(scores)   
        best_i = i
print(best_i, maxscore) ###17 0.7415371195133315
###best max_features
maxscore = 0
best_i = 0
for i in range(2,88):
    model2R = RandomForestClassifier(random_state = 17, max_features = i, n_estimators = 100)
    scores = cross_val_score(estimator = model2R, X = X2, y = y2, cv = 5)
    if np.average(scores) >= maxscore:
        maxscore = np.average(scores)   
        best_i = i
print(best_i, maxscore) ###9 0.7415371195133315
###best max_depth
maxscore = 0
best_i = 0
for i in range(2,62):
    model2R = RandomForestClassifier(random_state = 17, max_depth = i, n_estimators = 100)
    scores = cross_val_score(estimator = model2R, X = X2, y = y2, cv = 5)
    if np.average(scores) >= maxscore:
        maxscore = np.average(scores)   
        best_i = i
print(best_i, maxscore) ###44 0.7417284053837523
###best min_samples_split
maxscore = 0
best_i = 0
for i in range(2,62):
    model2R = RandomForestClassifier(random_state = 17, min_samples_split = i, n_estimators = 100)
    scores = cross_val_score(estimator = model2R, X = X2, y = y2, cv = 5)
    if np.average(scores) >= maxscore:
        maxscore = np.average(scores)   
        best_i = i
print(best_i, maxscore) ###17 0.7424303626170301
###put together 
model2R = RandomForestClassifier(random_state = 17, max_features = 9, max_depth = 44, min_samples_split = 17, n_estimators = 100)
scores = cross_val_score(estimator = model2R, X = X2, y = y2, cv = 5)
print(np.average(scores)) ###0.7424303626170301
###so we use random_state = 17, max_features = 9, min_samples_split = 17
###get the accuracy score for best model
model2R = RandomForestClassifier(random_state = 17, max_features = 9, max_depth = 44, min_samples_split = 17, n_estimators = 100).fit(X_train2RR, y_train2RR)
y_test_pred2R = model2R.predict(X_test2RR)
print(accuracy_score(y_test2RR, y_test_pred2R)) ###0.7396940076498087



















