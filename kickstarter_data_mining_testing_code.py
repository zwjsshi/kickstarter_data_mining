import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier

#############################################################################################################################
#############################################################################################################################
#############################################################################################################################

###data pre-processing for Q1 and Q2
kickstarter_df = pd.read_excel(r'C:\Users\zewen\Desktop\INSY662\project\individual project\Kickstarter-Grading-Sample.xlsx')
###change goal to usd based
kickstarter_df['goal'] = kickstarter_df['goal']*kickstarter_df['static_usd_rate']
###check if dataframe has null value
kickstarter_df.isnull().any().any() ###true -> there is null value 
###drop rows not faliure or successful
kickstarter_df = kickstarter_df[(kickstarter_df['state']=='failed')|(kickstarter_df['state']=='successful')]
###check if dataframe has null value
kickstarter_df.isnull().any().any() ###true-> there is no null value
###fill nan in category column as 'N/A'
kickstarter_df['category'] = kickstarter_df['category'].fillna(value = 'N/A')
###check if dataframe has null value
kickstarter_df.isnull().any().any() ###true-> there is no null value
kickstarter_df = kickstarter_df.reset_index()
kickstarter_df = kickstarter_df.drop('index', axis = 1)

#############################################################################################################################
#############################################################################################################################
#############################################################################################################################

###Question1
###final dataframe to use for question 1
kickstarter_df1 = kickstarter_df.drop(['project_id','name','pledged','state','disable_communication','country','deadline','state_changed_at'\
                     ,'created_at','launched_at','staff_pick','backers_count','static_usd_rate','spotlight'\
                     ,'name_len_clean','blurb_len_clean','state_changed_at_weekday'\
                     ,'state_changed_at_month','state_changed_at_day','state_changed_at_yr'\
                     ,'state_changed_at_hr','launch_to_state_change_days'], axis = 1)
###check if final1 table has nan value
kickstarter_df1.isnull().any().any() ###false -> there is no null value
###dummify categorical variables 
kickstarter_df1 = pd.get_dummies(kickstarter_df1, columns = ['currency'\
                                                 ,'category','deadline_weekday','created_at_weekday'\
                                                 ,'launched_at_weekday','deadline_month','deadline_yr'\
                                                 ,'created_at_month','created_at_yr'\
                                                 ,'launched_at_month','launched_at_yr'])

###prediction
X1 = kickstarter_df1.drop('usd_pledged', axis = 1)[['deadline_month_8', 'deadline_month_11', 'deadline_weekday_Saturday',\
                    'category_N/A', 'category_Software', 'deadline_month_5', 'created_at_yr_2016', 'deadline_yr_2012',\
                    'category_Wearables', 'launched_at_month_9', 'deadline_weekday_Monday', 'currency_GBP', 'currency_EUR',\
                    'created_at_weekday_Monday', 'created_at_weekday_Wednesday', 'created_at_yr_2015', 'created_at_month_9',\
                    'launched_at_yr_2015', 'launched_at_yr_2013', 'created_at_month_1', 'deadline_yr_2015', 'launched_at_month_8',\
                    'deadline_month_10', 'launched_at_month_3', 'category_Sound', 'deadline_month_6', 'created_at_weekday_Friday',\
                    'deadline_month_3', 'created_at_month_6', 'category_Flight', 'deadline_yr_2016', 'name_len', 'launched_at_weekday_Sunday',\
                    'deadline_month_7', 'deadline_yr_2014', 'launched_at_yr_2012', 'created_at_yr_2012', 'category_Hardware', 'category_Gadgets',\
                    'deadline_month_12', 'launched_at_month_1', 'launched_at_month_10', 'launched_at_weekday_Monday', 'launched_at_month_11',\
                    'created_at_weekday_Thursday', 'launched_at_yr_2016', 'launch_to_deadline_days', 'created_at_yr_2014', 'launched_at_month_7',\
                    'goal', 'created_at_weekday_Tuesday', 'created_at_yr_2013', 'created_at_month_5', 'created_at_month_10', 'launched_at_weekday_Tuesday',\
                    'created_at_month_2', 'created_at_month_7', 'deadline_weekday_Friday', 'deadline_weekday_Thursday', 'category_Robots',\
                    'create_to_launch_days', 'launched_at_yr_2014', 'launched_at_day', 'deadline_weekday_Tuesday', 'created_at_month_11',\
                    'launched_at_weekday_Wednesday', 'launched_at_month_2', 'launched_at_month_6', 'deadline_month_2', 'deadline_weekday_Wednesday',\
                    'created_at_month_4', 'launched_at_weekday_Thursday', 'deadline_month_1', 'deadline_month_4', 'launched_at_month_5',\
                    'deadline_yr_2017', 'created_at_month_8', 'currency_USD', 'deadline_yr_2013', 'deadline_month_9', 'created_at_month_3']]
y1 = kickstarter_df1["usd_pledged"]
###split the data for training and testing
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size = 0.3, random_state = 18)
RFR = RandomForestRegressor(random_state = 6, min_samples_split = 2, n_estimators = 100)
model1 = RFR.fit(X_train1,y_train1)
###using the model to predict the results based on the test dataset
y_test_pred1 = RFR.predict(X_test1)
###calculate the mean squared error of the prediction
mse = mean_squared_error(y_test1, y_test_pred1)
print(mse) ###7022160575.312807

#############################################################################################################################
#############################################################################################################################
#############################################################################################################################

###Question2
###final dataframe to use for question 1
kickstarter_df2 = kickstarter_df.drop(['project_id','name','pledged','disable_communication','country','deadline','state_changed_at'\
                     ,'created_at','launched_at','staff_pick','backers_count','static_usd_rate','usd_pledged','spotlight'\
                     ,'name_len_clean','blurb_len_clean','state_changed_at_weekday'\
                     ,'state_changed_at_month','state_changed_at_day','state_changed_at_yr'\
                     ,'state_changed_at_hr','launch_to_state_change_days'], axis = 1)
###check if final1 table has nan value
kickstarter_df2.isnull().any().any() ###false -> there is no null value
###dummify categorical variables 
kickstarter_df2 = pd.get_dummies(kickstarter_df2, columns = ['state','currency','category','deadline_weekday','created_at_weekday'\
                                                 ,'launched_at_weekday','deadline_month','deadline_yr'\
                                                 ,'created_at_month','created_at_yr'\
                                                 ,'launched_at_month','launched_at_yr'])
###failed = 1 successful = 0
kickstarter_df2 = kickstarter_df2.drop('state_successful', axis = 1)
kickstarter_df2 = kickstarter_df2.rename(columns={'state_failed': 'state'})

###prediction
X2 = kickstarter_df2.drop('state', axis = 1)[['goal', 'name_len', 'blurb_len', 'deadline_day', 'deadline_hr', 'created_at_day',\
                         'created_at_hr', 'launched_at_day', 'launched_at_hr', 'create_to_launch_days', 'launch_to_deadline_days',\
                         'currency_EUR', 'currency_GBP', 'currency_USD', 'category_Experimental', 'category_Festivals', 'category_Gadgets',\
                         'category_Hardware', 'category_Musical', 'category_N/A', 'category_Plays', 'category_Software', 'category_Wearables',\
                         'category_Web', 'deadline_weekday_Friday', 'deadline_weekday_Monday', 'deadline_weekday_Saturday',\
                         'deadline_weekday_Sunday', 'deadline_weekday_Thursday', 'deadline_weekday_Tuesday', 'deadline_weekday_Wednesday',\
                         'created_at_weekday_Friday', 'created_at_weekday_Monday', 'created_at_weekday_Saturday', 'created_at_weekday_Sunday',\
                         'created_at_weekday_Thursday', 'created_at_weekday_Tuesday', 'created_at_weekday_Wednesday', 'launched_at_weekday_Friday',\
                         'launched_at_weekday_Monday', 'launched_at_weekday_Saturday', 'launched_at_weekday_Thursday', 'launched_at_weekday_Tuesday',\
                         'launched_at_weekday_Wednesday', 'deadline_month_2', 'deadline_month_3', 'deadline_month_4', 'deadline_month_5', 'deadline_month_6',\
                         'deadline_month_7', 'deadline_month_8', 'deadline_month_9', 'deadline_month_10', 'deadline_month_11', 'deadline_month_12',\
                         'deadline_yr_2013', 'deadline_yr_2014', 'deadline_yr_2015', 'deadline_yr_2016', 'created_at_month_1', 'created_at_month_2',\
                         'created_at_month_3', 'created_at_month_4', 'created_at_month_5', 'created_at_month_6', 'created_at_month_7',\
                         'created_at_month_8', 'created_at_month_9', 'created_at_month_10', 'created_at_month_11', 'created_at_month_12',\
                         'created_at_yr_2014', 'created_at_yr_2015', 'created_at_yr_2016', 'launched_at_month_2', 'launched_at_month_3',\
                         'launched_at_month_4', 'launched_at_month_5', 'launched_at_month_6', 'launched_at_month_7', 'launched_at_month_8',\
                         'launched_at_month_9', 'launched_at_month_10', 'launched_at_month_11', 'launched_at_yr_2014', 'launched_at_yr_2015', 'launched_at_yr_2016']]
y2 = kickstarter_df2['state']
###split the data for training and testing
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size = 0.3, random_state = 18)
RFC = RandomForestClassifier(random_state = 17, max_features = 9, max_depth = 44, min_samples_split = 17, n_estimators = 100)
###get the cv score 
scores = cross_val_score(estimator = RFC, X = X2, y = y2, cv = 5)
print(np.average(scores)) ###0.7339215435727064
###get the accuracy score
model2 = RFC.fit(X_train2,y_train2)
y_test_pred2 = model2.predict(X_test2) 
ascore = accuracy_score(y_test2, y_test_pred2)
print(ascore) ###0.7468731387730793




























