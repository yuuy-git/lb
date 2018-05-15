# coding:utf-8

import pandas as pd
import numpy as np
from pathlib import Path
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('gender', type=str)
args = parser.parse_args()

if args.gender == 'male':
    gender = '男性'

if args.gender == 'female':
    gender == '女性'
######################################
# Setting paths, columns and dtypes. #
######################################
current_dir = Path('.')
path_dir = current_dir / 'datasets'
path_log = path_dir / 'log.tsv'
path_events = path_dir / 'events.tsv'
path_users = path_dir / 'users.tsv'

############################
# Reading and merging csv. #
############################
df_user = pd.read_table(path_users, header=0, parse_dates=True, na_values='n/a')
df_gender = df_user['user_id'][df_user['gender']==gender]
user_list = df_gender.values.tolist()
df = pd.DataFrame()
print('Reading CSV')
for user in user_list:
    path = Path('log_{}.csv'.format(user))
    csv = pd.read_csv(path)
    csv['created_year'] = pd.to_datetime(csv.created_on).dt.year.astype('int')
    csv['created_month'] = pd.to_datetime(csv.created_on).dt.month.astype('int')
    csv['event_month'] = pd.to_datetime(csv.event_start_at).dt.year.astype('int')
    csv['event_weekday'] = pd.to_datetime(csv.event_start_at).dt.weekday.astype('int')
    csv['event_hour'] = pd.to_datetime(csv.event_start_at).dt.hour.astype('int')
    csv['publish_month'] = pd.to_datetime(csv.first_published_at).dt.month.astype('int')
    csv['publish_weekday'] = pd.to_datetime(csv.first_published_at).dt.weekday.astype('int')
    csv['publish_hour'] = pd.to_datetime(csv.first_published_at).dt.hour.astype('int')
    csv = csv.drop(['gender', 'created_on', 'event_start_at', 'first_published_at'], axis=1)
    df = df.append(csv, ignore_index=True)

del df_user, df_gender, user_list
gc.collect()

######################################
# Making datasets for train and test #
######################################
print('Making datasets for train and test')

'''
:param: x : all columns except user_id, event_id and y(action_type)
:param: y : action_type column
'''
x = df.drop(['user_id', 'event_id', 'action_type'], axis=1).values
y = df.iloc[:, 2].values

##############################################################
# Splitting the datasets into the Training set and Test set. #
##############################################################
print ('Splitting the datasets into the Training set and Test set')

x_train, x_test, y_train, y_test = \
             train_test_split(x, y, test_size=0.25, random_state=0)

############################
# Setting Dataset for LGB. #
############################
target = "action_type"
predictors = ['age', 'prefecture_user', 'female_age_lower', 'female_age_upper',
              'male_age_lower', 'male_age_upper', 'prefecture_event',
              'female_price', 'male_price', 'interest', 'created_year',
              'created_month', 'event_month', 'event_weekday', 'event_hour',
              'publish_month', 'publish_weekday', 'publish_hour']

categorical_features = ['prefecture_user', 'prefecture_event', 'interest',
                        'created_year', 'created_month', 'event_month',
                        'event_weekday', 'event_hour', 'publish_month',
                        'publish_weekday', 'publish_hour']

lgb_train = lgb.Dataset(x_train, y_train,
                        feature_name=predictors,
                        categorical_feature=categorical_features)

lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

#######################
# Setting parameters. #
#######################

lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 4,
    'subsample_for_bin': 200000,  # Number of samples for constructing bin
    'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
    'reg_alpha': 0,  # L1 regularization term on weights
    'reg_lambda': 0,  # L2 regularization term on weights
    'nthread': 4,  # Number of threads for LightGBM
    'verbose': 0,
    'metric': 'auc',

    'learning_rate': 0.1,
    'num_leaves': 15,  # 2^max_depth - 1
    'max_depth': 4,  # -1 means no limit
    'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 100,  # Number of bucketed bin for feature values
    'subsample': 0.7,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'scale_pos_weight': 99
}

###########################
# Training and Predicting #
###########################

print('Start training...')

evals_result = {}
num_boost_round = 5
early_stopping_rounds = 30

booster = lgb.train(
     lgb_params,
     lgb_train,
     valid_sets=lgb_eval,  # List of data to be evaluated during training.
     evals_result=evals_result,
     num_boost_round=num_boost_round,
     early_stopping_rounds=early_stopping_rounds,
     #verbose_eval=1  # This tells us how it is going on in each iteretion.
)

print('Training is done.')

print('Start predicting...')

predictions = booster.predict(x_test, num_iteration=booster.best_iteration)

print('Prediction is done.')

###################################
# Showing and saving the results. #
###################################

print('Feature names:', booster.feature_name())
print('Calculate feature importances...')
print('Feature importances:', list(booster.feature_importance()))

print('Exporting to csv')
results = pd.DataFrame()

results_list = ['user_id', 'event_id']
for i, col in enumerate(results_list):
    results[col] = x_test[:, i]

prediction_list = ['click', 'favorite', 'entry']
for i, col in enumerate(prediction_list):
    results[col] = predictions[:, i]

results.to_csv('lgb_result_{}.csv'.format(args.gender), float_format='%.8f', index=False)

print('Saving the model...')
booster.save_model('lightgbm_model_{}.txt'.format(args.gender))
print('Above lightgbm_model.txt file is saved at your local file system,\
      mostly where jupyter notebook started')

print("All done.")
