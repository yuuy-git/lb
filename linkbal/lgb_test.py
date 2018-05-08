import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from pathlib import Path
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb


######################################
# Setting paths, columns and dtypes. #
######################################
current_dir = Path('.')
path_dir = current_dir / 'datasets'
path_log = path_dir / 'log.tsv'
path_events = path_dir / 'events.tsv'
path_users = path_dir / 'users.tsv'



log_cols = [
    'user_id', 'event_id', 'time_stamp', 'action_type',
    'num_of_people', 'payment_methond', 'total_price'
]

dtypes = {
        'user_id'          : 'int',
        'event_id'         : 'int',
        'time_stamp'       : 'str',
        'action_type'      : 'int',
        'num_of_people'    : 'float',
        'payment_methond'  : 'str',
        'total_price'      : 'float',
        'female_age_lower' : 'int',
        'female_age_upper' : 'int',
        'male_age_lower'   : 'int',
        'male_age_upper'   : 'int',
        'event_start_at'   : 'str',
        'prefecture'       : 'str',
        'first_published_at': 'str',
        'female_price'     : 'int',
        'male_price'       : 'int',
        'interest'         : 'str',
        'age'              : 'int',
        'gender'           : 'str',
        'createed_at'      : 'str'
}


prefecture_dict = {
    '北海道'  :   0,
    '青森県'  :   1,
    '岩手県'  :   2,
    '宮城県'  :   3,
    '秋田県'  :   4,
    '山形県'  :   5,
    '福島県'  :   6,
    '茨城県'  :   7,
    '栃木県'  :   8,
    '群馬県'  :   9,
    '埼玉県'  :  10,
    '千葉県'  :  11,
    '東京都'  :  12,
    '神奈川県':  13,
    '新潟県'  :  14,
    '富山県'  :  15,
    '石川県'  :  16,
    '福井県'  :  17,
    '山梨県'  :  18,
    '長野県'  :  19,
    '岐阜県'  :  20,
    '静岡県'  :  21,
    '愛知県'  :  22,
    '三重県'  :  23,
    '滋賀県'  :  24,
    '京都府'  :  25,
    '大阪府'  :  26,
    '兵庫県'  :  27,
    '奈良県'  :  28,
    '和歌山県':  29,
    '鳥取県'  :  30,
    '島根県'  :  31,
    '岡山県'  :  32,
    '広島県'  :  33,
    '山口県'  :  34,
    '徳島県'  :  35,
    '香川県'  :  36,
    '愛媛県'  :  37,
    '高知県'  :  38,
    '福岡県'  :  39,
    '佐賀県'  :  40,
    '長崎県'  :  41,
    '熊本県'  :  42,
    '大分県'  :  43,
    '宮崎県'  :  44,
    '鹿児島県':  45,
    '沖縄県'  :  46,
    'その他（海外等）': 47
}


gender_dict = {'男性':0, '女性':1}

payment_method_dict = {
    'クレカ'   :  0,
    'コンビニ' :  1,
    'eマネー'  :  2,
    '銀振'     :  3
}

interest_dict = {
    'アニメコン'                           :   0,
    '謎解きコン'                           :   1,
    'その他（スポーツコン）'                 :   2,
    '料理合コン（料理コン）'                 :   3,
    'スポーツコン'                         :   4,
    'フットサルコン'                       :   5,
    'ボルダリングコン'                      :   6,
    'テニスコン'                           :   7,
    '旅コン'                              :   8,
    'サバコン'                            :   9,
    '婚活バスツアー（お見合いバスツアー）'     :  10,
    '釣りコン'                            :  11,
    'その他'                              :  12
}

action_type_dict = {
    1 : 0,
    2 : 1,
    3 : 2
}

###########################################################
# Reading tsv and merge them on 'user_id' and 'event_id'. #
###########################################################

print ('Loading and merging csv')

df_log = pd.read_table(path_log, header=0, parse_dates=True, na_values='n/a')
df_events = pd.read_table(path_events, header=0, parse_dates=True, na_values='n/a')
df_users = pd.read_table(path_users, header=0, parse_dates=True, na_values='n/a')

df = pd.merge(df_log, df_events, on='event_id', how='left')
df = pd.merge(df, df_users, on='user_id', how='left', suffixes=('_event', '_user'))

del df_log, df_events, df_users

gc.collect()


##############################
# Replacing strings for int. #
##############################
df_rp = df.copy()

df_rp['prefecture_event'] = df_rp.prefecture_event.map(prefecture_dict)
df_rp['prefecture_user'] = df_rp.prefecture_user.map(prefecture_dict)
df_rp['gender'] = df_rp.gender.map(gender_dict)
df_rp['payment_method'] = df_rp.payment_method.map(payment_method_dict)
df_rp['interest'] = df_rp.interest.map(interest_dict)
df_rp['action_type'] = df_rp.action_type.map(action_type_dict)

df_rp = df_rp.fillna(9999)

del df
gc.collect()

######################################
# Making datasets for train and test #
######################################
print('Making datasets for train and test')

'''
:param: x : all columns except user_id, event_id and y(action_type)
:param: y : action_type column
'''
x_with_id = df_rp.drop(['action_type', 'time_stamp', 'event_start_at',
                       'first_published_at', 'created_on'], axis=1).values
y = df_rp.iloc[:, 3].values




##############################################################
# Splitting the datasets into the Training set and Test set. #
##############################################################
print ('Splitting the datasets into the Training set and Test set')

x_train_with_id, x_test_with_id, y_train, y_test = train_test_split(x_with_id, y, test_size = 0.25, random_state = 0)
x_train = x_train_with_id[:, 2:]
x_test = x_test_with_id[:, 2:]
x_train_id = x_train_with_id[:, [0, 1]]
x_test_id = x_test_with_id[:, [0, 1]]



############################
# Setting Dataset for LGB. #
############################
target = "action_type"
predictors = ['num_of_people','payment_method', 'total_price',
                'female_age_lower','female_age_upper',
                'male_age_lower','male_age_upper',
                'prefecture_event', 'female_price','male_price',
                'interest', 'age', 'gender', 'prefecture_user']

categorical_features = ['payment_method','prefecture_event',
                        'interest', 'gender', 'prefecture_user']

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
    'num_class': 3,
    'subsample_for_bin': 200000,  # Number of samples for constructing bin
    'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
    'reg_alpha': 0,  # L1 regularization term on weights
    'reg_lambda': 0,  # L2 regularization term on weights
    'nthread': 4,  # Number of threads for LightGBM
    'verbose': 0,
    'metric':'auc',

    'learning_rate': 0.1,
    'num_leaves': 15,  # 2^max_depth - 1
    'max_depth': 4,  # -1 means no limit
    'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 100,  # Number of bucketed bin for feature values
    'subsample': 0.7,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'scale_pos_weight':99
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
     valid_sets=lgb_eval, #  List of data to be evaluated during training.
     evals_result=evals_result,
     num_boost_round=num_boost_round,
     early_stopping_rounds=early_stopping_rounds,
#     verbose_eval=1  # This tells us how it is going on in each iteretion.
)

print('Training is done.')


print('Start predicting...')

predictions=booster.predict(x_test, num_iteration=booster.best_iteration)

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
    results[col] = x_test_id[:, i]

prediction_list = ['click', 'favorite', 'entry']
for i, col in enumerate(prediction_list):
    results[col]  = predictions[:, i]

results.to_csv('lgb_result.csv', float_format='%.8f', index=False)

print('Saving the model...')
booster.save_model('lightgbm_model.txt')
print('Above lightgbm_model.txt file is saved at your local file system, mostly where jupyter notebook started')

print("All done.")
