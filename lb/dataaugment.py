# coding:utf-8

import pandas as pd
from pathlib import Path

######################################
# Setting paths, columns and dtypes. #
######################################
print('Setting paths.')
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
        'user_id': 'int',
        'event_id': 'int',
        'time_stamp': 'str',
        'action_type': 'int',
        'num_of_people': 'float',
        'payment_methond': 'str',
        'total_price': 'float',
        'female_age_lower': 'int',
        'female_age_upper': 'int',
        'male_age_lower': 'int',
        'male_age_upper': 'int',
        'event_start_at': 'str',
        'prefecture': 'str',
        'first_published_at': 'str',
        'female_price': 'int',
        'male_price': 'int',
        'interest': 'str',
        'age': 'int',
        'gender': 'str',
        'createed_at': 'str'
}


prefecture_dict = {
    '北海道': 0,
    '青森県': 1,
    '岩手県': 2,
    '宮城県': 3,
    '秋田県': 4,
    '山形県': 5,
    '福島県': 6,
    '茨城県': 7,
    '栃木県': 8,
    '群馬県': 9,
    '埼玉県': 10,
    '千葉県': 11,
    '東京都': 12,
    '神奈川県': 13,
    '新潟県': 14,
    '富山県': 15,
    '石川県': 16,
    '福井県': 17,
    '山梨県': 18,
    '長野県': 19,
    '岐阜県': 20,
    '静岡県': 21,
    '愛知県': 22,
    '三重県': 23,
    '滋賀県': 24,
    '京都府': 25,
    '大阪府': 26,
    '兵庫県': 27,
    '奈良県': 28,
    '和歌山県': 29,
    '鳥取県': 30,
    '島根県': 31,
    '岡山県': 32,
    '広島県': 33,
    '山口県': 34,
    '徳島県': 35,
    '香川県': 36,
    '愛媛県': 37,
    '高知県': 38,
    '福岡県': 39,
    '佐賀県': 40,
    '長崎県': 41,
    '熊本県': 42,
    '大分県': 43,
    '宮崎県': 44,
    '鹿児島県': 45,
    '沖縄県': 46,
    'その他（海外等）': 47
}


gender_dict = {'男性': 0, '女性': 1}

payment_method_dict = {
    'クレカ': 0,
    'コンビニ': 1,
    'eマネー': 2,
    '銀振': 3
}

interest_dict = {
    'アニメコン': 0,
    '謎解きコン': 1,
    'その他（スポーツコン）': 2,
    '料理合コン（料理コン）': 3,
    'スポーツコン': 4,
    'フットサルコン': 5,
    'ボルダリングコン': 6,
    'テニスコン': 7,
    '旅コン': 8,
    'サバコン': 9,
    '婚活バスツアー（お見合いバスツアー）': 10,
    '釣りコン': 11,
    'その他': 12
}



column_list = [
    'prefecture_event', 'prefecture_user', 'gender',
    'payment_method', 'interest_dict', 'action_type'
]
dict_list = [
    prefecture_dict, prefecture_dict, gender_dict,
    payment_method_dict, interest_dict
]

################
# Reading tsv. #
################
print ('Loading and merging csv.')

df_log = pd.read_table(path_log, header=0,
                       parse_dates=True, na_values='n/a')
df_events = pd.read_table(path_events, header=0,
                          parse_dates=True, na_values='n/a')
df_users = pd.read_table(path_users, header=0,
                         parse_dates=True, na_values='n/a')

####################################################
# Replacing str for int and filling NaN with 9999. #
####################################################
print('Replacing str for int and filling NaN with 9999.')
df_events['prefecture'] = df_events.prefecture.map(prefecture_dict)
df_users['prefecture'] = df_users.prefecture.map(prefecture_dict)
df_users['gender'] = df_users.gender.map(gender_dict)
df_log['payment_method'] = df_log.payment_method.map(payment_method_dict)
df_events['interest'] = df_events.interest.map(interest_dict)

df_log = df_log.fillna(9999)
df_users = df_users.fillna(9999)
df_events = df_events.fillna(9999)

###########################################################
# Dropping rows of users that's not included in user.tsv. #
###########################################################
print("Dropping rows of users that's not inclued in user.tsc.")
df_log = df_log[df_log['user_id'].isin(df_users['user_id'])]

############################################
# Converting str data of date to Datetime. #
############################################
print("Converting str data of date to Datetime.")
df_users.iloc[:, 4] = pd.to_datetime(
    df_users.iloc[:, 4], format='%Y-%m-%d %H:%M:%S')
df_events.iloc[:, 5] = pd.to_datetime(
    df_events.iloc[:, 5], format='%Y-%m-%d %H:%M:%S')
df_events.iloc[:, 7] = pd.to_datetime(
    df_events.iloc[:, 7], format='%Y-%m-%d %H:%M:%S')
df_log.iloc[:, 2] = pd.to_datetime(
    df_log.iloc[:, 2], format='%Y-%m-%d %H:%M:%S')

################################################################
# Replacing '1970-01-01 00:00:00.000009999' with               #
#    the mean of difference between first_published_at         #
#       and event_start_at in all rows except the above value. #
################################################################
print("Handling NaT in 'first_published_at' column.")
dt_na = '1970-01-01 00:00:00.000009999'
df_events_2 = df_events[df_events['first_published_at'] != dt_na]
df_events_2['diff'] = df_events_2['first_published_at']\
                        - df_events_2['event_start_at']
diff_mean = df_events_2['diff'].mean()
df_events['first_published_at'] = df_events['first_published_at'].where(
                                    df_events['first_published_at'] != dt_na,
                                    df_events['event_start_at'] + diff_mean)

######################################################
# Making log csv for each user with action_type = 0. #
######################################################
# Making an event list
event_list = pd.DataFrame()
event_list['event_id'] = df_events.event_id
event_list['event_start_at'] = df_events.event_start_at

for i, v in enumerate(df_users['user_id']):
    print("Making a log_csv for 'user_id' = {}".format(v))
    # Merging event_list and df_test_user whrere event_start_at > created_on
    this_users_created_on = df_users.created_on[i]
    df_index = pd.DataFrame()
    df_index = event_list
    log_user = df_log[df_log['user_id'] == v]
    df_index = pd.merge(
        df_index[event_list.event_start_at > this_users_created_on],
        log_user[event_list.event_start_at > this_users_created_on],
        on='event_id', how='left')

    # Droping log.tsv's info and fill action_type=NaN with 0
    df_index = df_index.drop(['event_start_at', 'time_stamp', 'num_of_people',
                              'payment_method', 'total_price'], axis=1)
    df_index = df_index.iloc[:, [1, 0, 2]]
    df_index['user_id'] = v
    df_index = df_index.fillna(0)

    # Adding info of user.tsv and event.tsv to df_index
    df_index = pd.merge(df_index, df_users, on='user_id', how='left')
    df_index = pd.merge(df_index, df_events, on='event_id',
                        how='left', suffixes=('_user', '_event'))

    df_index.to_csv('log_{}.csv'.format(v), index=False)

print("All done.")
