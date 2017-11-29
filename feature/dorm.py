# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 20:43:36 2017

@author: huhu
"""
import pandas as pd
import numpy as np
test = pd.read_csv('f:/prize/PrizePredict/data/final_test/dorm_final_test.txt')
train = pd.read_csv('f:/prize/PrizePredict/data/train/dorm_train.txt')
#%%
train.columns=['id','time','dir']
test.columns=['id','time','dir']
train_test = pd.concat([train,test])
#%%
# 生成进出总次数
count = pd.DataFrame(train_test.groupby(['id'])['time'].count())   ##生成学生进出宿舍的总次数
group_show = train_test.groupby('id')

#%%
# 生成时间信息

train_test.time = pd.to_datetime(train_test.time, format='%Y/%m/%d %H:%M:%S')
train_test['month'] = train_test.time.dt.month
train_test['weekday'] = train_test.time.dt.weekday
train_test['days_in_month'] = train_test.time.dt.day
train_test['hour'] = train_test.time.dt.hour
#%%
##  生成进入宿舍和离开宿舍的统计信息
train_test_in = train_test[train_test.dir==1]
count['dormInMin'] =pd.DataFrame(train_test_in.groupby(['id'])['hour'].min())
count['dormInMean'] =pd.DataFrame(train_test_in.groupby(['id'])['hour'].mean())
count['dormInMax'] =pd.DataFrame(train_test_in.groupby(['id'])['hour'].max())
count['dormInMed'] =pd.DataFrame(train_test_in.groupby(['id'])['hour'].median())
train_test_out = train_test[train_test.dir==0]
count['dormOutMin'] =pd.DataFrame(train_test_out.groupby(['id'])['hour'].min())
count['dormOutMean'] =pd.DataFrame(train_test_out.groupby(['id'])['hour'].mean())
count['dormOutMax'] =pd.DataFrame(train_test_out.groupby(['id'])['hour'].max())
count['dormOutMed'] =pd.DataFrame(train_test_out.groupby(['id'])['hour'].median())
#%%  生成周末以及周中进出宿舍的次数
train_test_wd=train_test[train_test.weekday<5] 
train_test_we=train_test[train_test.weekday>=5] 
count['dormWDMin'] =pd.DataFrame(train_test_wd.groupby(['id'])['hour'].min())
count['dormWDMean'] =pd.DataFrame(train_test_wd.groupby(['id'])['hour'].mean())
count['dormWDMax'] =pd.DataFrame(train_test_wd.groupby(['id'])['hour'].max())
count['dormWDMed'] =pd.DataFrame(train_test_wd.groupby(['id'])['hour'].median())
count['dormWEMin'] =pd.DataFrame(train_test_we.groupby(['id'])['hour'].min())
count['dormWEMean'] =pd.DataFrame(train_test_we.groupby(['id'])['hour'].mean())
count['dormWEMax'] =pd.DataFrame(train_test_we.groupby(['id'])['hour'].max())
count['dormWEMed'] =pd.DataFrame(train_test_we.groupby(['id'])['hour'].median())
#%%  
## 按时间段生成进出次数
feature=train_test[train_test.hour<=7]
count['H7-count'] =pd.DataFrame(feature.groupby(['id'])['time'].count())
feature=train_test[(train_test.hour>=7)&(train_test.hour<=12)]
count['H7-12count'] =pd.DataFrame(feature.groupby(['id'])['time'].count())
feature=train_test[(train_test.hour>=12)&(train_test.hour<=19)]
count['H12-19count'] =pd.DataFrame(feature.groupby(['id'])['time'].count())
feature=train_test[train_test.hour>=19]
count['H19+count'] =pd.DataFrame(feature.groupby(['id'])['time'].count())
#%%
del count['time']
count.to_csv('f:/prize/PrizePredict/data/input/dorm.csv')