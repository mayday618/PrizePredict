# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 20:42:54 2017

@author: huhu
"""
import pandas as pd
import numpy as np
test = pd.read_csv('f:/prize/PrizePredict/data/final_test/library_final_test.txt')
train = pd.read_csv('f:/prize/PrizePredict/data/train/library_train.txt')
#%%
train.columns=['id','door','date']
test.columns=['id','door','date']
train_test = pd.concat([train,test])
count = pd.DataFrame(train_test.groupby(['id'])['date'].count())   ##生成进出图书馆的次数
group_show = train_test.groupby('id')

#%%
# 生成时间信息

train_test.time = pd.to_datetime(train_test.date, format='%Y/%m/%d %H:%M:%S')
train_test['month'] = train_test.time.dt.month
train_test['weekday'] = train_test.time.dt.weekday
train_test['days_in_month'] = train_test.time.dt.day
train_test['hour'] = train_test.time.dt.hour
#%%
##  生成进入图书馆和离开图书馆的统计信息
count['LibMin'] =pd.DataFrame(train_test.groupby(['id'])['hour'].min())
count['LibMean'] =pd.DataFrame(train_test.groupby(['id'])['hour'].mean())
count['LibMax'] =pd.DataFrame(train_test.groupby(['id'])['hour'].max())
count['LibMed'] =pd.DataFrame(train_test.groupby(['id'])['hour'].median())
#%%  生成周末以及周中进出图书馆的次数
train_test_wd=train_test[train_test.weekday<5] 
train_test_we=train_test[train_test.weekday>=5] 
count['LibWDMin'] =pd.DataFrame(train_test_wd.groupby(['id'])['hour'].min())
count['LibWDMean'] =pd.DataFrame(train_test_wd.groupby(['id'])['hour'].mean())
count['LibWDMax'] =pd.DataFrame(train_test_wd.groupby(['id'])['hour'].max())
count['LibWDMed'] =pd.DataFrame(train_test_wd.groupby(['id'])['hour'].median())
count['LibWEMin'] =pd.DataFrame(train_test_we.groupby(['id'])['hour'].min())
count['LibWEMean'] =pd.DataFrame(train_test_we.groupby(['id'])['hour'].mean())
count['LibWEMax'] =pd.DataFrame(train_test_we.groupby(['id'])['hour'].max())
count['LibWEMed'] =pd.DataFrame(train_test_we.groupby(['id'])['hour'].median())
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
del count['date']
count.to_csv('f:/prize/PrizePredict/data/input/library.csv')