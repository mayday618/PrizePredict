# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 16:53:44 2017

@author: huhu
"""

# -*-coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

print "Import package OK!"

print "Reading data..."
train = pd.read_table('f:/prize/PrizePredict/data/train/subsidy_train.txt',sep=',',header=-1)
train.columns = ['id','label']
test = pd.read_table('f:/prize/PrizePredict/data/final_test/subsidy_final_test.txt',sep=',',header=-1)
test.columns = ['id']
test['label'] = np.nan

train_test = pd.concat([train,test])
del train
del test

card_train = pd.read_table('f:/prize/PrizePredict/data/train/card_train.txt',sep=',',header=-1)
card_train.columns = ['id','pos','place','consume','time','price','rest']
card_test = pd.read_table('f:/prize/PrizePredict/data/final_test/card_final_test.txt',sep=',',header=-1)
card_test.columns = ['id','pos','place','consume','time','price','rest']
#
card_train_test = pd.concat([card_train,card_test])
print "All right!"

##release memery
del card_train
del card_test

##

print "Feature beginning..."


for var in ['loc21','loc829','loc818','loc213','loc72','loc283','loc91','loc245','loc65','loc161','loc996','loc277','loc842','loc75','loc263','loc840']:
    
    place = card_train_test[card_train_test.place == var]
    
    feature_p = pd.DataFrame(place.groupby(['id'])['pos'].count())
    feature_p['%s_sum'%var]=place.groupby(['id'])['price'].sum()
    feature_p['%s_avg'%var]=place.groupby(['id'])['price'].mean()
    feature_p['%s_max'%var]=place.groupby(['id'])['price'].max()
    feature_p['%s_min'%var]=place.groupby(['id'])['price'].min()
    feature_p['%s_median'%var]=place.groupby(['id'])['price'].median()
    
    del place
    feature_p.to_csv('f:/prize/PrizePredict/data/input/loc/card_%sfeature.csv'%var,index=True)
    feature_p=pd.read_csv('f:/prize/PrizePredict/data/input/loc/card_%sfeature.csv'%var)
    feature_p=feature_p.rename(columns={'pos' : '%s_count'%var})
    
    train_test = pd.merge(train_test, feature_p, how='left',on='id') 
    del feature_p

print "OK!"
##release memery
del card_train_test

print "Feature 3 end."

