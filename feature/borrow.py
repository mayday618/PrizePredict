# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 20:22:50 2017

@author: huhu
"""



import pandas as pd
import numpy as np
test_borrow = pd.read_csv('f:/prize/PrizePredict/data/final_test/borrow_final_test.txt',error_bad_lines=False)
train_borrow = pd.read_csv('f:/prize/PrizePredict/data/train/borrow_train.txt',error_bad_lines=False)

#%%
train_borrow.columns=['id','date','name','bookId']
test_borrow.columns=['id','date','name','bookId']
borrow_train_test = pd.concat([train_borrow,test_borrow])
borrow = pd.DataFrame(borrow_train_test.groupby(['id'])['name'].count())   ##生成借书数量
borrow.to_csv('f:/prize/PrizePredict/data/input/borrow.csv')