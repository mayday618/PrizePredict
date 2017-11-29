# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from  ggplot  import *   
import  matplotlib.pyplot   as plt  
"""
borrow_train = pd.read_csv("f:/prize/PrizePredict/data/train/borrow_train.txt",error_bad_lines=False)
borrow_test = pd.read_csv("f:/prize/PrizePredict/data/final_test/borrow_final_test.txt",error_bad_lines=False)

borrow_test.columns = ['id','time','book','bookId']
borrow_train.columns = ['id','time','book','bookId']

book_train = borrow_train['book']
book_test = borrow_test['book']

book_res = pd.concat([book_train,book_test],axis=0)

book_res.to_excel("f:/prize/book.xlsx",index=False)
"""
card_train = pd.read_csv("f:/prize/PrizePredict/data/train/card_train.txt",error_bad_lines=False)
card_test = pd.read_csv("f:/prize/PrizePredict/data/final_test/card_final_test.txt",error_bad_lines=False)

card_test.columns = ['id','payType','type','place','time','money','left']
card_train.columns = ['id','payType','type','place','time','money','left']
card=pd.concat([card_train,card_test],axis=0)
