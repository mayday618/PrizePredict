import pandas as pd
import numpy as np

card_train = pd.read_csv("f:/prize/PrizePredict/data/train/card_train.txt",error_bad_lines=False)
card_test = pd.read_csv("f:/prize/PrizePredict/data/final_test/card_final_test.txt",error_bad_lines=False)
card_test.columns = ['id','payType','type','place','time','money','left']
card_train.columns = ['id','payType','type','place','time','money','left']
print card_test.counts
print card_train.counts

