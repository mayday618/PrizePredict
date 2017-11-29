# -*-coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

print "Import package OK!"
print "Reading data..."
train = pd.read_table('f:/prize/PrizePredict/data/train/subsidy_train.txt',sep=',',header=-1)
train.columns = ['id','label']
test = pd.read_table('f:/prize/PrizePredict/data/final_test/test.txt',sep=',',header=-1)
test.columns = ['id']
test['label'] = np.nan
#%%
train_test = pd.concat([train,test])
#%%

del train
del test

score_train = pd.read_table('f:/prize/PrizePredict/data/train/score_train.txt',sep=',',header=-1)
score_train.columns = ['id','college','rank']
score_test = pd.read_table('f:/prize/PrizePredict/data/final_test/score_final_test.txt',sep=',',header=-1)
score_test.columns = ['id','college','rank']
score_train_test = pd.concat([score_train,score_test])
del score_train
del score_test


college = pd.read_csv('f:/prize/PrizePredict/data/input/college.csv')
college.columns = ['college','total_people']
score_train_test = pd.merge(score_train_test, college, how='left',on='college')
del college

score_train_test['rank_percent'] = score_train_test['rank']/score_train_test['total_people']
train_test = pd.merge(train_test,score_train_test,how='left',on='id')


del score_train_test
print "All right!"
#%%
# 合并借书数量信息
borrow = pd.read_csv('f:/prize/PrizePredict/data/input/borrow.csv')
train_test = pd.merge(train_test, borrow, how='left',on='id')
#%%
# 合并宿舍信息
dorm = pd.read_csv('f:/prize/PrizePredict/data/input/dorm.csv')
train_test = pd.merge(train_test, dorm, how='left',on='id')
#%%
# 合并图书馆信息
library = pd.read_csv('f:/prize/PrizePredict/data/input/library.csv')
train_test = pd.merge(train_test, library, how='left',on='id')
#%%
## 合并时间特征
print "Merge beginning..."

for m in range(1,13):
    card = pd.read_csv('f:/prize/PrizePredict/data/input/time/featureM%d.csv'%m) 
    card=card.rename(columns={'pos' : 'countM%d'%m}) 
    train_test = pd.merge(train_test, card, how='left',on='id')
    del card

## hours <7|7|8|9|17|18|19|19>

card = pd.read_csv('f:/prize/PrizePredict/data/input/time/featureH7-.csv') 
card=card.rename(columns={'pos' : 'countH7-'}) 
train_test = pd.merge(train_test, card, how='left',on='id')
del card


for h in [7,8,9,17,18,19]:        
    card = pd.read_csv('f:/prize/PrizePredict/data/input/time/featureH%d.csv'%h) 
    card=card.rename(columns={'pos' : 'countH%d'%h}) 
    train_test = pd.merge(train_test, card, how='left',on='id')
    del card

card = pd.read_csv('f:/prize/PrizePredict/data/input/time/featureH19+.csv') 
card=card.rename(columns={'pos' : 'countH19+'}) 
train_test = pd.merge(train_test, card, how='left',on='id')
del card

## week
card = pd.read_csv('f:/prize/PrizePredict/data/input/time/featureWD.csv') 
train_test = pd.merge(train_test, card, how='left',on='id')
card=card.rename(columns={'pos' : 'countWD'}) 
del card

card = pd.read_csv('f:/prize/PrizePredict/data/input/time/featureWE.csv')     
card=card.rename(columns={'pos' : 'countWE'}) 
train_test = pd.merge(train_test, card, how='left',on='id')
del card
#%%
## 合并一卡通支付信息
## bash feature
card = pd.read_csv('f:/prize/PrizePredict/data/input/card/card_bashfeature.csv') 
card=card.rename(columns={'pos' : 'price_count'}) 
train_test = pd.merge(train_test, card, how='left',on='id') #2512
del card


## consume feature
card_consume = pd.read_csv('f:/prize/PrizePredict/data/input/card/card_consumefeature.csv') 
card_consume=card_consume.rename(columns={'pos' : 'consume_count'}) 
train_test = pd.merge(train_test, card_consume, how='left',on='id') 
del card_consume


card_kaihu = pd.read_csv('f:/prize/PrizePredict/data/input/card/card_kaihufeature.csv') 
card_kaihu=card_kaihu.rename(columns={'pos' : 'kaihu_count'}) 
train_test = pd.merge(train_test, card_kaihu, how='left',on='id') 
del card_kaihu

card_xiaohu = pd.read_csv('f:/prize/PrizePredict/data/input/card/card_xiaohufeature.csv') 
card_xiaohu=card_xiaohu.rename(columns={'pos' : 'xiaohu_count'}) 
train_test = pd.merge(train_test, card_xiaohu, how='left',on='id') 
del card_xiaohu

card_buban = pd.read_csv('f:/prize/PrizePredict/data/input/card/card_bubanfeature.csv') 
card_buban=card_buban.rename(columns={'pos' : 'buban_count'}) 
train_test = pd.merge(train_test, card_buban, how='left',on='id') 
del card_buban

card_jiegua = pd.read_csv('f:/prize/PrizePredict/data/input/card/card_jieguafeature.csv') 
card_jiegua=card_jiegua.rename(columns={'pos' : 'jiegua_count'}) 
train_test = pd.merge(train_test, card_jiegua, how='left',on='id') 
del card_jiegua

card_change = pd.read_csv('f:/prize/PrizePredict/data/input/card/card_changefeature.csv') 
card_change=card_change.rename(columns={'pos' : 'change_count'}) 
train_test = pd.merge(train_test, card_change, how='left',on='id') 
del card_change

## place infor
card_canteen = pd.read_csv('f:/prize/PrizePredict/data/input/card/card_canteenfeature.csv')
card_canteen=card_canteen.rename(columns={'pos' : 'canteen_count'}) 
train_test = pd.merge(train_test, card_canteen, how='left',on='id') 
del card_canteen

card_boiled_water = pd.read_csv('f:/prize/PrizePredict/data/input/card/card_boiled_waterfeature.csv') 
card_boiled_water=card_boiled_water.rename(columns={'pos' : 'boiled_water_count'}) 
train_test = pd.merge(train_test, card_boiled_water, how='left',on='id') 
del card_boiled_water

card_bathe = pd.read_csv('f:/prize/PrizePredict/data/input/card/card_bathefeature.csv') 
card_bathe=card_bathe.rename(columns={'pos' : 'bathe_count'}) 
train_test = pd.merge(train_test, card_bathe, how='left',on='id') 
del card_bathe


card_shool_bus = pd.read_csv('f:/prize/PrizePredict/data/input/card/card_shool_busfeature.csv') 
card_shool_bus=card_shool_bus.rename(columns={'pos' : 'shool_bus_count'}) 
train_test = pd.merge(train_test, card_shool_bus, how='left',on='id') 
del card_shool_bus


card_shop = pd.read_csv('f:/prize/PrizePredict/data/input/card/card_shopfeature.csv') 
card_shop=card_shop.rename(columns={'pos' : 'shop_count'}) 
train_test = pd.merge(train_test, card_shop, how='left',on='id') 
del card_shop

card_wash_house = pd.read_csv('f:/prize/PrizePredict/data/input/card/card_wash_housefeature.csv') 
card_wash_house=card_wash_house.rename(columns={'pos' : 'wash_house_count'}) 
train_test = pd.merge(train_test, card_wash_house, how='left',on='id') 
del card_wash_house


card_library = pd.read_csv('f:/prize/PrizePredict/data/input/card/card_libraryfeature.csv') 
card_library=card_library.rename(columns={'pos' : 'library_count'}) 
train_test = pd.merge(train_test, card_library, how='left',on='id') 
del card_library


card_printhouse = pd.read_csv('f:/prize/PrizePredict/data/input/card/card_printhousefeature.csv') 
card_printhouse=card_printhouse.rename(columns={'pos' : 'printhouse_count'}) 
train_test = pd.merge(train_test, card_printhouse, how='left',on='id') 
del card_printhouse


card_dean = pd.read_csv('f:/prize/PrizePredict/data/input/card/card_deanfeature.csv') 
card_dean=card_dean.rename(columns={'pos' : 'dean_count'}) 
train_test = pd.merge(train_test, card_dean, how='left',on='id') 
del card_dean

card_other = pd.read_csv('f:/prize/PrizePredict/data/input/card/card_otherfeature.csv') 
card_other=card_other.rename(columns={'pos' : 'other_count'}) 
train_test = pd.merge(train_test, card_other, how='left',on='id') 
del card_other


card_hospital = pd.read_csv('f:/prize/PrizePredict/data/input/card/card_hospitalfeature.csv') 
card_hospital=card_hospital.rename(columns={'pos' : 'hospital_count'}) 
train_test = pd.merge(train_test, card_hospital, how='left',on='id') 
del card_hospital
#%%
## 合并
for var in ['loc21','loc829','loc818','loc213','loc72','loc283','loc91','loc245','loc65','loc161','loc996','loc277','loc842','loc75','loc263','loc840']:

    feature_p=pd.read_csv('f:/prize/PrizePredict/data/input/loc/card_%sfeature.csv'%var)
    feature_p=feature_p.rename(columns={'pos' : '%s_count'%var})
    train_test = pd.merge(train_test, feature_p, how='left',on='id') 
    del feature_p


print "Merge all right."
#%%  分出训练集和测试集

train = train_test[train_test['label'].notnull()]    #标签不为空的是训练集
test = train_test[train_test['label'].isnull()]      #标签为空的是测试集
#%%   
#缺失值填充
train = train.fillna(-1)          #缺失值填充为-1
test = test.fillna(-1)            #

train.to_csv('f:/prize/PrizePredict/data/input/train_time.csv',index=False)   #写
test.to_csv('f:/prize/PrizePredict/data/input/test_time.csv',index=False)

train = pd.read_csv('f:/prize/PrizePredict/data/input/train_time.csv')
test = pd.read_csv('f:/prize/PrizePredict/data/input/test_time.csv')
ids = test['id'].values
#%%
## 生成训练集特征和标签 生成测试集特征
Y_train=train.iloc[:,1:2]
del train['label']
del train['id']
X_train=train
del test['label']
del test['id']
X_test=test

#%%  随机森林测试特征重要性

feat_labels = train.columns
forest = RandomForestClassifier(n_estimators=10000,random_state=0,n_jobs=-1)
forest.fit(X_train,Y_train)
importances=forest.feature_importances_
indices=np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print ("%2d) %-*s %f" %(f+1,30,feat_labels[f],importances[indices[f]]))
    
#%%
from sklearn.feature_selection import SelectKBest  
from matplotlib import pyplot as plt  
slct = SelectKBest(k="all")  
slct.fit(X_train,Y_train)  
scores = slct.scores_  
#%%
# 2. 将特征按分数 从大到小 排序  
from pylab import *
names=train.columns
named_scores = zip(names, scores)  
sorted_named_scores = sorted(named_scores, key=lambda z: z[1], reverse=True)
  
sorted_scores = [each[1] for each in sorted_named_scores]  
sorted_names = [each[0] for each in sorted_named_scores]  
sorted_scores = sorted_scores[:20]
sorted_names = sorted_names[:20]
#y_pos = np.arange(len(names))           # 从上而下的绘图顺序  
y_pos = np.arange(20)
 # 3. 绘图  
figure(figsize=(20,16), dpi=80)
fig, ax = plt.subplots()  
ax.barh(y_pos, sorted_scores, height=0.7, align='center', color='red', tick_label=sorted_names)  
# ax.set_yticklabels(sorted_names)      # 也可以在这里设置 条条 的标签~  
ax.set_yticks(y_pos)  
ax.set_xlabel('Feature Score')  
ax.set_ylabel('Feature Name')  
ax.invert_yaxis()  
ax.set_title('importances of the features.')  
# 4. 添加每个 条条 的数字标签  
for score, pos in zip(sorted_scores, y_pos):  
    ax.text(score + 20, pos, '%.1f' % score, ha='center', va='bottom', fontsize=8)  
  
plt.show()  
    
#%%
# 随机森林特征可视化

plt.title('Feature Importances')
plt.bar(range(X))
    
    
    
#%%
nice_feature=pd.read_csv('f:/prize/PrizePredict/data/input/feature_imp/nice_feature.csv',header=None,index_col=0)
feature_imp_place20=pd.read_csv('f:/prize/PrizePredict/data/input/feature_imp/feature_imp_place20.csv')

target = 'label'
IDcol = 'id'
ids = test['id'].values

all_feature = [x for x in train.columns if x not in [target,IDcol]]
#predictors = [x for x in train.columns if x in all_feature]
predictors = [ x for x in all_feature if (x in nice_feature.index)|(x in feature_imp_place20.feature.values)]
#predictors = [ x for x in all_feature if (x in nice_feature.index)]
#%%
# Oversample
Oversampling1000 = train.loc[train.label == 1000]
Oversampling1500 = train.loc[train.label == 1500]
Oversampling2000 = train.loc[train.label == 2000]
for i in range(7):
    train = train.append(Oversampling1000)
for j in range(10):
    train = train.append(Oversampling1500)
for k in range(9):
    train = train.append(Oversampling2000)
#%%
    
##  xgboost模型

from xgboost import XGBClassifier

# model

param_dist = {
    'n_estimators': 60,
    'max_depth': 5,
    'learning_rate':0.2,
    'min_child_weight':2,
    'gamma':0,
    'subsample':1,
    'colsample_bytree':1,
    'reg_alpha':0.008}
'''
param_dist = {
    'n_estimators': 120,
    'max_depth': 7,
    'min_child_weight':7,
    'gamma':0,
    'subsample':0.4,
    'colsample_bytree':1, 
    'reg_alpha':0.1,
    'learning_rate':0.05}
'''
clf = XGBClassifier(**param_dist).fit(train[predictors],train[target])

result = clf.predict(test[predictors])

# Save results
test_result = pd.DataFrame(columns=["studentid","subsidy"])
test_result.studentid = ids
test_result.subsidy = result
test_result.subsidy = test_result.subsidy.apply(lambda x:int(x))

print '1000--'+str(len(test_result[test_result.subsidy==1000])) + ':741'
print '1500--'+str(len(test_result[test_result.subsidy==1500])) + ':465'
print '2000--'+str(len(test_result[test_result.subsidy==2000])) + ':354'

test_result.to_csv("f:/prize/PrizePredict/data/output/submitXGBNFeatureNoid.csv",index=False)

#%%  
## 逻辑回归
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred_LR = logreg.predict(X_test)

test_result_LR = pd.DataFrame(columns=["studentid","subsidy"])
test_result_LR.studentid = ids
test_result_LR.subsidy = Y_pred_LR
test_result_LR.subsidy = test_result_LR.subsidy.apply(lambda x:int(x))
test_result_LR.to_csv("f:/prize/PrizePredict/data/output/submitLRFeatureNoid.csv",index=False)
#%%  svc
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred_SVC = svc.predict(X_test)
test_result_SVC = pd.DataFrame(columns=["studentid","subsidy"])
test_result_SVC.studentid = ids
test_result_SVC.subsidy = Y_pred_SVC
test_result_SVC.subsidy = test_result_SVC.subsidy.apply(lambda x:int(x))
test_result_SVC.to_csv("f:/prize/PrizePredict/data/output/submitSVCFeatureNoid.csv",index=False)
#%%  knn
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred_knn = knn.predict(X_test)
test_result_knn = pd.DataFrame(columns=["studentid","subsidy"])
test_result_knn.studentid = ids
test_result_knn.subsidy = Y_pred_knn
test_result_knn.subsidy = test_result_knn.subsidy.apply(lambda x:int(x))
test_result_knn.to_csv("f:/prize/PrizePredict/data/output/submitknnFeatureNoid.csv",index=False)
#%%   朴素贝叶斯
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred_gs = gaussian.predict(X_test)
test_result_gs = pd.DataFrame(columns=["studentid","subsidy"])
test_result_gs.studentid = ids
test_result_gs.subsidy = Y_pred_gs
test_result_gs.subsidy = test_result_gs.subsidy.apply(lambda x:int(x))
test_result_gs.to_csv("f:/prize/PrizePredict/data/output/submitgsFeatureNoid.csv",index=False)
#%%   神经网络
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred_per = perceptron.predict(X_test)
test_result_per = pd.DataFrame(columns=["studentid","subsidy"])
test_result_per.studentid = ids
test_result_per.subsidy = Y_pred_per
test_result_per.subsidy = test_result_per.subsidy.apply(lambda x:int(x))
test_result_per.to_csv("f:/prize/PrizePredict/data/output/submitperFeatureNoid.csv",index=False)
#%%
# linear_svc
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred_linear_svc = linear_svc.predict(X_test)
test_result_linear_svc = pd.DataFrame(columns=["studentid","subsidy"])
test_result_linear_svc.studentid = ids
test_result_linear_svc.subsidy = Y_pred_linear_svc
test_result_linear_svc.subsidy = test_result_linear_svc.subsidy.apply(lambda x:int(x))
test_result_linear_svc.to_csv("f:/prize/PrizePredict/data/output/submitlinear_svcFeatureNoid.csv",index=False)
#%%
# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred_sgd = sgd.predict(X_test)
test_result_sgd = pd.DataFrame(columns=["studentid","subsidy"])
test_result_sgd.studentid = ids
test_result_sgd.subsidy = Y_pred_sgd
test_result_sgd.subsidy = test_result_sgd.subsidy.apply(lambda x:int(x))
test_result_sgd.to_csv("f:/prize/PrizePredict/data/output/submitsgdFeatureNoid.csv",index=False)
#%%

# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred_dt = decision_tree.predict(X_test)
test_result_dt = pd.DataFrame(columns=["studentid","subsidy"])
test_result_dt.studentid = ids
test_result_dt.subsidy = Y_pred_dt
test_result_dt.subsidy = test_result_dt.subsidy.apply(lambda x:int(x))
test_result_dt.to_csv("f:/prize/PrizePredict/data/output/submitdtFeatureNoid.csv",index=False)
#%%
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred_rf = random_forest.predict(X_test)

test_result_rf = pd.DataFrame(columns=["studentid","subsidy"])
test_result_rf.studentid = ids
test_result_rf.subsidy = Y_pred_rf
test_result_rf.subsidy = test_result_rf.subsidy.apply(lambda x:int(x))
test_result_rf.to_csv("f:/prize/PrizePredict/data/output/submitrfFeatureNoid.csv",index=False)





