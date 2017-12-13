# PrizePredict
the prize prediction from data castle
**创建这个项目来记录数据挖掘课程上的助学金预测项目**
* 2017.11.26
* 本题目来源于数据城堡上的一个比赛[大学生助学金精准资助预测](http://www.dcjingsai.com/common/cmpt/%E5%A4%A7%E5%AD%A6%E7%94%9F%E5%8A%A9%E5%AD%A6%E9%87%91%E7%B2%BE%E5%87%86%E8%B5%84%E5%8A%A9%E9%A2%84%E6%B5%8B_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html)
## 问题描述
* 基于学生每天产生的一卡通实时数据，利用大数据挖掘与分析技术、数学建模理论帮助管理者掌握学生在校期间的真实消费情况、学生经济水平、发现“隐性贫困”与疑似“虚假认定”学生，从而实现精准资助。
* 采用某高校2014、2015两学年的助学金获取情况作为标签，2013-2014、2014-2015两学年的学生在校行为数据作为原始数据，包括**消费数据、图书借阅数据、寝室门禁数据、图书馆门禁数据、学生成绩排名数据**，并以助学金获取金额作为结果数据进行模型优化和评价。
* 需利用学生在2013/09-2014/09的数据，预测学生在2014年的助学金获得情况；利用学生在2014/09-2015/09的数据，预测学生在2015年的助学金获得情况。虽然所有数据在时间上混合在了一起，即训练集和测试集中的数据都有2013/09-2015/09的数据，但是学生的行为数据和助学金数据是对应的。
* 此竞赛赛程分为两个阶段，以**测试集切换**为标志，2017年2月13日切换。
## 数据集描述
### 1）数据总体概述
    数据分为两组，分别是训练集和测试集，每一组都包含大约1万名学生的信息纪录
 * 图书借阅数据borrow_train.txt和borrow_test.txt、
 * 一卡通数据card_train.txt和card_test.txt、
 * 寝室门禁数据dorm_train.txt和dorm_test.txt、
 * 图书馆门禁数据library_train.txt和library_test.txt、
 * 学生成绩数据score_train.txt和score_test.txt
 * 助学金获奖数据subsidy_train.txt和subsidy_test.txt
 训练集和测试集中的学生id无交集，详细信息如下。
 注：数据中所有的记录均为“原始数据记录”直接经过脱敏而来，可能会存在一些重复的或者是异常的记录，请参赛者自行处理。
### 2）数据详细描述
#### （1）图书借阅数据borrow*.txt（*代表_train和_test）
   注：有些图书的编号缺失。字段描述和示例如下（第三条记录缺失图书编号）：
 ```
 学生id，借阅日期，图书名称，图书编号
    9708,2014/2/25,"我的英语日记/ (韩)南银英著 (韩)卢炫廷插图","H315 502"
    6956,2013/10/27,"解读联想思维: 联想教父柳传志","K825.38=76 547"
    9076,2014/3/28,"公司法 gong si fa = = Corporation law / 范健, 王建文著 eng"
```
#### （2）一卡通数据card*.txt
   字段描述和示例如下：
```
    学生id，消费类别，消费地点，消费方式，消费时间，消费金额，剩余金额
    1006,"POS消费","地点551","淋浴","2013/09/01 00:00:32","0.5","124.9"
    1406,"POS消费","地点78","其他","2013/09/01 00:00:40","0.6","373.82"
    13554,"POS消费","地点6","淋浴","2013/09/01 00:00:57","0.5","522.37"
```
#### （3）寝室门禁数据dorm*.txt
   字段描述和示例如下：
```
    学生id，具体时间，进出方向(0进寝室，1出寝室)	
    13126,"2014/01/21 03:31:11","1"
    9228,"2014/01/21 10:28:23","0"
```
#### （4）图书馆门禁数据library*.txt
   图书馆的开放时间为早上7点到晚上22点，门禁编号数据在2014/02/23之前只有“编号”信息，之后引入了“进门、出门”信息，还有些异常信息为null，请参赛者自行处理。
   字段描述和示例如下：
```
    学生id，门禁编号，具体时间
    3684,"5","2013/09/01 08:42:50"
    7434,"5","2013/09/01 08:50:08"
    8000,"进门2","2014/03/31 18:20:31"
    5332,"小门","2014/04/03 20:11:06"
    7397,"出门4","2014/09/04 16:50:51"
```
#### （5）学生成绩数据score*.txt。
   注：成绩排名的计算方式是将所有成绩按学分加权求和，然后除以学分总和，再按照学生所在学院排序。
```
    学生id,学院编号,成绩排名
    0,9,1
    1,9,2
    8,6,1565
    9,6,1570
```
#### （6）助学金数据（训练集中有金额，测试集中无金额）subsidy*.txt
   字段描述和示例如下：
```
    学生id,助学金金额（分隔符为半角逗号）
    10,0
    22,1000
    28,1000
    64,1500
    650,2000
```
## 目标
多分类问题，通过上面数据集的数据，预测出该学生获得的奖学金（0,1000,1500,2000）
## 参考项目地址
1. [第一名](https://github.com/mayday618/DataMiningCompetitionFirstPrize)
2. [第七名](https://github.com/kuhung/Student-Grants)
3. [第14名](https://github.com/971456267/datacastle)
4. 另外附上两个未知排名的代码
 * [InsaneLife](https://github.com/InsaneLife/Sdudent-Grant)
 * [ShawnXiha](https://github.com/ShawnXiha/DataCastle-subsidy/tree/master/Jupyter)
## 解决方案流程
/data文件夹存数据
* /train 训练集
* /final_test 最终测试集
* /input 提取出的中间特征
* /output 预测结果

/feature文件夹生成特征
* borrow.py 生成借书统计特征
* dorm.py 生成进出宿舍特征
* library.py 生成 进出图书馆特征
* place.py 生成在不同地方一卡通消费特征
* time.py 生成不同时间段一卡通消费特征
* cash.py 生成一卡通消费金额特征

/code文件夹
code.py
1. 合并特征并进行抽样 
2. 预处理
3. 生成训练集测试集
4. 基本模型和xgboost模型
