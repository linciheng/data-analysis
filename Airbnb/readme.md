
# 背景及描述

About this Dataset,In this challenge, you are given a list of users along with their demographics, web session records, and some summary statistics.
You are asked to predict which country a new user’s first booking destination will be. 
All the users in this dataset are from the USA.

There are 12 possible outcomes of the destination country: ‘US’, ‘FR’, ‘CA’, ‘GB’, ‘ES’, ‘IT’, ‘PT’, ‘NL’,’DE’, ‘AU’, ‘NDF’ (no destination found), and ‘other’. 
Please note that ‘NDF’ is different from ‘other’ because ‘other’ means there was a booking, but is to a country not included in the list, 
while ‘NDF’ means there wasn’t a booking.


# 数据描述：
 id: user id （用户id）
- date_account_created（帐号注册时间）: the date of account creation
- timestamp_first_active（首次活跃时间）: timestamp of the first activity, note that it can be earlier than date_account_created or date_first_booking because a user can search before signing up
- date_first_booking（首次订房时间）: date of first booking
- gender（性别）
- age（年龄）
- signup_method（注册方式）
- signup_flow（注册页面）: the page a user came to signup up from
- language（语言）: international language preference
- affiliate_channel（付费市场渠道）: what kind of paid marketing
- affiliate_provider（付费市场渠道名称）: where the marketing is e.g. google, craigslist, other
- first_affiliate_tracked（注册前第一个接触的市场渠道）: whats the first marketing the user interacted with before the signing up
- signup_app（注册app）
- first_device_type(设备类型)
- first_browser（浏览器类型）
- country_destination（订房国家-需要预测的量）: this is the target variable you are to predict 

# 环境
Python3 

# 总结
这个项目完成了Airbnb项目。对民俗预定结果进行了预测。

主要分为以下几个部分：

## 1 数据探索

数据探索部分主要基于pandas库，利用常见的:head()，value_counts()，describe()，isnull()，unique()等函数以及通过matplotlib作图对数据进行理解和探索；
使用get_dummies()函数对pandas中的数据进行onehot编码；
缺失值处理；
pandas时间数据处理；

## 2 特征工程

 特征工程部分主要是通过从日期中提取年月日，季节，weekday，对年龄进行分段，
 计算相关特征之间的差值，根据用户id进行分组，从而统计一些特征变量的次数，
 平均值，标准差等等，以及通过one hot encoding和labels encoding对数据进行编码来提取特征；
 
 ## 模型构建

构建模型部分主要基于sklearn包，xgboost包，通过调用不同的模型进行预测，
其中涉及到的模型有，逻辑回归模型Logistic Regression，
树模型：DecisionTree，RandomForest，AdaBoost，Bagging，ExtraTree，GraBoost，S
VM模型：SVM-rbf，SVM-poly，SVM-linear，xgboost，
以及通过改变模型的参数和数据量大小，来观察NDCG的评分结果，从而了解不同模型，不同参数和不同数据量大小对预测结果的影响.

参考：
原文：https://blog.csdn.net/Datawhale/article/details/80847662 
在原文中有不少地方都跑不通，进行改进；另有些地方的语句稍显复杂多余，都尽数改掉了。
但是在跑模型之前，因为数据量比较大，我改的时候写了一句非常弱智的命令，导致笔记本温度哦过高/内存溢出，反正电脑死机了，然后后面就没执行。
