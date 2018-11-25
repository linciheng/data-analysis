{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Aribnb房价预测\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#导入相关包\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#导入数据\n",
    "train = pd.read_csv('train_users_2.csv')\n",
    "test = pd.read_csv('test_users.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "train sets size (213451, 16)\n",
      "\n",
      "\n",
      "feature names in training set\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "Index(['id', 'date_account_created', 'timestamp_first_active',\n",
       "       'date_first_booking', 'gender', 'age', 'signup_method', 'signup_flow',\n",
       "       'language', 'affiliate_channel', 'affiliate_provider',\n",
       "       'first_affiliate_tracked', 'signup_app', 'first_device_type',\n",
       "       'first_browser', 'country_destination'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#查看数据集中的特征信息\n",
    "print('train sets size', train.shape)\n",
    "print('\\n')\n",
    "print('feature names in training set')\n",
    "train.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "test sets size (62096, 15)\n",
      "\n",
      "\n",
      "feature names in test set\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "Index(['id', 'date_account_created', 'timestamp_first_active',\n",
       "       'date_first_booking', 'gender', 'age', 'signup_method', 'signup_flow',\n",
       "       'language', 'affiliate_channel', 'affiliate_provider',\n",
       "       'first_affiliate_tracked', 'signup_app', 'first_device_type',\n",
       "       'first_browser'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print('test sets size', test.shape)\n",
    "print('\\n')\n",
    "print('feature names in test set')\n",
    "test.columns"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "这里会发现训练数据集和测试数据集的特征数量不等！从下面会发现，test set中没有{'country_destination'}，因为这就是我们需要预测的label！"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'country_destination'}"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "set(train.columns) - set(test.columns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 213451 entries, 0 to 213450\n",
      "Data columns (total 16 columns):\n",
      "id                         213451 non-null object\n",
      "date_account_created       213451 non-null object\n",
      "timestamp_first_active     213451 non-null int64\n",
      "date_first_booking         88908 non-null object\n",
      "gender                     213451 non-null object\n",
      "age                        125461 non-null float64\n",
      "signup_method              213451 non-null object\n",
      "signup_flow                213451 non-null int64\n",
      "language                   213451 non-null object\n",
      "affiliate_channel          213451 non-null object\n",
      "affiliate_provider         213451 non-null object\n",
      "first_affiliate_tracked    207386 non-null object\n",
      "signup_app                 213451 non-null object\n",
      "first_device_type          213451 non-null object\n",
      "first_browser              213451 non-null object\n",
      "country_destination        213451 non-null object\n",
      "dtypes: float64(1), int64(2), object(13)\n",
      "memory usage: 26.1+ MB\n"
     ]
    }
   ],
   "source": [
    "#查看数据详细信息\n",
    "train.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "tips：\n",
    "1.注意其中的样本数量213451，特征数量16（其中id对模型并无多大作用-index，对特征工程有用；而最后一个特征country_destination实为label）\n",
    "2.需要注意一些数据的类型：object，float，int\n",
    "3. 完全可以考虑删除缺失值较多的特征，没有问题"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style>\n",
       "    .dataframe thead tr:only-child th {\n",
       "        text-align: right;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>date_account_created</th>\n",
       "      <th>timestamp_first_active</th>\n",
       "      <th>date_first_booking</th>\n",
       "      <th>gender</th>\n",
       "      <th>age</th>\n",
       "      <th>signup_method</th>\n",
       "      <th>signup_flow</th>\n",
       "      <th>language</th>\n",
       "      <th>affiliate_channel</th>\n",
       "      <th>affiliate_provider</th>\n",
       "      <th>first_affiliate_tracked</th>\n",
       "      <th>signup_app</th>\n",
       "      <th>first_device_type</th>\n",
       "      <th>first_browser</th>\n",
       "      <th>country_destination</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>gxn3p5htnn</td>\n",
       "      <td>2010-06-28</td>\n",
       "      <td>20090319043255</td>\n",
       "      <td>NaN</td>\n",
       "      <td>-unknown-</td>\n",
       "      <td>NaN</td>\n",
       "      <td>facebook</td>\n",
       "      <td>0</td>\n",
       "      <td>en</td>\n",
       "      <td>direct</td>\n",
       "      <td>direct</td>\n",
       "      <td>untracked</td>\n",
       "      <td>Web</td>\n",
       "      <td>Mac Desktop</td>\n",
       "      <td>Chrome</td>\n",
       "      <td>NDF</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>820tgsjxq7</td>\n",
       "      <td>2011-05-25</td>\n",
       "      <td>20090523174809</td>\n",
       "      <td>NaN</td>\n",
       "      <td>MALE</td>\n",
       "      <td>38.0</td>\n",
       "      <td>facebook</td>\n",
       "      <td>0</td>\n",
       "      <td>en</td>\n",
       "      <td>seo</td>\n",
       "      <td>google</td>\n",
       "      <td>untracked</td>\n",
       "      <td>Web</td>\n",
       "      <td>Mac Desktop</td>\n",
       "      <td>Chrome</td>\n",
       "      <td>NDF</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4ft3gnwmtx</td>\n",
       "      <td>2010-09-28</td>\n",
       "      <td>20090609231247</td>\n",
       "      <td>2010-08-02</td>\n",
       "      <td>FEMALE</td>\n",
       "      <td>56.0</td>\n",
       "      <td>basic</td>\n",
       "      <td>3</td>\n",
       "      <td>en</td>\n",
       "      <td>direct</td>\n",
       "      <td>direct</td>\n",
       "      <td>untracked</td>\n",
       "      <td>Web</td>\n",
       "      <td>Windows Desktop</td>\n",
       "      <td>IE</td>\n",
       "      <td>US</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>bjjt8pjhuk</td>\n",
       "      <td>2011-12-05</td>\n",
       "      <td>20091031060129</td>\n",
       "      <td>2012-09-08</td>\n",
       "      <td>FEMALE</td>\n",
       "      <td>42.0</td>\n",
       "      <td>facebook</td>\n",
       "      <td>0</td>\n",
       "      <td>en</td>\n",
       "      <td>direct</td>\n",
       "      <td>direct</td>\n",
       "      <td>untracked</td>\n",
       "      <td>Web</td>\n",
       "      <td>Mac Desktop</td>\n",
       "      <td>Firefox</td>\n",
       "      <td>other</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>87mebub9p4</td>\n",
       "      <td>2010-09-14</td>\n",
       "      <td>20091208061105</td>\n",
       "      <td>2010-02-18</td>\n",
       "      <td>-unknown-</td>\n",
       "      <td>41.0</td>\n",
       "      <td>basic</td>\n",
       "      <td>0</td>\n",
       "      <td>en</td>\n",
       "      <td>direct</td>\n",
       "      <td>direct</td>\n",
       "      <td>untracked</td>\n",
       "      <td>Web</td>\n",
       "      <td>Mac Desktop</td>\n",
       "      <td>Chrome</td>\n",
       "      <td>US</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           id date_account_created  timestamp_first_active date_first_booking  \\\n",
       "0  gxn3p5htnn           2010-06-28          20090319043255                NaN   \n",
       "1  820tgsjxq7           2011-05-25          20090523174809                NaN   \n",
       "2  4ft3gnwmtx           2010-09-28          20090609231247         2010-08-02   \n",
       "3  bjjt8pjhuk           2011-12-05          20091031060129         2012-09-08   \n",
       "4  87mebub9p4           2010-09-14          20091208061105         2010-02-18   \n",
       "\n",
       "      gender   age signup_method  signup_flow language affiliate_channel  \\\n",
       "0  -unknown-   NaN      facebook            0       en            direct   \n",
       "1       MALE  38.0      facebook            0       en               seo   \n",
       "2     FEMALE  56.0         basic            3       en            direct   \n",
       "3     FEMALE  42.0      facebook            0       en            direct   \n",
       "4  -unknown-  41.0         basic            0       en            direct   \n",
       "\n",
       "  affiliate_provider first_affiliate_tracked signup_app first_device_type  \\\n",
       "0             direct               untracked        Web       Mac Desktop   \n",
       "1             google               untracked        Web       Mac Desktop   \n",
       "2             direct               untracked        Web   Windows Desktop   \n",
       "3             direct               untracked        Web       Mac Desktop   \n",
       "4             direct               untracked        Web       Mac Desktop   \n",
       "\n",
       "  first_browser country_destination  \n",
       "0        Chrome                 NDF  \n",
       "1        Chrome                 NDF  \n",
       "2            IE                  US  \n",
       "3       Firefox               other  \n",
       "4        Chrome                  US  "
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#查看数据\n",
    "train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2014-05-13    674\n",
       "2014-06-24    670\n",
       "2014-06-25    636\n",
       "2014-05-20    632\n",
       "2014-05-14    622\n",
       "Name: date_account_created, dtype: int64"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#对date_account_created数据进行统计\n",
    "# train set\n",
    "train.date_account_created.value_counts().head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2014-07-23    1105\n",
       "2014-07-22    1052\n",
       "2014-07-17     978\n",
       "2014-07-24     923\n",
       "2014-07-18     892\n",
       "Name: date_account_created, dtype: int64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#test set\n",
    "test.date_account_created.value_counts().head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "tips：\n",
    "这里之所以要查看‘创建账号’的时间信息，是为了分析其用户数量增长情况。\n",
    "但是对于最总预测有没有用，暂时没发现！\n",
    "对于预测目的来说应该是没用的，存疑？"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count         213451\n",
       "unique          1634\n",
       "top       2014-05-13\n",
       "freq             674\n",
       "Name: date_account_created, dtype: object"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.date_account_created.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "describe()函数可以用来描述数据的统计信息，包括max，min，mean等"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.legend.Legend at 0x8fccc88>"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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E4EJwMsisgunMXu+qm4I8tpw8QZzFU6ngZfc6ppncK4ywGkszsW5MnkJpxor0CLg226Pq1vd6qUpONVKm3Gq4n6J0/AM8AOc6riPcNpb7uniPHyGk2xOwjm3fgfhJLxPcKqeGoDfhHib84Gv4JyLhTiJwG+p+/oPcRrCXwry+odwktdDHaq8TgXedP8ei4DzVHVvuM9jEk9UbSEnY0ziich7wCWq+kyyYzGRsSsOY0zCici3cKoFn012LCZyKT31gjEm84jIKpx2sZn+9hWTXqyqyhhjTESsqsoYY0xEMrKqasiQIVpUVJTsMIwxJq2sXr16m6oO7Wq/jEwcRUVFVFdXJzsMY4xJKyISctaBQFZVZYwxJiKWOIwxxkTEEocxxpiIZGQbh8lMzc3N1NXVsXevzUCRbH379mX48OHk5OQkOxSTBJY4TNqoq6tjwIABFBUVISJdv8DEharS0NBAXV0dI0eOTHY4Jgmsqsqkjb1791JQUGBJI8lEhIKCArvy68EscZi0YkkjNdjfoWezxGGMManC54OiIsjKcu59vmRHFJQlDmM82rFjB4sXL47qtdOmTWPHjh1d7xhC//7hVr/tXmzh3HvvvWzevDnm72uC8PlgzhyorQVV537OnJRMHpY4jPEo3JdzS0v4RepWrlzJwIED4xEWYIkjIyxYAE1N7cuampzyFGOJw2QsX42PoluKyLo+i6JbivDVdO+X29VXX817771HcXExV1xxBatWrWLy5Mmcf/75jBkzBoCzzjqL8ePHM3r0aCorK9teW1RUxLZt29i4cSNHHHEE3/ve9xg9ejRTpkxhz549nY61YcMGJk2axIQJE/jZz37WVr5r1y5KS0s55phjGDNmDA8//HDQ2ELtt3v3bqZPn864ceM46qijWLFiBQCrV6/ma1/7GuPHj+eUU05hy5YtPPDAA1RXV1NWVkZxcXHQOE0MbdoUWXkyqWrG3caPH68m87z11lue912+drnmVeQp19F2y6vI0+Vrl0d9/A0bNujo0aPbnj/33HOal5en77//fltZQ0ODqqo2NTXp6NGjddu2baqqWlhYqPX19bphwwbNzs7W1157TVVVzznnHF22bFmnY51++um6ZMkSVVW97bbbtF+/fqqq2tzcrI2NjaqqWl9fr1/4whe0tbW1U2yh9nvggQf0u9/9btt+O3bs0H379umk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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#观察用户增长情况\n",
    "dac_train = train.date_account_created.value_counts()\n",
    "dac_test = test.date_account_created.value_counts()\n",
    "\n",
    "dac_train_date = pd.to_datetime(train.date_account_created.value_counts().index)\n",
    "dac_test_date = pd.to_datetime(test.date_account_created.value_counts().index)    #这里index\n",
    "\n",
    "dac_train_day = dac_train_date - dac_train_date.min()\n",
    "dac_test_day = dac_test_date - dac_train_date.min()      #这里要都减去train data的起始时间\n",
    "\n",
    "plt.scatter(dac_train_day.days, dac_train.values, color = 'g', label = 'train dataset')\n",
    "plt.scatter(dac_test_day.days, dac_test.values, color = 'r', label = 'test dataset')\n",
    "\n",
    "plt.title('Accounts created vs days')\n",
    "plt.xlabel('Days')\n",
    "plt.ylabel('Accounts created')\n",
    "plt.legend(loc = 'upper center')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "tips：\n",
    "1 这里注意一下，原数据中‘date_account_created’是object类型的，不能直接减运算，所以需要用pd.to_datetime转换成datetime类型；\n",
    "2 之所以要.index再转换，可以看一下dac_train中其index为日期，值为这个日期的数量\n",
    "3 **可以看到随着时间增长用户注册数量在上升，很自然的事情"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    20090319043255\n",
       "1    20090523174809\n",
       "2    20090609231247\n",
       "3    20091031060129\n",
       "4    20091208061105\n",
       "Name: timestamp_first_active, dtype: int64"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# timestamp_first_active（首次活跃时间）\n",
    "# 查看头几行数据\n",
    "train.timestamp_first_active.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "这里需要注意数据类型！"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1], dtype=int64)"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.timestamp_first_active.value_counts().unique()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "结果[1]表明timestamp_first_active没有重复数据"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count                  213451\n",
       "unique                 213451\n",
       "top       2013-07-01 05:26:34\n",
       "freq                        1\n",
       "first     2009-03-19 04:32:55\n",
       "last      2014-06-30 23:58:24\n",
       "Name: timestamp_first_active, dtype: object"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tfa_train_dt = train.timestamp_first_active.astype(str).apply(lambda x: pd.datetime(int(x[0:4]),int(x[4:6]), int(x[6:8]), int(x[8:10]), int(x[10:12]), int(x[12:])))\n",
    "tfa_train_dt.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "tips:\n",
    "这里要注意一下pd.datetime的应用。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "count          88908\n",
      "unique          1976\n",
      "top       2014-05-22\n",
      "freq             248\n",
      "Name: date_first_booking, dtype: object\n",
      "count    0.0\n",
      "mean     NaN\n",
      "std      NaN\n",
      "min      NaN\n",
      "25%      NaN\n",
      "50%      NaN\n",
      "75%      NaN\n",
      "max      NaN\n",
      "Name: date_first_booking, dtype: float64\n"
     ]
    }
   ],
   "source": [
    "#获取date_first_booking信息\n",
    "print(train.date_first_booking.describe())\n",
    "print(test.date_first_booking.describe())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "tips：\n",
    "可以看到train set中date_first_booking缺失值超过一半了，而在tes他set中没有date_first_booking数据。所以可以将这个feature直接删掉。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 213451 entries, 0 to 213450\n",
      "Data columns (total 15 columns):\n",
      "id                         213451 non-null object\n",
      "date_account_created       213451 non-null object\n",
      "timestamp_first_active     213451 non-null int64\n",
      "gender                     213451 non-null object\n",
      "age                        125461 non-null float64\n",
      "signup_method              213451 non-null object\n",
      "signup_flow                213451 non-null int64\n",
      "language                   213451 non-null object\n",
      "affiliate_channel          213451 non-null object\n",
      "affiliate_provider         213451 non-null object\n",
      "first_affiliate_tracked    207386 non-null object\n",
      "signup_app                 213451 non-null object\n",
      "first_device_type          213451 non-null object\n",
      "first_browser              213451 non-null object\n",
      "country_destination        213451 non-null object\n",
      "dtypes: float64(1), int64(2), object(12)\n",
      "memory usage: 24.4+ MB\n"
     ]
    }
   ],
   "source": [
    "train.drop(['date_first_booking'], axis = 1, inplace = True)\n",
    "test.drop(['date_first_booking'], axis = 1, inplace = True)\n",
    "train.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "30.0    6124\n",
       "31.0    6016\n",
       "29.0    5963\n",
       "28.0    5939\n",
       "32.0    5855\n",
       "Name: age, dtype: int64"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#对‘age’数据进行统计\n",
    "train.age.value_counts().head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1933.0    1\n",
       "1942.0    1\n",
       "112.0     1\n",
       "1938.0    1\n",
       "1952.0    1\n",
       "Name: age, dtype: int64"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.age.value_counts().tail()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "这些1千多的肯定就是异常值了！需要对异常值进行处理！"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.text.Text at 0x2275f2da438>"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#柱状图统计\n",
    "#先导入seaborn\n",
    "\n",
    "import seaborn as sns\n",
    "\n",
    "#首先将年龄进行分成4组missing values, too small age, reasonable age, too large age\n",
    "age_train = [train[train.age.isnull()].age.shape[0], train.query('age < 15').age.shape[0], train.query('age >= 15 & age <= 90').age.shape[0], train.query('age > 90').age.shape[0]]\n",
    "age_test = [train[train.age.isnull()].age.shape[0], train.query('age < 15').age.shape[0], train.query('age >= 15 & age <= 90').age.shape[0], train.query('age > 90').age.shape[0]]\n",
    "\n",
    "columns = ['Null', 'age < 15', 'age', 'age> 90']\n",
    "fig, (ax1, ax2) = plt.subplots(1,2,sharex = True, sharey = True, figsize = (10,5))\n",
    "\n",
    "sns.barplot(columns, age_train, ax = ax1)\n",
    "sns.barplot(columns,age_test, ax = ax2)\n",
    "\n",
    "ax1.set_title('training dataset')\n",
    "ax2.set_title('test dataset')\n",
    "ax1.set_ylabel('conts')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def feature_barplot(feature, df_train = train, df_test = test, figsize=(10,5), rot = 90, saveimg = False): \n",
    "    feat_train = df_train[feature].value_counts() \n",
    "    feat_test = df_test[feature].value_counts() \n",
    "    fig_feature, (axis1,axis2) = plt.subplots(1,2,sharex=True, sharey = True, figsize = figsize) \n",
    "    sns.barplot(feat_train.index.values, feat_train.values, ax = axis1) \n",
    "    sns.barplot(feat_test.index.values, feat_test.values, ax = axis2) \n",
    "    axis1.set_xticklabels(axis1.xaxis.get_majorticklabels(), rotation = rot) \n",
    "    axis2.set_xticklabels(axis1.xaxis.get_majorticklabels(), rotation = rot) \n",
    "    axis1.set_title(feature + ' of training dataset') \n",
    "    axis2.set_title(feature + ' of test dataset') \n",
    "    axis1.set_ylabel('Counts') \n",
    "    plt.tight_layout() \n",
    "    if saveimg == True: \n",
    "        figname = feature + \".png\" \n",
    "        fig_feature.savefig(figname, dpi = 75)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "feature_barplot('gender', saveimg = False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "feature_barplot('signup_method')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "feature_barplot('language')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "feature_barplot('affiliate_channel')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "feature_barplot('first_affiliate_tracked')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "feature_barplot('signup_app')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "feature_barplot('first_device_type')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'df' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-25-b82a39802200>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m()\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[0mdf_sessions\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mpd\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mread_csv\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'sessions.csv'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 2\u001b[1;33m \u001b[0mdf\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0msessions_head\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;36m3\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m: name 'df' is not defined"
     ]
    }
   ],
   "source": [
    "df_sessions = pd.read_csv('sessions.csv')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style>\n",
       "    .dataframe thead tr:only-child th {\n",
       "        text-align: right;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>user_id</th>\n",
       "      <th>action</th>\n",
       "      <th>action_type</th>\n",
       "      <th>action_detail</th>\n",
       "      <th>device_type</th>\n",
       "      <th>secs_elapsed</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>d1mm9tcy42</td>\n",
       "      <td>lookup</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Windows Desktop</td>\n",
       "      <td>319.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>d1mm9tcy42</td>\n",
       "      <td>search_results</td>\n",
       "      <td>click</td>\n",
       "      <td>view_search_results</td>\n",
       "      <td>Windows Desktop</td>\n",
       "      <td>67753.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>d1mm9tcy42</td>\n",
       "      <td>lookup</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Windows Desktop</td>\n",
       "      <td>301.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      user_id          action action_type        action_detail  \\\n",
       "0  d1mm9tcy42          lookup         NaN                  NaN   \n",
       "1  d1mm9tcy42  search_results       click  view_search_results   \n",
       "2  d1mm9tcy42          lookup         NaN                  NaN   \n",
       "\n",
       "       device_type  secs_elapsed  \n",
       "0  Windows Desktop         319.0  \n",
       "1  Windows Desktop       67753.0  \n",
       "2  Windows Desktop         301.0  "
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_sessions.head(3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style>\n",
       "    .dataframe thead tr:only-child th {\n",
       "        text-align: right;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>action</th>\n",
       "      <th>action_type</th>\n",
       "      <th>action_detail</th>\n",
       "      <th>device_type</th>\n",
       "      <th>secs_elapsed</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>d1mm9tcy42</td>\n",
       "      <td>lookup</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Windows Desktop</td>\n",
       "      <td>319.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "           id  action action_type action_detail      device_type  secs_elapsed\n",
       "0  d1mm9tcy42  lookup         NaN           NaN  Windows Desktop         319.0"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_sessions.rename(columns = {'user_id' : 'id'}, inplace = True)\n",
    "df_sessions.head(1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "将seession中的‘user_id'换成’id'方便后面进行文件合并。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(10567737, 6)"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_sessions.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 10567737 entries, 0 to 10567736\n",
      "Data columns (total 6 columns):\n",
      "id               object\n",
      "action           object\n",
      "action_type      object\n",
      "action_detail    object\n",
      "device_type      object\n",
      "secs_elapsed     float64\n",
      "dtypes: float64(1), object(5)\n",
      "memory usage: 483.8+ MB\n"
     ]
    }
   ],
   "source": [
    "df_sessions.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "id                 34496\n",
       "action             79626\n",
       "action_type      1126204\n",
       "action_detail    1126204\n",
       "device_type            0\n",
       "secs_elapsed      136031\n",
       "dtype: int64"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_sessions.isnull().sum()  #查看缺失值"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "id                34496\n",
       "action                0\n",
       "action_type           0\n",
       "action_detail         0\n",
       "device_type           0\n",
       "secs_elapsed     136031\n",
       "dtype: int64"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#填充缺失值,填充空值\n",
    "df_sessions.action = df_sessions.action.fillna('NAN') \n",
    "df_sessions.action_type = df_sessions.action_type.fillna('NAN') \n",
    "df_sessions.action_detail = df_sessions.action_detail.fillna('NAN') \n",
    "df_sessions.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "最后一个属性‘secs_elapsed’表示停留属性，这个应该是很有用的，并且是数值型的数据。"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 特征提取"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0            lookup\n",
       "1    search_results\n",
       "2            lookup\n",
       "3    search_results\n",
       "4            lookup\n",
       "Name: action, dtype: object"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_sessions.action.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "我们可以通过value_counts()来看个action的数量，发现其中会有大量的action数量很少，1、2等\n",
    "可以考虑将这些操作归为一类。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_sessions.action.value_counts().min()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#Action values with low frequency are changed to 'OTHER' \n",
    "act_freq = 100 #Threshold of frequency ，选择的阈值，将小于这个阈值的action都设置为others\n",
    "act = dict(zip(*np.unique(df_sessions.action, return_counts=True))) \n",
    "df_sessions.action = df_sessions.action.apply(lambda x: 'OTHER' if act[x] < act_freq else x) \n",
    "#np.unique(df_sessions.action, return_counts=True) 取以数组形式返回非重复的action值和它的数量 \n",
    "#zip（*（a,b））a,b种元素一一对应，返回zip object\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "2. 对特征action，action_detail，action_type，device_type，secs_elapsed进行细化\n",
    "- 首先将用户的特征根据用户id进行分组\n",
    "- 特征action：统计每个用户总的action出现的次数，各个action类型的数量，平均值以及标准差\n",
    "- 特征action_detail：统计每个用户总的action_detail出现的次数，各个action_detail类型的数量，平均值以及标准差\n",
    "- 特征action_type：统计每个用户总的action_type出现的次数，各个action_type类型的数量，平均值，标准差以及总的停留时长（进行log处理）\n",
    "- 特征device_type：统计每个用户总的device_type出现的次数，各个device_type类型的数量，平均值以及标准差\n",
    "- 特征secs_elapsed：对缺失值用0填充，统计每个用户secs_elapsed时间的总和，平均值，标准差以及中位数（进行log处理），（总和/平均数），secs_elapsed（log处理后）各个时间出现的次数\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "show              230\n",
       "index             229\n",
       "search_results    227\n",
       "Name: action, dtype: int64"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "f_act = df_sessions.action.value_counts().argsort()\n",
    "f_act.head(3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#对action特征进行细化 \n",
    "f_act = df_sessions.action.value_counts().argsort() \n",
    "f_act_detail = df_sessions.action_detail.value_counts().argsort() \n",
    "f_act_type = df_sessions.action_type.value_counts().argsort() \n",
    "f_dev_type = df_sessions.device_type.value_counts().argsort() \n",
    "\n",
    "#按照id进行分组 \n",
    "dgr_sess = df_sessions.groupby(['id']) \n",
    "#Loop on dgr_sess to create all the features.\n",
    "samples = [] #samples列表\n",
    "ln = len(dgr_sess) #计算分组后df_sessions的长度\n",
    "\n",
    "for g in dgr_sess:  #对dgr_sess中每个id的数据进行遍历\n",
    "    gr = g[1]   #data frame that comtains all the data for a groupby value 'zzywmcn0jv'\n",
    "\n",
    "    l = []  #建一个空列表，临时存放特征\n",
    "\n",
    "    #the id    for example:'zzywmcn0jv'\n",
    "    l.append(g[0]) #将id值放入空列表中\n",
    "\n",
    "    # number of total actions\n",
    "    l.append(len(gr))#将id对应数据的长度放入列表\n",
    "\n",
    "    #secs_elapsed 特征中的缺失值用0填充再获取具体的停留时长值\n",
    "    sev = gr.secs_elapsed.fillna(0).values   #These values are used later.\n",
    "\n",
    "    #action features 特征-用户行为 \n",
    "    #每个用户行为出现的次数，各个行为类型的数量，平均值以及标准差\n",
    "    c_act = [0] * len(f_act)\n",
    "    for i,v in enumerate(gr.action.values): #i是从0-1对应的位置，v 是用户行为特征的值\n",
    "        c_act[f_act[v]] += 1\n",
    "    _, c_act_uqc = np.unique(gr.action.values, return_counts=True)\n",
    "    #计算用户行为行为特征各个类型数量的长度，平均值以及标准差\n",
    "    c_act += [len(c_act_uqc), np.mean(c_act_uqc), np.std(c_act_uqc)]\n",
    "    l = l + c_act\n",
    "\n",
    "    #action_detail features 特征-用户行为具体\n",
    "    #(how many times each value occurs, numb of unique values, mean and std)\n",
    "    c_act_detail = [0] * len(f_act_detail)\n",
    "    for i,v in enumerate(gr.action_detail.values):\n",
    "        c_act_detail[f_act_detail[v]] += 1\n",
    "    _, c_act_det_uqc = np.unique(gr.action_detail.values, return_counts=True)\n",
    "    c_act_detail += [len(c_act_det_uqc), np.mean(c_act_det_uqc), np.std(c_act_det_uqc)]\n",
    "    l = l + c_act_detail\n",
    "\n",
    "    #action_type features  特征-用户行为类型 click等\n",
    "    #(how many times each value occurs, numb of unique values, mean and std\n",
    "    #+ log of the sum of secs_elapsed for each value)\n",
    "    l_act_type = [0] * len(f_act_type)\n",
    "    c_act_type = [0] * len(f_act_type)\n",
    "    for i,v in enumerate(gr.action_type.values):\n",
    "        l_act_type[f_act_type[v]] += sev[i] #sev = gr.secs_elapsed.fillna(0).values ，求每个行为类型总的停留时长\n",
    "        c_act_type[f_act_type[v]] += 1  \n",
    "    l_act_type = np.log(1 + np.array(l_act_type)).tolist() #每个行为类型总的停留时长，差异比较大，进行log处理\n",
    "    _, c_act_type_uqc = np.unique(gr.action_type.values, return_counts=True)\n",
    "    c_act_type += [len(c_act_type_uqc), np.mean(c_act_type_uqc), np.std(c_act_type_uqc)]\n",
    "    l = l + c_act_type + l_act_type    \n",
    "\n",
    "    #device_type features 特征-设备类型\n",
    "    #(how many times each value occurs, numb of unique values, mean and std)\n",
    "    c_dev_type  = [0] * len(f_dev_type)\n",
    "    for i,v in enumerate(gr.device_type .values):\n",
    "        c_dev_type[f_dev_type[v]] += 1 \n",
    "    c_dev_type.append(len(np.unique(gr.device_type.values))) \n",
    "    _, c_dev_type_uqc = np.unique(gr.device_type.values, return_counts=True)\n",
    "    c_dev_type += [len(c_dev_type_uqc), np.mean(c_dev_type_uqc), np.std(c_dev_type_uqc)]        \n",
    "    l = l + c_dev_type    \n",
    "\n",
    "    #secs_elapsed features  特征-停留时长     \n",
    "    l_secs = [0] * 5 \n",
    "    l_log = [0] * 15\n",
    "    if len(sev) > 0:\n",
    "        #Simple statistics about the secs_elapsed values.\n",
    "        l_secs[0] = np.log(1 + np.sum(sev))\n",
    "        l_secs[1] = np.log(1 + np.mean(sev)) \n",
    "        l_secs[2] = np.log(1 + np.std(sev))\n",
    "        l_secs[3] = np.log(1 + np.median(sev))\n",
    "        l_secs[4] = l_secs[0] / float(l[1]) #\n",
    "\n",
    "        #Values are grouped in 15 intervals. Compute the number of values\n",
    "        #in each interval.\n",
    "        #sev = gr.secs_elapsed.fillna(0).values \n",
    "        log_sev = np.log(1 + sev).astype(int)\n",
    "        #np.bincount():Count number of occurrences of each value in array of non-negative ints.  \n",
    "        l_log = np.bincount(log_sev, minlength=15).tolist()                    \n",
    "    l = l + l_secs + l_log\n",
    "\n",
    "    #The list l has the feature values of one sample.\n",
    "    samples.append(l)\n",
    "\n",
    "#preparing objects    \n",
    "samples = np.array(samples) \n",
    "samp_ar = samples[:, 1:].astype(np.float16) #取除id外的特征数据\n",
    "samp_id = samples[:, 0]   #取id，id位于第一列\n",
    "\n",
    "#为提取的特征创建一个dataframe     \n",
    "col_names = []    #name of the columns\n",
    "for i in range(len(samples[0])-1):  #减1的原因是因为有个id\n",
    "    col_names.append('c_' + str(i))  #起名字的方式    \n",
    "df_agg_sess = pd.DataFrame(samp_ar, columns=col_names)\n",
    "df_agg_sess['id'] = samp_id\n",
    "df_agg_sess.index = df_agg_sess.id #将id作为index\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style>\n",
       "    .dataframe thead tr:only-child th {\n",
       "        text-align: right;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>c_0</th>\n",
       "      <th>c_1</th>\n",
       "      <th>c_2</th>\n",
       "      <th>c_3</th>\n",
       "      <th>c_4</th>\n",
       "      <th>c_5</th>\n",
       "      <th>c_6</th>\n",
       "      <th>c_7</th>\n",
       "      <th>c_8</th>\n",
       "      <th>c_9</th>\n",
       "      <th>...</th>\n",
       "      <th>c_448</th>\n",
       "      <th>c_449</th>\n",
       "      <th>c_450</th>\n",
       "      <th>c_451</th>\n",
       "      <th>c_452</th>\n",
       "      <th>c_453</th>\n",
       "      <th>c_454</th>\n",
       "      <th>c_455</th>\n",
       "      <th>c_456</th>\n",
       "      <th>id</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>id</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>00023iyk9l</th>\n",
       "      <td>40.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>12.0</td>\n",
       "      <td>6.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>00023iyk9l</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0010k6l0om</th>\n",
       "      <td>63.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>8.0</td>\n",
       "      <td>12.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>8.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0010k6l0om</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>001wyh0pz8</th>\n",
       "      <td>90.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>27.0</td>\n",
       "      <td>30.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>8.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>001wyh0pz8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0028jgx1x1</th>\n",
       "      <td>31.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>1.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0028jgx1x1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>002qnbzfs5</th>\n",
       "      <td>789.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>111.0</td>\n",
       "      <td>102.0</td>\n",
       "      <td>104.0</td>\n",
       "      <td>57.0</td>\n",
       "      <td>28.0</td>\n",
       "      <td>9.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>002qnbzfs5</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 458 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "              c_0  c_1  c_2  c_3  c_4  c_5  c_6  c_7  c_8  c_9     ...      \\\n",
       "id                                                                 ...       \n",
       "00023iyk9l   40.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     ...       \n",
       "0010k6l0om   63.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     ...       \n",
       "001wyh0pz8   90.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     ...       \n",
       "0028jgx1x1   31.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     ...       \n",
       "002qnbzfs5  789.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0     ...       \n",
       "\n",
       "            c_448  c_449  c_450  c_451  c_452  c_453  c_454  c_455  c_456  \\\n",
       "id                                                                          \n",
       "00023iyk9l   12.0    6.0    2.0    3.0    3.0    1.0    0.0    1.0    0.0   \n",
       "0010k6l0om    8.0   12.0    2.0    8.0    4.0    3.0    0.0    0.0    0.0   \n",
       "001wyh0pz8   27.0   30.0    9.0    8.0    1.0    0.0    0.0    0.0    0.0   \n",
       "0028jgx1x1    1.0    2.0    3.0    5.0    4.0    1.0    0.0    0.0    0.0   \n",
       "002qnbzfs5  111.0  102.0  104.0   57.0   28.0    9.0    4.0    1.0    1.0   \n",
       "\n",
       "                    id  \n",
       "id                      \n",
       "00023iyk9l  00023iyk9l  \n",
       "0010k6l0om  0010k6l0om  \n",
       "001wyh0pz8  001wyh0pz8  \n",
       "0028jgx1x1  0028jgx1x1  \n",
       "002qnbzfs5  002qnbzfs5  \n",
       "\n",
       "[5 rows x 458 columns]"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_agg_sess.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "train_row = train.shape[0]\n",
    "labels = train['country_destination'].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#合并train和test文件\n",
    "df = pd.concat([train, test], axis = 0, ignore_index = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#转换为datetime类型\n",
    "import datetime\n",
    "tfa = df.timestamp_first_active.astype(str).apply(lambda x: datetime.datetime(int(x[:4]), \n",
    "                                                                              int(x[4:6]), \n",
    "                                                                              int(x[6:8]), \n",
    "                                                                              int(x[8:10]), \n",
    "                                                                              int(x[10:12]), \n",
    "                                                                              int(x[12:])))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0   2009-03-19 04:32:55\n",
       "1   2009-05-23 17:48:09\n",
       "Name: timestamp_first_active, dtype: datetime64[ns]"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tfa.head(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#提取年月日特征\n",
    "# create tfa_year, tfa_month, tfa_day feature\n",
    "df['tfa_year'] = np.array([x.year for x in tfa])\n",
    "df['tfa_month'] = np.array([x.month for x in tfa])\n",
    "df['tfa_day'] = np.array([x.day for x in tfa])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# 提取特征：weekday\n",
    "#isoweekday() 可以返回一周的星期几，e.g.星期日：0；星期一：1 \n",
    "df['tfa_wd'] = np.array([x.isoweekday() for x in tfa]) \n",
    "df_tfa_wd = pd.get_dummies(df.tfa_wd, prefix = 'tfa_wd') # one hot encoding \n",
    "df = pd.concat((df, df_tfa_wd), axis = 1) #添加df['tfa_wd'] 编码后的特征 \n",
    "df.drop(['tfa_wd'], axis = 1, inplace = True)#删除原有未编码的特征\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#提取特征：季节\n",
    "Y = 2000 \n",
    "seasons = [(0, (datetime.date(Y,1,1), datetime.date(Y,3,20))), #'winter' \n",
    "           (1, (datetime.date(Y,3,21), datetime.date(Y,6,20))), #'spring' \n",
    "           (2, (datetime.date(Y,6,21), datetime.date(Y,9,20))), #'summer' \n",
    "           (3, (datetime.date(Y,9,21), datetime.date(Y,12,20))), #'autumn' \n",
    "           (0, (datetime.date(Y,12,21), datetime.date(Y,12,31)))] #'winter' \n",
    "def get_season(dt): \n",
    "    dt = dt.date() #获取日期 \n",
    "    dt = dt.replace(year=Y) #将年统一换成2000年 \n",
    "    return next(season for season, (start, end) in seasons if start <= dt <= end) \n",
    "df['tfa_season'] = np.array([get_season(x) for x in tfa]) \n",
    "df_tfa_season = pd.get_dummies(df.tfa_season, prefix = 'tfa_season') # one hot encoding \n",
    "df = pd.concat((df, df_tfa_season), axis = 1) \n",
    "df.drop(['tfa_season'], axis = 1, inplace = True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#将date_account_created转换为datetime类型\n",
    "dac = pd.to_datetime(df.date_account_created)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#提取特征：年，月，日\n",
    "df['dac_year'] = np.array([x.year for x in dac]) \n",
    "df['dac_month'] = np.array([x.month for x in dac]) \n",
    "df['dac_day'] = np.array([x.day for x in dac])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#提取weekday特征\n",
    "df['dac_wd'] = np.array([x.isoweekday() for x in dac])\n",
    "df_dac_wd = pd.get_dummies(df.dac_wd, prefix = 'dac_wd') \n",
    "df = pd.concat((df, df_dac_wd), axis = 1) \n",
    "df.drop(['dac_wd'], axis = 1, inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#提取季节特征\n",
    "df['dac_season'] = np.array([get_season(x) for x in dac]) \n",
    "df_dac_season = pd.get_dummies(df.dac_season, prefix = 'dac_season') \n",
    "df = pd.concat((df, df_dac_season), axis = 1) \n",
    "df.drop(['dac_season'], axis = 1, inplace = True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "-1     275369\n",
       " 0          7\n",
       " 6          4\n",
       " 5          4\n",
       " 1          4\n",
       " 2          3\n",
       " 3          3\n",
       " 4          3\n",
       " 28         3\n",
       " 94         2\n",
       "dtype: int64"
      ]
     },
     "execution_count": 70,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#提取特征：date_account_created和timestamp_first_active之间的差值\n",
    "dt_span = dac.subtract(tfa).dt.days \n",
    "dt_span.value_counts().head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#create categorical feature: span = -1; -1 < span < 30; 31 < span < 365; span > 365 \n",
    "def get_span(dt): \n",
    "    # dt is an integer \n",
    "    if dt == -1: \n",
    "        return 'OneDay' \n",
    "    elif (dt < 30) & (dt > -1): \n",
    "        return 'OneMonth' \n",
    "    elif (dt >= 30) & (dt <= 365): \n",
    "        return 'OneYear' \n",
    "    else: \n",
    "        return 'other' \n",
    "df['dt_span'] = np.array([get_span(x) for x in dt_span]) \n",
    "df_dt_span = pd.get_dummies(df.dt_span, prefix = 'dt_span') \n",
    "df = pd.concat((df, df_dt_span), axis = 1) \n",
    "df.drop(['dt_span'], axis = 1, inplace = True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#删除原有特征\n",
    "df.drop(['date_account_created','timestamp_first_active'], axis = 1, inplace = True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "在数据探索阶段，我们发现大部分数据是集中在（15，90）区间的，但有部分年龄分布在（1900，2000）区间，我们猜测用户是把出生日期误填为年龄，故进行预处理"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\mail\\Anaconda3\\lib\\site-packages\\ipykernel_launcher.py:4: RuntimeWarning: invalid value encountered in less\n",
      "  after removing the cwd from sys.path.\n",
      "C:\\Users\\mail\\Anaconda3\\lib\\site-packages\\ipykernel_launcher.py:4: RuntimeWarning: invalid value encountered in greater\n",
      "  after removing the cwd from sys.path.\n"
     ]
    }
   ],
   "source": [
    "av = df.age.values\n",
    "#This are birthdays instead of age (estimating age by doing 2014 - value)\n",
    "#数据来自2014年，故用2014-value \n",
    "av = np.where(np.logical_and(av<2000, av>1900), 2014-av, av) \n",
    "df['age'] = av\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#将年龄进行分段\n",
    "age = df.age \n",
    "age.fillna(-1, inplace = True) #空值填充为-1 \n",
    "div = 15 \n",
    "def get_age(age): \n",
    "    # age is a float number  将连续型转换为离散型 \n",
    "    if age < 0: \n",
    "        return 'NA' #表示是空值 \n",
    "    elif (age < div): \n",
    "        return div #如果年龄小于15岁，那么返回15岁 \n",
    "    elif (age <= div * 2): \n",
    "        return div*2 #如果年龄大于15小于等于30岁，则返回30岁 \n",
    "    elif (age <= div * 3): \n",
    "        return div * 3 \n",
    "    elif (age <= div * 4): \n",
    "        return div * 4 \n",
    "    elif (age <= div * 5): \n",
    "        return div * 5 \n",
    "    elif (age <= 110): \n",
    "        return div * 6 \n",
    "    else: return 'Unphysical' #非正常年龄\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#创建新的‘age'特征\n",
    "df['age'] = np.array([get_age(x) for x in age]) \n",
    "df_age = pd.get_dummies(df.age, prefix = 'age') \n",
    "df = pd.concat((df, df_age), axis = 1) \n",
    "df.drop(['age'], axis = 1, inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#其他特征，直接onehot编码\n",
    "feat_toOHE = ['gender', \n",
    "              'signup_method', \n",
    "              'signup_flow', \n",
    "              'language', \n",
    "              'affiliate_channel', \n",
    "              'affiliate_provider', \n",
    "              'first_affiliate_tracked', \n",
    "              'signup_app', \n",
    "              'first_device_type', \n",
    "              'first_browser'] \n",
    "#对其他特征进行one-hot-encoding处理 \n",
    "for f in feat_toOHE: \n",
    "    df_ohe = pd.get_dummies(df[f], prefix=f, dummy_na=True) \n",
    "    df.drop([f], axis = 1, inplace = True) \n",
    "    df = pd.concat((df, df_ohe), axis = 1)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {},
   "outputs": [
    {
     "ename": "TypeError",
     "evalue": "'<' not supported between instances of 'str' and 'int'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-78-a40e60bed021>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m()\u001b[0m\n\u001b[0;32m      5\u001b[0m \u001b[0mdf_all\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mdf_all\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfillna\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m-\u001b[0m\u001b[1;36m2\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;31m#对没有sesssion data的特征进行缺失值处理\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      6\u001b[0m \u001b[1;31m#加了一列，表示每一行总共有多少空值，这也作为一个特征\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 7\u001b[1;33m \u001b[0mdf_all\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'all_null'\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0marray\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0msum\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mr\u001b[0m\u001b[1;33m<\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mfor\u001b[0m \u001b[0mr\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mdf_all\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mvalues\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;32m<ipython-input-78-a40e60bed021>\u001b[0m in \u001b[0;36m<listcomp>\u001b[1;34m(.0)\u001b[0m\n\u001b[0;32m      5\u001b[0m \u001b[0mdf_all\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mdf_all\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfillna\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m-\u001b[0m\u001b[1;36m2\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;31m#对没有sesssion data的特征进行缺失值处理\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      6\u001b[0m \u001b[1;31m#加了一列，表示每一行总共有多少空值，这也作为一个特征\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 7\u001b[1;33m \u001b[0mdf_all\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'all_null'\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0marray\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0msum\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mr\u001b[0m\u001b[1;33m<\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mfor\u001b[0m \u001b[0mr\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mdf_all\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mvalues\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mTypeError\u001b[0m: '<' not supported between instances of 'str' and 'int'"
     ]
    }
   ],
   "source": [
    "#将上面session和train及test得到的特征进行合并\n",
    "#将对session提取的特征整合到一起 \n",
    "df_all = pd.merge(df, df_agg_sess, how='left') \n",
    "df_all = df_all.drop(['id'], axis=1) #删除id \n",
    "df_all = df_all.fillna(-2) #对没有sesssion data的特征进行缺失值处理 \n",
    "#加了一列，表示每一行总共有多少空值，这也作为一个特征 \n",
    "#df_all['all_null'] = np.array([sum(r<0) for r in df_all.values]) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style>\n",
       "    .dataframe thead tr:only-child th {\n",
       "        text-align: right;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>country_destination</th>\n",
       "      <th>tfa_year</th>\n",
       "      <th>tfa_month</th>\n",
       "      <th>tfa_day</th>\n",
       "      <th>tfa_wd_1</th>\n",
       "      <th>tfa_wd_2</th>\n",
       "      <th>tfa_wd_3</th>\n",
       "      <th>tfa_wd_4</th>\n",
       "      <th>tfa_wd_5</th>\n",
       "      <th>tfa_wd_6</th>\n",
       "      <th>...</th>\n",
       "      <th>c_447</th>\n",
       "      <th>c_448</th>\n",
       "      <th>c_449</th>\n",
       "      <th>c_450</th>\n",
       "      <th>c_451</th>\n",
       "      <th>c_452</th>\n",
       "      <th>c_453</th>\n",
       "      <th>c_454</th>\n",
       "      <th>c_455</th>\n",
       "      <th>c_456</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>NDF</td>\n",
       "      <td>2009</td>\n",
       "      <td>3</td>\n",
       "      <td>19</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>-2.0</td>\n",
       "      <td>-2.0</td>\n",
       "      <td>-2.0</td>\n",
       "      <td>-2.0</td>\n",
       "      <td>-2.0</td>\n",
       "      <td>-2.0</td>\n",
       "      <td>-2.0</td>\n",
       "      <td>-2.0</td>\n",
       "      <td>-2.0</td>\n",
       "      <td>-2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>NDF</td>\n",
       "      <td>2009</td>\n",
       "      <td>5</td>\n",
       "      <td>23</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>-2.0</td>\n",
       "      <td>-2.0</td>\n",
       "      <td>-2.0</td>\n",
       "      <td>-2.0</td>\n",
       "      <td>-2.0</td>\n",
       "      <td>-2.0</td>\n",
       "      <td>-2.0</td>\n",
       "      <td>-2.0</td>\n",
       "      <td>-2.0</td>\n",
       "      <td>-2.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>2 rows × 661 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "  country_destination  tfa_year  tfa_month  tfa_day  tfa_wd_1  tfa_wd_2  \\\n",
       "0                 NDF      2009          3       19         0         0   \n",
       "1                 NDF      2009          5       23         0         0   \n",
       "\n",
       "   tfa_wd_3  tfa_wd_4  tfa_wd_5  tfa_wd_6  ...    c_447  c_448  c_449  c_450  \\\n",
       "0         0         1         0         0  ...     -2.0   -2.0   -2.0   -2.0   \n",
       "1         0         0         0         1  ...     -2.0   -2.0   -2.0   -2.0   \n",
       "\n",
       "   c_451  c_452  c_453  c_454  c_455  c_456  \n",
       "0   -2.0   -2.0   -2.0   -2.0   -2.0   -2.0  \n",
       "1   -2.0   -2.0   -2.0   -2.0   -2.0   -2.0  \n",
       "\n",
       "[2 rows x 661 columns]"
      ]
     },
     "execution_count": 79,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_all.head(2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#模型构建：\n",
    "主要选择了逻辑回归模型，树模型，SVM，以及xgboost模型"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#选出训练集和测试集并保存\n",
    "Xtrain = df_all.iloc[:train_row, :]\n",
    "Xtest = df_all.iloc[train_row:, :]\n",
    "Xtrain.to_csv(\"Airbnb_xtrain_v2.csv\")\n",
    "Xtest.to_csv(\"Airbnb_xtest_v2.csv\")\n",
    "#labels.tofile（）：Write array to a file as text or binary (default)\n",
    "labels.tofile(\"Airbnb_ytrain_v2.csv\", sep='\\n', format='%s') #存放目标变量"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "xtrain = Xtrain\n",
    "ytrain = labels"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import LabelEncoder \n",
    "from sklearn.preprocessing import StandardScaler \n",
    "from sklearn.preprocessing import LabelBinarizer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#对目标变量进行encoding\n",
    "le = LabelEncoder()\n",
    "ytrain_le = le.fit_transform(ytrain.values)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#提取10%的数据进行训练\n",
    "# Let us take 10% of the data for faster training. \n",
    "n = int(xtrain.shape[0]*0.1) \n",
    "xtrain_new = xtrain.iloc[:n, :] #训练数据 \n",
    "ytrain_new = ytrain_le[:n] #训练数据的目标变量"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#将所有的数据都规范化0-1\n",
    "X_scaler = StandardScaler()\n",
    "xtrain_new = X_scaler.fit_transform(xtrain_new)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "评分模型：NDCG\n",
    "NDCG是一种衡量排序质量的评价指标，该指标考虑了所有元素的相关性\n",
    "由于我们预测的目标变量并不是二分类变量，故我们用NDGG模型来进行模型评分，判断模型优劣\n",
    "一般二分类变量: 我们习惯于使用 f1 score, precision, recall, auc score来进行模型评分"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from sklearn.metrics import make_scorer \n",
    "def dcg_score(y_true, y_score, k=5): \n",
    "    \"\"\"\n",
    "    y_true : array, shape = [n_samples] #数据\n",
    "        Ground truth (true relevance labels).\n",
    "    y_score : array, shape = [n_samples, n_classes] #预测的分数\n",
    "        Predicted scores.\n",
    "    k : int\n",
    "    \"\"\" \n",
    "    order = np.argsort(y_score)[::-1] #分数从高到低排序 \n",
    "    y_true = np.take(y_true, order[:k]) #取出前k[0,k）个分数 \n",
    "    gain = 2 ** y_true - 1 \n",
    "    discounts = np.log2(np.arange(len(y_true)) + 2) \n",
    "    return np.sum(gain / discounts) \n",
    "\n",
    "def ndcg_score(ground_truth, predictions, k=5): \n",
    "    \"\"\"\n",
    "    Parameters\n",
    "    ----------\n",
    "    ground_truth : array, shape = [n_samples]\n",
    "        Ground truth (true labels represended as integers).\n",
    "    predictions : array, shape = [n_samples, n_classes] \n",
    "        Predicted probabilities. 预测的概率\n",
    "    k : int\n",
    "        Rank.\n",
    "    \"\"\" \n",
    "    lb = LabelBinarizer() \n",
    "    lb.fit(range(len(predictions) + 1)) \n",
    "    T = lb.transform(ground_truth) \n",
    "    scores = [] \n",
    "    # Iterate over each y_true and compute the DCG score \n",
    "    for y_true, y_score in zip(T, predictions): \n",
    "        actual = dcg_score(y_true, y_score, k) \n",
    "        best = dcg_score(y_true, y_true, k) \n",
    "        score = float(actual) / float(best) \n",
    "        scores.append(score) \n",
    "        return np.mean(scores)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#模型构建"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#逻辑回归\n",
    "from sklearn.linear_model import LogisticRegression \n",
    "from sklearn.model_selection import KFold \n",
    "from sklearn.model_selection import cross_val_score \n",
    "from sklearn.model_selection import train_test_split \n",
    "lr = LogisticRegression(C = 1.0, penalty='l2', multi_class='ovr') \n",
    "RANDOM_STATE = 2017 #随机种子 #k-fold cross validation（k-折叠交叉验证） \n",
    "kf = KFold(n_splits=5, random_state=RANDOM_STATE) #分成5个组 \n",
    "train_score = [] \n",
    "cv_score = [] \n",
    "# select a k  (value how many y): \n",
    "k_ndcg = 3 \n",
    "# kf.split: Generate indices to split data into training and test set. \n",
    "for train_index, test_index in kf.split(xtrain_new, ytrain_new): \n",
    "    #训练集数据分割为训练集和测试集，y是目标变量 \n",
    "    X_train, X_test = xtrain_new[train_index, :], xtrain_new[test_index, :] \n",
    "    y_train, y_test = ytrain_new[train_index], ytrain_new[test_index] \n",
    "    lr.fit(X_train, y_train) \n",
    "    y_pred = lr.predict_proba(X_test) \n",
    "    train_ndcg_score = ndcg_score(y_train, lr.predict_proba(X_train), k = k_ndcg) \n",
    "    cv_ndcg_score = ndcg_score(y_test, y_pred, k=k_ndcg) \n",
    "    train_score.append(train_ndcg_score) \n",
    "    cv_score.append(cv_ndcg_score) \n",
    "    print (\"\\nThe training score is: {}\".format(np.mean(train_score))) \n",
    "    print (\"\\nThe cv score is: {}\".format(np.mean(cv_score)))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#改变迭代次数\n",
    "# set the iterations \n",
    "iteration = [1,5,10,15,20, 50, 100] \n",
    "kf = KFold(n_splits=3, random_state=RANDOM_STATE) \n",
    "train_score = [] \n",
    "cv_score = [] \n",
    "# select a k: \n",
    "k_ndcg = 5 \n",
    "for i, item in enumerate(iteration): \n",
    "    lr = LogisticRegression(C=1.0, max_iter=item, tol=1e-5, solver='newton-cg', multi_class='ovr') \n",
    "    train_score_iter = [] \n",
    "    cv_score_iter = [] \n",
    "    for train_index, test_index in kf.split(xtrain_new, ytrain_new): \n",
    "        X_train, X_test = xtrain_new[train_index, :], xtrain_new[test_index, :] \n",
    "        y_train, y_test = ytrain_new[train_index], ytrain_new[test_index] \n",
    "        lr.fit(X_train, y_train) y_pred = lr.predict_proba(X_test) \n",
    "        train_ndcg_score = ndcg_score(y_train, lr.predict_proba(X_train), k = k_ndcg) \n",
    "        cv_ndcg_score = ndcg_score(y_test, y_pred, k=k_ndcg) \n",
    "        train_score_iter.append(train_ndcg_score) \n",
    "        cv_score_iter.append(cv_ndcg_score) \n",
    "    train_score.append(np.mean(train_score_iter)) \n",
    "    cv_score.append(np.mean(cv_score_iter))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "ymin = np.min(cv_score)-0.05 \n",
    "ymax = np.max(train_score)+0.05 \n",
    "plt.figure(figsize=(9,4)) \n",
    "plt.plot(iteration, train_score, 'ro-', label = 'training') \n",
    "plt.plot(iteration, cv_score, 'b*-', label = 'Cross-validation') \n",
    "plt.xlabel(\"iterations\") \n",
    "plt.ylabel(\"Score\") \n",
    "plt.xlim(-5, np.max(iteration)+10) \n",
    "plt.ylim(ymin, ymax) \n",
    "plt.plot(np.linspace(20,20,50), np.linspace(ymin, ymax, 50), 'g--') \n",
    "plt.legend(loc = 'lower right', fontsize = 12) \n",
    "plt.title(\"Score vs iteration learning curve\") \n",
    "plt.tight_layout()"
   ]
  },
  {
   "attachments": {
    "image.png": {
     "image/png": "iVBORw0KGgoAAAANSUhEUgAAAoYAAAEXCAYAAAAwWIUCAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAADysSURBVHhe7d0HfBR1+sfxJyR0UKoECIQiGIooilKlKioliDRplqNY7k7Pemc5EVA59AT09BDxLHCKf/FQBJSocIQiICieipRQJTQlSAch2fnP8+wsLCVAYLPJJp/3ix8z85tNtme/+yszUY5LAAAAkC8Eop8ug4vP55MCtgcAAAD5HsEQAAAAhmAIAAAAQzAEAACAIRgCAADAEAwBAABgCIYAcr2YmBhZv369txV+8+fPlxo1anhbobNx40aJioqS9PR0ryZ8sus+AYhsBEMgj1iwYIE0btxYChcuLEWLFpUmTZrI0qVLvb2RTYNT9erVbf3222+Xv/71r7aeXTSsrV271tsSadGihaxbt87byhvy4n0CcP4IhkAesGfPHmnbtq3c/8D9cuDAAdm5c6cMHz7cQmIoZWRkeGuRKyda58Itr9zHvPB6AyINwRDIA1JSUmx5S69bJDo62loMr7vuOrn00kutXr3++uty8cUXS8GCBaVWrVqybNkyq1+5cqW0bNnSQmTt2rVl2vRpVq+0de7uu++WG264wbpz58yZI7/99ps8/PDDEhsbKxdeeKHcddddcvDgQe8njtHL6e9cvny5VyOyY8cOu32//PKLpKWlSceOHY+2cDZr1syOun8qgRa88ePHy9tvvy1PP/203Z5OnTrZ/q1bt8rNN98sJUuWlMqVK8s//vEPq1dDhw6Vbt26Sb9+/aRQoUL280uWLJGrr77arrtMmTLyhz/8QQ4fPmyXb96iuS0vueQSu473J78vycnJUr58eatXZ3rMfv/730uHDh3ssW7UqNFZt8zt3r1bBgwYYLepXLly1jIaCEf6O1q3bi3FixeXEiVKSN++fWXX7l22T1WqVEmee+45qVu3rt1PDYda98ILL1id3taePXvKoUOH7PIn3qfTXVY9//zzdrvKli0r//rXv05qVQ3266+/yh133GGX1ee2S5cuVq+PfdOmTW09IPj3nPh6+9vf/ialS5c+LiBOnfqR3Ualr5eRI0dKlSpV7HHp0aOHXTeA8+AAiHhuoHCKFSvm3Hrrrc6nn37quB+O3h6/Dz6Y7Lgf0o4biBz3w9RZs2aNs3HjRscNQ07VqlWdZ5991nGDnDN79mzH/UB2Vq1aZT932223OW7IcBYsWOC4H86OGwCdP/3pT44byJydO3c6e/bscdxw5zz66KN2+RO54cB5/PHHvS3HeeWVVxw3sNq6/sydd95pt0HLvHnz7Ladiv6p0tus9DY98cQTtq70djVo0MAZNmyY3Qc3QDluOHSSkpJs/1NPPWU/7wYKu+yBAwecr7/+2lm0aJFz5MgRZ8OGDU7NmjWdMWPG2OVV8PUpNxA7blCz9bN5zIoUKeJ89dVX9vv79OnjuCHL9p1Ir1uvSy+nEhMTncGDBzv79u1zfv75Z+eKK65wxo0bZ/v09nz++eeOG9YcN1g7zZo3s+cioGLFik69evWcTZs22X0M1Onv2LJliz1fej9fffVV2xd8n9TpLjtz5kynVKlSjhvynf379ztuyD7pMQp24403Om5Is9ehPl5uCLX6t956y2nSpImtB5z43J74eouLi7P7HeCGfMcNjLauz5kbvJ3U1FR7XPSx69Wrl+0DkDn9W6tF32ful0j7G6TvVX0fEQyBPGLFihX2waof9vphq+Ft+/bttu/aa691XnzxRVsPpmFMP/D1j0OAfrBqmFL6+/r372/rSv+QREdHO2vXrvVqHGfhwoVOpUqVvK3jffHFFxbSAho3buxMmDjB1p988kmnc+fOmYaLYCeGh+BguHjxYqdChQrelt+IESOc22+/3db1vmiIOh0NGF26dPG2Th8Mz+YxGzBggK2rTz75xKlRo4a3dbzgYKjPVVRU1NFQpya9N8lp2bKlt3U8Dbr169f3tvzB7o033vC2/LTu3+/829tynEceecTCuDpVMMzsshrwg8O/PjYnPkYBW7dutX0nfjlRZxMMg19vSp9rvX6lX0T09adfapSG11mzZtm6Clx3IGgDOLXTBUO6koE8IiEhQdwPXuum1a7O1NRUuffee22fG0Ck5sU1bT3Ylq1bbFJHgQLH/hTotv5sQNWqVb01f1ew+4dE6tSpY92NWlq1aiVuqPEucbw2bdrYmMevvvpKfvrpJ5sMc1OXm2zfQw89ZN2w+vNxcXHWJXguNv600a4/cHu0aBesdi8HVIuv5q35ade7dmNrN6V2uz7wwAPy888/e3tP72weMzdkeWsixYoVszGgZ6IzlN2/1+KGzqP349b+tx69H/q8ugHUupj1Nt98c7eTbrN2qZ4otkKst+a/Lfv27fO2TpbZZfW+Bf9ufb4ys2nTJuvW1ftxLoJfb6pPnz7yzjvv2NCEDz/60IYABC6jM9W12znweMXHx1vX9Nk+lwBORjAE8iAdHzdw4ED53//+Z9vVqlWTtWtOHg9WqWIl+3B1vzl6Nf6AEvzBrx+0ATpmTAOR/ox+UAdKZpMd9LL9+/eXSZMmybvvvnt0HKDS5d///ncLHZ999pk888wzMnv2bNt3OsG3R1WJq2KBIPj2uN9+ZebMmd4lTv6ZwYMH2zg1Davut2QZNWqUhbKzcTaP2bnQ4KWP1/79+4/eD71tq1evtv1/+ctf7H7ottZPmfKfk8Zknng/Q0XHbQYH3+D1E+n90PsQPP4xoESJ4rYv4FRfKE68D/olRMfGJiXNlIkTJsqtt97q7RF73gPjXgNFHxMdLwng3BAMgTxg1apVFm42b95s2/rBPXHiRDskibrrrjsteH3zzTcWgHSwv4YibX3RiQw6sUDDlE5ImDx5stxyyy32cyfS4KITNe677z5rwVJbtmyxYJcZnSShkw60NbN//35erciMT2bY7dDbc8EFF9hkA52YciY66SV40sNVV11lk2B04oVOgtEWTZ3wcrpD9WgLnl6n3nd97F588UVvj5+2dmU2YSSrj9nZ0vulrZgPPvig3T4NOHob5s6da/u1TsO03ld9zJ99doTVh4O2VI4dO9ZaorUFWCf0ZEbvx4033ij33H2P7Nq1yx6jefPm2b4GDS6TH374wb6w6MSWIUOGWP2Z3Hbbbe7re7R9cdAJJgHaIq6BWV/LSlu0p0372NYBnBuCIZAHaGBYuHChXHbZZRawLr/8clvXWaaqW7fuMmzYMOnevbvNlNUAooe00S7JpKQkmT59uoUdbWX84IPJ1uKYGe3y1S7ghg0b2s/r7FwNV5nRIKW3T1vVrr/+Bq9WJGV1ilxzzTV2e/R3aXeudiufic7a/e6776zrUGe7apjU1kENvdpSpPdDZ7fqDN/MjBkzxoKqXrfOntVWzWA6G1bDkF7HZPfxCHYuj9nZ0jCvrYE1a9Y8Opt32zZ/V7KGMe2S19vUvn17mzUcLtpd+8gjj9iMYm0R1BnkSm/LqWjXrz622qKnAVy/tCidDa8zyps3b277rml5jdWfSe/evS2AX3/99dZqHaDBsGvXrjZbO/A6WrRosbcXwLmIcr+tn13/CQAALm051C5ebQ3ULyIAIksg+ukyuGhPBS2GAIAz0uMHamumdg/rcSwTExMJhUAeRDAEAJzRK6/804YE6Ixr7b4fN26ctwdAXkJXMgAAQD4SiH6BLuRAoSsZAAAAR0V0i6HOCNRZbshflv/sP/duvYvq2RIAAJxszZo1snfvXm/rmED0C7QUBoq2GEZ0MNRDEyxbtszbQn7R9u12tpx92yxbAgCAk2WWkwLRLxAIA4WuZAAAABxFMAQAAIAhGAIAAMAQDBFxnmj5uBUAABBaBENEnLbV21oBAAChRTBExPlu+3dWAABAaBEMEXH+NPN+KwAAILQIhgAAADBhC4ZJSUlSo0YNiY+Pl5EjR3q1x2zatElatWoll156qdStW1c+/fRTbw8AAADCISzBMCMjQwYOHCizZs2SlJQUmTBhgqxYscLb6zd8+HDp3bu3fP/99/Kf//xHBgwY4O0BAABAOIQlGC5ZskQSEhKkevXqUqhQIenfv79MnTrV2+sXFRUle/bssfXdu3dLlSpVbB0AAADhEZZguGXLFutCDoirEiepqanelt/QoUPlzTfflPLly0ubNm1k7Nix3p7jjR8/3s79p2Xbtm1eLfKTZ9s9YwUAAIRWWIKh4/i8tWO0hTDYpEmTZNCgQfLLL7/If//7X+nVq5edzPlEehk9IbSW2NhYrxb5SZO4JlYAAEBohSUYVq4cJxs3bvS2RFI3pbp1lb0tP20h7Nmzp603adJEDh48KGlpabYNBFuUusgKAAAIrbAEw0aNGtlkkw0bNsjhw4dl4sSJkpiY6O310/GHOjlFrVy50oJhuXLlbBvnz/fuJMkoUVsyosraUrdDLVzX8eenr5U//7Vltl0H8rdwvI4B5D689/3CEgxjYmJsbKCOHaxZs6b07dvXDkkzZMgQmTZ9ml1mzJgx1mpYr1496d69u7z33nsndTfnReF4IervdPo+ILLfa4F1l7odyusK63X8dthfkQ3XgfwtHK9jALkP7/1johyXtx5xdAKKjjWMVEdfiOIFHVNQov45VKK6dhHJ8Omxfk4u6emnrg+UdLf4MsTxLud0vdf9vfv8v/44JSTq3ROOKXmml0Mm+51+f3b/P+DfOE4xifr3ycetPBeB62h7+17bnv1WSVuG8jqQv4XjdQwg98n0vV+8rETvW+1tRJ7MclIg+ukyuOjcDoJhDtIWwqPfTnDWTg6GAABkj2gncj+nCYaRxH3wM6LLexsnixo7wn01Rh8rMVpi3HUtWldAonRZwNsffDlbBi4XLb4rbnKvb7f3m4NEl5ICK5K8jSBn6sI/xX7fJde5SXeXtxVEr2PV597G+Qlcx0nBMITXgfwtHK9jALlPpu99WgwjS6QGQ+enTeLrdYfIokxue4hfiKfusi4kUe+MkgJ9envb5yec1/FdIf+B0BscLuL+H9rrQP4WjtcxgNwnr773zyUYhmXyCTz6oE/8t/jim7ih8EeJuqOHW1nIv+8o94X42lBvPTT0Ra0vbg2cxl2G+sUezutoULCyPxRmw3UgfwvH6xhA7sN7/xhaDMPESUsTZ8A94kz9QqRRAykw+S2Jqhbv/5YyeIh/rKG+EN1QyIfQ6c1eP9uWbau3tSUAADgZLYa5lPNpkvjKX+WGwtkSNeJRiV70hYVCpSFQu411cKsuIz0UbtvmSKtW6bJ9e/Z93/jrF89I15eGZet1IH8Lx+sYQO7De9/NJd4S2WH/fvEN/r34OvQRiS8lBb6dJQX+8pBNCMmrhg/PkC8XigwbluHVhN6GjRmyb5eTrdeB/C0cr2MAuQ/vfbqSs43z1RLxdb5D5OetEvXAQCnwzFCRIjpZIm8qXDRdMo54G0E0A48bd/ws5hNfcWfaVlr3+987dphGp387q4ua6D9Tjl7HK6/k/YOhI/sFXmMn4jUG5G2ZvfcLFhQ5eDDG24o859KVTDAMtSNHxDd8hDjDXxIpVFoKJL0uUa1beTsj1+7djmzcqEVk/XpHNrjr69bqurbgiew/1fGzs8mJwRAAgFAqXETk5q4if/97tFSoELlfCgmGOcxZtVp8N98m8uNqierbRQq8Mlrkwgu9vTlPx0706pUh779/8gt9715/6NPAt8ENfus3uGWdu73BDX9u2es/QsxR2vhZrZqe49pduqVG9Sj5dKYjs2eLHWoxI11Eh0sOH+7vNj/x0IdZ3VZa95e/ZMg70W4wdNc1GPbtIzJiRN7tmkf46Wvs3UnHXse8xoD8Ifi978sQGTxIewoit7VQEQxzij6Yr4wV54/D3I2CEvXBi1Kg283+fblIv37p8t7/iTRtItKkqRv4vOC3br3InhOOf63fluLj3eCn4a+GLqMsBMbHR7mBMErKlIk6KbzdfHO6VKrkvpkGF5DXXvPJli0iU6aE9k2l11E0LkVu6VVAkibVzJbrQP4WjtcxgNwnL773CYY5wHFfOb6+7teKOYtE2jaTAu+Ml6jYWG9v7lC0aLr2cJ/MDXbXXidSU1v93LCnLX/V3OBXvXqUlC17cvADAACRg2AYBscdd1BKuMV/lPSosUOlwJ1uQMyFaUq7kFu3yZA1Kf7tgoVFunUVeeGFyBw7MWP1DFt2rN3RlgAA4GTnEgw5XE0WHD1ljoVCpTMujkjU83+WAncNzpWhUMXGRsn2bf51nWGVfsQ/9DFSB9S+sHCUFQAAEFoEwyywlsLjzqOo3JT91D+99dxp7VpH9u4VadxEZPHiAnKnm2G3bvV2AgAAeAiGWXG0pfAEmdXnEjNm+Gw5cUK0NGhQQF5+OYbB9AAA4CQEw6wInFz7RJnV5xLTpztSq7ZIjRrMJgEAAJkjGGZB1GtD3f8L+TeOKuTV5056fMLkZJHOnQmFAADg9AiGWVCgT2+JemfUsRZCd6nbWp9bfTHLf5qfTh3zTjCceNMEKwAAILQIhlmkITB632qJdtJsmZtDoZo+zSclS4o0bZp3nurKF1a2AgAAQotgmIf5fDq+UOTGG/2HqckrJi+fbAUAAIQWwTAP+/prn6SliXTKY+MLxy591QoAAAgtgmEeprORo9xn+IbreZoBAMCZkRjysGnTHGnSWKRMGWYkAwCAMyMY5lFbtjjy3XccpgYAAJw9gmEe9ckn/rOddOrEUwwAAM4OqSGP0m7kynEiderkvRbDyT0mWwEAAKFFMMyDDh7UA1trN7JIVB7sSS5brIwVAAAQWgTDPCg52Se/HXKDYR7tRp7wv4lWAABAaIUtOSQlJUmNGjUkPj5eRo4c6dUe88ADD0j9+vWtVK9eXQoXLuztQVZNn+6TwkVEWrXKm8HwrW/fsgIAAEIrLMkhIyNDBg4cKLNmzZKUlBSZMGGCrFixwtvrN2rUKPnhhx+sPPjgg9K3b19vD7LCcUSmThO57lqRIm44BAAAOFthCYZLliyRhIQEawksVKiQ9O/fX6ZOnertPdnEiROlT58+3hayYvlyR7Zu5jA1AAAg68ISDLds2WJdyAFxVeIkNTXV2zreTz/9ZK2Kbdq08WqON378eGnYsKGVbdu2ebUI0G5k1aFD3uxGBgAA2Scs6cFx/GElWFQm02Xfe+8960aOjo72ao43aNAgWbZsmZXY2FivFgEff+zI5ZeLVKxIiyEAAMiasATDypXjZOPGjd6WSOqmVLeusrd1PLqRz11amiNLlub9buTpfaZZAQAAoRWWYNioUSObbLJhwwY5fPiwhb/ExERv7zGrV692w02aNGnSxKtBVsxM8ok2znbqlLeDYbGCxawAAIDQCkswjImJsbGBOm6wZs2a1lVct25dGTJkiEybfqzlZ9KkSXLbbbdl2s2M09OznZQtJ9KwYd4eX/jq0nFWAABAaEU5Lm894ugEFB1rCJEjR0TKlU+XHt1FXn89xqvNm9q+3c6Ws2+bZUsAAHCyzHJSIPrpMrj4fL7wtBgi+335pU/279PxhTylAADg3JAi8ojpM3wSXVCkbVu64QEAwLkhGOYR0z4WadVSpGRJgiEAADg3BMM8YM0axy0iiZztBAAAnAeCYR4wY4b/AOIdO+aPp1MnnTDxBACA0CMY5gHTpztS+xKR6tVpMQQAAOeOYBjh9uxxZO5ckcTE/BMKRy0cbQUAAIQWwTDCfTHLkYwMkY4d8k8wnL56uhUAABBaBMMIN32aT0qWFGnalKcSAACcH9JEBPP59DR4Ih066GkHvUoAAIBzRDCMYEuX+uTXX0U6cZgaAAAQAgTDCKazkaPcZ/CG6/PX01i0UDErAAAgtAiGEWzaNEeaNhEpXTp/tRjO6D3NCgAACC2CYYTavNmR778X6Uw3MgAACBGCYYT65BP/2U46dcp/T+Ezc5+1AgAAQotgGKG0G7lKVZGEhPzXYjhr/SwrAAAgtAiGEejgQZHP3VzUqZNIFD3JAAAgRAiGEWjOHJ8c+U2kcz7sRgYAANmHZBGBpk/3SZEiIi1b8vQBAIDQIVlEGMcR+XiayHXXiYXD/Khs8bJWAABAaBEMI8wPP/hk65b8fZiayd3ftwIAAEKLYBhh9GwnqkMHnjoAABBapIsI8/HHjlzeUCQ2Nv+2GD4x6wkrAAAgtAiGESQtzZElS0US8/nZTr5MXWgFAACEFsEwgnw60yfi6PELOXghAAAIPYJhBJn2sSPlLxK5/HKeNgAAEHokjAhx5IjIp0kinTq6TxrPGgAAyAZEjAixYIFPDuzTw9TwlMVdGGcFAACEFikjQkyf4ZPogiJt2/KUTbjpbSsAACC0wpYykpKSpEaNGhIfHy8jR470ao83+YPJUqtWLaldu7b06dPHq4X6eKpI69YiJUp4FQAAACEWlmCYkZEhAwcOlFmzZklKSopMmDBBVqxY4e31W7NmjQwbOkyWLFkiq1evlhdffNHbg5QUR9atE0lkNrJ5MOkhKwAAILTCEgw17CUkJEj16tWlUKFC0r9/f5k6daq312/8+PHypz/9SUqVKmXb5cuXtyVEZszw2bJjR7qR1bJty6wAAIDQCkvS2LJli3UhB8RViZPU1FRvy2/lypXWUtikSRO56qqrrOv5VDRANmzY0Mq2bdu82rxNT4N3SYJItWq0GAIAgOwTlmDoOP4Wr2BRUceHnPT0dFm1apXMmzdPJk+eLP369ZNdu3d5e48ZNGiQLFu2zEpsbKxXm3ft2eO4j4lIYiKhEAAAZK+wBMPKleNk48aN3pZI6qZUt66yt+VXpUoV6dq1qxQsWFCqVasm9evXlzUpa7y9+dcXX/gkI0OPX0gwBAAA2SsswbBRo0Y22WTDhg1y+PBhmThxoiQmJnp7/TQUzp4929bT0tLkxx9/tFnM+d20aY6UvECkcWPGFwbULlfbCgAACK2wpI2YmBgbG9imTRupWbOm9O3bV+rWrStDhgyRadOn2WXat28v5cqVs8PVtGjRQv7xj5ekTJkyti+/0pbC6dNFOnbQx9CrhLzacawVAAAQWlGOy1uPODoBRcca5lWLF/ukeXOf/PudKLmlV7RXCwAAcGaZ5aRA9NNlcPH5fOFpMcS50dnIBdw8eH17nqZgd8242woAAAgtEkcu9vHHjjRtIlK6NBNPgq3esdoKAAAILYJhLpWa6sjy5SKdOxMKAQBAeBAMc6lPPvEf+7FTJ54iAAAQHueUOhYsWCBvvfWWre/YscMOQ4PQ0sPUVK0qcskltBgCAIDwyHIwHDZsmIwYMUKGDh1q20eOHJFbbrnF1hEaBw7oga1FOnfRM8R4lTiqYWxDKwAAILSyHAzff/99mTZtmpQoUcK2K1asKHv27LF1hMacZJ8buN1g2JFu5FN54fq/WwEAAKGV5eRRuHBhO89x4FzH+/fvtyVCZ/o0nxQpInLNNQRDAAAQPllOHn369JE777xTdu7cKa+//rqdzeSee+7x9uJ86TEnP/5YzwSjIdyrxHFu/eg2KwAAILSyHAwffPBB6dGjhwXEVatWybPPPit/+MMfvL04X99/75Nt2zhMzemk7k61AgAAQitLwTAjI0Patm0r1157rTz33HPy/PPP2zpCR892ojp0oBsZAACEV5bSR3R0tBQvXlx2797t1SDU9GwnDa8QqVCBFkMAABBeWW6WKlKkiFxyySUyYMAAuffee48WnL8dOxxZ+rVIIt3IAAAgB2Q5GHZO7CzPPf+ctGrdShpd1ehowfn7dKZPxNGznRAMT6dZXFMrAAAgtKIcl7d+1g4fPiwpKSm2Xrt2bSlYsKCth1vDhg1l2bJl3lbk69kzXebPF9m8OYYDWwMAgPOSWU4KRD9dBhefz5f1FsPk5GSJj4+Xu+66SwbfOViqVasm8+bN8/biXLlZW2Z+prOROdsJAADIGVkOhjqeUMOhhsEF8xfInDlzOFxNCCxY4JMD+7QbOctPSb7T44OeVgAAQGhlOYXouZG1+zigVq1aVofzM32GT6ILirRpQzA8k7T9aVYAAEBoZTmFNGnSxGYka6uhlkGDBknjxo29vThXH0/VUCjinYIaAAAg7LIcDMeOHSv169eXMWPGyOjRo6VevXry6quventxLrQbef16kdYtGVwIAAByTpaDYXp6uo0z/PDDD+Wjjz6SP/7xj3ZGFJy7hx/22fLHH7M8QRwAACBkshwMW7duLQcPHvS2xNbbaB8osqxo0XSJiUmXr77yb7/7rti21iNz7aq3swIAAEIry8FQg2CJoIFwur5//35vC1mxfn209AyaXFu4iEjv3iIbNkR7NTiVx1s+ZgUAAIRWloOhBsHggyV+/fXXUqxYMW8LWREbGyWHDvnXo90sqMcyvOACzpMMAAByRpaD4csvvyyJiYnSrFkzad6iudx8881MPjkPq1b7lzNnFpA7B4ts3erfRuY6TupsBQAAhNZZB8OlS5fK9u3bpVGjRrJu3Trp27evFIwpKJ07d5bq1at7l0JW1XAfuosv9h+/8OWXY2TKlBhvDzJz8PABKwAAILTOOhj+7ne/k0KFCtn6okWL5Mknn5R777tXypQpIwMHDrR6ZI1O5p47T6RVK68CAAAgB511MNRD0pQuXdrW33vvPTtkzc1db5Zhw4bJqlWrrB5Zs3y5T/bvE2nZijGFAAAg52UpGOoxDNXMmTOlbdu2tq4C9ciauXP9xy1seU2Wh3oCAACE3FknkltvvVVatGhhE090FrKuq7Vr10qpUqVs/XSSkpKkRo0aEh8fLyNHjvRqj3n77belZMmSdlYVLf/617+8PXlXcrIjleNEqlShxTArOtXuZAUAAIRWlOPy1s9o8eLFsm3bVrn22uukePHiVpeSkiL79u2Thg0b2vapaGtjtWrVZO7cuVK5cmW77AcffCB16tTxLuEPhkuWLLFZz2dLf0/woXMiiT7qsRXT5fr2IhMmMOEEAACEVmY5KRD9dBlcfD7f2bcYqsaNG0uXLjcdDYWqVq1apw2FSgNfQkKCzV7WCSz9+/eXqVOnenvzpzVrHEnbIdKS8yMDAIBcIiyD27Zs2WJdyAFxVeIkNTXV2zrm3Xfflbp160q3bt1OuV+NHz/egqiWbdu2ebWRZ+5c//mRW7ZkfGFWtX27nRUAABBaYUkljuMPQcGioo5vKdPjIW7dulV+/PFHad++vfTr18/bc7xBgwZZs6iW2NhYrzby6MSTsmW1xZUWQwAAkDuEJRhWrhwnGzdu9LZEUjel2ljDYHo8xMKFC9u6Hhdx4cKFtp5XzUnW1kINyF4FAABADgtLMNSzpaxYsUI2bNgghw8flokTJ9rs5mDB3cLTp0+TBg0aeFt5z08/ObI5VQ9sTSoEAAC5R1iCYUxMjI0NbNOmjdSsWdNOp6djCYcMGSLT3BCoXnrpJaldu7bUq1dPRo0aLf/+97+tPi+aNz8wvpBgCAAAco8sHa4mt4nUw9XceWe6vP++yI4dMRId7VXirL26dJwt72p0py0BAMDJsv1wNQiN5GQRPT44ofDcaCAkFAIAEHoEwzD7+WdH1qxhfOH5OHDkgBUAABBaBMMwm8/4wvPW6d3OVgAAQGgRDMMsea4jhYtovz8PPQAAyF1IJ2E2N1mkaRORggW9CgAAgFyCYBhGu3Y78v0PjC8EAAC5E8EwjL780hFx/zG+EAAA5EYEwzDS8yPrIWquvpqH/XzcfvntVgAAQGiRUMJozn8daXSVSNGiXgXOya2X9bcCAABCi2AYJvv3i+jBx1szvvC8pR3YaQUAAIQWwTBMFi/2SUYG4wtDocfkHlYAAEBoEQzDRMcXRrmPdtOmBEMAAJA7EQzDJDnZkcsaiFxwAcEQAADkTgTDMPjtN5FFi0RatfYqAAAAciGCYRh8/bVPjhxxg2FLHm4AAJB7kVTCQMcXqubN6UYOhbsb3WUFAACEFsEwDOa4wbD2JSJlyxIMQ6FHvR5WAABAaBEMs5keombBfD1+oVeB87Z592YrAAAgtAiG2ey773xy8IBISw5sHTL9P7rVCgAACC2CYTYLjC+8pgUPNQAAyN1IK9lMj19YtapI5cq0GAIAgNyNYJiNHMcNhnNFWnP8QgAAEAEIhtlo5UpHdu/i/MgAACAyEAyz0dy5Plu25MDWIfVg0wesAACA0CKxZCOdeHLRRSLVq9NiGEoda3e0AgAAQotgmE10fOGcOf7xhVHkwpBKSUuxAgAAQotgmE02bHBk+3bGF2aHO6ffZQUAAIQWwTCbML4QAABEGlJLNtHxhRdcKJKQQIshAACIDGELhklJSVKjRg2Jj4+XkSNHerUnmzJlikRFRcnXX3/t1USm5GSRlte4DzDRGwAARIiwxJaMjAwZOHCgzJo1S1JSUmTChAmyYsUKb+8xe/fulRdGvSCNGjXyaiLT1q2ObNgg0orzIwMAgAgSlmC4ZMkSSUhIkOrVq0uhQoWkf//+MnXqVG/vMU8++aQ89thjUrRoUa8mMs2bHxhfSDDMDk+0fNwKAAAIrbAEwy1btlgXckBclThJTU31tvy+/fZb2bhxo3TscPrj040fP14aNmxoZdu2bV5t7jI32ZEibra97DL6kbND2+ptrQAAgNAKS3JxHH8LWjAdRxjg8/nk3nvvldGjR3s1mRs0aJAsW7bMSmxsrFebu+jxC5u3EImJ8SoQUt9t/84KAAAIrbAEw8qV46w1MCB1U6pbV9nb8o8t/Oabb6Rp06ZSqVIlmT9/vrRv3z4iJ6Ds3OnIypUirelGzjZ/mnm/FQAAEFphCYY6mUQnm2zYsEEOHz4sEydOlMTERG+vyIUXXij79u2zLmctLVq0kM8++0yuvPJK7xKRY8ECx5aMLwQAAJEmLMEwJibGxga2adNGatasKX379pW6devKkCFDZNr0ad6l8obkuT6JLqhhmPGFAAAgskQ5Lm894ugEFB1rmJs0uipdShTXcYYMMMwubd9uZ8vZt82yJQAAOFlmOSkQ/XQZXHTOB81aIbR3ryP/+x/HLwQAAJGJFsMQ+uwzn3To4JOkpALSrh2ZO7ssSl1kyyZxTWwJAHmNttxs3rxZ9u/f79UAp1a8eHGb0FvgFKdaO5cWQ4JhCD3xRIY897wjO9NipEQJrxIAgCz65Zdf5LfffrMjdZzqAx9QGuR00m7hwoWlfPnyXu0xdCXnsLlzHbn8ciEUZjNtMQy0GgJAXrRr1y656KKLCIU4LX196Otk9+7dXs354xUXIocOiSxeItK6NeMLs9tjsx63AgB5VUZGhhQsWNDbAjKnr5P09HRv6/wRDENkyRKfZBzh+IUAgNAIPkMYkJlQv04IhiGi3cjiPjctmvNGBgAAkYlgGCLJyY7UrSNSqhTBEACAzNx1113y9NNPe1unl5XLIjQIhiFw5IjIl1+KtGrtVQAAkEfpTOlZs879BAOvvvqqPPHEE97W6WXlsggNgmEIfPutzyaftObA1mEx5obRVgAAx/O9O0kyStSWjKiyttTtcArlJAjkDIJhCNj4QleLFjyc4dCgQgMrAIBjNAQ6fR8Q2Z/mr3CXuh3KcNi/f3/ZunWrtG/fXmJiYuT555+3yQ9vvPGGxMbGSqvWrexy3bt3l9KlS9vx9Zq3aC4//vij1avbb79d/vrXv9p6cnKyHX9v1KhRcsEFF0iZMmXkrbfesn0qK5fduXOndOrUSQoVKiRXXHGF/VzTpk29vThbJJkQ0PGFNWqIVKhAi2E4zF4/2woA5Ae+ex+UjCbtzlicvn9yL33Y/0NHHbb6U10+uOh1nI2JEydKxYoV5bPPPrPWwZ49e1r9nDlzZO3atfLF51/YdseOHeWnn36SPXv2yNVXXX30cqeyY8cOOw5fWlqavPPOOzJgwAA7juOpnO6yd999t50FRLd132uvvWb1yBqC4Xny+UTmzdPjF3oVyHZPz33GCgAgWGbduNnfvTt06FALZUWLFrXtO+64Q0qWLGkthkOeGiLLly/P9CDMepBmbd3T4/HdeOON9ntWrVrl7T1eZpfV4z6+//77Mnz4cClWrJjUqVNHBg8e7P0UsoJgeJ6WL/fJ3r0iLRlfCADIBgVeekGiF806Y5HiZb2fOIFbf6rLBxe9jvMRFxfnrfkPzv3oo49KlSpVrFu3wkUVrF5b+U5Fu4S1WzqgRIkSsm/fPm/reJldVlsSVfDt0OtH1hEMz1NgfOE1jC8EAOSgqNeGuv8X8m8cVcirD51THVA5uG7Se5Nk8uTJMm/ePDvf8/aft1t94Py82aFcuXK23Lx5sy3Vpk2bvDVkBWnmPGkwrFhJpGpVWgwBADmnQJ/eEvXOqGMth+5St7U+lPRwNevWrfO2TrZ3z14pUqSIte4dOHBAHnv0MW9P9omOjpYePXrIkCFD7Dq1e3n8+PHeXmQFwfA86Jef5Ll6mBr9tuRVAgCQQzQERu9bLdFOmi1DHQrVk0/+1cb56fjBDz74wKs9Rmcu16hRQ8qWLSsXX3yxNG0WnpnB//znP23iSalSpaR37942o1kDKrImysnOtt1s1rBhQ1m2bJm3FX5r1jiSkJAhY8dGyaBB0V4tsltKWoota5WtZUsAyGtWrlzpfr4keFs4F3/5y1/s0Dpvv/22V5N3ZfZ6ySwnBaKfLoOLz+ejxfB8zJ3rs2XLljyM4aSBkFAIAAim3cfff/+9BZwlS5bIyy+/LN263eztxdki0ZwHHV9YurRI7dr0I4fTjNUzrAAAELB37147wLXOWu7SpYudSq9z50RvL84WwfA8zEkWacX4wrB7YeEoKwAABDRq1Eg2btxoh8vZsmWLdSWfagY1To9geI42bXIkdZMGQ150AAAgbyAYnqN58wPjCwmGAAAgbyAYnqO5yY4ULyFSvz4PIQAAyBtINedIxxde00IPqulVAAAARDiC4Tn45RdH1qQwvjCnTLxpghUAABBaBMNzMG++d37kawiGOaHyhZWtAABwvoYOHSr9+vWzdT2/sh7uRmc2n0rwZc9F7dq1JTk52dvKnQiG52Dmpz4RNxNWrkwwzAmTl0+2AgDIOZPem2Rn1tAgpedFvv7662XBggXe3shUpUoVSU9Pt3Mvny89JZ+eOjDY6tWrpZUe5y4XIxieg/9Mcf9zRP72t1N/o0D2Grv0VSsAgJNt2+a44SNdtm/PvjPejh49Wu4cfKc8OeRJ2b17t3td2+WP9/5Rpk6d6l3iGA1aiBxhC4ZJSUl2Uu34+HgZOXKkV3vMuHHjpE6dOlK/fn1p2rSprFixwtuTexQtmu5+M0qX3bv82+5Ntm2tBwAgNxg+PEO+XCgybFj2NF5oEHz44YflzTffkK43dZXixYtLwYIFpVPHTvLcc89Zd2u3bt2sy7VQoUJ2ruLffvtN7r//filbtqwVXdc6lZaWJh07dpTChQu7n6dFpVmzZnbOXqW/r1y5cvb7NUPMnj3b6k/Uvn17eeWVV7wtv3r16smHH31o6/fdd59UqFDBbs9ll10m8+fPt/oT6QGy9aDYgTC7YcMGueaaa+z627VrJ7/88ovVB3Tv3l1Kly5tt715i+by448/Wv348ePtfj/99NPWoqpnZFGVKlWSWbNm2frpHhPtbi5fvryMGjVKLrjgAmuRfeutt2xfdgtLMNS++oEDB9qDkZKSIhMmTDgp+PXu3dvqfvjhB3ns8cfsScxt1q+Pli5djs1ELlxEb7e+cJiaDADIHvc/kC6tW5+5FCzkb7zQRgvHzVWBxgutP9Xlg4tex9latGiRfa536XKTV3OyKVOmSM+ePeTQoUPSp08fefbZZy2MrVy50s5p/OWXX8ozzzxjl33hhResC3ffvn2yZ88eC4MazrTbVffp+Y+PHDki//3vf6VatWr2Myfqf2t/mThxorcllifWrFkjHW7sYNtXN77a6g4cOCC33XabdO7c2W7bmfTs2VOuvvpqu21PPfWUNWIF00D7008/2e2++qqr7fJq0KBBdj16Wj4NmdOnT7f6YKd7TNSOHTsshGtwfuedd2TAgAGya5fXMpWNwhIM9WTWCQkJUr16dUvr/fv3P6m5WRNxwP79+3PlaWxiY6PcbxwiPkfcbw8ihw/r7Ra3jrGGAICcdeUVImXLikR5n+y61O1GV/q3QyVtZ5qUKlXKWsIy06x5M0lM7CIFChSwVsA33nhDhg0bZq1g2gI4fPhwef311+2y2hqnp7DTgKXrLVq0sAyg4/wOHjxorXAaDLXHUVsNT+UmN6QuXbrUfofSINW3b19ryVN9+/S1Vje9zQ888IAFRA2ep6MTUTS/6O3W36Mthz169PD2+t1xxx1SsmRJ2z/kqSGyfPlyC3Nn43SPidLHTsco6mNy4403WsusBsjsFpZgqE+4PqEBcVXiJDU11ds65p///Kd9a7jv3vtOahIO0OZZHeyqZdu2bV5t+OhV3jlYZPHiArbcutXbAQBANhg9KkbmzDlzWbQoRrp18/+MNl6o7t1FFi489eWDi17H2Spbpqy1XJ1u7GC1+ONb9k7MAbq+1fsAfeihh2y2rk7KiIuLOzrcrGbNmjLutXHW6laiRAnp1avX0Z/RgBcoGuA0nHXt2lXee+8926/duMGzh7VL9uKLL7YAp+Xw4cPWInc6epv1ejWQBQS3WGqr6aOPPmq5RRu9KlxUweq1he9snO4xUYEgG6C3RVsus1tYgqGjbdonOFWL4D333GNP8JgXx1hyPhVtnl22bJmV2NhYrzZ8pkyJkZdfjpEGDQrYUrcRXpN7TLYCADheOBovmjRpYq15H3/8sVdzshM/43VsnY7fC9CWvYoVK9q6hrq///3v1mD02WefWXdqYCxh71t6u8F2oY3t09/5yCOPWL2G0kDRYKb69etrQ9W0q1tbBFu3bm312l2r3cAfffSRtUDqOD4Nco5z+sk5evs0iGkvZkDwfdBZ2ZMnT5Z58+bZ79z+83arD/zeM/V8nu4xyUlhCYaVK8cdd+dTN6W6dZkfh65Xz15HUz9worLFylgBABwvHI0XF154oTz//PPWjfrxx1MthGlX78yZM+XPf/6zd6nj6aFbNJxpK522qD355JM2Zk7N+GSGrF271gKVDivTVjINntrVq+MKNXQVKVJEihUrdtrDyNxww402j+Hxxx+38X3aFav27t1r3bHaZatBUhuetMXwTKpWrSqNGjWy262X10PxaBAM2Ltnr90ubdnTx+CxRx/z9vhp45Xer8yc7jHJSWEJhvrA6qBPnd2jD64OEE1MTPT2+ukg0YBPPv1E6tat620Bx5vwv4lWAAA5Q2fQjn11rAwZ8pSFuYsuukheeukluemmU09I0bDWuHFj6zKuVauWXHXVVVanUlanHJ35q8PEdAygditrINRuZu1C1TCqh8QZMWKE/cypaBexdh9ra6NOeAnQGct6u7RlUcOahrmzbZl7//33LRBqKNXgNnjwYG+P2HwJHfOoM4q1m7pps6beHj8Ned99953dri46c/UEp3tMclKUm9Cz70BHQT799FPrKta0fvfdd8tjjz3mvqCGSKOrGknnTp1tFrJeRpt39UEeO3bsGcOhvoC0Sxn5S9u329ly9m3+Kf8AkNfoTFWdtAmcjcxeL5nlpED002Vw0cMEhS0YZgeCYf5EMASQ1xEMkRWhDIZh6UoGAABA7kcwBAAAgCEYAgCQCwW6+4DTCfXrhGCIiDO9zzQrAJBX6WFZ9BAwwJno6yT4QNjni2CIiFOsYDErAJBX6Snnfv75Z5sMAGRGXx/6OtHD+YQKs5IRcV5d6j+J+V2N7rQlAOQ1+oG/efPm4866AZyKnrJPTxoSOKB3MA5Xg3yBw9UAAHBmHK4GAAAA54xgCAAAAEMwBAAAgCEYAgAAwET05JOSJUvKxRdf7G1ln+3bt0uFChW8LeQGPCe5E89L7sTzkjvxvOROeel5WbNmjezdu9fbOiYQ/QKTTgIl4mclhwuzn3MfnpPciecld+J5yZ14XnKn/PC8BKJfIBAGCrOSAQAAcBTBEAAAACb6KZe3jkzoOSuvvPJKbwu5Ac9J7sTzkjvxvOROPC+5U35+XrQ7mTGGAAAA+Ugg+ukyuDDGEAAAAEcRDAEAAGAIhqeRlJQkNWrUkPj4eBk5cqRXi3BLTU2VVq1a2TEra9euLS+99JLV//rrr9KuXTupVq2aLXft2mX1CK+MjAy59NJLpWPHjra9YcMGadSokT0vPXv2lMOHD1s9wmfX7l3SrVs3qVmzpr1vFi1axPslFxgzZoz9DUtISJBbbrlFDh06xPslB/zud7+TCy64wJ6HgMzeH9q9eu+991oOqFu3br44vBDBMBP6YTdw4ECZNWuWpKSkyIQJE2TFihXeXoRTTEyM/UHVA3V+/fXXMnr0aHsu/va3v0n79u3tD6sudRvhp0Fdg2HAww8/LI888rA9L2XKlJE33njD24Nwue/e+6RDhw6ydu1a+fHHHyWhTgLvlxy2ZcsWef755+V///ufrFy50j5j/u///o/3Sw644447JDk52dvyy+z9MXPmTFm1apXVv/nmmzJo0CCrz9PcNIxTWLhwoXPttdd6W44zYsQIK8h5nTt3dj7//HOnevXqztatW61Ol7qN8EpNTXVat27tzJ4923GDiOPz+ZzixYs7R44csf0nvo+Q/Xbv3u1UqlTJnotgvF9y1ubNm52LLrrI2blzp70/9P2SlJTE+yWHuEHPueSSS7ytzN8fgwcPdia9N8nWVfDlIpn+fdDifkFx0tPT7TV4+PBh59ChQw4thpnQb3fadBwQVyXOujSRszZu3ChfffWVXH311eL+oZXY2Fir16X7ZrV1hI92sbzwwgtSoID/T4n7oSdly5a1Vl4VFxcnmzZtsnWEx/r16+39oK0i2pKrPR/79+/n/ZLD3LAujz/+uJQvX15Kly4tpUqVskOi8H7JHTJ7f+jnfpW4KrauNBfoZfMygmEm3DztrR0TFRXlrSEn7Nu3TxITE2Xsq2NtfAhy1oxPZtj5RK+44gqv5tghEILxvgkv99u/fPPNN3LPPffI999/L8WLF2eMdC6gY9amTJki27Ztsy9Q+vfs05mfenuP4f2Su+THv2kEw0xUrhxnrVMBqZtS3brK3hbC7ciRI9K1a1e5/fbbpetNXa1Onw/9I6t0WbFiRVtHeCyYv0AmT55sLSE33XSTjcW57777JC0tzcKJ0m/b2gqC8NHHW98L2qquevToIUuXLuX9ksNmzfrCJgOVK1dOChYsKN17dLf3EO+X3CGz90eVKlVkU+qxVlzNBfo3Ly8jGGZCZ4npBAcdcKqzxCZOnGitVQg//cY2YMAAqV+/vtx///1erUj37t3l7bfftnVd6gcgwufZZ5+VX375xYZdfPTRR3LDDTfIO++8Y8spU/5jl3nrrbfk5ptvtnWEh7bi6szK1atX27ZOoNP3Du+XnFWlSlWZN2+eHDhwwP6mffH5F1KvXj3eL7lEZu+PLjd1kTffeNOes8WLF9swgECXc57l3llk4pNPPnHcP7CO+w3OeeaZZ7xahNv8+fO1Ld+pU6eO4/4htaLPjftN2yY+xMfH21IHdSNnzJkzxwbTq3Xr1jlXXHGFU7VqVadbt242mBnh9e233zqXX365vWfcL7TOr7/+yvslFxgyZIhTo0YNm/TQr18/e2/wfgm/Xr16OW7As8+VcuXKOf/6178yfX/oBI177rnHckBCQoKzdOlSq490p5t8winxAAAA8pFA9NNlcHHDIl3JAAAA8CMYAgAAwBAMAQAAYAiGAAAAMARDAAAAGIIhgHylcePGttQD1U56b5Kth8qIESO8Nb/AdQFApOBwNQDypeTkZHnuuedkxowZXs2ZZWRkSHR0tLd1Mj3nbeAsFgCQWwWiX+AwNYHC4WoA5Dsa3tRDDz0kX3zxhZ0VZMyYMRb6HnnkETv3c926deW1116zy2mAbNWqlfTp00cSEhKsrkuXLnLZZZdJ7dq1Zfz48Vb36KOP2u/Q39e3b1+rC1yX/sHV360/X6dOHXl/8vtWr7+7ZcuW0q1bNztdmv6cXlbp76tVq5bdlocfftjqACC70WIIIF8JtOqd2GKoAe/nn3+Wxx9/XH777Tdp0qSJfPjhh9bl3K5dO1mzZo2dak79+uuvdmqsgwcPSsOGDeXLL7+UMmXKnNRiGNie8uEUeeXlV+Szzz6zc+M2aNBAvv32W1m1apW0b99e1q1bZ+dm1escPXq0hcErr7xS1q5dayfs37V7l5S6sJT3WwHg/ASiny6DCy2GAOCZOXOmvP7669bip6FMQ2JKSorta968+dFQqF566SU7z62eU339+vVHL5eZ+fPmS79+/awb+qKLLpLrrrtOlixdYvtatGhhJ/AvUKCAXe+GjRukZMmSUrRoURk4cKB8+NGHUqxoMbssAGQ3giEAuPTb8rhx4+SHH36wsnnzZgtwqkSJErZU2tKoIXLJkiWyfPlym2By6NAhb++p6e/OTJEiRbw1seCYfiTdWhq/+eYbO7H/h1M+tFZFAAgHgiGAfElb5fbs2eNtidx4443yyiuvyJEjR2xbWwH3799v68H27NktZcuWlWLFillX8Pz58709IgULFjz688Fatmop7777ro1B3LFjh41tvPqqq729J9u3b5/dNr1NL774oixevNjbAwDZi2AIIF+69NJLrWVOu4R18smAAQOsG1nH/+kkEe3GPdUM4/btr7fwp+MAdTyidgUH3HfffTa5JDD5JOCmLjfZWES9rmuuuUZefOlFqVChgrf3ZBoMb7jhBruOZs2aybjXxnl7ACB7MfkEAAAgHwlEP10GFyafAAAA4CiCIQAAAAzBEAAAAIZgCAAAAEMwBAAAgCEYAgAAwCXy/x83k5zhxiYuAAAAAElFTkSuQmCC"
    }
   },
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![image.png](attachment:image.png)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#改变数据量大小\n",
    "# Chaning the sampling size # set the iter to the best iteration: iter = 20 \n",
    "perc = [0.01,0.02,0.05,0.1,0.2,0.5,1] \n",
    "kf = KFold(n_splits=3, random_state=RANDOM_STATE) \n",
    "train_score = [] \n",
    "cv_score = [] # select a k: \n",
    "k_ndcg = 5 \n",
    "for i, item in enumerate(perc): \n",
    "    lr = LogisticRegression(C=1.0, max_iter=20, tol=1e-6, solver='newton-cg', multi_class='ovr') \n",
    "    train_score_iter = [] \n",
    "    cv_score_iter = [] \n",
    "    n = int(xtrain_new.shape[0]*item) \n",
    "    xtrain_perc = xtrain_new[:n, :] \n",
    "    ytrain_perc = ytrain_new[:n] \n",
    "    for train_index, test_index in kf.split(xtrain_perc, ytrain_perc): \n",
    "        X_train, X_test = xtrain_perc[train_index, :], xtrain_perc[test_index, :] \n",
    "        y_train, y_test = ytrain_perc[train_index], ytrain_perc[test_index] \n",
    "        print(X_train.shape, X_test.shape) \n",
    "        lr.fit(X_train, y_train) \n",
    "        y_pred = lr.predict_proba(X_test) \n",
    "        train_ndcg_score = ndcg_score(y_train, lr.predict_proba(X_train), k = k_ndcg) \n",
    "        cv_ndcg_score = ndcg_score(y_test, y_pred, k=k_ndcg) \n",
    "        train_score_iter.append(train_ndcg_score) \n",
    "        cv_score_iter.append(cv_ndcg_score) \n",
    "    train_score.append(np.mean(train_score_iter)) \n",
    "    cv_score.append(np.mean(cv_score_iter))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "ymin = np.min(cv_score)-0.1 \n",
    "ymax = np.max(train_score)+0.1 \n",
    "plt.figure(figsize=(9,4)) \n",
    "plt.plot(np.array(perc)*100, train_score, 'ro-', label = 'training') \n",
    "plt.plot(np.array(perc)*100, cv_score, 'bo-', label = 'Cross-validation') \n",
    "plt.xlabel(\"Sample size (unit %)\") \n",
    "plt.ylabel(\"Score\") \n",
    "plt.xlim(-5, np.max(perc)*100+10) \n",
    "plt.ylim(ymin, ymax) \n",
    "plt.legend(loc = 'lower right', fontsize = 12) \n",
    "plt.title(\"Score vs sample size learning curve\") \n",
    "plt.tight_layout()\n"
   ]
  },
  {
   "attachments": {
    "image.png": {
     "image/png": "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t1999VXd36PHDXj22Wd19qi0VEkLlHyYSjCRc93NmDFDQ6G0Jk6ePElbFv2R7lzp3m3UqJF+v8zelDDqjwROuX/S8tOuXXqLV8L6BJ1BKvdHbku6gqXLNzPpWpSZqdIVK7OCy5UrhwcffFD3yYf5o488qrcxbNgw3Hrr2Z2zT2aPrl+/Xh+Lhx56GN9++62Gpcy6dbsOTz39lLZMymMgj8e33840ezPatXuXdv1Kl6O0KF599dUaRjL74osvMHHixLTZxLIsWrRIW2nlOZIWK7lf8vvLc+DbmuuPhEFpOW3dunXa47x06Y+6T1rHZDavtPbKrOjmLU5u1cwp8jPlNTd48P3auiezey+99FIUKJj1qZAeeOABfczk9SGPd9++fdNmsMs/Pc8//7z+oyC34/4DdCr+XpOXXHIJxo4dqzPkJShLF/pHH35k9hKRP5bz39OZ98kQEQUJaSl69913NVRT7pJgKyH3q6++0uBKRMHBjXiy9l3kPeu7sGWQiIhOm3R77z+wX8flyTkE5QNGWmaJKP9hGCQiotMmLbBVKlfRru+vv/4acXFxabOyiSh/YTcxERER0TnIjXiy9l3YTUxEREREafJFy6B0Q7gncCUiIiKidHLapqxOoeRGPLdF0F0ytwzmizAop1NwL8pPREREROn85SQ34rkh0F0yh0F2ExMRERGFMIZBIiIiohDGMEhEREQUwhgGiYiIiEIYwyARERFRCGMYJCIiIgphDINEREREIYxhkIiIiCiEMQwSERERhTCGQSIiIqIQxjBIREREFMIYBomIiIhCGMMgERERUQhjGCQiIiIKYQyDRERERCGMYZCIiIgohDEMEhEREYUwhkEiIiKiEMYwSERERBTCGAaJiIiIQljAw2B8fDxq1KiBmJgYjBgxwtSm27ZtG1q1aoWGDRuifv36mDVrltlDRERERLktoGEwNTUVAwYMwNy5c5GQkICxY8dizZo1Zq/Xc889h549e+K3337DV199hf79+5s9ROk8n09AatE6SLXK6FrKREREFHgBDYPLli1D3bp1Ub16dURFRSE2NhZTp041e70sy8LBgwd1+8CBA6hatapuE7kk+Nm9hwBH9nornLWUGQiJiIgCz7IdZvusTZkyBTNnzsQHH3yg5c/Gf4alS5birbfe0rLYtWsX2rRpgz179uDw4cNYvHgxGjdubPam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pCz1WR3vu8jPyK6lS5fqmLpu3a4zNSeTz/GbbroRx48fR69evfDiiy9qAFu7dq1eQ/h///sfXnjhBT321Vdf1e7Zw4cP4+DBgxoAJQtIl6rsk+sNJycn4/vvv0e1atX0ezKLvTUW48aNMyVgzZo12LBhAzp26Kjly5pepnVHjx5F37590aVLF71v/+amm27CZZddpvdt2LBheO+998weLwmxW7du1ft92aWX6fFi4MCB+nPkkngSLGfMmKH1vk71mIg9e/Zo8JawPH78ePTv3x/79+83e3NHQMOgvGgGDBigaTghIQFjx47VJ8XXeeedp79sbGysqSEiIqLsSk02G5n4qz9Te/ftRcmSJbXFy58Wl7dA167dEBYWpq19H330EZ599llt7ZKWvueeew4ffPCBHiutbnL5OAlVst2yZUsNgzJu79ixY9raJmFQGpOkUSkr1znBdPny5XobQvJE7969tcVO9O7VW1vX5D4PGTJEQ6GEzVORySTLli3T+y23Iy2EN954o9nr1a9fPxQrVkz3Dx02FH/88YcGuOw41WMi5LGTMYfymHTo0EFbYCU05qaAhkF5MOvWrYvq1atrE60EvszjCuRJbtiwof7yRERE5DV6VATmz//3pUq0+YZMpD6r430X+RnZVaZ0GW2hOtVYwGoxGVvwJOzJ57xLtnfu3KnbDz30kM6ylYkV0dHRab2HNWvWxHvvv6eta0WLFsXNN9+c9j0S6txFQpsEsu7du2PixIm6X7pofWf9SndrrVq1NLTJkpSUpC1vpyL3WX6uhDCXb8ukNHQ9/vjj2qop2aZC+QpaLy152XGqx0S44dUl90VaKHNTQBNZ5l84umo0EhNlUMPpk354mb0ky65du0wtERFRaHvpJQsFCpqCIWWpD6RmzZppq920adNMzcmkZc+XjJWT8XguacGrVKmSbkuQe+WVVzQXzJ49W7tK3bGBPW/piSVLluhYPbnNRx55ROsliLqLhDHRp09v7XmUbmxp+WvdurXWS1esdPF+88032tIo4/IkvNm2rfv9kfsn4evIkfTzC/v+DjKbetKkSTrMTW5z91+7td693cyPQWanekyCRUDDoG17B4L6+rcHyR/ph1+5cqUuFStWNLVEREShrVfPcIwZY3lbCJ2PWFlLWeoDqUSJEhg5cqR2kU6bNlWDl3TjxsXF4dFHHzVHZSSn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"
    }
   },
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![image.png](attachment:image.png)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#树模型\n",
    "from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier \n",
    "from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier \n",
    "from sklearn.tree import DecisionTreeClassifier \n",
    "from sklearn.ensemble import * \n",
    "from sklearn.svm import SVC, LinearSVC, NuSVC \n",
    "LEARNING_RATE = 0.1 \n",
    "N_ESTIMATORS = 50 \n",
    "RANDOM_STATE = 2017 \n",
    "MAX_DEPTH = 9 \n",
    "#建了一个tree字典 \n",
    "clf_tree ={ \n",
    "    'DTree': DecisionTreeClassifier(max_depth=MAX_DEPTH, \n",
    "                                    random_state=RANDOM_STATE), \n",
    "    'RF': RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, \n",
    "                                 random_state=RANDOM_STATE), \n",
    "    'AdaBoost': AdaBoostClassifier(n_estimators=N_ESTIMATORS, \n",
    "                                   learning_rate=LEARNING_RATE, \n",
    "                                   random_state=RANDOM_STATE), \n",
    "    'Bagging': BaggingClassifier(n_estimators=N_ESTIMATORS, \n",
    "                                 random_state=RANDOM_STATE), \n",
    "    'ExtraTree': ExtraTreesClassifier(max_depth=MAX_DEPTH, \n",
    "                                      n_estimators=N_ESTIMATORS, \n",
    "                                      random_state=RANDOM_STATE), \n",
    "    'GraBoost': GradientBoostingClassifier(learning_rate=LEARNING_RATE,\n",
    "                                           max_depth=MAX_DEPTH, \n",
    "                                           n_estimators=N_ESTIMATORS, \n",
    "                                           random_state=RANDOM_STATE) } \n",
    "train_score = [] \n",
    "cv_score = [] \n",
    "kf = KFold(n_splits=3, random_state=RANDOM_STATE) \n",
    "k_ndcg = 5 \n",
    "for key in clf_tree.keys(): \n",
    "    clf = clf_tree.get(key) \n",
    "    train_score_iter = [] \n",
    "    cv_score_iter = [] \n",
    "    for train_index, test_index in kf.split(xtrain_new, ytrain_new): \n",
    "        X_train, X_test = xtrain_new[train_index, :], xtrain_new[test_index, :] \n",
    "        y_train, y_test = ytrain_new[train_index], ytrain_new[test_index] \n",
    "        clf.fit(X_train, y_train) \n",
    "        y_pred = clf.predict_proba(X_test) \n",
    "        train_ndcg_score = ndcg_score(y_train, clf.predict_proba(X_train), k = k_ndcg) \n",
    "        cv_ndcg_score = ndcg_score(y_test, y_pred, k=k_ndcg) \n",
    "        train_score_iter.append(train_ndcg_score) \n",
    "        cv_score_iter.append(cv_ndcg_score) \n",
    "    train_score.append(np.mean(train_score_iter)) \n",
    "    cv_score.append(np.mean(cv_score_iter))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "train_score_tree = train_score \n",
    "cv_score_tree = cv_score \n",
    "ymin = np.min(cv_score)-0.05 \n",
    "ymax = np.max(train_score)+0.05 \n",
    "x_ticks = clf_tree.keys() \n",
    "plt.figure(figsize=(8,5)) \n",
    "plt.plot(range(len(x_ticks)), train_score_tree, 'ro-', label = 'training') \n",
    "plt.plot(range(len(x_ticks)),cv_score_tree, 'bo-', label = 'Cross-validation') \n",
    "plt.xticks(range(len(x_ticks)),x_ticks,rotation = 45, fontsize = 10) \n",
    "plt.xlabel(\"Tree method\", fontsize = 12) \n",
    "plt.ylabel(\"Score\", fontsize = 12) \n",
    "plt.xlim(-0.5, 5.5) \n",
    "plt.ylim(ymin, ymax) \n",
    "plt.legend(loc = 'best', fontsize = 12) \n",
    "plt.title(\"Different tree methods\") \n",
    "plt.tight_layout()\n"
   ]
  },
  {
   "attachments": {
    "image.png": {
     "image/png": "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"
    }
   },
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![image.png](attachment:image.png)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#SVM\n",
    "TOL = 1e-4 \n",
    "MAX_ITER = 1000 \n",
    "clf_svm = { \n",
    "    'SVM-rbf': SVC(kernel='rbf', \n",
    "                   max_iter=MAX_ITER, \n",
    "                   tol=TOL, \n",
    "                   random_state=RANDOM_STATE, decision_function_shape='ovr'), \n",
    "    'SVM-poly': SVC(kernel='poly', max_iter=MAX_ITER, \n",
    "                    tol=TOL, random_state=RANDOM_STATE, \n",
    "                    decision_function_shape='ovr'), \n",
    "    'SVM-linear': SVC(kernel='linear', max_iter=MAX_ITER, \n",
    "                      tol=TOL, random_state=RANDOM_STATE, \n",
    "                      decision_function_shape='ovr'), \n",
    "    'LinearSVC': LinearSVC(max_iter=MAX_ITER, tol=TOL, \n",
    "                           random_state=RANDOM_STATE, multi_class = 'ovr') \n",
    "train_score_svm = [] \n",
    "cv_score_svm = [] \n",
    "kf = KFold(n_splits=3, random_state=RANDOM_STATE) \n",
    "k_ndcg = 5 \n",
    "for key in clf_svm.keys(): \n",
    "    clf = clf_svm.get(key) \n",
    "    train_score_iter = [] \n",
    "    cv_score_iter = [] \n",
    "    for train_index, test_index in kf.split(xtrain_new, ytrain_new): \n",
    "        X_train, X_test = xtrain_new[train_index, :], xtrain_new[test_index, :] \n",
    "        y_train, y_test = ytrain_new[train_index], ytrain_new[test_index] \n",
    "        clf.fit(X_train, y_train) \n",
    "        y_pred = clf.decision_function(X_test) \n",
    "        train_ndcg_score = ndcg_score(y_train, clf.decision_function(X_train), k = k_ndcg) \n",
    "        cv_ndcg_score = ndcg_score(y_test, y_pred, k=k_ndcg) \n",
    "        train_score_iter.append(train_ndcg_score) \n",
    "        cv_score_iter.append(cv_ndcg_score) \n",
    "    train_score_svm.append(np.mean(train_score_iter)) \n",
    "    cv_score_svm.append(np.mean(cv_score_iter)) }\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "ymin = np.min(cv_score_svm)-0.05 \n",
    "ymax = np.max(train_score_svm)+0.05 \n",
    "x_ticks = clf_svm.keys() \n",
    "plt.figure(figsize=(8,5)) \n",
    "plt.plot(range(len(x_ticks)), train_score_svm, 'ro-', label = 'training') \n",
    "plt.plot(range(len(x_ticks)),cv_score_svm, 'bo-', label = 'Cross-validation') \n",
    "plt.xticks(range(len(x_ticks)),x_ticks,rotation = 45, fontsize = 10) \n",
    "plt.xlabel(\"Tree method\", fontsize = 12) \n",
    "plt.ylabel(\"Score\", fontsize = 12) \n",
    "plt.xlim(-0.5, 3.5) \n",
    "plt.ylim(ymin, ymax) \n",
    "plt.legend(loc = 'best', fontsize = 12) \n",
    "plt.title(\"Different SVM methods\") \n",
    "plt.tight_layout()\n"
   ]
  },
  {
   "attachments": {
    "image.png": {
     "image/png": "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"
    }
   },
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![image.png](attachment:image.png)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#xgboost\n",
    "import xgboost as xgb \n",
    "def customized_eval(preds, dtrain): \n",
    "    labels = dtrain.get_label() \n",
    "    top = [] \n",
    "    for i in range(preds.shape[0]): \n",
    "        top.append(np.argsort(preds[i])[::-1][:5]) \n",
    "        mat = np.reshape(np.repeat(labels,np.shape(top)[1]) == np.array(top).ravel(),np.array(top).shape).astype(int) \n",
    "        score = np.mean(np.sum(mat/np.log2(np.arange(2, mat.shape[1] + 2)),axis = 1)) \n",
    "        return 'ndcg5', score \n",
    "    # xgboost parameters \n",
    "    NUM_XGB = 200 \n",
    "    params = {} \n",
    "    params['colsample_bytree'] = 0.6 \n",
    "    params['max_depth'] = 6 \n",
    "    params['subsample'] = 0.8 \n",
    "    params['eta'] = 0.3 \n",
    "    params['seed'] = RANDOM_STATE \n",
    "    params['num_class'] = 12 \n",
    "    params['objective'] = 'multi:softprob' # output the probability instead of class. \n",
    "    train_score_iter = [] \n",
    "    cv_score_iter = [] \n",
    "    kf = KFold(n_splits = 3, random_state=RANDOM_STATE) \n",
    "    k_ndcg = 5 \n",
    "    for train_index, test_index in kf.split(xtrain_new, ytrain_new): \n",
    "        X_train, X_test = xtrain_new[train_index, :], xtrain_new[test_index, :] \n",
    "        y_train, y_test = ytrain_new[train_index], ytrain_new[test_index] \n",
    "        train_xgb = xgb.DMatrix(X_train, label= y_train) \n",
    "        test_xgb = xgb.DMatrix(X_test, label = y_test) \n",
    "        watchlist = [ (train_xgb,'train'), (test_xgb, 'test') ] \n",
    "        bst = xgb.train(params, train_xgb, NUM_XGB, watchlist, feval = customized_eval, verbose_eval = 3, early_stopping_rounds = 5) \n",
    "        #bst = xgb.train( params, dtrain, num_round, evallist ) \n",
    "        y_pred = np.array(bst.predict(test_xgb)) \n",
    "        y_pred_train = np.array(bst.predict(train_xgb)) \n",
    "        train_ndcg_score = ndcg_score(y_train, y_pred_train , k = k_ndcg) \n",
    "        cv_ndcg_score = ndcg_score(y_test, y_pred, k=k_ndcg) \n",
    "        train_score_iter.append(train_ndcg_score) \n",
    "        cv_score_iter.append(cv_ndcg_score) \n",
    "train_score_xgb = np.mean(train_score_iter) \n",
    "cv_score_xgb = np.mean(cv_score_iter) \n",
    "print (\"\\nThe training score is: {}\".format(train_score_xgb)) \n",
    "print (\"The cv score is: {}\\n\".format(cv_score_xgb))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#模型比较\n",
    "model_cvscore = np.hstack((cv_score_lr, cv_score_tree, cv_score_svm, cv_score_xgb)) \n",
    "model_name = np.array(['LinearReg','ExtraTree','DTree','RF','GraBoost','Bagging','AdaBoost','LinearSVC','SVM-linear','SVM-rbf','SVM-poly','Xgboost']) \n",
    "fig = plt.figure(figsize=(8,4)) \n",
    "sns.barplot(model_cvscore, model_name, palette=\"Blues_d\") \n",
    "plt.xticks(rotation=0, size = 10) \n",
    "plt.xlabel(\"CV score\", fontsize = 12) \n",
    "plt.ylabel(\"Model\", fontsize = 12) \n",
    "plt.title(\"Cross-validation score for different models\") \n",
    "plt.tight_layout()\n"
   ]
  },
  {
   "attachments": {
    "image.png": {
     "image/png": "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"
    }
   },
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![image.png](attachment:image.png)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
