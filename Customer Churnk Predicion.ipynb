{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "5eeb463a",
   "metadata": {},
   "source": [
    "# CUSTOMER CHURN PREDICTION"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "c87cf2a0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'C:\\\\Users\\\\binod\\\\ML Internship Tasks'"
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pwd"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "88029ab3",
   "metadata": {},
   "source": [
    "### Import All Required Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "5376b691",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, r2_score, precision_score, recall_score, f1_score"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "34f11311",
   "metadata": {},
   "source": [
    "### Load And Explore Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "2a428af1",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('C:/Users/binod/OneDrive/Documents/Machine Learning (ML)/Task 3.csv', header = 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "32c97c00",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>RowNumber</th>\n",
       "      <th>CustomerId</th>\n",
       "      <th>Surname</th>\n",
       "      <th>CreditScore</th>\n",
       "      <th>Geography</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Tenure</th>\n",
       "      <th>Balance</th>\n",
       "      <th>NumOfProducts</th>\n",
       "      <th>HasCrCard</th>\n",
       "      <th>IsActiveMember</th>\n",
       "      <th>EstimatedSalary</th>\n",
       "      <th>Exited</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>15634602</td>\n",
       "      <td>Hargrave</td>\n",
       "      <td>619</td>\n",
       "      <td>France</td>\n",
       "      <td>Female</td>\n",
       "      <td>42</td>\n",
       "      <td>2</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>101348.88</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>15647311</td>\n",
       "      <td>Hill</td>\n",
       "      <td>608</td>\n",
       "      <td>Spain</td>\n",
       "      <td>Female</td>\n",
       "      <td>41</td>\n",
       "      <td>1</td>\n",
       "      <td>83807.86</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>112542.58</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>15619304</td>\n",
       "      <td>Onio</td>\n",
       "      <td>502</td>\n",
       "      <td>France</td>\n",
       "      <td>Female</td>\n",
       "      <td>42</td>\n",
       "      <td>8</td>\n",
       "      <td>159660.80</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113931.57</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>15701354</td>\n",
       "      <td>Boni</td>\n",
       "      <td>699</td>\n",
       "      <td>France</td>\n",
       "      <td>Female</td>\n",
       "      <td>39</td>\n",
       "      <td>1</td>\n",
       "      <td>0.00</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>93826.63</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>15737888</td>\n",
       "      <td>Mitchell</td>\n",
       "      <td>850</td>\n",
       "      <td>Spain</td>\n",
       "      <td>Female</td>\n",
       "      <td>43</td>\n",
       "      <td>2</td>\n",
       "      <td>125510.82</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>79084.10</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   RowNumber  CustomerId   Surname  CreditScore Geography  Gender  Age  \\\n",
       "0          1    15634602  Hargrave          619    France  Female   42   \n",
       "1          2    15647311      Hill          608     Spain  Female   41   \n",
       "2          3    15619304      Onio          502    France  Female   42   \n",
       "3          4    15701354      Boni          699    France  Female   39   \n",
       "4          5    15737888  Mitchell          850     Spain  Female   43   \n",
       "\n",
       "   Tenure    Balance  NumOfProducts  HasCrCard  IsActiveMember  \\\n",
       "0       2       0.00              1          1               1   \n",
       "1       1   83807.86              1          0               1   \n",
       "2       8  159660.80              3          1               0   \n",
       "3       1       0.00              2          0               0   \n",
       "4       2  125510.82              1          1               1   \n",
       "\n",
       "   EstimatedSalary  Exited  \n",
       "0        101348.88       1  \n",
       "1        112542.58       0  \n",
       "2        113931.57       1  \n",
       "3         93826.63       0  \n",
       "4         79084.10       0  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "bf43faa4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>RowNumber</th>\n",
       "      <th>CustomerId</th>\n",
       "      <th>Surname</th>\n",
       "      <th>CreditScore</th>\n",
       "      <th>Geography</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Tenure</th>\n",
       "      <th>Balance</th>\n",
       "      <th>NumOfProducts</th>\n",
       "      <th>HasCrCard</th>\n",
       "      <th>IsActiveMember</th>\n",
       "      <th>EstimatedSalary</th>\n",
       "      <th>Exited</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>9995</th>\n",
       "      <td>9996</td>\n",
       "      <td>15606229</td>\n",
       "      <td>Obijiaku</td>\n",
       "      <td>771</td>\n",
       "      <td>France</td>\n",
       "      <td>Male</td>\n",
       "      <td>39</td>\n",
       "      <td>5</td>\n",
       "      <td>0.00</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>96270.64</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9996</th>\n",
       "      <td>9997</td>\n",
       "      <td>15569892</td>\n",
       "      <td>Johnstone</td>\n",
       "      <td>516</td>\n",
       "      <td>France</td>\n",
       "      <td>Male</td>\n",
       "      <td>35</td>\n",
       "      <td>10</td>\n",
       "      <td>57369.61</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>101699.77</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9997</th>\n",
       "      <td>9998</td>\n",
       "      <td>15584532</td>\n",
       "      <td>Liu</td>\n",
       "      <td>709</td>\n",
       "      <td>France</td>\n",
       "      <td>Female</td>\n",
       "      <td>36</td>\n",
       "      <td>7</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>42085.58</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9998</th>\n",
       "      <td>9999</td>\n",
       "      <td>15682355</td>\n",
       "      <td>Sabbatini</td>\n",
       "      <td>772</td>\n",
       "      <td>Germany</td>\n",
       "      <td>Male</td>\n",
       "      <td>42</td>\n",
       "      <td>3</td>\n",
       "      <td>75075.31</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>92888.52</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9999</th>\n",
       "      <td>10000</td>\n",
       "      <td>15628319</td>\n",
       "      <td>Walker</td>\n",
       "      <td>792</td>\n",
       "      <td>France</td>\n",
       "      <td>Female</td>\n",
       "      <td>28</td>\n",
       "      <td>4</td>\n",
       "      <td>130142.79</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>38190.78</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      RowNumber  CustomerId    Surname  CreditScore Geography  Gender  Age  \\\n",
       "9995       9996    15606229   Obijiaku          771    France    Male   39   \n",
       "9996       9997    15569892  Johnstone          516    France    Male   35   \n",
       "9997       9998    15584532        Liu          709    France  Female   36   \n",
       "9998       9999    15682355  Sabbatini          772   Germany    Male   42   \n",
       "9999      10000    15628319     Walker          792    France  Female   28   \n",
       "\n",
       "      Tenure    Balance  NumOfProducts  HasCrCard  IsActiveMember  \\\n",
       "9995       5       0.00              2          1               0   \n",
       "9996      10   57369.61              1          1               1   \n",
       "9997       7       0.00              1          0               1   \n",
       "9998       3   75075.31              2          1               0   \n",
       "9999       4  130142.79              1          1               0   \n",
       "\n",
       "      EstimatedSalary  Exited  \n",
       "9995         96270.64       0  \n",
       "9996        101699.77       0  \n",
       "9997         42085.58       1  \n",
       "9998         92888.52       1  \n",
       "9999         38190.78       0  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "5ad23069",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(10000, 14)"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "e60b4dce",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "140000"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.size"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "d60249cf",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>RowNumber</th>\n",
       "      <th>CustomerId</th>\n",
       "      <th>CreditScore</th>\n",
       "      <th>Age</th>\n",
       "      <th>Tenure</th>\n",
       "      <th>Balance</th>\n",
       "      <th>NumOfProducts</th>\n",
       "      <th>HasCrCard</th>\n",
       "      <th>IsActiveMember</th>\n",
       "      <th>EstimatedSalary</th>\n",
       "      <th>Exited</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>10000.00000</td>\n",
       "      <td>1.000000e+04</td>\n",
       "      <td>10000.000000</td>\n",
       "      <td>10000.000000</td>\n",
       "      <td>10000.000000</td>\n",
       "      <td>10000.000000</td>\n",
       "      <td>10000.000000</td>\n",
       "      <td>10000.00000</td>\n",
       "      <td>10000.000000</td>\n",
       "      <td>10000.000000</td>\n",
       "      <td>10000.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>5000.50000</td>\n",
       "      <td>1.569094e+07</td>\n",
       "      <td>650.528800</td>\n",
       "      <td>38.921800</td>\n",
       "      <td>5.012800</td>\n",
       "      <td>76485.889288</td>\n",
       "      <td>1.530200</td>\n",
       "      <td>0.70550</td>\n",
       "      <td>0.515100</td>\n",
       "      <td>100090.239881</td>\n",
       "      <td>0.203700</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>2886.89568</td>\n",
       "      <td>7.193619e+04</td>\n",
       "      <td>96.653299</td>\n",
       "      <td>10.487806</td>\n",
       "      <td>2.892174</td>\n",
       "      <td>62397.405202</td>\n",
       "      <td>0.581654</td>\n",
       "      <td>0.45584</td>\n",
       "      <td>0.499797</td>\n",
       "      <td>57510.492818</td>\n",
       "      <td>0.402769</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.00000</td>\n",
       "      <td>1.556570e+07</td>\n",
       "      <td>350.000000</td>\n",
       "      <td>18.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.00000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>11.580000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>2500.75000</td>\n",
       "      <td>1.562853e+07</td>\n",
       "      <td>584.000000</td>\n",
       "      <td>32.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.00000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>51002.110000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>5000.50000</td>\n",
       "      <td>1.569074e+07</td>\n",
       "      <td>652.000000</td>\n",
       "      <td>37.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>97198.540000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.00000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>100193.915000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>7500.25000</td>\n",
       "      <td>1.575323e+07</td>\n",
       "      <td>718.000000</td>\n",
       "      <td>44.000000</td>\n",
       "      <td>7.000000</td>\n",
       "      <td>127644.240000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.00000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>149388.247500</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>10000.00000</td>\n",
       "      <td>1.581569e+07</td>\n",
       "      <td>850.000000</td>\n",
       "      <td>92.000000</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>250898.090000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>1.00000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>199992.480000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         RowNumber    CustomerId   CreditScore           Age        Tenure  \\\n",
       "count  10000.00000  1.000000e+04  10000.000000  10000.000000  10000.000000   \n",
       "mean    5000.50000  1.569094e+07    650.528800     38.921800      5.012800   \n",
       "std     2886.89568  7.193619e+04     96.653299     10.487806      2.892174   \n",
       "min        1.00000  1.556570e+07    350.000000     18.000000      0.000000   \n",
       "25%     2500.75000  1.562853e+07    584.000000     32.000000      3.000000   \n",
       "50%     5000.50000  1.569074e+07    652.000000     37.000000      5.000000   \n",
       "75%     7500.25000  1.575323e+07    718.000000     44.000000      7.000000   \n",
       "max    10000.00000  1.581569e+07    850.000000     92.000000     10.000000   \n",
       "\n",
       "             Balance  NumOfProducts    HasCrCard  IsActiveMember  \\\n",
       "count   10000.000000   10000.000000  10000.00000    10000.000000   \n",
       "mean    76485.889288       1.530200      0.70550        0.515100   \n",
       "std     62397.405202       0.581654      0.45584        0.499797   \n",
       "min         0.000000       1.000000      0.00000        0.000000   \n",
       "25%         0.000000       1.000000      0.00000        0.000000   \n",
       "50%     97198.540000       1.000000      1.00000        1.000000   \n",
       "75%    127644.240000       2.000000      1.00000        1.000000   \n",
       "max    250898.090000       4.000000      1.00000        1.000000   \n",
       "\n",
       "       EstimatedSalary        Exited  \n",
       "count     10000.000000  10000.000000  \n",
       "mean     100090.239881      0.203700  \n",
       "std       57510.492818      0.402769  \n",
       "min          11.580000      0.000000  \n",
       "25%       51002.110000      0.000000  \n",
       "50%      100193.915000      0.000000  \n",
       "75%      149388.247500      0.000000  \n",
       "max      199992.480000      1.000000  "
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "9155e0d9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 10000 entries, 0 to 9999\n",
      "Data columns (total 14 columns):\n",
      " #   Column           Non-Null Count  Dtype  \n",
      "---  ------           --------------  -----  \n",
      " 0   RowNumber        10000 non-null  int64  \n",
      " 1   CustomerId       10000 non-null  int64  \n",
      " 2   Surname          10000 non-null  object \n",
      " 3   CreditScore      10000 non-null  int64  \n",
      " 4   Geography        10000 non-null  object \n",
      " 5   Gender           10000 non-null  object \n",
      " 6   Age              10000 non-null  int64  \n",
      " 7   Tenure           10000 non-null  int64  \n",
      " 8   Balance          10000 non-null  float64\n",
      " 9   NumOfProducts    10000 non-null  int64  \n",
      " 10  HasCrCard        10000 non-null  int64  \n",
      " 11  IsActiveMember   10000 non-null  int64  \n",
      " 12  EstimatedSalary  10000 non-null  float64\n",
      " 13  Exited           10000 non-null  int64  \n",
      "dtypes: float64(2), int64(9), object(3)\n",
      "memory usage: 1.1+ MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "0bc833d7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "False"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().values.any()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "e7ec3e84",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'Geography',\n",
       "       'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',\n",
       "       'IsActiveMember', 'EstimatedSalary', 'Exited'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "f53613f1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RangeIndex(start=0, stop=10000, step=1)"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.index"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "b4c8a149",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['France' 'Spain' 'Germany']\n",
      "['Female' 'Male']\n",
      "[1 3 2 4]\n",
      "[1 0]\n",
      "[1 0]\n",
      "[1 0]\n"
     ]
    }
   ],
   "source": [
    "print(df[\"Geography\"].unique())\n",
    "print(df[\"Gender\"].unique())\n",
    "print(df[\"NumOfProducts\"].unique())\n",
    "print(df[\"HasCrCard\"].unique())\n",
    "print(df[\"IsActiveMember\"].unique())\n",
    "print(df[\"Exited\"].unique())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "7f9e7259",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RowNumber          0\n",
       "CustomerId         0\n",
       "Surname            0\n",
       "CreditScore        0\n",
       "Geography          0\n",
       "Gender             0\n",
       "Age                0\n",
       "Tenure             0\n",
       "Balance            0\n",
       "NumOfProducts      0\n",
       "HasCrCard          0\n",
       "IsActiveMember     0\n",
       "EstimatedSalary    0\n",
       "Exited             0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "db0467c0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[RangeIndex(start=0, stop=10000, step=1),\n",
       " Index(['RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'Geography',\n",
       "        'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',\n",
       "        'IsActiveMember', 'EstimatedSalary', 'Exited'],\n",
       "       dtype='object')]"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.axes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "f75b39f1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>RowNumber</th>\n",
       "      <th>CustomerId</th>\n",
       "      <th>Surname</th>\n",
       "      <th>CreditScore</th>\n",
       "      <th>Geography</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Tenure</th>\n",
       "      <th>Balance</th>\n",
       "      <th>NumOfProducts</th>\n",
       "      <th>HasCrCard</th>\n",
       "      <th>IsActiveMember</th>\n",
       "      <th>EstimatedSalary</th>\n",
       "      <th>Exited</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>15634602</td>\n",
       "      <td>Hargrave</td>\n",
       "      <td>619</td>\n",
       "      <td>France</td>\n",
       "      <td>Female</td>\n",
       "      <td>42</td>\n",
       "      <td>2</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>101348.88</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>15647311</td>\n",
       "      <td>Hill</td>\n",
       "      <td>608</td>\n",
       "      <td>Spain</td>\n",
       "      <td>Female</td>\n",
       "      <td>41</td>\n",
       "      <td>1</td>\n",
       "      <td>83807.86</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>112542.58</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>15619304</td>\n",
       "      <td>Onio</td>\n",
       "      <td>502</td>\n",
       "      <td>France</td>\n",
       "      <td>Female</td>\n",
       "      <td>42</td>\n",
       "      <td>8</td>\n",
       "      <td>159660.80</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113931.57</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>15701354</td>\n",
       "      <td>Boni</td>\n",
       "      <td>699</td>\n",
       "      <td>France</td>\n",
       "      <td>Female</td>\n",
       "      <td>39</td>\n",
       "      <td>1</td>\n",
       "      <td>0.00</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>93826.63</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   RowNumber  CustomerId   Surname  CreditScore Geography  Gender  Age  \\\n",
       "0          1    15634602  Hargrave          619    France  Female   42   \n",
       "1          2    15647311      Hill          608     Spain  Female   41   \n",
       "2          3    15619304      Onio          502    France  Female   42   \n",
       "3          4    15701354      Boni          699    France  Female   39   \n",
       "\n",
       "   Tenure    Balance  NumOfProducts  HasCrCard  IsActiveMember  \\\n",
       "0       2       0.00              1          1               1   \n",
       "1       1   83807.86              1          0               1   \n",
       "2       8  159660.80              3          1               0   \n",
       "3       1       0.00              2          0               0   \n",
       "\n",
       "   EstimatedSalary  Exited  \n",
       "0        101348.88       1  \n",
       "1        112542.58       0  \n",
       "2        113931.57       1  \n",
       "3         93826.63       0  "
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.iloc[0:4]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "d542e582",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0       15634602\n",
       "1       15647311\n",
       "2       15619304\n",
       "3       15701354\n",
       "4       15737888\n",
       "          ...   \n",
       "9995    15606229\n",
       "9996    15569892\n",
       "9997    15584532\n",
       "9998    15682355\n",
       "9999    15628319\n",
       "Name: CustomerId, Length: 10000, dtype: int64"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.loc[:, \"CustomerId\"]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1d7d6712",
   "metadata": {},
   "source": [
    "### Data Visulization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "179ab8b8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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PDw8XJJKncU5evSoddYoPDxchYWFys3NvWzNkSNHSu332LFjpc4+lZg6darcbrc1HTx4sPwHCAAAqjSvBiJjjJ544gm9//77+vjjj9WsWTOP5c2aNVN4eLjWr19vjRUWFio1NVXdunWTJHXu3Fl+fn4eNTk5Odq5c6dVExMTI7fbrW3btlk1W7duldvttmou5u/vr+DgYI8JAADcmLx6D9G4ceP07rvv6h//+Ifq1KljnQlyOp0KDAyUw+FQUlKSkpOT1apVK7Vq1UrJycmqVauWhgwZYtWOGjVKkyZNUkhIiOrXr6/Jkyerffv26t27tySpbdu26tevn0aPHq1FixZJksaMGaO4uLgresIMAADc2LwaiF577TVJUo8ePTzGlyxZohEjRkiSpkyZovz8fCUmJio3N1fR0dFat26d6tSpY9XPnz9fvr6+Gjx4sPLz89WrVy8tXbpUPj4+Vs2KFSs0YcIE62m0+Ph4LVy4sHIPEAAAVAtV6j1EVRnvIQK8h/cQATcu3kMEAABQRRCIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7RGIAACA7Xk1EP3v//6vBg4cqIiICDkcDv3973/3WG6M0YwZMxQREaHAwED16NFDu3bt8qgpKCjQ+PHj1aBBAwUFBSk+Pl6HDh3yqMnNzVVCQoKcTqecTqcSEhJ08uTJSj46AABQXXg1EJ05c0Y/+9nPtHDhwjKXz507V/PmzdPChQuVnp6u8PBw9enTR6dOnbJqkpKStHr1aq1atUqffvqpTp8+rbi4OBUVFVk1Q4YMUWZmplJSUpSSkqLMzEwl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/LLSPg/Abjp16qSbb75Z77//vjX2/vvvKzIyUh07drTGUlJSdOedd6pu3boKCQlRXFycx893Wb7++mvdc889ql27tsLCwpSQkKDjx49X2rHAuwhEuOF8+OGHeuSRRzRhwgR9/fXXWrRokZYuXarf/e53HnX/9V//pWHDhikzM1O33HKLhgwZorFjx2rq1Knavn27JOmJJ56w6k+fPq177rlHH330kb744gvFxsZq4MCB+v7778vswxijAQMGyOVyae3atcrIyFCnTp3Uq1cvnThxovI+AMBmHn30US1ZssSaf+uttzRy5EiPmjNnzmjixIlKT0/Xhg0bVKNGDd1///0qLi4uc5s5OTnq3r27brvtNm3fvl0pKSk6cuSIBg8eXKnHAi8yQDU1fPhw4+PjY4KCgqzpl7/8pbnrrrtMcnKyR+2yZctMo0aNrHlJZvr06db8li1bjCTz5ptvWmMrV640AQEBl+2hXbt25g9/+IM136RJEzN//nxjjDEbNmwwwcHB5ty5cx7rtGjRwixatOiqjxeAp+HDh5t7773XHDt2zPj7+5vs7Gyzf/9+ExAQYI4dO2buvfdeM3z48DLXPXr0qJFkduzYYYwxJjs720gyX3zxhTHGmGeffdb07dvXY52DBw8aSWbPnj2VeVjwElv96g7ceH7xi1/otddes+aDgoLUsmVLpaene5wRKioq0rlz53T27FnVqlVLktShQwdreVhYmCSpffv2HmPnzp1TXl6egoODdebMGc2cOVNr1qzR4cOHdeHCBeXn51/yDFFGRoZOnz6tkJAQj/H8/PyfPFUP4Mo1aNBAAwYM0Ntvv22dmW3QoIFHzb59+/Tss88qLS1Nx48ft84Mff/994qKiiq1zYyMDG3cuFG1a9cutWzfvn1q3bp15RwMvIZAhGqtJAD9WHFxsWbOnKlBgwaVqg8ICLD+7OfnZ/3Z4XBccqzkP86nn35aH374oV5++WW1bNlSgYGB+uUvf6nCwsIyeysuLlajRo20adOmUsvq1q17ZQcI4IqMHDnSusT96quvllo+cOBARUZGavHixYqIiFBxcbGioqIu+/M7cOBAzZkzp9SyRo0aVWzzqBIIRLjhdOrUSXv27CkVlK7VJ598ohEjRuj++++X9J97ivbv33/ZPlwul3x9fdW0adMK7QWAp379+lnhJjY21mPZDz/8oKysLC1atEh33XWXJOnTTz+97PY6deqkv/3tb2ratKl8ffmqtANuqsYN57nnntM777yjGTNmaNeuXcrKytJ7772n6dOnX9N2W7Zsqffff1+ZmZn68ssvNWTIkEvekClJvXv3VkxMjO677z59+OGH2r9/vzZv3qzp06dbN20DqBg+Pj7KyspSVlaWfHx8PJbVq1dPISEheuONN7R37159/PHHmjhx4mW3N27cOJ04cUK/+tWvtG3bNn333Xdat26dRo4cqaKioso8FHgJgQg3nNjYWK1Zs0br16/X7bffrq5du2revHlq0qTJNW13/vz5qlevnrp166aBAwcqNjZWnTp1umS9w+HQ2rVrdffdd2vkyJFq3bq1Hn74Ye3fv9+6ZwlAxQkODlZwcHCp8Ro1amjVqlXKyMhQVFSUnnrqKb300kuX3VZERIQ+++wzFRUVKTY2VlFRUXryySfldDpVowZfnTcihzHGeLsJAAAAbyLmAgAA2yMQAQAA2yMQAQAA2yMQAQAA2yMQAQAA2yMQAQAA2yMQAQAA2yMQAcAV6NGjh5KSkrzdBoBKQiACUG24XC49+eSTatmypQICAhQWFqY777xTr7/+us6ePevt9gBUY/zGOgDVwnfffaef//znqlu3rpKTk9W+fXtduHBB33zzjd566y1FREQoPj7e221eUlFRkRwOB7/2Aaii+MkEUC0kJibK19dX27dv1+DBg9W2bVu1b99eDzzwgP75z39q4MCBkiS3260xY8YoNDRUwcHB6tmzp7788ktrOzNmzNBtt92mZcuWqWnTpnI6nXr44Yd16tQpq+bMmTMaNmyYateurUaNGumVV14p1U9hYaGmTJmim266SUFBQYqOjtamTZus5UuXLlXdunW1Zs0atWvXTv7+/jpw4EDlfUAArgmBCECV98MPP2jdunUaN26cgoKCyqxxOBwyxmjAgAFyuVxau3atMjIy1KlTJ/Xq1UsnTpywavft26e///3vWrNmjdasWaPU1FS9+OKL1vKnn35aGzdu1OrVq7Vu3Tpt2rRJGRkZHvt79NFH9dlnn2nVqlX66quv9OCDD6pfv3769ttvrZqzZ89q9uzZ+tOf/qRdu3YpNDS0gj8ZABXGAEAVl5aWZiSZ999/32M8JCTEBAUFmaCgIDNlyhSzYcMGExwcbM6dO+dR16JFC7No0SJjjDHPP/+8qVWrlsnLy7OWP/300yY6OtoYY8ypU6dMzZo1zapVq6zlP/zwgwkMDDRPPvmkMcaYvXv3GofDYf7973977KdXr15m6tSpxhhjlixZYiSZzMzMivkQAFQq7iECUG04HA6P+W3btqm4uFhDhw5VQUGBMjIydPr0aYWEhHjU5efna9++fdZ806ZNVadOHWu+UaNGOnr0qKT/nD0qLCxUTEyMtbx+/fpq06aNNf/555/LGKPWrVt77KegoMBj3zVr1lSHDh2u4YgBXC8EIgBVXsuWLeVwOLR7926P8ebNm0uSAgMDJUnFxcVq1KiRx708JerWrWv92c/Pz2OZw+FQcXGxJMkY85P9FBcXy8fHRxkZGfLx8fFYVrt2bevPgYGBpUIcgKqJQASgygsJCVGfPn20cOFCjR8//pL3EXXq1Ekul0u+vr5q2rRpufbVsmVL+fn5KS0tTTfffLMkKTc3V9988426d+8uSerYsaOKiop09OhR3XXXXeXaD4CqhZuqAVQLf/zjH3XhwgV16dJF7733nrKysrRnzx4tX75cu3fvlo+Pj3r37q2YmBjdd999+vDDD7V//35t3rxZ06dP1/bt269oP7Vr19aoUaP09NNPa8OGDdq5c6dGjBjh8bh869atNXToUA0bNkzvv/++srOzlZ6erjlz5mjt2rWV9REAqEScIQJQLbRo0UJffPGFkpOTNXXqVB06dEj+/v5q166dJk+erMTERDkcDq1du1bTpk3TyJEjdezYMYWHh+vuu+9WWFjYFe/rpZde0unTpxUfH686depo0qRJcrvdHjVLlizRrFmzNGnSJP373/9WSEiIYmJidM8991T0oQO4DhzmSi6YAwAA3MC4ZAYAAGyPQAQAAGyPQAQAAGyPQAQAAGyPQAQAAGyPQAQAAGyPQAQAAGyPQAQAAGyPQAQAAGyPQAQAAGyPQAQAAGyPQAQAAGzv/wGRYULfm5/hBwAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x='Gender', data=df)\n",
    "plt.title('Gender Distribution')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "b3f6cec0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABJEAAANYCAYAAAB0DnxAAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAABe50lEQVR4nO3dd5xV9YH///fQBikzUqRFBAsSEJQIiqBRFLEkWGKyKUaiG1tiZaMxcU0imu9i1sTuxhg3irFEk6xmUwx2XRWxEEdFkVhAURkwCgMonfv7gx83DO0AooP6fD4e83jMuedzz/3cO8M48/Jzzq0olUqlAAAAAMA6NGroCQAAAACw+RORAAAAACgkIgEAAABQSEQCAAAAoJCIBAAAAEAhEQkAAACAQiISAAAAAIVEJAAAAAAKiUgAAAAAFGrS0BMAAFhfs6fOzmXbXlbePvr+o9N9SPeGm9BGurT7pal7tS5Jss+5+2TIqCENO6EkY4aMyasPvpok2eXoXXL4mMOTbJ6v+eY4p03lgVEP5MHzHkySVHerzsipIxt2QgCwEhEJADaB995+LzXX1eSVu1/JjGdnZMGsBWnUtFFadWqV1l1ap/uQ7unxuR7psluXVFRUNPR0eZ9WjkBJUtG4Io2bNU7zLZunauuqdPpMp+z89Z3Tbe9uH+g8asbU5H//9X/L2+eWzv1AH+/D8HEJRCvHoHoqksqqyrTv2T47HrpjBp42MJWtKz+weXxcXk8ANg8iEgC8T3/71d9y58g7s2jeovo7FiTvzH0n77z4Tl598NU8eN6DOX3K6dmy+5YNMk8+OKWlpSyZvyTz5s/LvOnz8uYTb+Zvv/xbuu/bPUfceERad2ldb/xnz/lsFtYtTJJ0Hdy1Iaa8mgHfHpAdh++YJOnQp0MDz2bdtmi7RYb9dFh5u832bRpwNhuolCysW5g3Hn8jbzz+RmqurckxDx6Tqq2rGnpmAFBIRAKA9+HRSx7NXd+56583VCTb7rttPjXwU6msqsz8d+ZnxtMz8upDr2bJ/CUNN9EPwJIFS5avwGnauKGn0qDabNcmA749IEsWLsnsKbPz4l9ezLzaeUmSqfdPzbV7XZvjHjsuLbdqWb5P/+P7N9R0V7Nw7sJUtq5Mn6/0aeiprLfKqsoMPnNwQ09jg+z173tlizZbZOHchfn7H/+e2praJMmsV2blr6f+NV+5/SsNPEMAKCYiAcBG+scL/8g9Z91T3m7RvkW+9qevZes9tl5t7OL3FueZm55Js1bNVts3/anpefzyxzP1wamZ++bcNG7aOO0/3T59vtYnu520W5o0X/0/13PemJPxl4zPS2Nfyuyps7NsybJUfaoq3YZ0yx4j90jHvh1Xu8/sV2fn3rPvzct3vpwlC5ak02c6ZZ9z98ncN+au9ZSoVa+TM/D0gbnvnPvy+qOvZ8HsBeWVVQ+NfihvPPZG3nr+rbz39ntZNHdRmrVqlnY7tkvPw3pm4OkD06xl/ed+XsV55c8Pu+6wtOzQMg+Nfii1T9WmcbPG2W7Ydtn/J/unzXbrXmXy3O+ey7ifjsvMZ2emacum2XH4jjnw4gOzRdstkiQPnv9gHjj3gSTJlt23zGmvnFbvlMLap2tzdb+ry9sn1pyYTrt0Wudjrqyqa1W9oLFk4ZLccfIdeepXTyVJZk+ZnbGnj80Xb/5iecy6rok0+Y+T88TPn0jtU7WZ/878NNmiSVpu1TId+nbIpwZ+Knt9b6/UvVZX7xSlFVZ+TVccd9VT3s6ee3YeGPVAnv/d85nzxpzs/YO9M2TUkLVeE2lNJt46MeN+Oi5vPfdWmrVulp6H9szQ0UPTssM/Q9m6TrVb2ylWq54mmCTX73t9+fNu+3TLMQ8cs16naD3/++fz1LVPZfqE6Zn/zvw0a90sHfp0SJ+v9smux+2axs3+GT/XdLw5r8/J+EvH563n3lrj99WG6n98//IqxM/++2fz894/z6xXZiVJ/v6Xv2fJwiVpUln8q/n8d+bnscsfy+Q/Ts47L72TpQuXpmWHlum6Z9fsfuru2WbPbcpj1/f1BID1JSIBwEYaf9n4LFuyrLw9/OrhawxISdK0RdM1rj55/MrHM3bk2JSWlsq3LV24NG8++WbefPLNTLxlYkbcPSLNq5uX97/6f6/mlsNuyYLZC+oda9YrszLrlVl55oZncsg1h6Tf0f3K+2ZPnZ1fDfpVeYVMkkx7ZFpuPPDG9Di4x3o93xnPzMh1e12Xxe8tXm3fQ6MfyuJ369++YPaC8ik7E2+ZmGPHHbvGiJYkT//66Uy9f2p5e/F7i/P8757P1Aem5thxx6btDm3XeL+HL3g4L9/1cnl7yYIlefr6p/POS+/kmw9/M0nS/8T+eeg/HsrSRUsze+rsvHL3K9n+gO3L93nut8+VP++8a+cNCkhr0qSySQ755SGZPmF6ebXJxFsm5oCfHbDaaW2rWjW8JMmiuYuyaO6izHplVib/7+QM+rdB72t+Nx50Y6Y9Mm2j7z/uZ+Py4l9eLG8vWbAkT/3qqbz64Ks5dvyxadGuxfua3/u1bOmy3HbkbfW+rkmyYNaCvPbQa3ntodfy9PVP56i7jqr372pl9/3gvnqv0Zq+r96PJpVN0nnXzuWItGzxssx/e37h98dbz7+VGw+8MXNen1Pv9jmvz8lztz6X5377XPb7f/vls//+2fc9RwBYExEJADbS1Pumlj9v3qZ5Pv2FT2/Q/V975LX89bS/Jv9/P9pmr22y7f7bZsHsBXn6+qezYNaCvPnEm/nLt/9SXsWyYPaC3PqFW8sBqWnLpvnMNz+TJls0yTM3PJN50+dl2eJl+dNxf0rnXTuXVyTdccod9QJSj8/1SOf+nfPiX17Mi3e8mPVR+1RtGjVtlH7H9Eub7dvkrefeSqOmjZIk1dtUp0OfDqnepjrN2zRPSsmsKbPy3K3PZfG7izPz2Zl54udPZM+z9lzza3n/1HTu3zk9Ptcjbz33VibdNilJ8t5b7+XP3/pzvnHPN9Z4v5fvejlbD9o62w7dNi/++cVytJn2yLRMe3Raug7qmlYdW6X3l3rn2ZufTZL87b//Vi8iPf+758uf9/vXfuv1WhSpaFSRXY7ZJbUjl88npWTqA1PT98i+67zfk1c9Wf68y25dsuPwHbNsybLMmTYnrz/2ev4x6R9J/nlNoDeffDPP3frPWLLydYLWdq2laY9MS9c9u2bbodtm0dxFG3wtnhf/8mK679s923x2m0x7ZFqm3DslSfLOS+/knu/dk0P/+9ANOt7KPnvOZzN76uw8PPrh8m39v9U/bbdfHhGruhbP9aH/eKheQFrxXGfUzMjkP05Okrzx+Bv584l/zpdu+dIajzHtkWmF31fvx5KFSzL9b9PL242aNsoW7da9wmnZkmW59Qu3lgNSoyaNssvRu6Rlx5Z5/nfP550X30lKyX3n3JdOn+mUHgf32CSvJwCsTEQCgI208mqAdj3a1TtFatXTY1bo9cVe+fLvv5wkefSiR8sBafsDt8/X//r18jF2OGiH3HTQTUmWr2IZduGwVG1dlZoxNZn/zvzy8b5y21fKQWTAiQNy5aevzLLFy7JsybI8dvljOfSaQzP3zbn1QtFOX9mp/MfzZ8/5bH6xyy/y9uS31+s5f/V/v7rGlUsnP39yFtQtyLRx01L3Wl0Wv7s4W/XaKl36d8mr/7f8FKmX73x5rRFpq522yrHjji2fYvSnE/6Uv13ztyTJlHun5J2X3yn/4buyrffYOv/6f/+aRk0aZdC/DcpPO/y0vKrrzSffLP+xv9spu5Uj0uT/nZx333o3Lbdqmdqa2uV/fCdp3KxxYeTZEO12bFdve84bc9Yy8p+WLPjndbMOvvzg1Va2zZ46O42bNU6T5k0y+MzBqRlTUy8irc91gvp8tU+OuPmIjX6XwO0P2D5fH7v8e7VUKuWmg24qrwZ75sZncvDlB6dpi6Ybdez+x/dfLXr0+Uqf9X43sWVLl+Wxyx4rb2+z1zY55sFjUtFo+XP932P/NzXX1iRZvgLtgJ8dsMaItr7fVxtiwjUTskWbLbJo3qJM/uPk8iqkZHnULTqV7e9//nve/vs//51+7r8+l/4nLF/duOd398zl219e/tkw/uLx6XFwj/f9egLAqkQkANgUNuLv8ZVPl3n5zpdzfqPz1zywlLw+/vX0/lLvTBv3z/u07NCy3oqaNtu1yTZ7bVM+Lez1ca8nyfIVD/88Wy67fGOX8udNKpukz9f65MFRa3gr8lV03KXjGgNSaVkp93z/njx22WNZumjpWu+/6ik4K9vpKzvVu0bNzkftXI5ISTJ9wvQ1RqTPHPuZNGqyfDXUFm23SIv2LfLujHeTLD99aYWug7qmc//OmT5hepYuWpqnf/10Bp8xOM/97p8BpudhPTf6ejdrVCoesqptPrtNZjwzI0lyw7AbsvWgrdO2R9ts1XurdNu72xqvdbWh9vzenhsdkJKk71F9y/evqKhI36/3LUekpQuXZubEmfnU7p963/PcGG9PfrteZO1zZJ9yQEqSfkf3K0eklJJpj07LTv+y02rHWd/vqw2xcshZ2Zbdt8zBlx9ceP+V/+0ny/+NrNB8y+bpeVjP1FxXs8axALCpiEgAsJFaf6p1eRXLOy++k1KpVP7jeuW3IH9o9ENr/MNz5T92i7z71up/wK58EeMVWnVs9c/jz1p+/FWvndSqU6t1bq/NqitrVnjs8scy7qfjCu+/ZOHa351u1efSsmP97RXPZVXV3arrba+8mqO0rH7F2f2U3cvXG3rqV09l8BmDP5BT2VZYedVIklR9qvjUoaGjh2bWK7Py0l9fyqJ5i/LK3a/klbtfKe/vtk+3fP2Or2/0Sp9k7V/H9bWxX6uV/32s63vh/Vj1sYvmurYgtCHfVxusIqlsXZl2O7bLjofumD1O3yOVVZWFd1v5uTVr1Wy174GVn9vi9xZn6aKl9cIsAGwKIhIAbKRth25bjkjz35mfyX+cnE8ftvy6SCu/BfnjVz6+xj9Wm7dpnvfeei9J0n3f7unxubVf4HrF6TPN2/zzQsDvznx3tXHzZvzzukdbtFm+qqb5lvUvHrzq/Va+VtK6rC1crHw6VYc+HXLEzUek/afbp3HTxrn7rLvXKzCtOqcVqz5WWPU5rNC46Sp/JK9jgU2fr/XJ3d+9O+/94738Y9I/8viVj5e/fq27tK63quv9WrZ0WWrG1NSb1/qcQlRZVZmv3/H1zHl9Tl4f/3re/vvbeev5t/LC7S9k8XuL8+qDr+aRCx+p925uG+r9BKhk/b9WK68ASpIl85eUH3vF676prfieL8+taK5t3v/31fpa8U6GG2vl57Zo3qIsfm9xva/lys+taYumAhIAH4hGDT0BAPio2v2U3VPR+J9/Xf7lW3/J9Kemr+Me9a184eN5tfMy4NsDMvjMwfU++p/QP1Vdq9KpX6fV7vPuzHfrvTPZrFdm5bWHXytvbz14+fV0ugzoUu+P4Im3TCx/vmThkkz8zT+3N8Z7b79X/rz7vt3TsW/HNG7aOIvnLy5fyLjIc7c+l6WL/3kq3DM3PlNvf5cBXd7XHJPlq0k+c9xnytt3f/fu8uc7f2PnNGq8aX4tWrJwSf584p8z4+kZ5dv6fLVP4TtvJcnMiTOzdPHSVG1dld5f6p3P/vtnc8SNR9Sb9/QJ9S/IvLI1vXPepvbsjc+mVFq+GqdUKuXZm54t72vcrHH5lLtVw9/rjy0/vXLp4qUZf8n4tR7//Tyndj3b1TslceLNE+utHKq5vuafgyuy1ndT3ByteqH0lf+NLJi9IJP/d/IaxzbE9wgAH19WIgHARuqwU4fs++N9c9+/35dkeQi6ZrdrssNBO6Rz/85pUtkks1+dvdrqhxUGnTFoeWQpJf+Y9I9c1eeqfPqIT6dF+xaZ/878zKiZkVcfejWtOrVKn6/0SZLscvQu+b8f/1/5VLhbj7i13ruzLVu8LMnyd24aeOrAJMtPV9tx+I75+5/+niR5+vqns7BuYTrs3CEv/vnF9b6o9tq079m+vLLkb9f8bfnpOlWVef53z6/3sd967q38atCv0uPzPfLWxH++O1uyPEyt6XpIG2O3b++WcT8dl9LSUr2LWPc7pt9GH3POtDkZ97NxWbpoaWZNmZUX//xivdVdW3bfMgdddtB6HeuuM+/KG4+/ke2GbpeqrlVpsVWLzH1zbvlaN0n9OLPqKXL/c+T/pOvgrqloVJGdR+xc7/TGTeXlu17Or4f+Ot327pbXHn6t/O5sSdL3633Lq2M69++8PF7+/w3nt0f8NjsctENmPDMjbz3/1lqP33KrlmnUtFH5e/m+c+5LbU1tGjdrnO5Duq8zKDZq3Ci7n7Z7+Rpfrz38Wq7b+7pst/92qa2prRdaen+pd6q7Vq/tUJudHYfvmLY92pb/rd1x8h154/E30qpTqzz32+fqnR67x7/tUf78/byeALAqEQkA3ofPnv3ZNGvVLPecdU+WLFiS0tJSXvzLi3nxLy+ucfzKb+Pd7bPdctBlB+XOf7szpaWlzJ46O+MvXvsKjWT5KS3/8vt/ya1fuDUL6xZm8buL8/gVj9cb06hJowy/eng67vzPizAffMXBefOJN8tx44U/vJAX/vBCUrH8neFevvP/X9G0Eaft7Pm9PfPS2JeybMmyLFmwJI9fvnw+zVo1S68jetULQmuzw8E75KWxL9VbZZMsf72G/2L4hk9qLaq3qU7PQ3vmhdtfKN/WdXDXtO/ZfqOPOeuVWfVWNa2s+5DuOeKmI9Jyq9WvX7U2C2YtyPO/f36N+5o0b5LdT929vL31oK3TqnOrzJu+/Os6+X8nl0NJ9yHdP5CI1H1I90y9f2r5Au4rtNmuTYZdOKy8XfWpqvT5ap/ySrcFsxeUV8Ftf8D29VbRraxxs8bZcfiO5a9RbU1tamtqkyTDfjqsMHrsfc7emfnMzPL33bRHptW7iH2yPHANv3rTfV99GBo1aZSv3PaV3HjgjZn75twsW7IsT/3qqdXGDTlvSL1TY9/v6wkAKxORAOB9GnjqwOz0LztlwjUTMuWeKfnHC//I/Fnz06hJo7Ro3yLte7bP1oO3Ts9Deq72B9vAUwem22e75fErH8+rD76aOa/PSUXjirTu0jptt2+bHQ7eIZ8+/NP17rPtvtvm289+O49e/GhevvPlzJ46O6VlpbTu3Drdh3TPwJED02mXTvXus2W3LXPs+GNz7/fvzUtjX8qShUvSqV+n7P3DvTPj6RnliLS2aw+tyzZ7bZOj7jwq9//w/rw54c00ad4k2+y5TYb+ZGgm/c+k9YpIO315pwz6zqA8eP6DmT5heho1bZTt9t8u+/9k/7TdYdOsQlph91N3rxeR+n2z3/s/aMXy0+Wab9k8VVtXpWO/jtn56ztv8FupD/7u4LT/dPu88dgbqZtWt/yaWRXLg8w2n90mg84YVO8d2ppUNsnX7/h67j7r7rzx2BtZOGfh+38uBfY5d5/0+9d+GX/p+Pxj0j/SrFWz7Hjojhk6emhatG9Rb+xh1x6Wlh1b5rlblq+UadujbXY/Zfdsf+D2uXy7y9f6GIdcc0gqqyrz0tiX8t5b723QxawbNWmUf/n9v+S53z6XmutqMn3C9CyYvSDNWjXLVjttlZ2+slP6n9C/3sWyPyo69OmQbz3zrTx22WP5+5/+nrdffDtLFy1Nyw4t03Vw1+x+6u7p9tluq93v/byeALCyitKKk9oBgI+t0rJSli1ZttrFdpctXZZrB1+bNx5/I0my3bDtMuKuER/KnM6rOK/8+WHXHfa+TinbEHPfnJuLP3VxkuUXID6j9oxUti5+dywAgE+6j97/ggEANtjCOQtzRY8r0ufIPunUr1NadmiZuW/MTc2YmnJASpKBpw1swFl+sKY+MDWL5i3KY5c9Vr5t52/sLCABAKwnEQkAPiHe+8d75esVraZi+bVUdhy+44c6pw/T9fteX297i3ZbZJ8f7dNAswEA+OgRkQDgE6Bpi6bZ6+y9MvX+qZn1yqzMnzU/jZs2TlXXqmyz1zbpf2L/fGq3TzX0ND8Uzds0L1+zqXXn1g09HQCAjwzXRAIAAACgUKOGngAAAAAAmz8RCQAAAIBCrom0npYtW5Y333wzrVu3TkVFRUNPBwAAAGCTKJVKmTt3brp06ZJGjda+3khEWk9vvvlmunbt2tDTAAAAAPhATJs2LVtvvfVa94tI66l16+Xv3jJt2rRUVVU18GwAAAAANo05c+aka9eu5faxNiLSelpxCltVVZWIBAAAAHzsFF2+x4W1AQAAACgkIgEAAABQSEQCAAAAoJCIBAAAAEAhEQkAAACAQiISAAAAAIVEJAAAAAAKiUgAAAAAFBKRAAAAACgkIgEAAABQSEQCAAAAoJCIBAAAAEAhEQkAAACAQiISAAAAAIVEJAAAAAAKiUgAAAAAFBKRAAAAACgkIgEAAABQSEQCAAAAoJCIBAAAAEAhEQkAAACAQiISAAAAAIVEJAAAAAAKiUgAAAAAFBKRAAAAACgkIgEAAABQSEQCAAAAoJCIBAAAAEAhEQkAAACAQiISAAAAAIVEJAAAAAAKiUgAAAAAFGrQiDRq1KhUVFTU++jUqVN5f6lUyqhRo9KlS5dsscUWGTJkSJ577rl6x1i4cGFOPfXUtG/fPi1btsyhhx6a119/vd6YWbNmZcSIEamurk51dXVGjBiR2bNnfxhPEQAAAOBjoUlDT2CnnXbKPffcU95u3Lhx+fMLL7wwF198ccaMGZMdd9wx/+///b8MGzYskydPTuvWrZMkI0eOzJ/+9KfccsstadeuXc4444wMHz48EyZMKB/ryCOPzOuvv56xY8cmSU444YSMGDEif/rTnz7EZ7rp9f/urxt6CsAn2ISffqOhpwAAAHyIGjwiNWnSpN7qoxVKpVIuvfTSnHPOOTniiCOSJNdff306duyYm2++OSeeeGLq6uryq1/9KjfccEP233//JMmNN96Yrl275p577smBBx6YSZMmZezYsRk/fnwGDhyYJLnmmmsyaNCgTJ48OT179lzjvBYuXJiFCxeWt+fMmbOpnzoAAADAR0aDXxPpxRdfTJcuXbLtttvmq1/9al555ZUkyZQpU1JbW5sDDjigPLaysjL77LNPxo0blySZMGFCFi9eXG9Mly5d0qdPn/KYRx99NNXV1eWAlCR77LFHqqury2PW5IILLiif/lZdXZ2uXbtu0ucNAAAA8FHSoBFp4MCB+fWvf50777wz11xzTWprazN48OC8/fbbqa2tTZJ07Nix3n06duxY3ldbW5tmzZqlTZs26xzToUOH1R67Q4cO5TFrcvbZZ6eurq78MW3atPf1XAEAAAA+yhr0dLaDDz64/Hnfvn0zaNCgbL/99rn++uuzxx57JEkqKirq3adUKq1226pWHbOm8UXHqaysTGVl5Xo9DwAAAICPuwY/nW1lLVu2TN++ffPiiy+Wr5O06mqhmTNnllcnderUKYsWLcqsWbPWOWbGjBmrPdZbb7212ionAAAAANZss4pICxcuzKRJk9K5c+dsu+226dSpU+6+++7y/kWLFuXBBx/M4MGDkyT9+/dP06ZN642ZPn16Jk6cWB4zaNCg1NXV5fHHHy+Peeyxx1JXV1ceAwAAAMC6NejpbGeeeWYOOeSQbLPNNpk5c2b+3//7f5kzZ06OPvroVFRUZOTIkRk9enR69OiRHj16ZPTo0WnRokWOPPLIJEl1dXWOPfbYnHHGGWnXrl3atm2bM888M3379i2/W1uvXr1y0EEH5fjjj8/VV1+dJDnhhBMyfPjwtb4zGwAAAAD1NWhEev311/O1r30t//jHP7LVVltljz32yPjx49OtW7ckyVlnnZX58+fnpJNOyqxZszJw4MDcddddad26dfkYl1xySZo0aZIvf/nLmT9/foYOHZoxY8akcePG5TE33XRTTjvttPK7uB166KG58sorP9wnCwAAAPARVlEqlUoNPYmPgjlz5qS6ujp1dXWpqqpq6OkkSfp/99cNPQXgE2zCT7/R0FMAAAA2gfVtHpvVNZEAAAAA2DyJSAAAAAAUEpEAAAAAKCQiAQAAAFBIRAIAAACgkIgEAAAAQCERCQAAAIBCIhIAAAAAhUQkAAAAAAqJSAAAAAAUEpEAAAAAKCQiAQAAAFBIRAIAAACgkIgEAAAAQCERCQAAAIBCIhIAAAAAhUQkAAAAAAqJSAAAAAAUEpEAAAAAKCQiAQAAAFBIRAIAAACgkIgEAAAAQCERCQAAAIBCIhIAAAAAhUQkAAAAAAqJSAAAAAAUEpEAAAAAKCQiAQAAAFBIRAIAAACgkIgEAAAAQCERCQAAAIBCIhIAAAAAhUQkAAAAAAqJSAAAAAAUEpEAAAAAKCQiAQAAAFBIRAIAAACgkIgEAAAAQCERCQAAAIBCIhIAAAAAhUQkAAAAAAqJSAAAAAAUEpEAAAAAKCQiAQAAAFBIRAIAAACgkIgEAAAAQCERCQAAAIBCIhIAAAAAhUQkAAAAAAqJSAAAAAAUEpEAAAAAKCQiAQAAAFBIRAIAAACgkIgEAAAAQCERCQAAAIBCIhIAAAAAhUQkAAAAAAqJSAAAAAAUEpEAAAAAKCQiAQAAAFBIRAIAAACgkIgEAAAAQCERCQAAAIBCIhIAAAAAhUQkAAAAAAqJSAAAAAAUEpEAAAAAKCQiAQAAAFBIRAIAAACgkIgEAAAAQCERCQAAAIBCIhIAAAAAhUQkAAAAAAqJSAAAAAAUEpEAAAAAKCQiAQAAAFBIRAIAAACgkIgEAAAAQCERCQAAAIBCIhIAAAAAhUQkAAAAAAqJSAAAAAAUEpEAAAAAKCQiAQAAAFBIRAIAAACgkIgEAAAAQCERCQAAAIBCIhIAAAAAhUQkAAAAAAqJSAAAAAAUEpEAAAAAKCQiAQAAAFBIRAIAAACgkIgEAAAAQCERCQAAAIBCIhIAAAAAhUQkAAAAAAqJSAAAAAAUEpEAAAAAKCQiAQAAAFBIRAIAAACgkIgEAAAAQCERCQAAAIBCIhIAAAAAhUQkAAAAAAqJSAAAAAAUEpEAAAAAKCQiAQAAAFBIRAIAAACgkIgEAAAAQCERCQAAAIBCIhIAAAAAhUQkAAAAAAqJSAAAAAAUEpEAAAAAKCQiAQAAAFBIRAIAAACgkIgEAAAAQCERCQAAAIBCIhIAAAAAhUQkAAAAAAqJSAAAAAAUEpEAAAAAKCQiAQAAAFBIRAIAAACgkIgEAAAAQCERCQAAAIBCIhIAAAAAhUQkAAAAAAqJSAAAAAAUEpEAAAAAKCQiAQAAAFBIRAIAAACgkIgEAAAAQCERCQAAAIBCm01EuuCCC1JRUZGRI0eWbyuVShk1alS6dOmSLbbYIkOGDMlzzz1X734LFy7Mqaeemvbt26dly5Y59NBD8/rrr9cbM2vWrIwYMSLV1dWprq7OiBEjMnv27A/hWQEAAAB8PGwWEemJJ57IL3/5y+y88871br/wwgtz8cUX58orr8wTTzyRTp06ZdiwYZk7d255zMiRI3P77bfnlltuycMPP5x58+Zl+PDhWbp0aXnMkUcemZqamowdOzZjx45NTU1NRowY8aE9PwAAAICPugaPSPPmzcvXv/71XHPNNWnTpk359lKplEsvvTTnnHNOjjjiiPTp0yfXX3993nvvvdx8881Jkrq6uvzqV7/KRRddlP333z+f+cxncuONN+bZZ5/NPffckySZNGlSxo4dm//+7//OoEGDMmjQoFxzzTX585//nMmTJzfIcwYAAAD4qGnwiHTyySfn85//fPbff/96t0+ZMiW1tbU54IADyrdVVlZmn332ybhx45IkEyZMyOLFi+uN6dKlS/r06VMe8+ijj6a6ujoDBw4sj9ljjz1SXV1dHrMmCxcuzJw5c+p9AAAAAHxSNWnIB7/lllvyt7/9LU888cRq+2pra5MkHTt2rHd7x44d8+qrr5bHNGvWrN4KphVjVty/trY2HTp0WO34HTp0KI9ZkwsuuCDnnXfehj0hAAAAgI+pBluJNG3atJx++um58cYb07x587WOq6ioqLddKpVWu21Vq45Z0/ii45x99tmpq6srf0ybNm2djwkAAADwcdZgEWnChAmZOXNm+vfvnyZNmqRJkyZ58MEHc/nll6dJkyblFUirrhaaOXNmeV+nTp2yaNGizJo1a51jZsyYsdrjv/XWW6utclpZZWVlqqqq6n0AAAAAfFI1WEQaOnRonn322dTU1JQ/BgwYkK9//eupqanJdtttl06dOuXuu+8u32fRokV58MEHM3jw4CRJ//7907Rp03pjpk+fnokTJ5bHDBo0KHV1dXn88cfLYx577LHU1dWVxwAAAACwbg12TaTWrVunT58+9W5r2bJl2rVrV7595MiRGT16dHr06JEePXpk9OjRadGiRY488sgkSXV1dY499ticccYZadeuXdq2bZszzzwzffv2LV+ou1evXjnooINy/PHH5+qrr06SnHDCCRk+fHh69uz5IT5jAAAAgI+uBr2wdpGzzjor8+fPz0knnZRZs2Zl4MCBueuuu9K6devymEsuuSRNmjTJl7/85cyfPz9Dhw7NmDFj0rhx4/KYm266Kaeddlr5XdwOPfTQXHnllR/68wEAAAD4qKoolUqlhp7ER8GcOXNSXV2durq6zeb6SP2/++uGngLwCTbhp99o6CkAAACbwPo2jwa7JhIAAAAAHx0iEgAAAACFRCQAAAAAColIAAAAABQSkQAAAAAoJCIBAAAAUEhEAgAAAKCQiAQAAABAIREJAAAAgEIiEgAAAACFRCQAAAAAColIAAAAABQSkQAAAAAoJCIBAAAAUEhEAgAAAKCQiAQAAABAIREJAAAAgEIiEgAAAACFRCQAAAAAColIAAAAABQSkQAAAAAoJCIBAAAAUEhEAgAAAKCQiAQAAABAIREJAAAAgEIiEgAAAACFRCQAAAAAColIAAAAABQSkQAAAAAoJCIBAAAAUEhEAgAAAKCQiAQAAABAIREJAAAAgEIiEgAAAACFRCQAAAAAColIAAAAABQSkQAAAAAoJCIBAAAAUEhEAgAAAKCQiAQAAABAIREJAAAAgEIiEgAAAACFRCQAAAAAColIAAAAABQSkQAAAAAoJCIBAAAAUEhEAgAAAKCQiAQAAABAIREJAAAAgEIiEgAAAACFRCQAAAAAColIAAAAABQSkQAAAAAoJCIBAAAAUEhEAgAAAKCQiAQAAABAIREJAAAAgEIiEgAAAACFRCQAAAAAColIAAAAABQSkQAAAAAoJCIBAAAAUEhEAgAAAKCQiAQAAABAIREJAAAAgEIiEgAAAACFRCQAAAAAColIAAAAABQSkQAAAAAoJCIBAAAAUEhEAgAAAKCQiAQAAABAIREJAAAAgEIiEgAAAACFRCQAAAAAColIAAAAABQSkQAAAAAoJCIBAAAAUEhEAgAAAKCQiAQAAABAIREJAAAAgEIiEgAAAACFRCQAAAAAColIAAAAABQSkQAAAAAoJCIBAAAAUEhEAgAAAKCQiAQAAABAIREJAAAAgEIiEgAAAACFRCQAAAAAColIAAAAABQSkQAAAAAoJCIBAAAAUEhEAgAAAKCQiAQAAABAIREJAAAAgEIiEgAAAACFRCQAAAAAColIAAAAABQSkQAAAAAoJCIBAAAAUEhEAgAAAKCQiAQAAABAIREJAAAAgEIiEgAAAACFRCQAAAAAColIAAAAABQSkQAAAAAoJCIBAAAAUEhEAgAAAKCQiAQAAABAIREJAAAAgEIiEgAAAACFRCQAAAAAColIAAAAABQSkQAAAAAoJCIBAAAAUEhEAgAAAKCQiAQAAABAIREJAAAAgEIiEgAAAACFRCQAAAAAColIAAAAABQSkQAAAAAoJCIBAAAAUEhEAgAAAKCQiAQAAABAIREJAAAAgEIiEgAAAACFRCQAAAAAColIAAAAABQSkQAAAAAoJCIBAAAAUEhEAgAAAKBQg0akq666KjvvvHOqqqpSVVWVQYMG5a9//Wt5f6lUyqhRo9KlS5dsscUWGTJkSJ577rl6x1i4cGFOPfXUtG/fPi1btsyhhx6a119/vd6YWbNmZcSIEamurk51dXVGjBiR2bNnfxhPEQAAAOBjoUEj0tZbb52f/OQnefLJJ/Pkk09mv/32y2GHHVYORRdeeGEuvvjiXHnllXniiSfSqVOnDBs2LHPnzi0fY+TIkbn99ttzyy235OGHH868efMyfPjwLF26tDzmyCOPTE1NTcaOHZuxY8empqYmI0aM+NCfLwAAAMBHVUWpVCo19CRW1rZt2/z0pz/NN7/5zXTp0iUjR47M9773vSTLVx117Ngx//mf/5kTTzwxdXV12WqrrXLDDTfkK1/5SpLkzTffTNeuXXPHHXfkwAMPzKRJk9K7d++MHz8+AwcOTJKMHz8+gwYNygsvvJCePXuu17zmzJmT6urq1NXVpaqq6oN58huo/3d/3dBTAD7BJvz0Gw09BQAAYBNY3+ax2VwTaenSpbnlllvy7rvvZtCgQZkyZUpqa2tzwAEHlMdUVlZmn332ybhx45IkEyZMyOLFi+uN6dKlS/r06VMe8+ijj6a6urockJJkjz32SHV1dXnMmixcuDBz5syp9wEAAADwSdXgEenZZ59Nq1atUllZmW9961u5/fbb07t379TW1iZJOnbsWG98x44dy/tqa2vTrFmztGnTZp1jOnTosNrjdujQoTxmTS644ILyNZSqq6vTtWvX9/U8AQAAAD7KGjwi9ezZMzU1NRk/fny+/e1v5+ijj87zzz9f3l9RUVFvfKlUWu22Va06Zk3ji45z9tlnp66urvwxbdq09X1KAAAAAB87DR6RmjVrlh122CEDBgzIBRdckF122SWXXXZZOnXqlCSrrRaaOXNmeXVSp06dsmjRosyaNWudY2bMmLHa47711lurrXJaWWVlZfld41Z8AAAAAHxSNXhEWlWpVMrChQuz7bbbplOnTrn77rvL+xYtWpQHH3wwgwcPTpL0798/TZs2rTdm+vTpmThxYnnMoEGDUldXl8cff7w85rHHHktdXV15DAAAAADr1qQhH/zf//3fc/DBB6dr166ZO3dubrnlljzwwAMZO3ZsKioqMnLkyIwePTo9evRIjx49Mnr06LRo0SJHHnlkkqS6ujrHHntszjjjjLRr1y5t27bNmWeemb59+2b//fdPkvTq1SsHHXRQjj/++Fx99dVJkhNOOCHDhw9f73dmAwAAAPika9CINGPGjIwYMSLTp09PdXV1dt5554wdOzbDhg1Lkpx11lmZP39+TjrppMyaNSsDBw7MXXfdldatW5ePcckll6RJkyb58pe/nPnz52fo0KEZM2ZMGjduXB5z00035bTTTiu/i9uhhx6aK6+88sN9sgAAAAAfYRWlUqnU0JP4KJgzZ06qq6tTV1e32Vwfqf93f93QUwA+wSb89BsNPQUAAGATWN/msdldEwkAAACAzY+IBAAAAEAhEQkAAACAQiISAAAAAIVEJAAAAAAKiUgAAAAAFBKRAAAAACgkIgEAAABQSEQCAAAAoJCIBAAAAEAhEQkAAACAQiISAAAAAIVEJAAAAAAKiUgAAAAAFBKRAAAAACgkIgEAAABQSEQCAAAAoJCIBAAAAEAhEQkAAACAQiISAAAAAIVEJAAAAAAKiUgAAAAAFBKRAAAAACgkIgEAAABQSEQCAAAAoJCIBAAAAEAhEQkAAACAQiISAAAAAIVEJAAAAAAKbVRE2m677fL222+vdvvs2bOz3Xbbve9JAQAAALB52aiINHXq1CxdunS12xcuXJg33njjfU8KAAAAgM1Lkw0Z/Mc//rH8+Z133pnq6ury9tKlS3Pvvfeme/fum2xyAAAAAGweNigiHX744UmSioqKHH300fX2NW3aNN27d89FF120ySYHAAAAwOZhgyLSsmXLkiTbbrttnnjiibRv3/4DmRQAAAAAm5cNikgrTJkyZVPPAwAAAIDN2EZFpCS59957c++992bmzJnlFUorXHvtte97YgAAAABsPjYqIp133nk5//zzM2DAgHTu3DkVFRWbel4AAAAAbEY2KiL94he/yJgxYzJixIhNPR8AAAAANkONNuZOixYtyuDBgzf1XAAAAADYTG1URDruuONy8803b+q5AAAAALCZ2qjT2RYsWJBf/vKXueeee7LzzjunadOm9fZffPHFm2RyAAAAAGweNioiPfPMM+nXr1+SZOLEifX2ucg2AAAAwMfPRkWk+++/f1PPAwAAAIDN2EZdEwkAAACAT5aNWom07777rvO0tfvuu2+jJwQAAADA5mejItKK6yGtsHjx4tTU1GTixIk5+uijN8W8AAAAANiMbFREuuSSS9Z4+6hRozJv3rz3NSEAAAAANj+b9JpIRx11VK699tpNeUgAAAAANgObNCI9+uijad68+aY8JAAAAACbgY06ne2II46ot10qlTJ9+vQ8+eST+eEPf7hJJgYAAADA5mOjIlJ1dXW97UaNGqVnz545//zzc8ABB2ySiQEAAACw+dioiHTddddt6nkAAAAAsBnbqIi0woQJEzJp0qRUVFSkd+/e+cxnPrOp5gUAAADAZmSjItLMmTPz1a9+NQ888EC23HLLlEql1NXVZd99980tt9ySrbbaalPPEwAAAIAGtFHvznbqqadmzpw5ee655/LOO+9k1qxZmThxYubMmZPTTjttU88RAAAAgAa2USuRxo4dm3vuuSe9evUq39a7d+/813/9lwtrAwAAAHwMbdRKpGXLlqVp06ar3d60adMsW7bsfU8KAAAAgM3L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ly7rHPeJiEjt27dP48aNV1t1NHPmzNVWJ61QWVmZysrKerdtueWWH9QUocFUVVX5Dx/AZsrPaIDNm5/TfJysawXSCp+IC2s3a9Ys/fv3z913313v9rvvvjuDBw9uoFkBAAAAfHR8IlYiJcl3vvOdjBgxIgMGDMigQYPyy1/+Mq+99lq+9S3X9AAAAAAo8omJSF/5ylfy9ttv5/zzz8/06dPTp0+f3HHHHenWrVtDTw0aRGVlZc4999zVTtsEoOH5GQ2wefNzmk+qilLR+7cBAAAA8In3ibgmEgAAAADvj4gEAAAAQCERCQAAAIBCIhIAwEfQqFGj0q9fv4aeBgDwCSIiwUfAMccck4qKitU+XnrppYaeGgBrMHPmzJx44onZZpttUllZmU6dOuXAAw/Mo48+uske48wzz8y99967yY4H8FFWW1ub008/PTvssEOaN2+ejh07Zq+99sovfvGLvPfeew09PfjYaNLQEwDWz0EHHZTrrruu3m1bbbVVve1FixalWbNmH+a0AFiDL37xi1m8eHGuv/76bLfddpkxY0buvffevPPOO5vsMVq1apVWrVptsuMBfFS98sor2XPPPbPllltm9OjR6du3b5YsWZK///3vufbaa9OlS5cceuihG3zcxYsXp2nTph/AjOGjy0ok+IhY8X+yV/4YOnRoTjnllHznO99J+/btM2zYsCTJxRdfnL59+6Zly5bp2rVrTjrppMybN698rDFjxmTLLbfMnXfemV69eqVVq1Y56KCDMn369HqPee2112annXZKZWVlOnfunFNOOaW8r66uLieccEI6dOiQqqqq7Lfffnn66ac/nBcDYDM2e/bsPPzww/nP//zP7LvvvunWrVt23333nH322fn85z+fJKmoqMhVV12Vgw8+OFtssUW23Xbb/O53v6t3nO9973vZcccd06JFi2y33Xb54Q9/mMWLF5f3r3o62zHHHJPDDz88P/vZz9K5c+e0a9cuJ598cr37AHwcnXTSSWnSpEmefPLJfPnLX06vXr3St2/ffPGLX8xf/vKXHHLIIUmKf39d8XP12muvzXbbbZfKysqUSqVUVFTk6quvzvDhw9OiRYv06tUrjz76aF566aUMGTIkLVu2zKBBg/Lyyy+Xj/Xyyy/nsMMOS8eOHdOqVavstttuueeee+rNu3v37hk9enS++c1vpnXr1tlmm23yy1/+srx/v/32q/f7d5K8/fbbqayszH333fdBvJRQSESCj7jrr78+TZo0ySOPPJKrr746SdKoUaNcfvnlmThxYq6//vrcd999Oeuss+rd77333svPfvaz3HDDDfm///u/vPbaaznzzDPL+6+66qqcfPLJOeGEE/Lss8/mj3/8Y3bYYYckSalUyuc///nU1tbmjjvuyIQJE7Lrrrtm6NChm/T/sgN8FK1YIfSHP/whCxcuXOu4H/7wh/niF7+Yp59+OkcddVS+9rWvZdKkSeX9rVu3zpgxY/L888/nsssuyzXXXJNLLrlknY99//335+WXX87999+f66+/PmPGjMmYMWM21VMD2Oy8/fbbueuuu3LyySenZcuWaxxTUVGx3r+/vvTSS/ntb3+b//mf/0lNTU359h//+Mf5xje+kZqamnz605/OkUcemRNPPDFnn312nnzyySSpF3zmzZuXz33uc7nnnnvy1FNP5cADD8whhxyS1157rd7cLrroogwYMCBPPfVUTjrppHz729/OCy+8kCQ57rjjcvPNN9f7b8lNN92ULl26ZN99933frx1slBKw2Tv66KNLjRs3LrVs2bL88aUvfam0zz77lPr161d4/9/+9reldu3albevu+66UpLSSy+9VL7tv/7rv0odO3Ysb3fp0qV0zjnnrPF49957b6mqqqq0YMGCerdvv/32pauvvnpDnx7Ax87vf//7Ups2bUrNmzcvDR48uHT22WeXnn766fL+JKVvfetb9e4zcODA0re//e21HvPCCy8s9e/fv7x97rnnlnbZZZfy9tFHH13q1q1bacmSJeXb/uVf/qX0la98ZRM8I4DN0/jx40tJSrfddlu929u1a1f+vfmss85ar99fzz333FLTpk1LM2fOrDcmSekHP/hBefvRRx8tJSn96le/Kt/2m9/8ptS8efN1zrV3796lK664orzdrVu30lFHHVXeXrZsWalDhw6lq666qlQqlUoLFiwotW3btnTrrbeWx/Tr1680atSodT4OfJCsRIKPiH333Tc1NTXlj8svvzxJMmDAgNXG3n///Rk2bFg+9alPpXXr1vnGN76Rt99+O++++255TIsWLbL99tuXtzt37pyZM2cmWX5B2DfffDNDhw5d41wmTJiQefPmpV27duX/496qVatMmTKl3jJegE+qL37xi3nzzTfzxz/+MQceeGAeeOCB7LrrrvVWBQ0aNKjefQYNGlRvJdLvf//77LXXXunUqVNatWqVH/7wh6v9H+xV7bTTTmncuHF5e+Wf7QAfZxUVFfW2H3/88dTU1GSnnXbKwoUL1/v3127duq123dEk2Xnnncufd+zYMUnSt2/ferctWLAgc+bMSZK8++67Oeuss9K7d+9sueWWadWqVV544YXVfo6vfNyKiop06tSp/HO7srIyRx11VK699tokSU1NTZ5++ukcc8wxG/MSwSbhwtrwEdGyZcvy6WSr3r6yV199NZ/73OfyrW99Kz/+8Y/Ttm3bPPzwwzn22GPrXRdj1YsErljmmyRbbLHFOueybNmydO7cOQ888MBq+7bccsv1fEYAH2/NmzfPsGHDMmzYsPzoRz/Kcccdl3PPPXedv/yv+CNo/Pjx+epXv5rzzjsvBx54YKqrq3PLLbfkoosuWudjruln+7Jly973cwHYXO2www6pqKgonwK2wnbbbZfkn7/Xru/vr2s7JW7ln68rflav6bYVP3O/+93v5s4778zPfvaz7LDDDtliiy3ypS99KYsWLVrrcVccZ+Wf28cdd1z69euX119/Pddee22GDh2abt26rXGO8GEQkeBj5sknn8ySJUty0UUXpVGj5YsNf/vb327QMVq3bp3u3bvn3nvvXeP51rvuumtqa2vTpEmTdO/efVNMG+Bjr3fv3vnDH/5Q3h4/fny+8Y1v1Nv+zGc+kyR55JFH0q1bt5xzzjnl/a+++uqHNleAj4p27dpl2LBhufLKK3PqqaeuNQJ92L+/PvTQQznmmGPyhS98IcnyayRNnTp1g4/Tt2/fDBgwINdcc01uvvnmXHHFFZt4prBhnM4GHzPbb799lixZkiuuuCKvvPJKbrjhhvziF7/Y4OOMGjUqF110US6//PK8+OKL+dvf/lb+j9b++++fQYMG5fDDD8+dd96ZqVOnZty4cfnBD35QvrAgwCfV22+/nf322y833nhjnnnmmUyZMiW/+93vcuGFF+awww4rj/vd736Xa6+9Nn//+99z7rnn5vHHHy9flHWHHXbIa6+9lltuuSUvv/xyLr/88tx+++0N9ZQANms///nPs2TJkgwYMCC33nprJk2alMmTJ+fGG2/MCy+8kMaNG3/ov7/usMMOue2228qnoB155JEbvTL0uOOOy09+8pMsXbq0HKWgoYhI8DHTr1+/XHzxxfnP//zP9OnTJzfddFMuuOCCDT7O0UcfnUsvvTQ///nPs9NOO2X48OF58cUXkyxfZnvHHXdk7733zje/+c3suOOO+epXv5qpU6eWzxEH+KRq1apVBg4cmEsuuSR77713+vTpkx/+8Ic5/vjjc+WVV5bHnXfeebnllluy88475/rrr89NN92U3r17J0kOO+yw/Nu//VtOOeWU9OvXL+PGjcsPf/jDhnpKAJu17bffPk899VT233//nH322dlll10yYMCAXHHFFTnzzDPz4x//+EP//fWSSy5JmzZtMnjw4BxyyCE58MADs+uuu27Usb72ta+lSZMmOfLII9O8efNNPFPYMBWlFRdBAQDgQ1FRUZHbb789hx9+eENPBYDN3LRp09K9e/c88cQTGx2iYFNxTSQAAADYzCxevDjTp0/P97///eyxxx4CEpsFp7MBAADAZmbFmyxMmDBho65xCh8Ep7MBAAAAUMhKJAAAAAAKiUgAAAAAFBKRAAAAACgkIgEAAABQSEQCAAAAoJCIBACwCXTv3j2XXnppgz3+kCFDMnLkyE1+3FGjRqVfv37l7WOOOSaHH374Jn+cNT0WALB5EZEAgA9MbW1tTj/99Oywww5p3rx5OnbsmL322iu/+MUv8t577zX09DZ7Y8aMSUVFRSoqKtK4ceO0adMmAwcOzPnnn5+6urp6Y2+77bb8+Mc/Xq/jbkhwOvPMM3Pvvfdu6NQLVVRU5A9/+MOH8lgAwKbRpKEnAAB8PL3yyivZc889s+WWW2b06NHp27dvlixZkr///e+59tpr06VLlxx66KENOsfFixenadOmDTqHIlVVVZk8eXJKpVJmz56dcePG5YILLsh1112XRx55JF26dEmStG3bdpM+bqlUytKlS9OqVau0atVqkx57bT7MxwIANpyVSADAB+Kkk05KkyZN8uSTT+bLX/5yevXqlb59++aLX/xi/vKXv+SQQw4pj62rq8sJJ5yQDh06pKqqKvvtt1+efvrpese76qqrsv3226dZs2bp2bNnbrjhhnr7X3jhhey1115p3rx5evfunXvuuafeapepU6emoqIiv/3tbzNkyJA0b948N954Y95+++187Wtfy9Zbb50WLVqkb9+++c1vflPv2EOGDMkpp5ySU045JVtuuWXatWuXH/zgBymVSvXGvffee/nmN7+Z1q1bZ5tttskvf/nL8r799tsvp5xySr3xb7/9diorK3Pfffet9XWsqKhIp06d0rlz5/Tq1SvHHntsxo0bl3nz5uWss86qN8eVVxf9/Oc/T48ePcorwL70pS8lWX462oMPPpjLLrusvMpp6tSpeeCBB1JRUZE777wzAwYMSGVlZR566KG1nmJ23nnnlb9eJ554YhYtWlTet6ZT+/r165dRo0aV9yfJF77whVRUVJS3V32sZcuW5fzzz8/WW2+dysrK9OvXL2PHji3vX/E1ve2227LvvvumRYsW2WWXXfLoo4+u9fUEADaeiAQAbHJvv/127rrrrpx88slp2bLlGsdUVFQkWb7i5fOf/3xqa2tzxx13ZMKECdl1110zdOjQvPPOO0mS22+/PaeffnrOOOOMTJw4MSeeeGL+9V//Nffff3+S5bHh8MMPT4sWLfLYY4/ll7/8Zc4555w1Pu73vve9nHbaaZk0aVIOPPDALFiwIP3798+f//znTJw4MSeccEJGjBiRxx57rN79rr/++jRp0iSPPfZYLr/88lxyySX57//+73pjLrroogwYMCBPPfVUTjrppHz729/OCy+8kCQ57rjjcvPNN2fhwoXl8TfddFO6dOmSfffdd4Ne3w4dOuTrX/96/vjHP2bp0qWr7X/yySdz2mmn5fzzz8/kyZMzduzY7L333kmSyy67LIMGDcrxxx+f6dOnZ/r06enatWv5vmeddVYuuOCCTJo0KTvvvPMaH//ee+/NpEmTcv/99+c3v/lNbr/99px33nnrPf8nnngiSXLddddl+vTp5e1VXXbZZbnooovys5/9LM8880wOPPDAHHrooXnxxRfrjTvnnHNy5plnpqamJjvuuGO+9rWvZcmSJes9HwBgPZUAADax8ePHl5KUbrvttnq3t2vXrtSyZctSy5YtS2eddVapVCqV7r333lJVVVVpwYIF9cZuv/32pauvvrpUKpVKgwcPLh1//PH19v/Lv/xL6XOf+1ypVCqV/vrXv5aaNGlSmj59enn/3XffXUpSuv3220ulUqk0ZcqUUpLSpZdeWjj/z33uc6UzzjijvL3PPvuUevXqVVq2bFn5tu9973ulXr16lbe7detWOuqoo8rby5YtK3Xo0KF01VVXlUqlUmnBggWltm3blm699dbymH79+pVGjRq11nlcd911perq6jXuu+qqq0pJSjNmzCjP8fTTTy+VSqXS//zP/5SqqqpKc+bMWeN9Vx67wv33319KUvrDH/5Q7/Zzzz23tMsuu5S3jz766FLbtm1L7777br25tGrVqrR06dLya3HJJZfUO84uu+xSOvfcc8vbK39t1vZYXbp0Kf3Hf/xHvTG77bZb6aSTTiqVSv/8mv73f/93ef9zzz1XSlKaNGnSGp87ALDxrEQCAD4wK1YbrfD444+npqYmO+20U3lFzoQJEzJv3ry0a9eufE2cVq1aZcqUKXn55ZeTJJMmTcqee+5Z71h77rlnJk2alCSZPHlyunbtmk6dOpX377777muc04ABA+ptL126NP/xH/+RnXfeuTyHu+66K6+99lq9cXvssUe95zNo0KC8+OKL9VYCrbxyZ8VpaDNnzkySVFZW5qijjsq1116bJKmpqcnTTz+dY445Zi2v3rqV/v9T6VZ9jZNk2LBh6datW7bbbruMGDEiN91003pfyHzV12dNdtlll7Ro0aK8PWjQoMybNy/Tpk1bz9kXmzNnTt588811ft1XWPl179y5c5KUX3cAYNNxYW0AYJPbYYcdUlFRUT6Va4XtttsuSbLFFluUb1u2bFk6d+6cBx54YLXjbLnlluXPV40lpVKp3ilxa4opa7Lq6XUXXXRRLrnkklx66aXp27dvWrZsmZEjR9a7xs/6WvUi3RUVFVm2bFl5+7jjjku/fv3y+uuv59prr83QoUPTrVu3DX6cZHlYq6qqSrt27Vbb17p16/ztb3/LAw88kLvuuis/+tGPMmrUqDzxxBP1XtM1Wdvph+tjxdegUaNGq10vavHixe/rmCus6Wu98uu+Yt/KrzsAsGlYiQQAbHLt2rXLsGHDcuWVV+bdd99d59hdd901tbW1adKkSXbYYYd6H+3bt0+S9OrVKw8//HC9+40bNy69evVKknz605/Oa6+9lhkzZpT3r+06O6t66KGHcthhh+Woo47KLrvsku222261a+4kyfjx41fb7tGjRxo3brxej5Mkffv2zYABA3LNNdfk5ptvzje/+c31vu/KZs6cmZtvvjmHH354GjVa869zTZo0yf77758LL7wwzzzzTKZOnVq+gHezZs3WeC2l9fX0009n/vz55e3x48enVatW2XrrrZMkW221VaZPn17eP2fOnEyZMqXeMZo2bbrOOVRVVaVLly7r/LoDAB8uEQkA+ED8/Oc/z5IlSzJgwIDceuutmTRpUiZPnpwbb7wxL7zwQjm+7L///hk0aFAOP/zw3HnnnZk6dWrGjRuXH/zgB3nyySeTJN/97nczZsyY/OIXv8iLL76Yiy++OLfddlvOPPPMJMtP39p+++1z9NFH55lnnskjjzxSvrB20QqlHXbYIXfffXfGjRuXSZMm5cQTT0xtbe1q46ZNm5bvfOc7mTx5cn7zm9/kiiuuyOmnn77Br8txxx2Xn/zkJ1m6dGm+8IUvFI4vlUqpra3N9OnTM2nSpFx77bUZPHhwqqur85Of/GSN9/nzn/+cyy+/PDU1NXn11Vfz61//OsuWLUvPnj2TLH93tMceeyxTp07NP/7xjw1etbNo0aIce+yxef755/PXv/415557bk455ZRy0Npvv/1yww035KGHHsrEiRNz9NFHrxbbunfvnnvvvTe1tbWZNWvWGh/nu9/9bv7zP/8zt956ayZPnpzvf//7qamp2ajXHQB4/5zOBgB8ILbffvs89dRTGT16dM4+++y8/vrrqaysTO/evXPmmWfmpJNOSrI88txxxx0555xz8s1vfjNvvfVWOnXqlL333jsdO3ZMkhx++OG57LLL8tOf/jSnnXZatt1221x33XUZMmRIkqRx48b5wx/+kOOOOy677bZbtttuu/z0pz/NIYcckubNm69znj/84Q8zZcqUHHjggWnRokVOOOGEHH744amrq6s37hvf+Ebmz5+f3XffPY0bN86pp56aE044YYNfl6997WsZOXJkjjzyyMK5JctX8XTu3DkVFRWpqqpKz549c/TRR+f0009PVVXVGu+z5ZZb5rbbbsuoUaOyYMGC9OjRI7/5zW+y0047JUnOPPPMHH300endu3fmz5+/2iqhIkOHDk2PHj2y9957Z+HChfnqV7+aUaNGlfefffbZeeWVVzJ8+PBUV1fnxz/+8WqPcdFFF+U73/lOrrnmmnzqU5/K1KlTV3uc0047LXPmzMkZZ5yRmTNnpnfv3vnjH/+YHj16bNB8AYBNo6K06gnrAAAfA4888kj22muvvPTSS9l+++3f17GGDBmSfv365dJLL33f85o2bVq6d++eJ554Irvuuuv7Ph4AwIfFSiQA4GPh9ttvT6tWrdKjR4+89NJLOf3007Pnnnu+74C0qSxevDjTp0/P97///eyxxx4CEgDwkSMiAQAfC3Pnzs1ZZ52VadOmpX379tl///1z0UUXNfS0yh555JHsu+++2XHHHfP73/++oacDALDBnM4GAAAAQCHvzgYAAABAIREJAAAAgEIiEgAAAACFRCQAAAAAColIAAAAABQSkQAAAAAoJCIBAAAAUEhEAgAAAKDQ/wd/Ba8Qqf+tfgAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 1400x1000 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(14,10))\n",
    "sns.countplot(x='Geography', data=df)\n",
    "plt.xlabel('Geography Distribution')\n",
    "plt.ylabel('Count')\n",
    "plt.title('Geography Distribution Plot',fontsize=14, fontweight=\"bold\", color = \"purple\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "702e36d0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.FacetGrid at 0x1d47b4cebd0>"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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JpMJ1dXVyu1HH30U/fv3x/nz57FhwwaMGjUK4eHhTf49RGphi5pIR4KCgjBmzBisXbsWV69evWG5gQMHIjc3Fx4eHujWrZvdJTg4GADQu3dv7N271+7n9u3bh969ewMAevXqhczMTFy8eFF5/PDhw42q57fffot7770X06ZNQ79+/dClSxecPn26XrkDBw7Uu929e3e4u7s36vcAQHR0NAYNGoR3330XH374IR555JFG/yyRFjCoiXTmrbfewvXr1zFo0CB8/PHHSEtLw8mTJ7F582akp6fD3d0do0ePRmxsLCZPnowvvvgCZ8+exb59+/D888/j+++/BwA8/fTT2LhxI9atW4fTp09j5cqV2Lp1KxYsWAAAGDNmDLp27YqHHnoIx44dw3fffYfnnnsOQP2WeF3dunXDzp07sW/fPqSlpeGxxx5Dbm5uvXJZWVl46qmncPLkSXz00UdYs2YN5s6d2+S/yaOPPopXXnkFVVVV+M1vftPknydSE4OaSGe6du2KH374AaNHj8aiRYvQr18/DBo0CGvWrMGCBQvw17/+FSaTCdu3b8ddd92FRx55BD169MD999+Ps2fPIjQ0FAAwefJk/O1vf8Nrr72GPn36YP369Xj//fcxcuRIAIC7uzs+++wzlJSU4I477sCjjz6K559/HgDg7e190zq+8MILGDhwIMaNG4eRI0fCarVi8uTJ9co9+OCDKC0txZ133oknnngCs2fPxsyZM5v8N3nggQfg4eGB+Pj4W9aNSGtMQtQ5KZGIqJm+++47DB8+HGfOnEHXrl1b9FwjR45E//79HbJEaFZWFiIiInD48GEMHDiwxc9H5EycTEZEzbZt2za0adMG3bt3x5kzZzB37lwMGzasxSHtKJWVlcjJycGzzz6LIUOGMKTJJTGoiajZrly5goULFyIrKwvBwcEYPXo0VqxYoXa1FN999x3uvvtu9OjRA//85z/Vrg5Rs7Drm4iISMM4mYyIiEjDGNREREQaxqAmIiLSMAY1ERGRhjGoiYiINIxBTUREpGEMaiIiIg1jUBMREWnY/wcSVXMnUohhSwAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 500x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.displot(df.Geography, kde =True, color = \"red\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "85cd62b1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.FacetGrid at 0x1d47b6c79d0>"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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atSo1vYH6YsvvtAdd9yhxo0ba/LkyYqPj9fZs2f12Wefaf78+YqOjlbfvn2veL9nzpxRvXr1rsKMcUVq4Vdp43tmwIAB5v7776+yvmvXrmb48OHm17/+tYmIiDB33XWXMcaYadOmmfbt25uGDRuam2++2Tz55JPm+PHjzv0WLFhg3G63yc7ONrfddpsJCQkxqamp5quvvvLZ/6uvvmratm1r6tevbzwejxk+fLiz7dixY2bw4MHmxhtvNKGhoaZ79+4mPz//6nwB6qji4mIjybz33nsXHSPJvPzyy6Z3794mODjYNG/e3Lzxxhs+Y8aOHWtiY2NNgwYNTIsWLcyECRNMeXm5s33SpEnm9ttvd5bP/X17/vnnjcfjMeHh4WbYsGE+98GlpaammptvvtmUlpZecHtlZaUx5ru/j849N6+++qpp0aKFcblcprKy0kgyc+bMMT/96U9NgwYNzG233WZycnLMrl27TNeuXU3Dhg1N586dze7du5197d692/Tt29dERkaakJAQk5iYaNasWeMzr2bNmpk//vGP5j/+4z9Mo0aNTExMjJk7d66zvXv37j7/DhhjzJEjR0z9+vXNunXravx1u17w0jeuyKJFixQYGKgPPvhAc+fOlSTdcMMNevHFF7Vt2zYtWrRI77zzjsaOHetzv5MnT+rPf/6zFi9erH/84x/av3+/xowZ42yfPXu2hg8friFDhujTTz/VW2+9pdatW0uSjDH66U9/qsLCQq1atUp5eXnq1KmTevToUatnenXduTPdN9980+eT4843ceJEPfjgg/r444/Vv39/PfrooyooKHC2h4aGauHChfrnP/+pF154QfPmzdOMGTMu+djvvvuuPv/8c7377rtatGiRFi5cqIULF9bWoX2vHT16VKtXr9bw4cMVEhJywTEul+uyv492796tN954Q3/729+Un5/vrH/22Wf12GOPKT8/X7fddpvS0tI0dOhQjRs3Tlu2bJEkjRgxwhlfWlqqe++9V2vXrtVHH32k1NRU3Xfffdq/f7/P3KZNm6bExER99NFHGjZsmJ588knt2LFDkjRo0CAtW7bM5+/j0qVLFR0dre7du9f4a3fd8Pf/FGCfAQMGmICAABMSEuLcfv7zn5uuXbuajh07fuf933jjDRMREeEsL1iwwEjy+d/2Sy+9ZKKiopzl6OhoM378+Avub926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",
      "text/plain": [
       "<Figure size 500x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.displot(df.Geography, color = \"yellow\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "d50ba296",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABK4AAANrCAYAAABvGND2AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAABPyUlEQVR4nO3deZyVdd3/8fdh30EUARMXxAUV3MANdyQXxOV2yS0jMxWXXHMrcsn0rlzTXDIXNLXUzEy9U0LQXAMXFPcdTVFTAUVDkPP7wx+TI4jMODlf4Pl8POZR5zrXuc7nzDBdZ15d13Uq1Wq1GgAAAAAoTJPGHgAAAAAA5ka4AgAAAKBIwhUAAAAARRKuAAAAACiScAUAAABAkYQrAAAAAIokXAEAAABQJOEKAAAAgCIJVwAAAAAUSbgCABrEY489lu9973tZYYUV0rp167Ru3TorrrhiDjjggIwbN66xx2s0lUolhxxyyH/1OcaMGZNKpVLz1aJFi3Tp0iUDBgzIj370o7zyyitzPOaKK65IpVLJyy+/XKfnOu2003LTTTfV6TFze67NNtssq6++ep2282Vuu+22nHTSSXO9b7nllsvQoUMb9Pnmx+d/Nk2bNk3Xrl2z66675qmnnqpZ7+WXX06lUskVV1xR5+d48sknc9JJJ9X5ZwkACwLhCgD4yi6++OKss846efDBB3PYYYfllltuya233prDDz88TzzxRPr3758XXnihscdc6J122mm5//77M3r06Fx66aXZbLPNctlll6V37965+uqra607ePDg3H///enevXudn6Ou4aq+z1VXt912W04++eS53venP/0pw4cP/68+/7x89mdz7LHHZuTIkRkwYED++c9/fuVtP/nkkzn55JOFKwAWSs0aewAAYMF277335qCDDsrgwYNzww03pEWLFjX3bbHFFjn44INz/fXXp3Xr1o045bx9+OGHadOmTWOP8ZWtuOKKWX/99Wtub7/99jnqqKOy5ZZbZujQoenbt2/69OmTJOnSpUu6dOnyX53no48+SqtWrb6W5/oya621VqM+/2d/Nptsskk6deqU733ve7niiivyox/9qFFnA4CSOeIKAPhKTjvttDRt2jQXX3xxrWj1WbvuumuWWmqpWsvGjRuX7bffPp07d06rVq2y1lpr5brrrpvjsRMmTMgOO+yQxRZbLK1atcqaa66ZESNGzLHeE088kW9+85tp06ZNunTpkoMPPji33nprKpVKxowZU7Pe7FPU7r777my44YZp06ZN9t133yTJH/7wh3zzm99M9+7d07p16/Tu3TvHHXdcpk2bVuu5hg4dmnbt2uWJJ57IwIED07Zt23Tp0iWHHHJIPvzww7l+D6666qr07t07bdq0yRprrJFbbrml5r6///3vqVQqufbaa+d43JVXXplKpZKxY8fOdbtfpnPnzrn44oszc+bMnH322TXL53b63iOPPJLtttsuSy65ZFq2bJmllloqgwcPzmuvvZbk09Mep02blhEjRtSc+rbZZpvV2t4dd9yRfffdN126dEmbNm0yffr0eZ6W+Pe//z3rr79+WrdunW984xsZPnx4Pvnkk5r7Z59q99mfYTLnqXVDhw7Nr3/965o5Z3/Nfs65nSo4ceLE7L333jWvt3fv3jnzzDMza9asOZ7njDPOyFlnnZXll18+7dq1ywYbbJAHHnigDj+J2mZHrLmdxvlZ99xzTwYOHJj27dunTZs22XDDDXPrrbfW3H/FFVdk1113TZJsvvnmNa+7PqccAkCJhCsAoN4++eSTjB49Ov369avTaWCjR4/OgAEDMnny5Fx00UX585//nDXXXDPf+ta3av3B/cwzz2TDDTfME088kV/96le58cYbs+qqq2bo0KH5xS9+UbPeG2+8kU033TTPPPNMLrzwwlx55ZV5//33v/DaUm+88Ub23nvv7Lnnnrntttty0EEHJUmee+65bLvttrn00kvz17/+NYcffniuu+66DBkyZI5tzJgxI9tuu20GDhyYm266KYccckguvvjifOtb35pj3VtvvTXnn39+TjnllPzxj39M586ds9NOO+XFF19Mkmy88cZZa621asLLZ51//vnp379/+vfvP9/f38/r379/unfvnrvvvvsL15k2bVoGDRqUN998M7/+9a8zcuTInHPOOVlmmWXy/vvvJ0nuv//+tG7dOttuu23uv//+3H///bngggtqbWffffdN8+bNc9VVV+WGG25I8+bNv/A5J02alN133z177bVX/vznP2eXXXbJqaeemsMOO6zOr3H48OHZZZddauac/fVF/y7ffvvtbLjhhrnjjjvy05/+NDfffHO23HLLHH300XP9d/PZ78nVV1+dadOmZdttt82UKVPqPGuSPP/880kyzyPR7rrrrmyxxRaZMmVKLr300lx77bVp3759hgwZkj/84Q9JPj0N87TTTquZcfbrHjx4cL3mAoDiVAEA6mnSpEnVJNXdd999jvtmzpxZnTFjRs3XrFmzau5bZZVVqmuttVZ1xowZtR6z3XbbVbt371795JNPqtVqtbr77rtXW7ZsWZ04cWKt9bbZZptqmzZtqpMnT65Wq9XqD3/4w2qlUqk+8cQTtdbbaqutqkmqo0ePrlm26aabVpNUR40aNc/XNmvWrOqMGTOqd911VzVJdfz48TX3fec736kmqZ577rm1HvOzn/2smqR6zz331CxLUu3atWt16tSptb5vTZo0qZ5++uk1yy6//PJqkuojjzxSs+wf//hHNUl1xIgR85x19OjR1STV66+//gvXWW+99aqtW7ee4/leeumlarVarY4bN66apHrTTTfN87natm1b/c53vjPH8tnb22effb7wvtnPVa3+5+fw5z//uda63//+96tNmjSpvvLKK7Ve22d/htVqtfrSSy9Vk1Qvv/zymmUHH3xw9Yve3i677LK15j7uuOOqSaoPPvhgrfWGDRtWrVQq1WeeeabW8/Tp06c6c+bMmvVm/2yuvfbauT7fbLPn/8Mf/lCdMWNG9cMPP6zefffd1V69elWbNm1a8+9qbq9n/fXXry655JLV999/v2bZzJkzq6uvvnp16aWXrvmduv766+f6PQKAhYEjrgCA/4p11lknzZs3r/k688wzk3x6pMnTTz+dvfbaK0kyc+bMmq9tt902b7zxRp555pkkyZ133pmBAwemR48etbY9dOjQfPjhh7n//vuTfHpkyuqrr55VV1211np77LHHXGdbbLHFssUWW8yx/MUXX8yee+6Zbt26pWnTpmnevHk23XTTJKn1CXCzzX4Ns+25555JPj2i7LM233zztG/fvuZ2165ds+SSS9Y6TWyPPfbIkksuWeuoq/POOy9dunSZ61FcdVWtVud5f69evbLYYovl2GOPzUUXXZQnn3yyXs+z8847z/e67du3z/bbb19r2Z577plZs2bN8+iwhnDnnXdm1VVXzbrrrltr+dChQ1OtVnPnnXfWWj548OA0bdq05nbfvn2TfPmpfrN961vfSvPmzdOmTZtssskm+eSTT3LDDTfUbOfzpk2blgcffDC77LJL2rVrV7O8adOm+fa3v53XXnut5vcEABZmwhUAUG9LLLFEWrduPdc/3q+55pqMHTs2N998c63lb775ZpLk6KOPrhW2mjdvXnPK3r/+9a8kyTvvvDPXU71mXy/rnXfeqfnPrl27zrHe3JYlmes2P/jgg2y88cZ58MEHc+qpp2bMmDEZO3ZsbrzxxiSfXmj8s5o1a5bFF1+81rJu3brVmmu2z6+XJC1btqy1zZYtW+aAAw7INddck8mTJ+ftt9/Oddddl/322y8tW7ac6+uoi4kTJ85xnbHP6tixY+66666sueaaOeGEE7LaaqtlqaWWyoknnpgZM2bM9/PU5ZTRuf18vuh72NDm99/WbJ//Gc7+mXz+38UX+fnPf56xY8fm4YcfzsSJE/Piiy9mxx13/ML133vvvVSr1TrNCAALI58qCADUW9OmTbPFFlvkjjvuyBtvvFHrj+zZRz99/oLcSyyxRJLk+OOPz//8z//Mdbsrr7xykk9jwRtvvDHH/a+//nqtbS2++OI1QeyzJk2aNNftVyqVOZbdeeedef311zNmzJiao6ySZPLkyXPdxsyZM/POO+/UChqzn29uoWp+DBs2LP/7v/+byy67LP/+978zc+bMHHjggfXa1mf94x//yKRJk/K9731vnuv16dMnv//971OtVvPYY4/liiuuyCmnnJLWrVvnuOOOm6/nmtv39ovM62c2+3vYqlWrJMn06dNrrTc7btbX/P7baig9e/ZMv3795nv9xRZbLE2aNPlaZwSAEjniCgD4So4//vh88sknOfDAA+fryJyVV145K664YsaPH59+/frN9Wv2aXUDBw6sCUqfdeWVV6ZNmzY1n8y26aabZsKECXOc3vb73/9+vl/H7ODy+aObLr744i98zNVXX13r9jXXXJMkNZ+0V1fdu3fPrrvumgsuuCAXXXRRhgwZkmWWWaZe25rt3XffzYEHHpjmzZvniCOOmK/HVCqVrLHGGjn77LPTqVOnPPzwwzX3ff5Isa/i/fffn+OIvGuuuSZNmjTJJptskuTTTwNMkscee6zWep9/3OzZkvk7CmrgwIF58skna7225D+f4rj55pvP9+v4b2jbtm3WW2+93HjjjbVez6xZs/K73/0uSy+9dFZaaaUkdT/6CwAWJI64AgC+kgEDBuTXv/51Dj300Ky99trZf//9s9pqq9UcLfLHP/4xSdKhQ4eax1x88cXZZpttstVWW2Xo0KH5xje+kXfffTdPPfVUHn744Vx//fVJkhNPPDG33HJLNt988/zkJz9J586dc/XVV+fWW2/NL37xi3Ts2DFJcvjhh+eyyy7LNttsk1NOOSVdu3bNNddck6effjpJ0qTJl/9/dRtuuGEWW2yxHHjggTnxxBPTvHnzXH311Rk/fvxc12/RokXOPPPMfPDBB+nfv3/uu+++nHrqqdlmm22y0UYb1fv7edhhh2W99dZLklx++eV1euxzzz2XBx54ILNmzco777yTBx98MJdeemmmTp2aK6+8MqutttoXPvaWW27JBRdckB133DE9e/ZMtVrNjTfemMmTJ2fQoEE16/Xp0ydjxozJX/7yl3Tv3j3t27evOUKurhZffPEMGzYsEydOzEorrZTbbrstl1xySYYNG1YT7Lp165Ytt9wyp59+ehZbbLEsu+yyGTVqVM0pnJ/Vp0+fJJ+elrfNNtukadOm6du3b1q0aDHHukcccUSuvPLKDB48OKecckqWXXbZ3HrrrbngggsybNiwmijUmE4//fQMGjQom2++eY4++ui0aNEiF1xwQSZMmJBrr722JrauvvrqSZLf/OY3ad++fVq1apXll1++3kf+AUBRGvXS8ADAQuPRRx+tfve7360uv/zy1ZYtW1ZbtWpV7dWrV3WfffaZ6yf4jR8/vrrbbrtVl1xyyWrz5s2r3bp1q26xxRbViy66qNZ6jz/+eHXIkCHVjh07Vlu0aFFdY401an3y2mwTJkyobrnlltVWrVpVO3fuXP3e975XHTFixByfCLjppptWV1tttbm+hvvuu6+6wQYbVNu0aVPt0qVLdb/99qs+/PDDc3za23e+851q27Ztq4899lh1s802q7Zu3brauXPn6rBhw6offPBBrW0mqR588MFzPNfnP+Xus5Zbbrlq796953rf3Mz+5LrZX82aNasuvvji1Q022KB6wgknVF9++eU5HvP5T/p7+umnq3vssUd1hRVWqLZu3brasWPH6rrrrlu94ooraj3u0UcfrQ4YMKDapk2bapLqpptuWmt7Y8eO/dLnqlb/83MYM2ZMtV+/ftWWLVtWu3fvXj3hhBPm+LTJN954o7rLLrtUO3fuXO3YsWN17733rvkUxM/+XKZPn17db7/9ql26dKlWKpVazzm37/crr7xS3XPPPauLL754tXnz5tWVV165+stf/rLmUy2r1f98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2WmnnfLWW2+lZcuWNct79eqVY445Jvvvv/8cj5k+fXqmT59ec3vq1Knp0aNH1jj0ojRt2boO3w0AAABgYfPQL/dp7BEazNSpU9OxY8dMmTIlHTp0mOe6C801rjbffPNceOGFNbfbtm2bPfbYI/369Ztj3dGjR+e0007Lk08+malTp2bmzJn597//nWnTpqVt27ZJkjZt2tREqyTp3r173nrrrSTJW2+9lddffz0DBw6c6ywPPfRQPvjggyy++OK1ln/00Ud54YUX5vqY008/PSeffHLdXjQAAADAQmyhCVdt27atOVXv88s/65VXXsm2226bAw88MD/96U/TuXPn3HPPPfne976XGTNm1KzXvHnzWo+rVCqZfXBa69bzPgJq1qxZ6d69e8aMGTPHfZ06dZrrY44//vgceeSRNbdnH3EFAAAAsKhaaMLV/Bo3blxmzpyZM888M02afHqJr+uuu65O22jfvn2WW265jBo1Kptvvvkc96+99tqZNGlSmjVrluWWW26+ttmyZctapxUCAAAALOoW+ouzf94KK6yQmTNn5rzzzsuLL76Yq666KhdddFGdt3PSSSflzDPPzK9+9as899xzefjhh3PeeeclSbbccstssMEG2XHHHXP77bfn5Zdfzn333Zcf//jHGTduXEO/JAAAAICF0iIXrtZcc82cddZZ+fnPf57VV189V199dU4//fQ6b+c73/lOzjnnnFxwwQVZbbXVst122+W5555L8ulphbfddls22WST7LvvvllppZWy++675+WXX07Xrl0b+iUBAAAALJQWik8VXBjNvsK+TxUEAAAAFtVPFVzkjrgCAAAAYMEgXAEAAABQJOEKAAAAgCIJVwAAAAAUSbgCAAAAoEjCFQAAAABFEq4AAAAAKJJwBQAAAECRhCsAAAAAiiRcAQAAAFAk4QoAAACAIglXAAAAABRJuAIAAACgSMIVAAAAAEUSrgAAAAAoknAFAAAAQJGEKwAAAACKJFwBAAAAUCThCgAAAIAiCVcAAAAAFEm4AgAAAKBIwhUAAAAARRKuAAAAACiScAUAAABAkYQrAAAAAIokXAEAAABQJOEKAAAAgCIJVwAAAAAUSbgCAAAAoEjCFQAAAABFEq4AAAAAKJJwBQAAAECRhCsAAAAAiiRcAQAAAFAk4QoAAACAIglXAAAAABRJuAIAAACgSMIV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4447GnoWAAAAAKgx36cK/upXv6r577169crw4cPzwAMPpE+fPmnevHmtdX/wgx803IQAAAAALJIq1Wq1Oj8rLr/88vO3wUolL7744lcaimTq1Knp2LFjJhzfO+1bNW3scQCARrTMTx5v7BEAABrM7OYxZcqUdOjQYZ7rzvcRVy+99NJXHgwAAAAA5le9rnH1WdVqNfN50BYAAAAAzLd6h6tLL700q6++elq1apVWrVpl9dVXz29/+9uGnA0AAACARdh8nyr4WcOHD8/ZZ5+dQw89NBtssEGS5P77788RRxyRl19+OaeeemqDDgkAAADAoqde4erCCy/MJZdckj322KNm2fbbb5++ffvm0EMPFa4AAAAA+MrqdargJ598kn79+s2xfJ111snMmTO/8lAAAAAAUK9wtffee+fCCy+cY/lvfvOb7LXXXl95KAAAAACo16mCyacXZ7/jjjuy/vrrJ0keeOCBvPrqq9lnn31y5JFH1qx31llnffUpAQAAAFjk1CtcTZgwIWuvvXaS5IUXXkiSdOnSJV26dMmECRNq1qtUKg0wIgAAAACLonqFq9GjRzf0HAAAAABQS72ucQUAAAAA/231vsbV2LFjc/3112fixIn5+OOPa9134403fuXBAAAAAFi01euIq9///vcZMGBAnnzyyfzpT3/KjBkz8uSTT+bOO+9Mx44dG3pGAAAAABZB9QpXp512Ws4+++zccsstadGiRc4999w89dRT2W233bLMMss09IwAAAAALILqFa5eeOGFDB48OEnSsmXLTJs2LZVKJUcccUR+85vfNOiAAAAAACya6hWuOnfunPfffz9J8o1vfCMTJkxIkkyePDkffvhhw00HAAAAwCKrXuFq4403zsiRI5Mku+22Ww477LB8//vfzx577JGBAwfWaVuTJk3KYYcdll69eqVVq1bp2rVrNtpoo1x00UUiGAAAAMAirF6fKnj++efn3//+d5Lk+OOPT/PmzXPPPffkf/7nfzJ8+PD53s6LL76YAQMGpFOnTjnttNPSp0+fzJw5M88++2wuu+yyLLXUUtl+++3rPN+MGTPSvHnzOj8OAAAAgHLU+YirmTNn5i9/+UuaNPn0oU2aNMkxxxyTm2++OWeddVYWW2yx+d7WQQcdlGbNmmXcuHHZbbfd0rt37/Tp0yc777xzbr311gwZMiRJMmXKlOy///5Zcskl06FDh2yxxRYZP358zXZOOumkrLnmmrnsssvSs2fPtGzZMtVqNZVKJRdffHG22267tGnTJr17987999+f559/Pptttlnatm2bDTbYIC+88ELNtl544YXssMMO6dq1a9q1a5f+/fvnb3/7W625l1tuuZx22mnZd9990759+yyzzDK1ru21xRZb5JBDDqn1mHfeeSctW7bMnXfeOf/fbAAAAIBFWJ3DVbNmzTJs2LBMnz79Kz3xO++8kzvuuCMHH3xw2rZtO9d1KpVKqtVqBg8enEmTJuW2227LQw89lLXXXjsDBw7Mu+++W7Pu888/n+uuuy5//OMf8+ijj9Ys/+lPf5p99tknjz76aFZZZZXsueeeOeCAA3L88cdn3LhxSVIrMn3wwQfZdttt87e//S2PPPJIttpqqwwZMiQTJ06sNduZZ56Zfv365ZFHHslBBx2UYcOG5emnn06S7LfffrnmmmtqfY+uvvrqLLXUUtl8883n+lqnT5+eqVOn1voCAAAAWJTV6xpX6623Xh555JGv9MTPP/98qtVqVl555VrLl1hiibRr1y7t2rXLsccem9GjR+fxxx/P9ddfn379+mXFFVfMGWeckU6dOuWGG26oedzHH3+cq666KmuttVb69u2bSqWSJPnud7+b3XbbLSuttFKOPfbYvPzyy9lrr72y1VZbpXfv3jnssMMyZsyYmu2sscYaOeCAA9KnT5+suOKKOfXUU9OzZ8/cfPPNtebcdtttc9BBB6VXr1459thjs8QSS9RsZ+edd06lUsmf//znmvUvv/zyDB06tGauzzv99NPTsWPHmq8ePXp8lW8vAAAAwAKvXte4Ouigg3LUUUfltddeyzrrrDPHEVN9+/ad7219PuT84x//yKxZs7LXXntl+vTpeeihh/LBBx9k8cUXr7XeRx99VOsUv2WXXTZdunSZY/ufnaVr165Jkj59+tRa9u9//ztTp05Nhw4dMm3atJx88sm55ZZb8vrrr2fmzJn56KOP5jji6rPbrVQq6datW956660kScuWLbP33nvnsssuy2677ZZHH30048ePz0033fSF34fjjz8+Rx55ZM3tqVOnilcAAADAIq1e4epb3/pWkuQHP/hBzbLZp/VVKpV88sknX7qNXr16pVKp1JxeN1vPnj2TJK1bt06SzJo1K927d691VNRsnTp1qvnvX3S64Wcv0j47ks1t2axZs5IkP/zhD3P77bfnjDPOSK9evdK6devssssu+fjjj79wu7O3M3sbyaenC6655pp57bXXctlll2XgwIFZdtll5zpj8mnsatmy5RfeDwAAALCoqVe4eumll77yEy+++OIZNGhQzj///Bx66KFfGJ7WXnvtTJo0Kc2aNctyyy33lZ/3y/z973/P0KFDs9NOOyX59JpXL7/8cp2306dPn/Tr1y+XXHJJrrnmmpx33nkNPCkAAADAwq1e4WpeRw7VxQUXXJABAwakX79+Oemkk9K3b980adIkY8eOzdNPP5111lknW265ZTbYYIPsuOOO+fnPf56VV145r7/+em677bbsuOOO6devX4PMMluvXr1y4403ZsiQIalUKhk+fHitI6nqYr/99sshhxySNm3a1IQwAAAAAOZPvcLV5y9UPlulUkmrVq3Sq1evLL/88l+6nRVWWCGPPPJITjvttBx//PF57bXX0rJly6y66qo5+uijc9BBB6VSqeS2227Lj370o+y77755++23061bt2yyySY116xqSGeffXb23XffbLjhhlliiSVy7LHH1vsT/vbYY48cfvjh2XPPPdOqVasGnhQAAABg4VapVqvVuj6oSZMmNde0qrWxz1znaqONNspNN92UxRZbrMGGXdC8+uqrWW655TJ27NisvfbadXrs1KlT07Fjx0w4vnfat2r6X5oQAFgQLPOTxxt7BACABjO7eUyZMiUdOnSY57pN6vMEI0eOTP/+/TNy5MhMmTIlU6ZMyciRI7Puuuvmlltuyd1335133nknRx99dL1ewIJuxowZmThxYo499tisv/76dY5WAAAAANTzVMHDDjssv/nNb7LhhhvWLBs4cGBatWqV/fffP0888UTOOeec7Lvvvg026ILk3nvvzeabb56VVlopN9xwQ2OPAwAAALBAqle4euGFF+Z6KFeHDh3y4osvJklWXHHF/Otf//pq0y2gNttsszlOowQAAACgbup1quA666yTH/7wh3n77bdrlr399ts55phj0r9//yTJc889l6WXXrphpgQAAABgkVOvI64uvfTS7LDDDll66aXTo0ePVCqVTJw4MT179syf//znJMkHH3yQ4cOHN+iwAAAAACw66hWuVl555Tz11FO5/fbb8+yzz6ZarWaVVVbJoEGD0qTJpwdx7bjjjg05JwAAAACLmHqFqySpVCrZeuuts/XWWzfkPAAAAACQpJ7XuEqSu+66K0OGDEmvXr2y4oorZvvtt8/f//73hpwNAAAAgEVYvcLV7373u2y55ZZp06ZNfvCDH+SQQw5J69atM3DgwFxzzTUNPSMAAAAAi6BKtVqt1vVBvXv3zv77758jjjii1vKzzjorl1xySZ566qkGG3BRNXXq1HTs2DETju+d9q2aNvY4AEAjWuYnjzf2CAAADWZ285gyZUo6dOgwz3XrdcTViy++mCFDhsyxfPvtt89LL71Un00CAAAAQC31Clc9evTIqFGj5lg+atSo9OjR4ysPBQAAAAD1+lTBo446Kj/4wQ/y6KOPZsMNN0ylUsk999yTK664Iueee25DzwgAAADAIqhe4WrYsGHp1q1bzjzzzFx33XVJPr3u1R/+8IfssMMODTogAAAAAIumeoWrJNlpp52y0047NeQsAAAAAFCjXte4SpLJkyfnt7/9bU444YS8++67SZKHH344//znPxtsOAAAAAAWXfU64uqxxx7LlltumY4dO+bll1/Ofvvtl86dO+dPf/pTXnnllVx55ZUNPScAAAAAi5h6HXF15JFHZujQoXnuuefSqlWrmuXbbLNN7r777gYbDgAAAIBFV73C1dixY3PAAQfMsfwb3/hGJk2a9JWHAgAAAIB6hatWrVpl6tSpcyx/5pln0qVLl688FAAAAADUK1ztsMMOOeWUUzJjxowkSaVSycSJE3Pcccdl5513btABAQAAAFg01StcnXHGGXn77bez5JJL5qOPPsqmm26aXr16pV27dvnZz37W0DMCAAAAsAiq16cKdujQIffcc09Gjx6dhx56KLNmzcraa6+dLbfcsqHnAwAAAGARVacjrj766KPccsstNbfvuOOOvP7665k0aVJuu+22HHPMMfn3v//d4EMCAAAAsOip0xFXV155ZW655ZZst912SZLzzz8/q622Wlq3bp0kefrpp9O9e/ccccQRDT8pAAAAAIuUOh1xdfXVV2ffffetteyaa67J6NGjM3r06Pzyl7/Mdddd16ADAgAAALBoqlO4evbZZ7PSSivV3G7VqlWaNPnPJtZdd908+eSTDTcdAAAAAIusOp0qOGXKlDRr9p+HvP3227XunzVrVqZPn94wkwEAAACwSKvTEVdLL710JkyY8IX3P/bYY1l66aW/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BgAAAMBCZaEPV2+99VYOOOCALLPMMmnZsmW6deuWrbbaKvfff3+DPcfRRx+dUaNGNdj2AAAAAEiaNfYA/20777xzZsyYkREjRqRnz5558803M2rUqLz77rsN9hzt2rVLu3btGmx7AAAAACzkR1xNnjw599xzT37+859n8803z7LLLpt11103xx9/fAYPHpwkqVQqufDCC7PNNtukdevWWX755XP99dfX2s6xxx6blVZaKW3atEnPnj0zfPjwzJgxo+b+z58qOHTo0Oy4444544wz0r179yy++OI5+OCDaz0GAAAAgHlbqMPV7COhbrrppkyfPv0L1xs+fHh23nnnjB8/PnvvvXf22GOPPPXUUzX3t2/fPldccUWefPLJnHvuubnkkkty9tlnz/O5R48enRdeeCGjR4/OiBEjcsUVV+SKK65oqJcGAAAAsNBbqMNVs2bNcsUVV2TEiBHp1KlTBgwYkBNOOCGPPfZYrfV23XXX7LfffllppZXy05/+NP369ct5551Xc/+Pf/zjbLjhhlluueUyZMiQHHXUUbnuuuvm+dyLLbZYzj///KyyyirZbrvtMnjw4HleB2v69OmZOnVqrS8AAACARdlCHa6ST69x9frrr+fmm2/OVlttlTFjxmTttdeudfTTBhtsUOsxG2ywQa0jrm644YZstNFG6datW9q1a5fhw4dn4sSJ83ze1VZbLU2bNq253b1797z11ltfuP7pp5+ejh071nz16NGjjq8UAAAAYOGy0IerJGnVqlUGDRqUn/zkJ7nvvvsydOjQnHjiifN8TKVSSZI88MAD2X333bPNNtvklltuySOPPJIf/ehH+fjjj+f5+ObNm8+xvVmzZn3h+scff3ymTJlS8/Xqq6/O56sDAAAAWDgtEuHq81ZdddVMmzat5vYDDzxQ6/4HHnggq6yySpLk3nvvzbLLLpsf/ehH6devX1ZcccW88sorDT5Ty5Yt06FDh1pfAAAAAIuyZo09wH/TO++8k1133TX77rtv+vbtm/bt22fcuHH5xS9+kR122KFmveuvvz79+vXLRhttlKuvvjr/+Mc/cumllyZJevXqlYkTJ+b3v/99+vfvn1tvvTV/+tOfGuslAQAAACwyFupw1a5du6y33no5++yz88ILL2TGjBnp0aNHvv/97+eEE06oWe/kk0/O73//+xx00EHp1q1brr76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KvsaqB/7G6g7+5y5E64KcNBBB2XxxRfP2WefnYsvvjiffPJJkqRp06ZZZ511cuWVV2a33XZr5CmhTN4UwfwbMmRIvv/97+fSSy9Nv379at03bty4HHjggdl+++0baTool30N1I39DdSffc6cXOOqMDNmzMi//vWvJMkSSyyR5s2bN/JEUK5vf/vbeeyxx77wTdH3v//99OnTJ1deeWUjTQhlmTx5cvbYY4/cfvvt6dSpU5ZccslUKpW8+eabmTJlSrbaaqtcc8016dSpU2OPCsWwr4G6s7+B+rHPmTvhClhgeVME9fPUU0/lgQceyKRJk5Ik3bp1ywYbbJBVVlmlkSeD8tjXQP3Z30Dd2OfMnXAFLPC8KQLgv82+BoCvi31ObcIVACxCqtVq/va3v+W+++7LpEmTUqlU0rVr1wwYMCADBw5cZK+dAEDDsr8BGopwBSzQvCmC+ffPf/4z2223XR5//PGsvvrq6dq1a6rVat56661MmDAha6yxRm6++eZ84xvfaOxRoSj2NVA39jdQf/Y5cxKugAWWN0VQNzvssEM++OCD/O53v0v37t1r3ffGG29k7733Tvv27XPTTTc1zoBQIPsaqDv7G6gf+5y5E66ABZY3RVA37dq1y7333ps11lhjrvc/8sgj2XjjjfPBBx98zZNBuexroO7sb6B+7HPmrlljDwBQX6NGjcq99947x/+oJ0n37t1zxhlnZOONN26EyaBMrVu3zrvvvvuF97/33ntp3br11zgRlM++BurO/gbqxz5n7po09gAA9eVNEdTN7rvvnu985zu54YYbMmXKlJrlU6ZMyQ033JDvfve72XPPPRtxQiiPfQ3Unf0N1I99ztw54gpYYM1+U3TWWWdl0KBB6dixY5JP3xSNHDkyRx11lDdF8BlnnnlmZs6cmb322iszZ85MixYtkiQff/xxmjVrlu9973v55S9/2chTQlnsa6Du7G+gfuxz5s41roAF1scff5zDDjssl1122Re+KTrnnHNqlgOfmjp1asaNG5c333wzSdKtW7ess8466dChQyNPBuWxr4H6s7+BurHPmTvhCljgeVMEwH+bfQ0AXxf7nNqEKwBYhEybNi3XXHNN7rvvvkyaNCmVSiVdu3bNgAEDsscee6Rt27aNPSIACwH7G6ChCFfAAs2bIph/Tz75ZAYNGpQPP/wwm266abp27ZpqtZq33nord911V9q2bZs77rgjq666amOPCkWxr4G6sb+B+rPPmZNwBSywvCmCutl8883TrVu3jBgxYo5rI3z88ccZOnRo3njjjYwePbqRJoTy2NdA3dnfQP3Y58ydcAUssLwpgrpp06ZNxo0b94VvdiZMmJB11103H3744dc8GZTLvgbqzv4G6sc+Z+6aNfYAAPX14IMPZty4cXP9VI0WLVrkhBNOyLrrrtsIk0GZFltssTz33HNf+IfE888/n8UWW+xrngrKZl8DdWd/A/VjnzN3TRp7AID6mv2m6It4UwS1ff/73893vvOdnHHGGRk/fnwmTZqUN998M+PHj88ZZ5yRfffdNwcccEBjjwlFsa+BurO/gfqxz5k7R1wBC6zZb4p+/OMfZ9CgQenatWsqlUomTZqUkSNH5rTTTsvhhx/e2GNCMU466aS0bt06Z511Vo455phUKpUkSbVaTbdu3XLcccflmGOOaeQpoSz2NVB39jdQP/Y5c+caV8AC7ec//3nOPffcmk/cSP7zpujwww/3pgi+wEsvvZRJkyYlSbp165bll1++kSeCctnXQP19dn/TtWvX9OzZs5EngrLZ58xJuAIWCv4IB+C/zb4GvpoWLVpk/Pjx6d27d2OPAsWzz/kP4QpYaL366qs58cQTc9lllzX2KFCMjz76KA899FA6d+48x0Vz//3vf+e6667LPvvs00jTQZmeeuqpPPDAA9lwww2z8sor5+mnn865556b6dOnZ++9984WW2zR2CNCUY488si5Lj/33HOz9957Z/HFF0+SnHXWWV/nWLDAee+99zJixIg899xzWWqppbLPPvukR48ejT3W1064AhZa48ePz9prr51PPvmksUeBIjz77LP55je/mYkTJ6ZSqWTjjTfOtddem+7duydJ3nzzzSy11FJ+Z+Az/vrXv2aHHXZIu3bt8uGHH+ZPf/pT9tlnn6yxxhqpVqu56667cvvtt4tX8BlNmjTJGmuskU6dOtVaftddd6Vfv35p27ZtKpVK7rzzzsYZEAq11FJL5fHHH8/iiy+el156KQMGDEi1Wk2fPn3y1FNP5f33388DDzyQVVZZpbFH/VoJV8AC6+abb57n/S+++GKOOuoof4TD/7fTTjtl5syZufzyyzN58uQceeSRmTBhQsaMGZNllllGuIK52HDDDbPFFlvk1FNPze9///scdNBBGTZsWH72s58lSX70ox9l7NixueOOOxp5UijH6aefnksuuSS//e1va0Xd5s2bZ/z48XMc8Qt8qkmTJpk0aVKWXHLJ7LHHHpk0aVJuvfXWtGnTJtOnT88uu+ySVq1a5frrr2/sUb9WwhWwwGrSpEkqlUrm9T9jlUrFH+Hw/3Xt2jV/+9vf0qdPn5plBx98cG655ZaMHj06bdu2Fa7gczp27JiHHnoovXr1yqxZs9KyZcs8+OCDWXvttZMkEyZMyJZbbllzHRLgU2PHjs3ee++dIUOG5PTTT0/z5s2FK/gSnw1XPXv2nCP+Pvjgg9lll13y6quvNuKUX78mjT0AQH117949f/zjHzNr1qy5fj388MONPSIU5aOPPkqzZs1qLfv1r3+d7bffPptuummeffbZRpoMFgxNmjRJq1atap3+1L59+0yZMqXxhoJC9e/fPw899FDefvvt9OvXL48//njNJ6QBX2z278n06dPTtWvXWvd17do1b7/9dmOM1aiEK2CBtc4668wzTn3Z0ViwqFlllVUybty4OZafd9552WGHHbL99ts3wlRQtuWWWy7PP/98ze37778/yyyzTM3tV199teY6cUBt7dq1y4gRI3L88cdn0KBBjuiF+TBw4MCsvfbamTp16hz/p+LEiROzxBJLNNJkjafZl68CUKYf/vCHmTZt2hfe36tXr4wePfprnAjKttNOO+Xaa6/Nt7/97TnuO//88zNr1qxcdNFFjTAZlGvYsGG1/theffXVa93/f//3fy7MDl9i9913z0YbbZSHHnooyy67bGOPA8U68cQTa91u06ZNrdt/+ctfsvHGG3+dIxXBNa4AAAAAKJJTBQEAAAAoknAFAAAAQJGEKwAAAACKJFwBAAAAUCThCgBgATdp0qQceuih6dmzZ1q2bJkePXpkyJAhGTVq1Nc6R6VSyU033fS1PicAsHBr1tgDAABQfy+//HIGDBiQTp065Re/+EX69u2bGTNm5Pbbb8/BBx+cp59+urFHBACot0q1Wq029hAAANTPtttum8ceeyzPPPNM2rZtW+u+yZMnp1OnTpk4cWIOPfTQjBo1Kk2aNMnWW2+d8847L127dk2SDB06NJMnT651tNThhx+eRx99NGPGjEmSbLbZZunbt29atWqV3/72t2nRokUOPPDAnHTSSUmS5ZZbLq+88krN45dddtm8/PLL/82XDgAsApwqCACwgHr33Xfz17/+NQcffPAc0SpJOnXqlGq1mh133DHvvvtu7rrrrowcOTIvvPBCvvWtb9X5+UaMGJG2bdvmwQcfzC9+8YuccsopGTlyZJJk7NixSZLLL788b7zxRs1tAICvwqmCAAALqOeffz7VajWrrLLKF67zt7/9LY899lheeuml9OjRI0ly1VVXZbXVVsvYsWPTv3//+X6+vn375sQTT0ySrLjiijn//PMzatSoDBo0KF26dEnyaSzr1q3bV3hVAAD/4YgrAIAF1OwrPlQqlS9c56mnnkqPHj1qolWSrLrqqunUqVOeeuqpOj1f3759a93u3r173nrrrTptAwCgLoQrAIAF1IorrphKpTLPAFWtVucatj67vEmTJvn8ZU9nzJgxx2OaN29e63alUsmsWbPqMzoAwHwRrgAAFlCdO3fOVlttlV//+teZNm3aHPdPnjw5q666aiZOnJhXX321ZvmTTz6ZKVOmpHfv3kmSLl265I033qj12EcffbTO8zRv3jyffPJJnR8HAPBFhCsAgAXYBRdckE8++STrrrtu/vjHP+a5557LU089lV/96lfZYIMNsuWWW6Zv377Za6+98vDDD+cf//hH9tlnn2y66abp169fkmSLLbbIuHHjcuWVV+a5557LiSeemAkTJtR5luWWWy6jRo3KpEmT8t577zX0SwUAFkHCFQDAAmz55ZfPww8/nM033zxHHXVUVl999QwaNCijRo3KhRdemEqlkptuuimLLbZYNtlkk2y55Zbp2bNn/vCHP9RsY6uttsrw4cNzzDHHpH///nn//fezzz771HmWM888MyNHjkyPHj2y1lprNeTLBAAWUZXq5y9oAAAAAAAFcMQVAAAAAEUSrgAAAAAoknAFAAAAQJGEKwAAAACKJFwBAAAAUCThCgAAAIAiCVcAAAAAFEm4AgAAAKBIwhUAAAAARRKuAAAAACiScAUAAABAkYQrAAAAAIr0/wDzWdNQiCxE+gAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 1400x1000 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize = (14,10))\n",
    "count1 = df.Geography.value_counts()\n",
    "sns.barplot(x = count1, y = count1.index, orient = 'h')\n",
    "plt.xlabel('Count')\n",
    "plt.ylabel('Geography')\n",
    "plt.title('Geography Distribution Plot')\n",
    "plt.xticks(rotation=90)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "10fb1753",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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krAMAAAAAErCMgAAAAAoCcsAAAAAoCQsAwAAAICSsAwAAAAASsIyAAAAACgJywAAAACgJCwDAAAAgJKwDAAAAABKwjIAAAAAKAnLAAAAAKAkLAMAAACAkrAMAAAAAErCMgAAAAAoCcsAAAAAoNSmsOzvf/975s2b1161AAAAAECXalNY9sEPfjArr7xyjj766Nx1113tVRMAAAAAdIk2D8OcOHFifvazn2WrrbbKhhtumDPOOCNPP/10O5QGAAAAAJ2r3eYsq1areeyxx3LSSSdlnXXWyfbbb5/zzjsvkydPbq9DAAAAAECHalNY9oUvfCGrr756qtVqkqRSqSQpgrPbbrstRx55ZFZaaaXst99+ueyyyzJnzpy2VwwAAAAAHaRNYdmPf/zjjB8/Pvfcc09OPPHEvPOd72wKzqrVaqrVambNmpXLLrss++23X0aOHJnPfe5zufPOO9uleAAAAABoT5XqgnSrnTz11FP5y1/+kssuuyzjxo1rCs2aDlj2Pttmm21yzjnnZOONN27Pw9NGU6ZMyZAhQ/LA8aMyuF/Pri4HAOhCq590f1eXAADQbhZkHo2NjWloaKjbrld7H3jttdfOF77whay66qqZNWtW7rnnnqaAbIFqtZpbb70122yzTW6//fZstNFG7V0GAAAAALRau03wnyS33XZbjjjiiIwcOTIf//jHc++99y4yj1m1Wk3v3r2b2r/++us58cQT27MEAAAAAFhmbQ7LJkyYkDPOOCPrr79+tt9++5x//vlpbGxcZO6yHj165MMf/nCuuuqqTJs2LZdcckl69So6td1+++1tLQEAAAAA2kWbhmHuuuuuufHGGxeZl2zhIZerrLJKDjvssBx22GFZeeWVm5YffPDBOf/88zN27Ni88sorbSkBAAAAANpNm8KyG264oSkcq1QqqVarqVQq2WuvvXLkkUdm7733To8etTuvDR8+vC2HBgAAAIB21y4T/Fer1ay00kr59Kc/nc985jNZbbXVlrrN5z//+XzgAx9oj8MDAAAAQLtoU1hWqVSy22675YgjjsiHPvSh9OzZs8Xb7rjjjtlxxx3bcngAAAAAaFdtCsueeOKJrLXWWu1VCwAAAAB0qTZdDVNQBgAAAMBbSbvMWfboo4/m6quvzlNPPZXXX3+96cqYi6tUKrngggva45AAAAAA0O7aFJbNnz8/n/vc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MqoPHD8qg/v17OpyAIAutPpJ93d1CQAA7WZB5tHY2JiGhoa67do0Z9kC2223Xf7xj39kxowZefzxxzN16tQ0NDRkvfXWS79+/drjEAAAAADQ4dolLFugf//+2WSTTdpzlwAAAADQadotLHv44Yfz5z//Offff38aGxszZMiQbLzxxtl3333zjne8o70OAwAAAAAdps1h2Zw5c/LFL34x5513Xhaf/uxPf/pTTj755HzmM5/JT37yk/Tu3buthwMAAACADtPmsGz06NH5wx/+0BSUVSqVpnXVajXVajXnnntupk6dmksuuaSthwMAAACADtOjLRtfe+21+f3vf5/kjZBsQUC2cHhWrVbzu9/9Lv/617/aWC4AAAAAdJw29Sy74IILmu736NEj++yzT3bdddcsv/zyeeWVV3Ldddflr3/9a+bPn58k+dWvfpXddtutbRUDAAAAQAdpU1g2bty4pvu/+MUv8ulPf3qR9UcccUTOO++8HHHEEalUKrn99tvbcjgAAAAA6FBtGob50ksvJUn69u2bMWPG1Gxz6KGHpm/fvqlWq3n55ZfbcjgAAAAA6FBtCst69uyZJJk7d27mzp1bs83C63r0aNPhAAAAAKBDtSm9WnnllZMk8+bNy3e/+92abc4666zMmzcvlUqlqT0AAAAAdEdtmrNsu+22y+OPP54kOfnkk3PllVfmfe97X9ME/9dee23uvvvupvbbb79926oFAAAAgA7UprDsyCOPzIUXXpgkqVarueuuuxYJx6rV6iLtP/vZz7blcAAAAADQodo0DHOLLbbIMccck2q1mkqlkkqlkmq12nRbsCxJjjnmmLznPe9pl6IBAAAAoCO0ecb973//+znttNPSv3//Zj3JqtVq+vXrl9NPPz1nn312Ww8FAAAAAB2qTcMwF/jGN76RI444In//+9/zwAMPpLGxMUOGDMlGG22U97///RkxYkR7HAYAAAAAOlS7hGVJMmLEiIwePbq9dgcAAAAAna7dwrIkmTNnTh588MFMnDgxPXr0yC677NKeuwcAAACADtUuYdn48eNz0kkn5S9/+UtmzpyZJBk5cmSee+657LPPPmlsbExDQ0Muu+yy9jgcAAAAAHSINodlN954Yz7ykY9kypQpi0zwv+D+8ssvn8svvzyVSiXXXXdddt1117YeEgAAAAA6RJuuhvnyyy839RxL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",
      "text/plain": [
       "<Figure size 1400x1000 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize = (14,10))\n",
    "count1 = df.Geography.value_counts()\n",
    "sns.barplot(x = count1, y = count1.index, orient = 'h')\n",
    "plt.xlabel('Count', fontsize = 16, fontweight = 'bold')\n",
    "plt.ylabel('Geography', fontsize = 16, fontweight = 'bold')\n",
    "plt.title('Geography Distribution Plot', fontsize = 24, fontweight = 'bold', color = 'blue')\n",
    "plt.xticks(rotation=90, fontsize = 12, fontweight = 'bold', color = 'green')\n",
    "plt.yticks(fontsize = 12, fontweight = 'bold', color = 'green')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "ffc56d1d",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\binod\\AppData\\Local\\Temp\\ipykernel_8032\\2666996055.py:1: FutureWarning: The default value of numeric_only in DataFrame.corr is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.\n",
      "  correlation_matrix = df.corr()\n"
     ]
    },
    {
     "data": {
      "image/png": "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/fjjj+revbskKT093bK1bdtWKSkpOnLkSI6xXblyRSNHjlSFChXk5uYmNzc3+fj46OrVq0pKSrqXy9WmTZskZb3eZ555Rt7e3lmut3r16ipTpozls6enp4KDg62m8d9O5kr077//vtLT07VkyRJ16tRJPj4+2fY/ffq0BgwYoNKlS8vNzU3u7u4qW7asJN3VdVetWjXbKfrZqVChghYtWqTPP/9cjz/+uBo3bqxx48bd8VgAADgaU+gBAE6nSJEiypcvn44fP35H/TOncGc3TbpEiRJZEtVbTae+ed+ff/4pSRo+fLiGDx+e7TE3Ptd9s27dumnjxo0aPXq06tSpI19fXxkMBrVt29ayINvdOnfunNzc3FS0aFGrdoPBoICAgCxT2gsXLpzlHEaj8a7Hf+655zR+/HhNnjxZ+/fv15w5c7LtZzKZ1KpVK508eVKjR49WeHi4vL29ZTKZVL9+/bsa926nvj/22GPy9/fXn3/+qaioKLm6ut7V8QAAOBIJPADA6bi6uqpFixb66quv9Pvvv6tUqVK37J+ZoKakpGTpe/LkSavn36V/E92c3Lwv89jo6Gh17Ngx22NCQkKybb906ZLWrl2rsWPH6pVXXrG0p6am3tc77AsXLqz09HSdOXPGKok3m806deqU6tSpc8/nvpXSpUurZcuWGj9+vEJCQtSwYcNs+/3www86cOCAYmNj1atXL0v7zz//fNdj3ur/q+wMGDBAf/31l6pUqaLBgwercePGKliw4F2PCwCAIzCFHgDglKKjo2U2m9WvXz9dv349y/60tDStWbNGkizT4T/44AOrPt99952SkpLUokWLe44jJCREFStW1IEDB1S7du1st/z582d7rMFgkNlszrKK/uLFi5WRkWHVltnnTqrTmddz8/V+9tlnunr16n1d7+0MGzZM7dq1u+U72DOT7puve8GCBVn63s11387ixYv1wQcf6O2339aXX36pixcv2uxVgwAAPAhU4AEATqlBgwaaN2+eBg4cqFq1aumFF15QlSpVlJaWpu+//14LFy5UWFiY2rVrp5CQED3//POaM2eOXFxc1KZNG8sq9KVLl9bQoUPvK5YFCxaoTZs2ioiIUGRkpEqWLKnz588rKSlJ+/fv1yeffJLtcb6+vmrSpImmTZumIkWKKDAwUFu3btWSJUtUoEABq75hYWGSpIULFyp/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6tXrfm8JAAAAANyxvLqYnKOQwN9C8+bNNW/ePEnS+fPnNXfuXLVp00Z79uxR2bJl9emnn2rmzJn6+eefdeXKFaWnp8vX1zfH850+fVpjxozRpk2b9OeffyojI0PXrl3TiRMnrPpVrVrV8mdvb2/lz59fp0+fliQlJCSocePGluT9RmfOnNFvv/2mPn36qF+/fpb29PR0+fn55RhXamqqUlNTrdrS09zk5m68xd0BAAAAADxIJPC34O3trQoVKlg+16pVS35+flq0aJEef/xxdenSRePHj1dERISlKj59+vQczxcZGakzZ85o5syZKlu2rIxGoxo0aKDr169b9bs5OTcYDDKZTJIkLy+vHM+f2WfRokWqV6+e1T5XV9ccj4uJidH48eOt2po/NVotnh6b4zEAAAAAcDtmAxV4WyKBvwsGg0EuLi76+++/tXPnTpUtW1ajRo2y7P/1119vefz27ds1d+5ctW3bVpL022+/6ezZs3cVQ9WqVfXuu+8qLS0tS6Lv7++vkiVL6pdfflH37t3v+JzR0dGKioqyanvrC340AAAAACA3IUu7hdTUVJ06dUqSdOHCBb399tu6cuWK2rVrp0uXLunEiRP66KOPVKdOHf3f//2fVq9efcvzVahQQe+//75q166ty5cv6+WXX75lRT07L774oubMmaMuXbooOjpafn5+2rVrl+rWrauQkBCNGzdOgwcPlq+vr9q0aaPU1FTt3btXFy5cyJKkZzIajTIarafLu7ln3FVcAAAAAAD7YhX6W4iLi1Px4sVVvHhx1atXT999950++eQTNWvWTO3bt9fQoUP14osvqnr16oqPj9fo0aNveb6lS5fqwoULqlGjhnr06KHBgwerWLFidxVT4cKFtWnTJl25ckVNmzZVrVq1tGjRIks1vm/fvlq8eLFiY2MVHh6upk2bKjY2VkFBQfd8HwAAAADgXpjNBodteZHBbDabHR0Ecp/XP6ICj+y5ueXNX4b3yujB/ch09ZrJ0SHkKtV7VHZ0CLlKTOuFjg4h14jo1sjRIeQqLjwfa4V/V/6npD/34kZdGznn/fj52HGHjV2hfN4rYjKFHgAAAABgF2YmfdsUdxMAAAAAACdABR4AAAAAYBdmOefU/9yKCjwAAAAAAE6ABB4AAAAAACfAFHoAAAAAgF0whd62qMADAAAAAOAEqMADAAAAAOyC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dKklgU9ISNAzzzxzX3E89thjcnNz05dffqkuXbros88+U/78+S2J+b59+/Tzzz8rf/78Vsf9888/OnbsmFWbu7u7nn32WS1btky//PKLgoODs1Svz5w5o99++019+vRRv379LO3p6elZFta78djM//Nv/JLD399fp0+ftjqmWrVqypcvn+VzgwYNdOXKFf322286ffq0/vnnnyyJ+fXr11WjRg2rttq1a2dzt6ylpqZmeRYkPc1dbu539sMOAAAAAM7Mw8NDtWrV0oYNG/Tkk09a2jds2KD27dtne0yDBg20Zs0aq7b169erdu3acnd3t0uc953Au7q6asOGDYqPj9f69es1Z84cjRo1Srt371ZQUJA+/PBD/fPPP6pXr57lGLPZLJPJpMTERFWuXFleXl73G4Y8PDz09NNP68MPP1SXLl304YcfqnPnznJz+/cSTSaTatWqlWWavSQVLVo0S1vv3r1Vr149/fDDD+rdu3eW/SaTSdK/0+hvvDbp33tyoxv/z8us/t/clnm+27mx7//93/+pZMmSVvtv/obJ29v7tueMiYmxmp4vSY92GqNWncfeUUwAAAAA4OyioqLUo0cP1a5dWw0aNNDChQt14sQJDRgwQNK/U/L/+OMPvffee5KkAQMG6O2331ZUVJT69eunb7/9VkuWLNGKFSvsFuN9J/DSv0llo0aN1KhRI40ZM0Zly5bV6tWrFRUVpSVLlmjYsGGKjIy0Ombw4MFaunSp3nzzTVWtWlUbN27MkkRm8vDwUEZGxm3j6N69u1q1aqXDhw9r8+bNmjhxomVfzZo1tXLlSsuicLdTpUoVValSRQcPHlS3bt2y7Pf391fJkiX1yy+/qHv37rc93906cOCA/v77b8uXG7t27ZKPj49KlSqlggULymg06sSJE1bT5e9VdHS0oqKirNrmfmWfb4wAAAAAPDwcOYX+bnXu3Fnnzp3ThAkTlJKSorCwMK1bt87yqHhKSorVO+GDgoK0bt06DR06VO+8845KlCih2bNn66mnnrJbjPedwO/evVsbN25Uq1atVKxYMe3evVtnzpxRaGioEhIStH//fi1fvjzL+9+7du2qUaNGKSYmRtHR0QoPD9fAgQM1YMAAeXh4aPPmzXrmmWdUpEgRBQYGavfu3UpOTpaPj48KFSqUbSxNmzaVv7+/unfvrsDAQNWvX9+yr3v3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",
      "text/plain": [
       "<Figure size 1200x800 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "correlation_matrix = df.corr()\n",
    "plt.figure(figsize = (12,8))\n",
    "sns.heatmap(correlation_matrix, cmap = 'coolwarm', annot = False,  fmt=\".2f\")\n",
    "plt.title(\"Correlation Matrix\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e675454d",
   "metadata": {},
   "source": [
    "### Preprocess the data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "e01c3110",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>RowNumber</th>\n",
       "      <th>CustomerId</th>\n",
       "      <th>Surname</th>\n",
       "      <th>CreditScore</th>\n",
       "      <th>Geography</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Tenure</th>\n",
       "      <th>Balance</th>\n",
       "      <th>NumOfProducts</th>\n",
       "      <th>HasCrCard</th>\n",
       "      <th>IsActiveMember</th>\n",
       "      <th>EstimatedSalary</th>\n",
       "      <th>Exited</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>15634602</td>\n",
       "      <td>Hargrave</td>\n",
       "      <td>619</td>\n",
       "      <td>France</td>\n",
       "      <td>Female</td>\n",
       "      <td>42</td>\n",
       "      <td>2</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>101348.88</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>15647311</td>\n",
       "      <td>Hill</td>\n",
       "      <td>608</td>\n",
       "      <td>Spain</td>\n",
       "      <td>Female</td>\n",
       "      <td>41</td>\n",
       "      <td>1</td>\n",
       "      <td>83807.86</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>112542.58</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>15619304</td>\n",
       "      <td>Onio</td>\n",
       "      <td>502</td>\n",
       "      <td>France</td>\n",
       "      <td>Female</td>\n",
       "      <td>42</td>\n",
       "      <td>8</td>\n",
       "      <td>159660.80</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113931.57</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>15701354</td>\n",
       "      <td>Boni</td>\n",
       "      <td>699</td>\n",
       "      <td>France</td>\n",
       "      <td>Female</td>\n",
       "      <td>39</td>\n",
       "      <td>1</td>\n",
       "      <td>0.00</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>93826.63</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>15737888</td>\n",
       "      <td>Mitchell</td>\n",
       "      <td>850</td>\n",
       "      <td>Spain</td>\n",
       "      <td>Female</td>\n",
       "      <td>43</td>\n",
       "      <td>2</td>\n",
       "      <td>125510.82</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>79084.10</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   RowNumber  CustomerId   Surname  CreditScore Geography  Gender  Age  \\\n",
       "0          1    15634602  Hargrave          619    France  Female   42   \n",
       "1          2    15647311      Hill          608     Spain  Female   41   \n",
       "2          3    15619304      Onio          502    France  Female   42   \n",
       "3          4    15701354      Boni          699    France  Female   39   \n",
       "4          5    15737888  Mitchell          850     Spain  Female   43   \n",
       "\n",
       "   Tenure    Balance  NumOfProducts  HasCrCard  IsActiveMember  \\\n",
       "0       2       0.00              1          1               1   \n",
       "1       1   83807.86              1          0               1   \n",
       "2       8  159660.80              3          1               0   \n",
       "3       1       0.00              2          0               0   \n",
       "4       2  125510.82              1          1               1   \n",
       "\n",
       "   EstimatedSalary  Exited  \n",
       "0        101348.88       1  \n",
       "1        112542.58       0  \n",
       "2        113931.57       1  \n",
       "3         93826.63       0  \n",
       "4         79084.10       0  "
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "f776c0af",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.drop(labels = [\"RowNumber\",\"CustomerId\", \"Surname\"], axis = 1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "681fdd21",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.get_dummies(df, drop_first = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "f1c25e83",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CreditScore</th>\n",
       "      <th>Age</th>\n",
       "      <th>Tenure</th>\n",
       "      <th>Balance</th>\n",
       "      <th>NumOfProducts</th>\n",
       "      <th>HasCrCard</th>\n",
       "      <th>IsActiveMember</th>\n",
       "      <th>EstimatedSalary</th>\n",
       "      <th>Exited</th>\n",
       "      <th>Geography_Germany</th>\n",
       "      <th>Geography_Spain</th>\n",
       "      <th>Gender_Male</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>619</td>\n",
       "      <td>42</td>\n",
       "      <td>2</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>101348.88</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>608</td>\n",
       "      <td>41</td>\n",
       "      <td>1</td>\n",
       "      <td>83807.86</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>112542.58</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>502</td>\n",
       "      <td>42</td>\n",
       "      <td>8</td>\n",
       "      <td>159660.80</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113931.57</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>699</td>\n",
       "      <td>39</td>\n",
       "      <td>1</td>\n",
       "      <td>0.00</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>93826.63</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>850</td>\n",
       "      <td>43</td>\n",
       "      <td>2</td>\n",
       "      <td>125510.82</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>79084.10</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   CreditScore  Age  Tenure    Balance  NumOfProducts  HasCrCard  \\\n",
       "0          619   42       2       0.00              1          1   \n",
       "1          608   41       1   83807.86              1          0   \n",
       "2          502   42       8  159660.80              3          1   \n",
       "3          699   39       1       0.00              2          0   \n",
       "4          850   43       2  125510.82              1          1   \n",
       "\n",
       "   IsActiveMember  EstimatedSalary  Exited  Geography_Germany  \\\n",
       "0               1        101348.88       1                  0   \n",
       "1               1        112542.58       0                  0   \n",
       "2               0        113931.57       1                  0   \n",
       "3               0         93826.63       0                  0   \n",
       "4               1         79084.10       0                  0   \n",
       "\n",
       "   Geography_Spain  Gender_Male  \n",
       "0                0            0  \n",
       "1                1            0  \n",
       "2                0            0  \n",
       "3                0            0  \n",
       "4                1            0  "
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "bb8db820",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df.drop(\"Exited\", axis = 1)\n",
    "y = df[\"Exited\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "fba15552",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "75ccdc85",
   "metadata": {},
   "outputs": [],
   "source": [
    "Scaler = StandardScaler()\n",
    "X_train = Scaler.fit_transform(X_train)\n",
    "X_test = Scaler.transform(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "adfa3961",
   "metadata": {},
   "source": [
    "### Train Logistic Regression Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "73e97987",
   "metadata": {},
   "outputs": [],
   "source": [
    "lr_model = LogisticRegression()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "f0b6eb16",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LogisticRegression</label><div class=\"sk-toggleable__content\"><pre>LogisticRegression()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "LogisticRegression()"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lr_model.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "480a88d9",
   "metadata": {},
   "outputs": [],
   "source": [
    "lr_predictions = lr_model.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "59bddb84",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Logistic Regression Model:\n",
      "[[1543   64]\n",
      " [ 314   79]]\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.83      0.96      0.89      1607\n",
      "           1       0.55      0.20      0.29       393\n",
      "\n",
      "    accuracy                           0.81      2000\n",
      "   macro avg       0.69      0.58      0.59      2000\n",
      "weighted avg       0.78      0.81      0.77      2000\n",
      "\n",
      "Accuracy:  0.811\n",
      "r2_Score:  -0.19705296959390473\n",
      "Precision_score:  0.5524475524475524\n",
      "Recall_score:  0.2010178117048346\n",
      "f1_score:  0.2947761194029851\n"
     ]
    }
   ],
   "source": [
    "print(\"Logistic Regression Model:\")\n",
    "print(confusion_matrix(y_test, lr_predictions))\n",
    "print(classification_report(y_test, lr_predictions))\n",
    "print(\"Accuracy: \", accuracy_score(y_test, lr_predictions))\n",
    "print(\"r2_Score: \", r2_score(y_test, lr_predictions))\n",
    "print(\"Precision_score: \", precision_score(y_test, lr_predictions))\n",
    "print(\"Recall_score: \", recall_score(y_test, lr_predictions))\n",
    "print(\"f1_score: \", f1_score(y_test, lr_predictions))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fe53dc9b",
   "metadata": {},
   "source": [
    "### Train Random Forests Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "a588ec3d",
   "metadata": {},
   "outputs": [],
   "source": [
    "rf_model = RandomForestClassifier(n_estimators=1000, n_jobs = -1, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "3df19ce0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-2 {color: black;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-2\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-2\" type=\"checkbox\" checked><label for=\"sk-estimator-id-2\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">RandomForestClassifier</label><div class=\"sk-toggleable__content\"><pre>RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=42)</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=42)"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rf_model.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "b42b23d2",
   "metadata": {},
   "outputs": [],
   "source": [
    "rf_predictions = rf_model.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "e51826be",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Random Forest Model:\n",
      "[[1547   60]\n",
      " [ 209  184]]\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.88      0.96      0.92      1607\n",
      "           1       0.75      0.47      0.58       393\n",
      "\n",
      "    accuracy                           0.87      2000\n",
      "   macro avg       0.82      0.72      0.75      2000\n",
      "weighted avg       0.86      0.87      0.85      2000\n",
      "\n",
      "Accuracy:  0.8655\n",
      "r2_Score:  0.1481289713736499\n",
      "Precision_score:  0.7540983606557377\n",
      "Recall_score:  0.4681933842239186\n",
      "f1_score:  0.5777080062794349\n"
     ]
    }
   ],
   "source": [
    "print(\"Random Forest Model:\")\n",
    "print(confusion_matrix(y_test, rf_predictions))\n",
    "print(classification_report(y_test, rf_predictions))\n",
    "print(\"Accuracy: \", accuracy_score(y_test, rf_predictions))\n",
    "print(\"r2_Score: \", r2_score(y_test, rf_predictions))\n",
    "print(\"Precision_score: \", precision_score(y_test, rf_predictions))\n",
    "print(\"Recall_score: \", recall_score(y_test, rf_predictions))\n",
    "print(\"f1_score: \", f1_score(y_test, rf_predictions))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8dc6e996",
   "metadata": {},
   "source": [
    "### Train Gradient Boosting Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "78d36aad",
   "metadata": {},
   "outputs": [],
   "source": [
    "gb_model = GradientBoostingClassifier(n_estimators=1000, learning_rate = 0.02, max_depth = 1, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "55f9b4da",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-3 {color: black;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-3\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>GradientBoostingClassifier(learning_rate=0.02, max_depth=1, n_estimators=1000,\n",
       "                           random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-3\" type=\"checkbox\" checked><label for=\"sk-estimator-id-3\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">GradientBoostingClassifier</label><div class=\"sk-toggleable__content\"><pre>GradientBoostingClassifier(learning_rate=0.02, max_depth=1, n_estimators=1000,\n",
       "                           random_state=42)</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "GradientBoostingClassifier(learning_rate=0.02, max_depth=1, n_estimators=1000,\n",
       "                           random_state=42)"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "gb_model.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "e705506e",
   "metadata": {},
   "outputs": [],
   "source": [
    "gb_predictions = gb_model.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "36f379e6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Gradient Boosting Model:\n",
      "[[1550   57]\n",
      " [ 224  169]]\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.87      0.96      0.92      1607\n",
      "           1       0.75      0.43      0.55       393\n",
      "\n",
      "    accuracy                           0.86      2000\n",
      "   macro avg       0.81      0.70      0.73      2000\n",
      "weighted avg       0.85      0.86      0.84      2000\n",
      "\n",
      "Accuracy:  0.8595\n",
      "r2_Score:  0.11012728979924014\n",
      "Precision_score:  0.7477876106194691\n",
      "Recall_score:  0.4300254452926209\n",
      "f1_score:  0.5460420032310177\n"
     ]
    }
   ],
   "source": [
    "print(\"Gradient Boosting Model:\")\n",
    "print(confusion_matrix(y_test, gb_predictions))\n",
    "print(classification_report(y_test, gb_predictions))\n",
    "print(\"Accuracy: \", accuracy_score(y_test, gb_predictions))\n",
    "print(\"r2_Score: \", r2_score(y_test, gb_predictions))\n",
    "print(\"Precision_score: \", precision_score(y_test, gb_predictions))\n",
    "print(\"Recall_score: \", recall_score(y_test, gb_predictions))\n",
    "print(\"f1_score: \", f1_score(y_test, gb_predictions))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
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
   "version": "3.11.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
