#Library Imports
import numpy as np
import pandas as pd
import math
import os
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error

from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

train=pd.read_csv('train.csv', encoding = 'cp949')
test=pd.read_csv('test.csv', encoding = 'cp949')
submission=pd.read_csv('sample_submission.csv', encoding = 'cp949')

#건물별로 '비전기냉방설비운영'과 '태양광보유'를 판단해 test set의 결측치를 보간해줍니다
train[['num', '비전기냉방설비운영','태양광보유']]
ice={}
hot={}
count=0
for i in range(0, len(train), len(train)//60):
    count +=1
    ice[count]=train.loc[i,'비전기냉방설비운영']
    hot[count]=train.loc[i,'태양광보유']

for i in range(len(test)):
    test.loc[i, '비전기냉방설비운영']=ice[test['num'][i]]
    test.loc[i, '태양광보유']=hot[test['num'][i]]

#시간 변수와 요일 변수를 추가해봅니다.
def time(x):
    return int(x[-2:])
train['time']=train['date_time'].apply(lambda x: time(x))
test['time']=test['date_time'].apply(lambda x: time(x))

def weekday(x):
    return pd.to_datetime(x[:10]).weekday()
train['weekday']=train['date_time'].apply(lambda x :weekday(x))
test['weekday']=test['date_time'].apply(lambda x :weekday(x))

test.interpolate(method='values')

train_x=train.drop('전력사용량(kWh)', axis=1)
train_y=train[['전력사용량(kWh)']]

train_x.drop('date_time', axis=1, inplace=True)
test.drop('date_time', axis=1, inplace=True)

cross=KFold(n_splits=5, shuffle=True, random_state=42)
folds=[]
for train_idx, valid_idx in cross.split(train_x, train_y):
    folds.append((train_idx, valid_idx))

models={}
for fold in range(5):
    print(f'===================={fold+1}=======================')
    train_idx, valid_idx=folds[fold]
    X_train=train_x.iloc[train_idx, :]
    y_train=train_y.iloc[train_idx, :]
    X_valid=train_x.iloc[valid_idx, :]
    y_valid=train_y.iloc[valid_idx, :]
    
    model=LGBMRegressor(n_estimators=100)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], 
             early_stopping_rounds=30, verbose=100)
    models[fold]=model
    
    print(f'================================================\n\n')

for i in range(5):
    submission['answer'] += models[i].predict(test)/5

submission.to_csv('baseline_submission3.csv', index=False)