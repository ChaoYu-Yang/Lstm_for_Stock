import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler  # 匯入資料預處理模組

def readTrain():
    # filepath = "test.xlsx"
    # if not os.path.isfile(filepath):
    #     print('there is no file')
    # all_df = pd.read_excel(filepath)
    # print(all_df[:2])
  train = pd.read_excel("test.xlsx")
  cols = ['date', 'stock_num', 'M近四季常續性EPS', '本益比-TSE',
            '營收成長率', '稅前淨利率', '稅後淨利成長率', '市值(百萬元)', '股東權益總額', 'ROE(B)－常續利益']
  train = train[cols]
  return train

def augFeatures(train):
  train["Date"] = pd.to_datetime(train["date"])
  # train["year"] = train["Date"].dt.year
  # train["month"] = train["Date"].dt.month
  # train["date"] = train["Date"].dt.day
  # train["day"] = train["Date"].dt.dayofweek
  return train


def normalize(train):
  train = train.drop(["Date"], axis=1)
  train_norm = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
  return train_norm

# read SPY.csv
train = readTrain()

# Augment the features (year, month, date, day)
train_Aug = augFeatures(train)

# Normalization
train_norm = normalize(train_Aug)
# 讀取欄位名稱
# cols = ['date', 'stock_num', 'M近四季常續性EPS', '本益比-TSE',
#         '營收成長率', '稅前淨利率', '稅後淨利成長率', '市值(百萬元)', '股東權益總額', 'ROE(B)－常續利益']
# all_df = all_df[cols]

# all_df = all_df.dropna()  # 將所有的空值去除
# all_df.isnull().sum()  # 顯示現在df之中有多少個空值。
# all_df = all_df.sort_values('date')
# ndarray = all_df.values

# train_size = int(len(all_df) * 0.80)  # 分割訓練資料長度
# test_size = len(all_df) - train_size
#
# train_data = all_df[0:train_size, :]
# test_data = all_df[train_size: len(all_df), :]
#
# scaler = MinMaxScaler()
# train_data = train_data.reshape(-1, 1)
# test_data = test_data.reshape(-1, 1)

# if __name__ == '__main__':
#     print(all_df[:2])
#     print(all_df.head)
#     print(ndarray.shape)
