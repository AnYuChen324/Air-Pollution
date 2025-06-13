#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 15:28:43 2025

@author: anyuchen
"""
from pandasql import sqldf
import os
import sqlite3 
import pandas as pd
import numpy as np

#                                                                  =====  1.讀取CSV  =====
df_mmse_0 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/twb/TWBR10907-07_調查問卷.csv")
df_mmse = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/twb/TWBR10907-07_調查問卷.csv")
df_mmse_base = pd.read_excel(r"/Users/anyuchen/Desktop/空氣污染/dataset/twb/TWB_MMSE20230611.xlsx")

#  有些受測者有測驗兩次，因此都以第一次為基準FOLLOW = Baseline
df_mmse = df_mmse[df_mmse['FOLLOW'] == 'Baseline']

#  選取只需要的資料欄位：'Release_No','EDUCATION','G_1','G_1_A','G_2','G_3','G_4','G_5','G_5_A','G_5_B','G_5_C','G_5_D','G_5_E',先不要算SENTANCE欄位，因為不知道如何定義
cols = ['Release_No','EDUCATION','G_1','G_1_A','G_2','G_3','G_4','G_5','G_5_A','G_5_B','G_5_C','G_5_D','G_5_E']
df_mmse = df_mmse[cols]

#  將EDUCATION資料轉為數值型態,但因為有幾筆資料為R/N,不清楚其意義，故刪除
#  1=未受過正規教受，不識字、2=自修，識字、3=小學、4=國(初)中、5=高中(職)、6=大學(專)、7=研究所及以上
df_mmse = df_mmse.drop(df_mmse[(df_mmse['EDUCATION'] == 'R') | (df_mmse['EDUCATION'] == 'N')].index, axis=0)
df_mmse['EDUCATION'] = df_mmse['EDUCATION'].astype(int)

#  建立MMSE評分字典
mmse_table = {
    'G_1':5,
    'G_1_A':5,
    'G_2':3,
    'G_3':5,
    'G_4':3,
    'G_5':2,
    'G_5_A':1,
    'G_5_B':1,
    'G_5_C':1,
    'G_5_D':1,
    'G_5_E':3}

df_mmse = df_mmse.fillna(value=mmse_table)

#  計算總和MMSE分數
mmse_col = ['G_1','G_1_A','G_2','G_3','G_4','G_5','G_5_A','G_5_B','G_5_C','G_5_D','G_5_E']
df_mmse['MMSE_score'] = df_mmse[mmse_col].sum(axis = 1)

#  抓出符合下列三個條件，並將符合的設為1,否則為0
df_mmse['MMSE'] = np.where(
    ((df_mmse['EDUCATION'] >= 4) & (df_mmse['MMSE_score'] < 24)) | 
    ((df_mmse['EDUCATION'] == 3) & (df_mmse['MMSE_score'] < 21)) | 
    ((df_mmse['EDUCATION'] <= 2) & (df_mmse['MMSE_score'] < 16)),
    1,
    0)

#print(df_mmse['Release_No'].dtype)
#print(df_mmse_base['Release_No'].dtype)

#  去除空白欄位
df_mmse['Release_No'] = df_mmse['Release_No'].str.strip()
df_mmse_base['Release_No'] = df_mmse_base['Release_No'].str.strip()

pysqldf = lambda q:sqldf(q,globals())
query = '''
    SELECT 
        df.Release_No,
        df.MMSE,
        base.MMSEgroup
    FROM df_mmse AS df
    LEFT JOIN df_mmse_base AS base
    ON df.Release_No = base.Release_No;
'''


#  將合併好的地點資料放在twb_location中
match = pysqldf(query)
#  把result0加入到全域命名空間中
globals()['match'] = match

#  把match不需要的欄位MMSEgroup拿掉
match = match.drop(['MMSEgroup'],axis = 1)

#  將計算好的MMSE資料存成Excel檔
match.to_excel('/Users/anyuchen/Desktop/空氣污染/dataset/MMSE.xlsx',index = False)


#                                                             =====  和對與原先的資料是否相同  =====
'''
final_mmse = np.where(match,columns=['Release_No','']

mmse1_mmg_0 = (match['MMSE'] == 1) & (match['MMSEgroup'] == 0)
mmse0_mmg_1 = (match['MMSE'] == 0) & (match['MMSEgroup'] == 1)

Num_mmse1_mmg_0 = match.loc[mmse1_mmg_0,'Release_No']
Num_mmse0_mmg_1 = match.loc[mmse0_mmg_1,'Release_No']

#  確認是否有不交集的失智症判斷，df.loc[條件, 欄位]
result_base = df_mmse_base.loc[df_mmse_base['MMSEgroup'] == 1,'Release_No']
result = df_mmse.loc[df_mmse['MMSE'] == 1,'Release_No']

#  顯示每個在release_base中的是否也出現在result
mask = result_base.isin(result)
match = result_base[mask]
print(match)

only_in_base = set(result) - set(result_base)

df_mmse_0 = df_mmse_0[df_mmse_0['FOLLOW'] == 'Baseline']
df_mmse_0 = df_mmse_0[cols]
df_mmse_0['score'] = df_mmse_0[mmse_col].sum(axis = 1)
'''























