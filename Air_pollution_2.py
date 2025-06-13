#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 15:58:50 2025

@author: anyuchen
"""

from pandasql import sqldf
import os
import sqlite3 
import pandas as pd
import re
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager as fm

#面臨到的問題：
#1.行政區域無法與空氣污染站完全匹配
#2.問卷中的PLACE_CURR有三碼跟四碼，需要去下載合併三碼跟四碼的台灣行政區域資料

#  定義SQL查詢，定義一個幫你「用SQL查詢Pandas DataFrame」的工具函式
pysqldf = lambda q:sqldf(q,globals())



#                                                                  =====    讀取CSV  =====

#  讀取空氣污染站的資料
air_pollution = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/台灣空氣污染站經緯度.csv")

#  讀取台灣行政中心經緯度
city_location = pd.read_excel(r"/Users/anyuchen/Desktop/空氣污染/dataset/台灣行政區域3碼經緯度.xlsx")

#  先抓一個月的問卷來嘗試計算與受測者最近的空氣污染站
twb = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/twb/TWBR10907-07_調查問卷.csv")

#  台灣區域碼與鄉鎮市3碼與4碼
city_4num = pd.read_excel(r"/Users/anyuchen/Desktop/空氣污染/dataset/台灣區域4碼表.xlsx")

#  區域碼以0為開頭的讀進來會變成三碼，要再補0
city_4num['num'] = city_4num['num'].astype(str).str.zfill(4)



#                                                            =====  生成台灣區域碼三四碼對照表  =====
#  將欄位area的內容操作調整為[新竹市]東　區 -> 新竹市東區
city_4num['city'] = city_4num['area'].str.extract(r"\[(.*?)\]")
city_4num['District'] = city_4num['area'].str.replace(r"\[.*?\]" , "" , regex = True)\
                                        .str.replace(r"\s+", "" , regex = True)\
                                        .str.replace(u'\u3000' , '')  #去除全形空格
city_4num['location'] = city_4num['city'] + city_4num['District']

#  刪除掉用不到的欄位city,District
city_4num.drop(columns={'city','District','area'},inplace = True)

#  調整鄉鎮市的名字，讓3碼4碼核對順利
replace_dict = {
    '桃園縣中壢市': '桃園市中壢區','桃園縣平鎮市': '桃園市平鎮區',
    '桃園縣龍潭鄉': '桃園市龍潭區','桃園縣楊梅鎮': '桃園市楊梅區',
    '桃園縣新屋鄉': '桃園市新屋區','桃園縣觀音鄉': '桃園市觀音區',
    '桃園縣桃園市': '桃園市桃園區','桃園縣龜山鄉': '桃園市龜山區',
    '桃園縣八德市': '桃園市八德區','桃園縣大溪鎮': '桃園市大溪區',
    '桃園縣復興鄉': '桃園市復興區','桃園縣大園鄉': '桃園市大園區',
    '桃園縣蘆竹鄉': '桃園市蘆竹區','臺東縣台東市': '臺東縣臺東市',
    '彰化縣員林鎮': '彰化縣員林市','苗栗縣頭份鎮': '苗栗縣頭份市',
    '苗栗縣通宵鎮': '苗栗縣通霄鎮','彰化縣溪洲鄉': '彰化縣溪州鄉',
    '雲林縣台西鄉': '雲林縣臺西鄉','屏東縣霧台鄉': '屏東縣霧臺鄉',
    '屏東縣三地門': '屏東縣三地門鄉','臺東縣太麻里': '臺東縣太麻里鄉',
    '金門縣金寧鎮': '金門縣金寧鄉'
    }
city_4num['location'] = city_4num['location'].replace(replace_dict)

#  要先生成一個對照表為台灣行政區域的郵遞區號跟行政區號表：四碼city_4num,三碼city_location
city_num = """
    SELECT * FROM city_location
    LEFT JOIN city_4num
    ON city_location.行政區名 = city_4num.location
"""

#  將合併好的地點資料放在twb_location中
city_num = pysqldf(city_num)
#  把result0加入到全域命名空間中
globals()['city_num'] = city_num

#  整理欄位名字
city_num.rename(columns={'3碼郵遞區號':'3num' , 'num':'4num' , '中心點經度':'lon' , '中心點緯度':'lat'},inplace = True)


#                                                      =====   問卷跟中心經緯度合併，透過3,4碼行政區域碼  =====

#  step1:先暫時將問卷有需要的資料抓出來
twb = twb[['Release_No','SURVEY_DATE','ID_BIRTH','AGE', 'SEX', 'EDUCATION', 'MARRIAGE',
                     'INCOME_SELF', 'INCOME_FAMILY', 'DRK', 'SMK_CURR', 'DRUG_USE',
                     'SLEEP_QUALITY', 'ILL_ACT', 'SPO_HABIT', 'SEVERE_DYSMENORRHEA_SELF',
                     'SEVERE_DYSMENORRHEA_MOM', 'SEVERE_DYSMENORRHEA_SIS', 'MYOMA_SELF',
                     'MYOMA_MOM', 'MYOMA_SIS', 'COOK_FREQ', 'WET_HVY', 'WATER_BOILED',
                     'VEGE_YR', 'HORMOME_MED', 'HERBAL_MED', 'DRUG_KIND_A', 'DRUG_NAME_A',
                     'JOB_EXPERIENCE','PLACE_CURR']]

#  Step2:先將問卷的PLACE_CURR(4碼數字)與台灣區域代碼合併,這樣資料就有縣市鄉鎮
"""
twb_3num = pd.merge(twb , city_num , left_on='PLACE_CURR' , right_on='3num' , how='left')
twb_4num = pd.merge(twb_3num , city_num , left_on='PLACE_CURR' , right_on='4num' , how='left')
"""

#  把唯一一筆Place_curr = R以及Place_curr為nan的資料刪除(同時處理字串 "nan" 和真正的 NaN 缺失值)
twb = twb[~twb['PLACE_CURR'].isin(['R','nan']) & ~twb['PLACE_CURR'].isna()]

#  發現有四碼的區域碼後面多了.0,顯示為R的要刪除
twb['PLACE_CURR'] = twb['PLACE_CURR'].astype(str).str.replace(r'\.0$','',regex=True)

#  再將資料轉為數值型態
#twb['PLACE_CURR'] = twb['PLACE_CURR'].astype(int)

#  先合併四碼行政區域碼，並只抓2014的資料
twb_4num = """
    SELECT 
        twb.*,
        city_num.lon AS num4_lon,
        city_num.lat AS num4_lat,
        city_num.location AS num4_location
    FROM twb
    LEFT JOIN city_num
    ON twb.PLACE_CURR = city_num."4num"
    WHERE SURVEY_DATE BETWEEN '2014/01/01' and '2020/12/31'
    """

#  將合併好的地點資料放在twb_location中
twb_4num = pysqldf(twb_4num)
#  把result0加入到全域命名空間中
globals()['twb_4num'] = twb_4num 

#  Step3:合併三碼的行政區域碼
twb_final = """
    SELECT 
        twb_4num.*,
        city_num.lon AS num3_lon,
        city_num.lat AS num3_lat,
        city_num.location AS num3_location
    FROM twb_4num
    LEFT JOIN city_num
    ON twb_4num.PLACE_CURR = city_num."3num"
    """

#  將合併好的地點資料放在twb_final中
twb_final = pysqldf(twb_final)
#  把result0加入到全域命名空間中
globals()['twb_final'] = twb_final
    
#  Step4:把num4的缺失值用num3填補
twb_final['num4_lon'] = twb_final['num4_lon'].fillna(twb_final['num3_lon'])
twb_final['num4_lat'] = twb_final['num4_lat'].fillna(twb_final['num3_lat'])
twb_final['num4_location'] = twb_final['num4_location'].fillna(twb_final['num3_location'])

#  Step5:把最後核對不了的3碼行政區域碼前面補0：把num3跟num4合併失敗的，將num3前面補0,再次合併num4
#  先將'PLACE_CURR'轉為字串
twb_final['PLACE_CURR'] = twb_final['PLACE_CURR'].astype(str)
twb_final['PLACE_CURR'] = twb_final['PLACE_CURR'].apply(lambda x : x.zfill(4) if len(x)==3 else x)

#  把num3的所有欄位都drop掉
twb_final.drop(columns=['num3_lon','num3_lat','num3_location'] , inplace = True)

#  Step6:再與四碼行政區域碼合併一次
twb_final2 = """
    SELECT 
        twb_final.*,
        city_num.lon AS num4_lon2,
        city_num.lat AS num4_lat2,
        city_num.location AS num4_location2
    FROM twb_final
    LEFT JOIN city_num
    ON twb_final.PLACE_CURR = city_num."4num"
    """

#  將合併好的地點資料放在twb_location中
twb_final2 = pysqldf(twb_final2)
#  把result0加入到全域命名空間中
globals()['twb_final2'] = twb_final2

#  再把新的num4_lon2,num4_lat2去填補num4_lon,num4_lat的缺失值
twb_final2['num4_lon'] = twb_final2['num4_lon'].fillna(twb_final2['num4_lon2'])
twb_final2['num4_lat'] = twb_final2['num4_lat'].fillna(twb_final2['num4_lat2'])
twb_final2['num4_location'] = twb_final2['num4_location'].fillna(twb_final2['num4_location2'])

#  把'num4_lon2','num4_lat2','num4_location2'從twb_final2中drop掉
twb_final2.drop(columns=['num4_lon2','num4_lat2','num4_location2'] , inplace = True)

#  更新欄位名稱
twb_final2.rename(columns={'num4_lon':'lon','num4_lat':'lat'} , inplace = True)

#確認資料經緯度是否有nan值
miss_cols = twb_final2[twb_final2['lon'].isna()]



#                                                                  =====  計算兩點距離(使用高效的cKDTree)  =====
#  因受測者居住地不一定有監測站，因此以距離受測者居住地最近的監測站來填補缺失值
"""
#  建立計算經緯度之間距離的函式haversine()
def haversine(lon1, lat1, lon2, lat2):
    #將兩點的經度緯度角度換算成弧度,把四個變數lon1(受測者經度),lat1(受測者緯度),lon2(監測站經度),lat2(監測站緯度)，丟入radians換算
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    #計算經度緯度的差距
    dlon = lon1 - lon2
    dlat = lat1 - lat2
    #haversine公式
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球半徑 (km)
    
    return c * r

#  尋找受測者位置對應的最接近監測站
def near_station(row):
    distance = air_pollution.apply(lambda station : haversine(row['lon'],row['lat'],station['lon'],station['lat']),axis=1)
    #找出最小距離的監測站，並回傳名單
    return air_pollution.loc[distance.idxmin() , 'station_name']
"""

#  高效的cKDTree
from scipy.spatial import cKDTree
import numpy as np

#  Step1:建立空氣污染監測站的KD-Tree(經緯度)
air_pollution.rename(columns = {'twd97lon':'lon','twd97lat':'lat'} , inplace=True)

station_coords = air_pollution[['lon','lat']].to_numpy()
station_tree = cKDTree(station_coords)

#  Step2:將受測者的經緯度轉為陣列
twb_coords = twb_final2[['lon','lat']].to_numpy()

#  Step3:查詢每個受測者最近的監測站索引與距離，k=3代表找出最接近的3個監測站(因為有發現有監測站是沒有資料的,Ex:員林監測站)
distances , indices = station_tree.query(twb_coords , k=3)

#  Step4:建立前三順位監測站的Dataframe
nearest_df = pd.DataFrame({
    'nearest_station_1':air_pollution.loc[indices[:,0],'sitename'].values,
    'nearest_station_1_id':air_pollution.loc[indices[:,0],'siteid'].values,
    'nearest_station_1_distance':distances[:,0],
    'nearest_station_2':air_pollution.loc[indices[:,1],'sitename'].values,
    'nearest_station_2_id':air_pollution.loc[indices[:,1],'siteid'].values,
    'nearest_station_2_distance':distances[:,1],
    'nearest_station_3':air_pollution.loc[indices[:,2],'sitename'].values,
    'nearest_station_3_id':air_pollution.loc[indices[:,2],'siteid'].values,
    'nearest_station_3_distance':distances[:,2],
    })

#  Step5:將最近的監測站名稱與距離加到twb_degree
twb_final2 = pd.concat([twb_final2.reset_index(drop=True),nearest_df],axis = 1)
#print(twb_final2.isna().sum())


#                                                  /////  以上先處理好了行政區域碼的對照,並加入了問卷資料  /////
#  接下來處理空氣污染監測站的資料

#                                                                   =====  讀取csv檔  =====

#  2012空氣污染監測資料
df_2012_01 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2012-01).csv")
df_2012_02 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2012-02).csv")
df_2012_03 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2012-03).csv")
df_2012_04 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2012-04).csv")
df_2012_05 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2012-05).csv")
df_2012_06 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2012-06).csv")
df_2012_07 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2012-07).csv")
df_2012_08 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2012-08).csv")
df_2012_09 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2012-09).csv")
df_2012_10 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2012-10).csv")
df_2012_11 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2012-11).csv")
df_2012_12 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2012-12).csv")

df_2012 = pd.concat([df_2012_01,df_2012_02,df_2012_03,df_2012_04,df_2012_05,df_2012_06,df_2012_07,df_2012_08,df_2012_09,df_2012_10,df_2012_11,df_2012_12],axis=0)


#  2013空氣污染監測資料
df_2013_01 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2013-01).csv")
df_2013_02 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2013-02).csv")
df_2013_03 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2013-03).csv")
df_2013_04 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2013-04).csv")
df_2013_05 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2013-05).csv")
df_2013_06 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2013-06).csv")
df_2013_07 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2013-07).csv")
df_2013_08 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2013-08).csv")
df_2013_09 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2013-09).csv")
df_2013_10 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2013-10).csv")
df_2013_11 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2013-11).csv")
df_2013_12 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2013-12).csv")

df_2013 = pd.concat([df_2013_01,df_2013_02,df_2013_03,df_2013_04,df_2013_05,df_2013_06,df_2013_07,df_2013_08,df_2013_09,df_2013_10,df_2013_11,df_2013_12],axis=0)

#  2014空氣污染監測資料
df_2014_01 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2014-01).csv")
df_2014_02 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2014-02).csv")
df_2014_03 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2014-03).csv")
df_2014_04 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2014-04).csv")
df_2014_05 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2014-05).csv")
df_2014_06 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2014-06).csv")
df_2014_07 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2014-07).csv")
df_2014_08 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2014-08).csv")
df_2014_09 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2014-09).csv")
df_2014_10 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2014-10).csv")
df_2014_11 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2014-11).csv")
df_2014_12 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2014-12).csv")

df_2014 = pd.concat([df_2014_01,df_2014_02,df_2014_03,df_2014_04,df_2014_05,df_2014_06,df_2014_07,df_2014_08,df_2014_09,df_2014_10,df_2014_11,df_2014_12],axis=0)
#print(df_2014.columns.tolist())

#  2015空氣污染監測資料
df_2015_01 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2015-01).csv")
df_2015_02 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2015-02).csv")
df_2015_03 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2015-03).csv")
df_2015_04 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2015-04).csv")
df_2015_05 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2015-05).csv")
df_2015_06 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2015-06).csv")
df_2015_07 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2015-07).csv")
df_2015_08 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2015-08).csv")
df_2015_09 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2015-09).csv")
df_2015_10 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2015-10).csv")
df_2015_11 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2015-11).csv")
df_2015_12 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2015-12).csv")

df_2015 = pd.concat([df_2015_01,df_2015_02,df_2015_03,df_2015_04,df_2015_05,df_2015_06,df_2015_07,df_2015_08,df_2015_09,df_2015_10,df_2015_11,df_2015_12],axis=0)

#  2016空氣污染監測資料
df_2016_01 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2016-01).csv")
df_2016_02 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2016-02).csv")
df_2016_03 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2016-03).csv")
df_2016_04 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2016-04).csv")
df_2016_05 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2016-05).csv")
df_2016_06 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2016-06).csv")
df_2016_07 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2016-07).csv")
df_2016_08 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2016-08).csv")
df_2016_09 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2016-09).csv")
df_2016_10 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2016-10).csv")
df_2016_11 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2016-11).csv")
df_2016_12 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2016-12).csv")

df_2016 = pd.concat([df_2016_01,df_2016_02,df_2016_03,df_2016_04,df_2016_05,df_2016_06,df_2016_07,df_2016_08,df_2016_09,df_2016_10,df_2016_11,df_2016_12],axis=0)

#  2017空氣污染監測資料
df_2017_01 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2017-01).csv")
df_2017_02 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2017-02).csv")
df_2017_03 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2017-03).csv")
df_2017_04 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2017-04).csv")
df_2017_05 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2017-05).csv")
df_2017_06 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2017-06).csv")
df_2017_07 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2017-07).csv")
df_2017_08 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2017-08).csv")
df_2017_09 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2017-09).csv")
df_2017_10 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2017-10).csv")
df_2017_11 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2017-11).csv")
df_2017_12 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2017-12).csv")

df_2017 = pd.concat([df_2017_01,df_2017_02,df_2017_03,df_2017_04,df_2017_05,df_2017_06,df_2017_07,df_2017_08,df_2017_09,df_2017_10,df_2017_11,df_2017_12],axis=0)

#  2018空氣污染監測資料
df_2018_01 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2018-01).csv")
df_2018_02 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2018-02).csv")
df_2018_03 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2018-03).csv")
df_2018_04 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2018-04).csv")
df_2018_05 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2018-05).csv")
df_2018_06 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2018-06).csv")
df_2018_07 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2018-07).csv")
df_2018_08 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2018-08).csv")
df_2018_09 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2018-09).csv")
df_2018_10 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2018-10).csv")
df_2018_11 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2018-11).csv")
df_2018_12 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2018-12).csv")

df_2018 = pd.concat([df_2018_01,df_2018_02,df_2018_03,df_2018_04,df_2018_05,df_2018_06,df_2018_07,df_2018_08,df_2018_09,df_2018_10,df_2018_11,df_2018_12],axis=0)

#  2019空氣污染監測資料
df_2019_01 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2019-01).csv")
df_2019_02 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2019-02).csv")
df_2019_03 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2019-03).csv")
df_2019_04 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2019-04).csv")
df_2019_05 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2019-05).csv")
df_2019_06 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2019-06).csv")
df_2019_07 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2019-07).csv")
df_2019_08 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2019-08).csv")
df_2019_09 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2019-09).csv")
df_2019_10 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2019-10).csv")
df_2019_11 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2019-11).csv")
df_2019_12 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2019-12).csv") 

df_2019 = pd.concat([df_2019_01,df_2019_02,df_2019_03,df_2019_04,df_2019_05,df_2019_06,df_2019_07,df_2019_08,df_2019_09,df_2019_10,df_2019_11,df_2019_12],axis=0)

#  2020空氣污染監測資料
df_2020_01 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2020-01).csv")
df_2020_02 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2020-02).csv")
df_2020_03 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2020-03).csv")
df_2020_04 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2020-04).csv")
df_2020_05 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2020-05).csv")
df_2020_06 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2020-06).csv")
df_2020_07 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2020-07).csv")
df_2020_08 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2020-08).csv")
df_2020_09 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2020-09).csv")
df_2020_10 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2020-10).csv")
df_2020_11 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2020-11).csv")
df_2020_12 = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/空氣品質監測日平均值(一般污染物) (2020-12).csv") 

df_2020 = pd.concat([df_2020_01,df_2020_02,df_2020_03,df_2020_04,df_2020_05,df_2020_06,df_2020_07,df_2020_08,df_2020_09,df_2020_10,df_2020_11,df_2020_12],axis=0)


#                                                                  =====  2.空氣污染監測站資料處理  =====
#  1.更改欄位名稱，把“”拿掉
def renamerow(df):
    df.rename(columns={'"siteid"':"siteid",
                       '"sitename"':"sitename",
                       '"itemid"':"itemid",
                       '"itemname"':"itemname",
                       '"itemengname"':"itemengname",
                       '"itemunit"':"itemunit",
                       '"monitordate"':"monitordate",
                       '"concentration"':"concentration"},inplace=True)
    return df

df_2013=renamerow(df_2013)
df_2014=renamerow(df_2014)
df_2015=renamerow(df_2015)
df_2016=renamerow(df_2016)
df_2017=renamerow(df_2017)
df_2018=renamerow(df_2018)
df_2019=renamerow(df_2019)
df_2020=renamerow(df_2020)

#  2.轉換欄位，把污染物變成欄(長格式變寬格式)
def pivot_col(df):
    
    #將concentration轉為數值資料,並將無法轉成數值資料nan顯示
    df['concentration'] = pd.to_numeric(df['concentration'],errors='coerce')
    
    #建立寬表格
    df_wide = df.pivot_table(
        index = ['monitordate','siteid','sitename'],
        columns = 'itemengname',
        values = 'concentration').reset_index()
    
    #回傳資料表
    return df_wide

df_2013_wide = pivot_col(df_2013)
df_2014_wide = pivot_col(df_2014)
df_2015_wide = pivot_col(df_2015)
df_2016_wide = pivot_col(df_2016)
df_2017_wide = pivot_col(df_2017)
df_2018_wide = pivot_col(df_2018)
df_2019_wide = pivot_col(df_2019)
df_2020_wide = pivot_col(df_2020)

#  df_2017_wide多了一個欄位WD_HR'，但都是nan,故drop掉
df_2017_wide.drop(columns = ['WD_HR'] , inplace = True)

#  將2013,2014的空氣污染資料合併再一起,並只抓取2013/06/60-2016/06/30
pollution_1314 = """
    SELECT *
    FROM (
        SELECT * FROM df_2013_wide
        UNION ALL  
        SELECT * FROM df_2014_wide
        UNION ALL 
        SELECT * FROM df_2015_wide
        UNION ALL  
        SELECT * FROM df_2016_wide
        UNION ALL  
        SELECT * FROM df_2017_wide
        UNION ALL  
        SELECT * FROM df_2018_wide
        UNION ALL  
        SELECT * FROM df_2019_wide
        UNION ALL  
        SELECT * FROM df_2020_wide
        ) AS merge_pollution
    ORDER BY monitordate;
    """

#  將合併好的地點資料放在pollution_1314中
pollution_1314 = pysqldf(pollution_1314)
#  把result0加入到全域命名空間中
globals()['pollution_1314'] = pollution_1314

print(pollution_1314.columns)

#  變更pollution_1314的PM2.5的欄位名稱,SQL才有辦法使用
pollution_1314.rename(columns={'PM2.5':'PM25'} , inplace = True)

#  只抓監測站都共同擁有的污染物質，計算每個欄位的nan比例
na_ratio = pollution_1314.isna().mean()

#篩選出nan比例低於10%的欄位，在轉成list
threshold = 0.1
valid_columns = na_ratio[na_ratio < threshold].index.tolist()

base_columns = ['monitordate', 'siteid', 'sitename']
final_columns = base_columns + [col for col in valid_columns if col not in base_columns]

#  污染監測站的缺失值填補
#  取得過濾後的資料
pollution_1314_filter = pollution_1314[final_columns].copy()
print(pollution_1314_filter.isna().sum())

#  首先先將2013,2014空氣污染資料與經緯度合併，以便之後以鄰近資料填補缺失值
query = """
    SELECT 
        p.monitordate,
        p.siteid,
        p.sitename,
        p.AMB_TEMP,
        p.CO,
        p.NO, 
        p.NO2, 
        p.NOx, 
        p.O3,
        p.PM10, 
        p.PM25 ,
        p.RH,
        p.SO2,
        p.WIND_SPEED,
        p.WS_HR,
        a.lon,
        a.lat
    FROM pollution_1314_filter AS p
    LEFT JOIN air_pollution AS a
    ON p.siteid = a.siteid
"""
#  將合併好的經緯資料放在pollution_1314中
pollution_1314_fillter_1 = pysqldf(query)
#  把result0加入到全域命名空間中
globals()['pollution_1314_fillter_1'] = pollution_1314_fillter_1


#  先尋找鄰近監測站的距離，在尋找同一天填寫問卷的日期做填補
#  建立cKDTree來尋找鄰近監測站填補缺失值
#  Step1:建立空氣污染監測站的KD-Tree(經緯度)，為了建立KD-Tree
#  在前面統計受測者資料時已經建立了台灣監測站經緯度資料：station_tree
#  填補缺失值：監測站本身沒有該測量，
#  siteid = 10(淡水)本身就沒有RH,WIND_SPEED,WS_HR,AMB_TEMP
#  siteid = 16(大同)本身就沒有RH,WIND_SPEED,WS_HR,AMB_TEMP,O3
#  siteid = 64(陽明)本身就沒有WIND_SPEED,WS_HR

#  Step2:將各監測站歷年的經緯度轉為陣列，拿來當作查詢點
pollution_coords = pollution_1314_fillter_1[['lon','lat']].to_numpy()

#  Step3:查詢每個監測站索引與距離，k=2代表找出最接近的2個監測站
distances_pollution , indices_pollution = station_tree.query(pollution_coords , k=3)

#  Step4:建立前2順位監測站的Dataframe,因為前面已經根據air_pollution建立KD-Tree,而這個資料是從air_pollution計算的，所以要從air_pollution取值
nearest_pollution_df = pd.DataFrame({
    'nearest_station_1':air_pollution.loc[indices_pollution[:,0],'sitename'].values,
    'nearest_station_1_id':air_pollution.loc[indices_pollution[:,0],'siteid'].values,
    'nearest_station_1_distance':distances_pollution[:,0],
    'nearest_station_2':air_pollution.loc[indices_pollution[:,1],'sitename'].values,
    'nearest_station_2_id':air_pollution.loc[indices_pollution[:,1],'siteid'].values,
    'nearest_station_2_distance':distances_pollution[:,1],
    'nearest_station_3':air_pollution.loc[indices_pollution[:,2],'sitename'].values,
    'nearest_station_3_id':air_pollution.loc[indices_pollution[:,2],'siteid'].values,
    'nearest_station_3_distance':distances_pollution[:,2]
    })

#  Step5:將找到的鄰近資料合併回原本的資料中
pollution_1314_final = pd.concat([pollution_1314_fillter_1.reset_index(drop = True),nearest_pollution_df],axis = 1)

#  Step6:將有缺失值的污染值就鄰近監測站補植，並根據同一天填寫日期
#  先確認有哪些特徵欄位有缺失值
print(pollution_1314_final.isna().sum())

#  iterrows()是pandas DataFrame的一個方法，用來逐行遍歷資料,iterrows()會回傳一個產生器(generator)，每次迭代會回傳(index, row)這對tuple
def pollution_filled(df,feature):
    for idx,row in df.iterrows():
        if pd.isna(row[feature]):
            date = row['monitordate']
            
            #建立前三siteid
            nearest_sites = [row['nearest_station_1_id'],row['nearest_station_2_id'],row['nearest_station_3_id']]
            
            filled = False   #追蹤是否成功使用前三個鄰近監測站補值，預設這筆資料還沒被填補
            for site_id in nearest_sites:
                pollution_value = df[
                    (df['siteid'] == site_id) &
                    (df['monitordate'] == date)][feature]
            
                #如果找到鄰近數值，且該資料不為缺失值，就以該值填補
                if not pollution_value.empty and not pd.isna(pollution_value.values[0]):
                    df.at[idx,feature] = pollution_value.values[0]
                    filled = True   #已填補完成
                    #補到一個就停止往下補
                    break
                        
            #如果前三鄰近資料補完依舊有缺失值，就使用前後三天的同站資料
            if not filled:
                #先把date轉成datetime
                date = pd.to_datetime(row['monitordate']) 
                #先把monitordate轉乘datetime格式
                df['monitordate'] = pd.to_datetime(df['monitordate'])
                pollution_value = df[
                    (df['siteid'] == row['siteid']) & 
                    (df['monitordate'].between(date - pd.Timedelta(days=3) , date + pd.Timedelta(days=3)))][feature].dropna()
                #確定前後三天有資料再補
                if not pollution_value.empty:
                    df.at[idx,feature] = pollution_value.mean()
            
    return df
                
#  先抓出數值行欄位
value_columns = ['AMB_TEMP', 'CO', 'NO', 'NO2','NOx', 'O3', 'PM10', 'PM25', 'RH', 'SO2', 'WIND_SPEED', 'WS_HR']

#  為了保護原始資料不被污染，因此建立副本
pollution_1314_final_1 = pollution_1314_final.copy()

#  因為所有空氣污染的特徵欄位均有缺失值，因此會寫一個函式來一次呼叫
for feature in value_columns:
    pollution_1314_final_1 = pollution_filled(pollution_1314_final , feature)
            
#  確認是否還有有哪些特徵欄位有缺失值
print(pollution_1314_final_1.isna().sum())

print(pollution_1314_final_1.columns)

#  只留下有需要的欄位，第幾監測站補值那些特徵欄位不需要了
pollution_1314_final = pollution_1314_final_1.drop(columns=['lon','lat', 'nearest_station_1','nearest_station_1_id','nearest_station_1_distance',
                                                            'nearest_station_2','nearest_station_2_id', 'nearest_station_2_distance','nearest_station_3',
                                                            'nearest_station_3_id','nearest_station_3_distance'],axis = 1)

#                                                        =====  計算最接近的監測站一年平均污染物質  =====

"""
['monitordate', 'siteid', 'sitename', 'AMB_TEMP', 'CO', 'NO', 'NO2',
       'NOx', 'O3', 'PM10', 'PM25', 'RH', 'SO2', 'WIND_SPEED', 'WS_HR']

AND julianday(main.SURVEY_DATE) - julianday(sub.monitordate) BETWEEN 0 AND 183
"""

#  需要先把SURVEY_DATE(年/月/日)跟monitordate(年-月-日)的日期格式改成一樣的
#  先轉成datetime格式
twb_final2['SURVEY_DATE'] = pd.to_datetime(twb_final2['SURVEY_DATE'] , format='%Y/%m/%d')
#  再轉成字串格式：年-月-日
twb_final2['SURVEY_DATE'] = twb_final2['SURVEY_DATE'].dt.strftime('%Y-%m-%d')

#  確認nearest_station或siteid格式，因兩個欄位的資料型態不同，要都變更成object才可以
pollution_1314_final['siteid'] = pollution_1314_final['siteid'].astype(str)
#twb_final2['nearest_station_id'] = twb_final2['nearest_station_id'].astype(str)


#  要把SURVEY_DATE,monitordate資料改為datetime資料
twb_final2['SURVEY_DATE'] = pd.to_datetime(twb_final2['SURVEY_DATE'])
pollution_1314_final['monitordate'] = pd.to_datetime(pollution_1314_final['monitordate'])

#  先建立一個包含所有污染資料的監測站ID的集合Set,為了後續方便查找,因資料型態為str,需轉成數字才能跟twb_final2的nearest_station_id匹配
valid_sites = set(pollution_1314_final['siteid'].astype(float).unique())

#  定義函式：如果沒有第一近的監測站，就取第二近的以此類推
def valid_pollution_station(row):
    if row['nearest_station_1_id'] in valid_sites:
        return row['nearest_station_1_id']
    elif row['nearest_station_2_id'] in valid_sites:
        return row['nearest_station_2_id']
    elif row['nearest_station_3_id'] in valid_sites:
        return row['nearest_station_3_id']
    else:
        return np.nan   #都沒有時就射Nan

#  apply(..., axis=1)：表示會對每一列（每個 row）呼叫 get_valid_station(row)
twb_final2['final_station_id'] = twb_final2.apply(valid_pollution_station,axis = 1)

#  合併問卷資料跟空氣污染值,根據監測站siteid
Avg_pollution_1314 = """
    SELECT 
        main.*,
        
        --空氣污然欄位
        
        AVG(sub.AMB_TEMP) AS avg_AMB_TEMP,
        AVG(sub.CO) AS avg_CO,
        AVG(sub.NO) AS avg_NO,
        AVG(sub.NO2) AS avg_NO2,
        AVG(sub.NOx) AS avg_NOx,
        AVG(sub.O3) AS avg_O3,
        AVG(sub.PM10) AS avg_PM10,
        AVG(sub.PM25) AS avg_PM25,
        AVG(sub.RH) AS avg_RH,
        AVG(sub.SO2) AS avg_SO2,
        AVG(sub.WIND_SPEED) AS avg_WIND_SPEED,
        AVG(sub.WS_HR) AS avg_WS_HR
        
    FROM twb_final2 AS main
    LEFT JOIN pollution_1314_final AS sub
        ON main.final_station_id = sub.siteid
        AND julianday(main.SURVEY_DATE) - julianday(sub.monitordate) BETWEEN 0 AND 365
    GROUP BY main.Release_No
    """
#  將合併好的地點資料放在Avg_pollution_1314中
Avg_pollution_1314 = pysqldf(Avg_pollution_1314)
#  把result0加入到全域命名空間中
globals()['Avg_pollution_1314'] = Avg_pollution_1314

#  把不需要的欄位刪除，因已經合併平均值了，所以監測站經緯度可刪除
Avg_pollution_1314 = Avg_pollution_1314.drop(columns=['PLACE_CURR', 'lon', 'lat', 'num4_location', 'nearest_station_1',
                                                      'nearest_station_1_id', 'nearest_station_1_distance','nearest_station_2', 
                                                      'nearest_station_2_id','nearest_station_2_distance', 'nearest_station_3',
                                                      'nearest_station_3_id', 'nearest_station_3_distance'] , axis = 1)

#  確認缺失值
print(Avg_pollution_1314.isna().sum())


'''
#---------------檢查為何有nan值，而且都在彰化------------------
nan_rows = Avg_pollution_1314(Avg_pollution_1314['avg_PM25'].isnull()) 

print(nan_rows[['Release_No', 'nearest_station_id', 'SURVEY_DATE']].head())

#  發現有nan整列的都是彰化鄉鎮，會發現這些資料的監測站都是員林站
nan_in_changhua = Avg_pollution_1314[(Avg_pollution_1314['num4_location2'].str.contains('彰化',na=False)) & (Avg_pollution_1314[['avg_AMB_TEMP', 'avg_CO', 'avg_NO', 'avg_NO2', 'avg_NOx',
                         'avg_O3', 'avg_PM10', 'avg_PM25', 'avg_RH','avg_SO2', 'avg_WIND_SPEED', 'avg_WS_HR']].isnull().any(axis=1))]
#print(nan_in_changhua)
#  先確認污染資料中是否有員林站的資料： pollution_1314_filter 裡是否有員林站資料。
nan_yuanlin = pollution_1314_filter[pollution_1314_filter['sitename'] == '員林']
#  結果發現員林站的空氣污染監測站這半年內完全沒資料

#  回頭找看看原始污染資料一開始是否有員林站資料:會發現沒有員林監測站的資料
nan_yualin_original = pollution_1314[pollution_1314['sitename'] == '員林']

#  解決方法：1.直接從一開始刪除員林監測站  2.選取第二接近的監測站(較優)！

#----------------------------------------------------------------
'''

#  因不同監測站對於污染物的監測不同，因此nan值用0補
col_to_fill = [
    'avg_AMB_TEMP','avg_CO','avg_NO','avg_NO2','avg_NOx','avg_O3',
    'avg_PM10','avg_PM25','avg_RH','avg_SO2','avg_WIND_SPEED','avg_WS_HR']

#Avg_pollution_1314[col_to_fill] = Avg_pollution_1314[col_to_fill].fillna(0)

#  畫出問卷中各污染物的半年平均的圖
#這個會生成一個Series(污染物不是單獨一欄特徵欄位，而是索引值)，因此要轉成DataFrame
avg_twb_pollutant = Avg_pollution_1314[col_to_fill].mean().reset_index()
avg_twb_pollutant.columns = ['Pollutant','Mean']


plt.figure(figsize = (14,6))
ax = sns.barplot(data = avg_twb_pollutant , x='Pollutant' , y= 'Mean' , palette='muted' , hue = 'Pollutant', legend=False)
plt.xticks(rotation=90 , ha='right')
plt.title("TWB mean value of pollutant")
plt.xlabel('Pollutant')
plt.ylabel("Mean value")

#加上數值標籤
for i,bar in enumerate(ax.patches):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.1f}",
        ha = 'center',
        fontsize = 8)

plt.tight_layout()
#plt.savefig("/Users/anyuchen/Desktop/空氣污染/graph/TWB mean value of pollutant.png",dpi = 300,bbox_inches='tight')
plt.show()


#                                                     =====  計算每個空氣污染監測站的統計資料，並繪出圖  =====

#  Step1:定義繪圖計算函式-根據每個監測站、每年、每個污染物繪圖
def avg_pollution_station_for_a_year(df):
    
    #先設定所有需要繪圖的污染物欄位
    columns = ['AMB_TEMP', 'CH4', 'CO', 'CO2','NMHC', 'NO', 'NO2', 'NOx', 'O3', 'PH_RAIN', 'PM10', 'PM2.5',
               'RAIN_COND', 'RAIN_INT', 'RH', 'SO2', 'THC', 'WIND_SPEED', 'WS_HR']
    
    #並計算有所污染物的平均,標準差,最大最小值
    avg_pollution = df.groupby('sitename')[columns].agg(['mean','std','min','max']).reset_index()
    
    #  因使用groupby.agg()會產生多層欄位，需移除多層欄位名稱，並要保留原本的sitename
    avg_pollution.columns = ['sitename' if col[0] == 'sitename' else f"{col[0]}_{col[1]}" for col in avg_pollution.columns]

    #  將資料欄位轉為長格式，在繪圖上才方便繪圖
    mean_col = [col for col in avg_pollution.columns if 'mean' in col and col != 'sitename']
    mean_melt = avg_pollution.melt(id_vars = 'sitename',value_vars = mean_col , var_name = 'Pollutant' , value_name = 'mean_val')
    
    #清理欄位
    mean_melt['Pollutant'] = mean_melt['Pollutant'].str.replace('_mean',"")
    
    #  根據不同污染物繪圖來比較不同的監測站的差異
    pollutants = mean_melt['Pollutant'].unique()
    
    #  繪製個監測站的統計圖表
    # 指定字體檔案路徑
    font_path = "/System/Library/Fonts/STHeiti Medium.ttc"
    my_font = fm.FontProperties(fname=font_path)

    #  遞迴所有污染物質
    output_dir = "/Users/anyuchen/Desktop/空氣污染/graph"
    os.makedirs(output_dir, exist_ok=True)

    for pollutant in pollutants:
        plt.figure(figsize=(14,6))
        data_subset = mean_melt[mean_melt['Pollutant'] == pollutant]
        
        ax = sns.barplot(data=data_subset,x='sitename',y='mean_val',palette='muted',hue = 'sitename', legend = False)
            
        plt.xticks(rotation=90,ha='right', fontproperties=my_font)
        plt.title(f"{pollutant} - Mean value by Monitoring Station (2014)", fontproperties=my_font)
        plt.ylabel('Mean Value', fontproperties=my_font)
        plt.xlabel('Monitoring Station', fontproperties=my_font)
        
        #加上數值標籤
        #ax.patch是一個存放圖中每個bar的list,包含bar的座標,寬度,高度
        for i,bar in enumerate(ax.patches):
            height = bar.get_height()   # 取得這個長條的高度（也就是 mean_val 數值）
            ax.text(
                bar.get_x() + bar.get_width() / 2,   # 計算這個 bar 的中間位置（X 軸）
                height,                              # Y 軸：數值標示顯示在長條的頂部
                f"{height:.1f}",                     # 顯示的文字內容，保留一位小數
                ha='center',                         # 水平對齊方式（center：置中）
                fontsize = 8,    
                rotation = 90)
            
        plt.tight_layout()
        save_path = f"{output_dir}/{pollutant}__mean_value.png"
        plt.savefig(save_path,dpi = 300,bbox_inches='tight')
        plt.show()
        
    #avg_pollution 是每個站每個污染物的統計彙整（多層指標版）,mean_melt 是繪圖用的長格式資料
    return avg_pollution , mean_melt
        
avg_pollution_2013 , meam_melt_2013 = avg_pollution_station_for_a_year(df_2013_wide)
avg_pollution_2014 , meam_melt_2014 = avg_pollution_station_for_a_year(df_2014_wide)



#                                                            =====  整理問卷基本資料(非空污資料)  =====

#  透過Release NO合併X,y
df_MMSE = pd.read_excel(r"/Users/anyuchen/Desktop/空氣污染/dataset/MMSE.xlsx")
print(df_MMSE['MMSE'].value_counts())
Avg_pollution_1314 = Avg_pollution_1314.merge(df_MMSE,on='Release_No',how='left')
print(Avg_pollution_1314['MMSE'].value_counts())

#  只保留10%缺失的欄位：'AGE', 'SEX', 'EDUCATION','MARRIAGE','DRK','SPO_HABIT'
missing_ratio = Avg_pollution_1314.isnull().mean()
low_missing = missing_ratio[missing_ratio<0.1]
#print(low_missing)
"""
Release_No        0.000000
SURVEY_DATE       0.000000
ID_BIRTH          0.000000
AGE               0.000000
SEX               0.000000
EDUCATION         0.000000
MARRIAGE          0.000000
DRK               0.000000
SPO_HABIT         0.000000
avg_AMB_TEMP      0.002749
avg_CO            0.000196
avg_NO            0.000000
avg_NO2           0.000000
avg_NOx           0.000000
avg_O3            0.001374
avg_PM10          0.000000
avg_PM25          0.000000
avg_RH            0.002749
avg_SO2           0.000000
avg_WIND_SPEED    0.002945
avg_WS_HR         0.002945
MMSE              0.000000
"""
base_info_col = ['AGE', 'SEX', 'EDUCATION','MARRIAGE','DRK','SPO_HABIT','avg_AMB_TEMP',
                  'avg_CO','avg_NO', 'avg_NO2', 'avg_NOx','avg_O3','avg_PM10','avg_PM25',
                 'avg_RH', 'avg_SO2','avg_WIND_SPEED', 'avg_WS_HR','MMSE']

#  確認所有欄位有哪些數值
for col in base_info_col:
    print(f"{col}: {Avg_pollution_1314[col].unique()}")
    print(f"{col}: {Avg_pollution_1314[col].info()}")
    print(f"{col}: {Avg_pollution_1314[col].isna().sum()}")

#  保留缺失值低於0.1的特徵欄位
Avg_pollution_1314_final = Avg_pollution_1314[base_info_col]

#  刪掉有缺失值問卷資料:DRK,SPO_HABIT,確認執行完沒有None值了
Avg_pollution_1314_final = Avg_pollution_1314_final.dropna(subset=base_info_col)

#  因婚姻狀況有2筆是R,不知道意義是什麼故刪掉
Avg_pollution_1314_final = Avg_pollution_1314_final[Avg_pollution_1314_final['MARRIAGE']!='R']

#  確認是否最終資料有缺失值
#  建立所有資料對照表
#  學歷
education_map = {'1': '未受教育','2': '小學','3': '國中','4': '高中/高職','5': '專科','6': '大學','7': '研究所以上'}
print(Avg_pollution_1314_final['EDUCATION'].value_counts())

#  婚姻狀況
marriage_map = {'1': '未婚','2': '已婚','3': '離婚','4': '喪偶','R': '拒答/不明'}
print(Avg_pollution_1314_final['MARRIAGE'].value_counts())

#  喝酒習慣
drk_map = {'1': '不喝','2': '偶爾喝','3': '經常喝',None: '缺漏'}
print(Avg_pollution_1314_final['DRK'].value_counts())
#  將拒答或其他非正常答覆(=R/N)的資料刪除
Avg_pollution_1314_final = Avg_pollution_1314_final[~Avg_pollution_1314_final['DRK'].isin(['R','N'])]

#  運動狀況
spo_habit_map = {'1': '規律運動','2': '不規律或不運動',None: '缺漏'}
print(Avg_pollution_1314_final['SPO_HABIT'].value_counts())
#  將拒答或其他非正常答覆(=R/N)的資料刪除
Avg_pollution_1314_final = Avg_pollution_1314_final[~Avg_pollution_1314_final['SPO_HABIT'].isin(['R'])]

#  因有些資料數值為float，應改為int資料型態:DRK,SMK_CURR,DRUG_USE,ILL_ACT,SPO_HABIT,COOK_FREQ,WATER_BOILED,EGE_YR,HORMOME_MEDHERBAL_MED,JOB_EXPERIENCE
for col in base_info_col:
    Avg_pollution_1314_final[col] = Avg_pollution_1314_final[col].astype(float).astype(int)


#  因合併失敗，確認兩者的Release_No格式統一,為字串去除空白,轉大寫
#df_MMSE['Release_No'] = df_MMSE['Release_No'].astype(str).str.strip().str.upper()
#Avg_pollution_1314['Release_No'] = Avg_pollution_1314['Release_No'].astype(str).str.strip().str.upper()

print(Avg_pollution_1314_final.columns)

print(Avg_pollution_1314_final.info())


#                                                                ====  資料標準化:數值/類別  =====
'''
數值資料：
['AGE', 'avg_AMB_TEMP', 'avg_CO', 'avg_NO', 'avg_NO2', 'avg_NOx', 'avg_O3', 'avg_PM10', 'avg_PM25', 'avg_RH', 'avg_SO2', 'avg_WIND_SPEED','avg_WS_HR']
'''
num_col = ['AGE', 'avg_AMB_TEMP', 'avg_CO', 'avg_NO', 'avg_NO2', 'avg_NOx', 'avg_O3', 'avg_PM10', 'avg_PM25', 'avg_RH', 'avg_SO2', 'avg_WIND_SPEED','avg_WS_HR']
category_col = ['SEX', 'EDUCATION', 'MARRIAGE', 'DRK', 'SPO_HABIT','MMSE']

model_data = Avg_pollution_1314_final.copy()

#  數值資料需要標準化：
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler_data = scaler.fit_transform(model_data[num_col])

#  把標準化後的資料放回原本的data
model_data[num_col] = scaler_data

#  類別資料需要One Hot Encoding/Label Encoding,，二元類別不需要OneHot,one hot encoding是針對沒有次序性的多類別欄位
'''
類別資料：['SEX', 'EDUCATION', 'MARRIAGE', 'DRK', 'SPO_HABIT','MMSE']
'''
multiple_col = ['EDUCATION', 'MARRIAGE', 'DRK']
#  先確認資料型態是否為數值or字串，若為整數資料需要改為字串或類別，才能dummies
print(model_data.info())
#  資料都回數值，需要改成類別
model_data[multiple_col] = model_data[multiple_col].astype(str)
print(model_data.info())
#  將多元類別資料做one hot encoding,因為在python中做dummies會產生布林值，如果想要呈現0/1要再轉回數值
model_data = pd.get_dummies(model_data,columns = multiple_col , drop_first=True).astype(int)

print(model_data['MMSE'].value_counts())

#                                                                    =====  模型訓練(要使用平衡後的資料)  =====
#  需要丟進模型訓練的欄位
'''
['AGE', 'SEX', 'EDUCATION', 'MARRIAGE', 'DRK', 'SPO_HABIT','avg_AMB_TEMP', 'avg_CO', 'avg_NO', 'avg_NO2', 'avg_NOx', 'avg_O3',
 'avg_PM10', 'avg_PM25', 'avg_RH', 'avg_SO2', 'avg_WIND_SPEED','avg_WS_HR', 'MMSE']
'''

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from itertools import cycle  #用於生成色彩循環
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

#  切割自變數、因變數：是否確診失智的資料
base_x = model_data.drop('MMSE',axis =1)
base_y = model_data['MMSE']
print(base_y.shape)
print(base_x.shape)

#  切割訓練測試資料
X_train,X_test,y_train,y_test = train_test_split(base_x, base_y, test_size = 0.2,random_state=24)

#  針對訓練資料做SMOTE資料平衡：因變數y存在資料不平衡:0為4016筆/1為58筆,SMOTE完以後都是4016
#print(base_y.value_counts())
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=24)
X_train_sm , y_train_sm = smote.fit_resample(X_train, y_train)

print("Before SMOTE:" , y_train.value_counts())
print("Before SMOTE:" , y_train_sm.value_counts())

#  先計算正負樣本比例:負類樣本/正類樣本
scale = (base_y == 0).sum() / (base_y == 1).sum()

#  建立模型及設定超參數(邏輯回歸Logistic Regression做baseline)
models_param_grids = {
    'Logistic Regression':{
        'model':LogisticRegression(class_weight = 'balanced'),
        'params':{
            'C':[0.01,0.1,1,2],
            'penalty':['l2'],
            'max_iter':[1000]
            }
        },
    'Random Forest':{
        'model':RandomForestClassifier(),
        'params':{
            'n_estimators':[50,100],
            'criterion':['gini','entropy'],
            'max_depth':[None,10,20,30]
            }
        },
    #'SVM':{
        #'model':SVC(probability=True),
        #'params':{
            #'kernel':['linear','rbf'],
            #'C':[0.1,1,10],
            #'gamma':['scale','auto']
            #}
       # },  
    #新增一個XGBoost:處理極度不平衡資料
    #use_label_encoder = False  防止警告訊息
    #eval_metric = 'logloss'  必填，不然會跳警告
    'XGBoost':{
        'model':XGBClassifier(eval_metric = 'logloss',random_state = 24),
        'params':{
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [50, 100],
            #'scale_pos_weight': [scale]  因有做SMOTE資料平衡，不需要再平衡
            }
        }
    }

#  4.Grid_Search自動搜尋最佳參數
#  需要將結果儲存在results中，因為繪製ROC只需要y_prob的結果，因此需要將所有模型的結果儲存在一起，在繪圖時一次叫出來即可
results={}
for name,config in models_param_grids.items():
    print(f"Performing GridSearchCV for {name}...")
    grid_search=GridSearchCV(estimator=config["model"],
                             param_grid=config["params"],
                             cv=5,
                             scoring='recall',  #主要關心是「能抓出失智者」，你應該以：Recall（靈敏度）或 F1-score 為主要指標，而不是 Accuracy。
                             verbose=2,
                             n_jobs=-1)
    grid_search.fit(X_train_sm,y_train_sm)
    #取得最佳模型
    best_model=grid_search.best_estimator_
    y_prob=best_model.predict_proba(X_test)[:,1]  #獲取正類別概率
    #呼叫model_train來做評估，因為model_train()函式回傳return model,y_pred,yprob,但因為我的程式碼中只需要y_pred,y_prob,因此填了一個_為忽略
    y_pred=best_model.predict(X_test)
    results[name]=(best_model,y_prob)
    
    print(f"Best parameters for {name} : {grid_search.best_params_}")
    #輸出分類報告
    print(f"====Classification Report for {name}====\n",classification_report(y_test,y_pred))
    

#  再透過Random Forest or XGBoost提升準確率,並分析特徵重要性




























































