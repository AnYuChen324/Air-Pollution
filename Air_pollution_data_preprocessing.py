#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 18:05:56 2025

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
from glob import glob

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

#  受測者調查問卷
twb_survey = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/twb/TWBR10907-07_調查問卷.csv")

#  受測者健康問卷
twb_health = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/twb/TWBR10907-07_健康問卷.csv" , encoding="big5")

#  受測者體檢問卷
twb_examination = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/twb/TWBR10907-07_體檢與檢驗.csv" , encoding="big5")

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



#                                                              =====  健康、體檢、調查問卷前處理  =====

#  只保留有可能有影響的特徵欄位：且部分問卷有第一次測試與第二次，一律都以第一次測試為主FOLLOW=Baseline
#  因為health特徵欄位缺失值過多，每個都超過10％，就不採納
twb_examination = twb_examination[twb_examination['FOLLOW'] == 'Baseline']
twb_survey = twb_survey[twb_survey['FOLLOW'] == 'Baseline']

#  合併兩個問卷資料，沒有使用SQL語法是因為會有重複欄位
twb = pd.merge(twb_examination, twb_survey, how='inner', on='Release_No')

#  將重複欄位FOLLOW_x,FOLLOW_y刪除一個，並改回FOLLOW
twb = twb.drop(columns = ['FOLLOW_x'])
twb = twb.rename(columns = {'FOLLOW_y':'FOLLOW'})

'''  =====  會得到共122068筆資料，683個欄位  =====  '''

#  保留對失智症存在較高影響力的特徵（整合行為、生活與臨床數據）
selected_features = [
    # 基本人口學特徵
    'AGE', 'SEX', 'EDUCATION', 'MARRIAGE', 'FOLLOW',

    # 行為與生活習慣
    'SMK_EXPERIENCE', 'DRK', 'SPO_HABIT', 'SUPP', 'INCENSE_CURR', 'COOK_FUEL', 'COOK_SIXTHMONTHS',
    'INCOME_SELF', 'INCOME_FAMILY',

    # 身體測量
    'BODY_HEIGHT', 'BODY_WEIGHT', 'BODY_FAT_RATE', 'BODY_WAISTLINE',

    # 血壓測量（血管健康與腦血流息息相關）
    'SIT_1_SYSTOLIC_PRESSURE', 'SIT_1_DIASTOLIC_PRESSURE',
    'SIT_2_SYSTOLIC_PRESSURE', 'SIT_2_DIASTOLIC_PRESSURE',
    'SIT_3_SYSTOLIC_PRESSURE', 'SIT_3_DIASTOLIC_PRESSURE',

    # 血糖與糖化血色素（糖尿病是失智重要風險因子）
    'FASTING_GLUCOSE', 'HBA1C',

    # 血脂與膽固醇（影響腦血管與阿茲海默症）
    'T_CHO', 'TG', 'HDL_C', 'LDL_C',

    # 腎功能指標（與代謝性疾病及神經退化有關）
    'CREATININE', 'URIC_ACID', 'MICROALB', 'BUN',

    # 肝功能與代謝指標（輔助觀察全身慢性狀況）
    'SGOT', 'SGPT', 'GAMMA_GT', 'ALBUMIN',

    # 血球與免疫指標（可觀察系統性發炎）
    'WBC', 'RBC', 'PLATELET', 'HB', 'HCT',
    
    #居住區域
    'PLACE_CURR','SURVEY_DATE','Release_No',
]

#  只抓影響力較大的特徵作為訓練資料
twb = twb[selected_features]

#  檢查缺失值欄位
twb.isnull().sum()

#  檢查缺失值欄位的比例
twb.isnull().mean()*100

#  將缺失值的資料合併成表格，會比較清楚
twb_miss_data = pd.DataFrame({
    'Missing_Count':twb.isnull().sum(),
    'Missing_Ratio(%)':twb.isnull().mean()*100})

#  抓出有缺值得欄位
twb_missing_summary = twb_miss_data[twb_miss_data['Missing_Count'] > 0]
twb_missing_summary = twb_missing_summary.sort_values(by='Missing_Count',ascending = False)

import missingno as msno
# 條狀圖：缺失比例
msno.bar(twb)

# 矩陣圖：顯示缺失分布情形
msno.matrix(twb)

#  只保留nan值<10%的欄位
def low_miss_col(df):
    missing_ratio =  df.isnull().mean()
    low_miss = missing_ratio[missing_ratio<0.1]   #會產生低於10%的欄位跟比例(索引值為特徵欄,索引值0=比例)
    df = df[low_miss.index]
    
    #回傳低於10%的資料
    return df

twb = low_miss_col(twb)

#  重新檢視缺失值狀態
# twb.isnull().sum()
# twb.isnull().mean() * 100

#  刪除有nan值的資料
twb_clean = twb.dropna().copy()

#  將血壓取平均值，較公正
twb_clean['avg_SYSTOLIC_PRESSURE'] = twb_clean[['SIT_1_SYSTOLIC_PRESSURE','SIT_2_SYSTOLIC_PRESSURE']].mean(axis = 1, skipna=False)
twb_clean['avg_DIASTOLIC_PRESSURE'] = twb_clean[['SIT_1_DIASTOLIC_PRESSURE','SIT_2_DIASTOLIC_PRESSURE']].mean(axis = 1, skipna=False)

#  刪除不需要的欄位
twb_clean.drop(columns = ['SIT_1_SYSTOLIC_PRESSURE','SIT_2_SYSTOLIC_PRESSURE','SIT_1_DIASTOLIC_PRESSURE','SIT_2_DIASTOLIC_PRESSURE'] , inplace = True)



#                                                      =====   問卷twb_clean跟中心經緯度合併，透過3,4碼行政區域碼  =====
"""
twb_3num = pd.merge(twb , city_num , left_on='PLACE_CURR' , right_on='3num' , how='left')
twb_4num = pd.merge(twb_3num , city_num , left_on='PLACE_CURR' , right_on='4num' , how='left')
"""

#  把唯一一筆Place_curr = R以及Place_curr為nan的資料刪除(同時處理字串 "nan" 和真正的 NaN 缺失值)
twb_clean = twb_clean[~twb_clean['PLACE_CURR'].isin(['R','nan']) & ~twb_clean['PLACE_CURR'].isna()]

#  發現有四碼的區域碼後面多了.0,顯示為R的要刪除
twb_clean.loc[:,'PLACE_CURR'] = twb_clean['PLACE_CURR'].astype(str).str.replace(r'\.0$','',regex=True)

#  再將資料轉為數值型態
#twb['PLACE_CURR'] = twb['PLACE_CURR'].astype(int)

#  先合併四碼行政區域碼
twb_4num = """
    SELECT 
        twb.*,
        city_num.lon AS num4_lon,
        city_num.lat AS num4_lat,
        city_num.location AS num4_location
    FROM twb_clean AS twb
    LEFT JOIN city_num
    ON twb.PLACE_CURR = city_num."4num"
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

#  把num3的所有欄位都drop掉
twb_final.drop(columns=['num3_lon','num3_lat','num3_location'] , inplace = True)

#  Step5:把最後核對不了的3碼行政區域碼前面補0：把num3跟num4合併失敗的，將num3前面補0,再次合併num4
#  先將'PLACE_CURR'轉為字串
twb_final['PLACE_CURR'] = twb_final['PLACE_CURR'].astype(str)
twb_final['PLACE_CURR'] = twb_final['PLACE_CURR'].apply(lambda x : x.zfill(4) if len(x)==3 else x)

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

# 檢查還有沒有缺經緯度的
print("還有缺經緯度的：", twb_final2[twb_final2['lon'].isnull()]['PLACE_CURR'].drop_duplicates().to_list())


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
print(twb_final2.isna().sum())


#                                                  /////  以上問卷部分告一段落，已處理好了行政區域碼的對照,並加入了問卷資料  /////

#  接下來處理空氣污染監測站的資料

#                                                                   =====  讀取csv檔  =====

base_path = r"/Users/anyuchen/Desktop/空氣污染/dataset"
#  自動抓取所有監測日數值的CSV檔
file_list = sorted(glob(os.path.join(base_path, '空氣品質監測日平均值(一般污染物) (*.csv')))

#  -----  1.更改欄位名稱，把“”拿掉  -----
def renamerow(df):
    #將欄位中的 " 拿掉並去掉空白
    df.columns = [col.replace('"','').strip() for col in df.columns]
    return df

#  批次載入每日的空氣污染監測資料
dfs = [renamerow(pd.read_csv(file)) for file in file_list]
df_all = pd.concat(dfs, axis=0, ignore_index=True)   #將所有資料concat合併成一個大資料
df_all['concentration'] = pd.to_numeric(df_all['concentration'], errors='coerce')   #並針對concentration欄位資料轉為數值型態，若無法轉換直接轉為nan

#  -----  2.轉換欄位，把污染物變成欄(長格式變寬格式)  -----
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

pollution_all = pivot_col(df_all)

#  變更pollution_1220的PM2.5的欄位名稱,SQL才有辦法使用
pollution_all.rename(columns={'PM2.5':'PM25'} , inplace = True)

#  將整籃空氣污染值都為nan的欄位刪除
pollution_all = pollution_all.dropna(axis = 1,how = 'all')

#  -----  3. 篩選出nan比例低於10%的欄位  -----
#  只抓監測站都共同擁有的污染物質，計算每個欄位的nan比例
na_ratio = pollution_all.isna().mean()
#  篩選出nan比例低於10%的欄位，再轉成list
threshold = 0.1
valid_columns = na_ratio[na_ratio < threshold].index.tolist()
base_columns = ['monitordate', 'siteid', 'sitename']
final_columns = base_columns + [col for col in valid_columns if col not in base_columns]
#  取得過濾後的資料,只保留需要的空氣污染值欄位：RH,WIND_SPEED需要拿掉(對失智症無影響）
pollution_final = pollution_all[final_columns].drop(columns=['RH','WIND_SPEED'])

#  ----- 4. 合併監測站經緯度資訊  -----
pollution_final = pollution_final.merge(air_pollution[['siteid', 'lon', 'lat']],on='siteid', how='left')
'''
query_1 = """
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
        p.SO2,
        a.lon,
        a.lat
    FROM pollution_final AS p
    LEFT JOIN air_pollution AS a
    ON p.siteid = a.siteid
"""
#  將合併好的經緯資料放在pollution_1220中
pollution_1220_fillter_1 = pysqldf(query_1)
#  把result0加入到全域命名空間中
globals()['pollution_1220_fillter_1'] = pollution_1220_fillter_1
'''

#  刪除掉沒有經緯度的資料，因為也抓不到鄰近資料
pollution_final = pollution_final.dropna(subset = ['lon','lat']).reset_index(drop=True)

#  確保air_pollution中的siteid的值是唯一，避免如果有重複id資料量會暴增，保證在合併時是一對一的 join。
air_pollution = air_pollution.drop_duplicates(subset = ['siteid']).reset_index(drop=True)

#  根據前面建立好的經緯度station_coords來跟空氣污染監測經緯度計算距離
#  Step1:將各監測站歷年的經緯度轉為陣列，拿來當作查詢點
pollution_coords = pollution_final[['lon','lat']].to_numpy()

#  Step2:查詢每個監測站索引與距離，k=3代表找出最接近的3個監測站
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
pollution_final = pd.concat([pollution_final.reset_index(drop = True),nearest_pollution_df],axis = 1)

  -----  5. 缺失值補植：先找鄰近監測站、再找同站前後三天均值  -----
#  先確認有哪些特徵欄位有缺失值
print(pollution_final.isna().sum())


from datetime import timedelta
def pollution_filled(df,feature):   #df為要填補缺失值的資料表,feature為需要補植的特徵欄位
    df = df.copy()
    #在一開始就先當日期型態處理好，轉成Datetime格式
    df['monitordate'] = pd.to_datetime(df['monitordate'])
    
    #為了提升效能，使用字典查找，因此建立查找表
    #將df索引設為siteid,monitordate組合，[feature]索引位置取出特徵值，會形成：索引是 (siteid, monitordate)，值是污染物數值，並轉成字典格式
    search_table = df.set_index(['siteid','monitordate'])[feature].to_dict()
    
    #建立每個siteid對應日期與數值(方便查找前後三天的值)
    #先將資料根據siteid分類，針對每個siteid的資料根據'monitordate'做索引，取出污染物的特徵欄位，並確保依日期排序
    site_table = df.groupby('siteid', group_keys=False).apply(lambda x:x.set_index('monitordate')[feature].sort_index().to_dict())
    
    #idx 是該行的索引,row 是該行的資料,df.loc抓出df的特徵欄位中有nan值的整筆資料
    for idx,row in df.loc[df[feature].isna()].iterrows():
        
        #取出該筆資料的日期，以便尋找鄰近戰貨日期前後的資料
        date = row['monitordate']
        filled = False   #追蹤是否成功使用前三個鄰近監測站補值，預設這筆資料還沒被填補
        
        #建立前三siteid
        nearest_sites = [row['nearest_station_1_id'],row['nearest_station_2_id'],row['nearest_station_3_id']]
            
        #對每個最近的監測站尋找該站相同日期的污染物
        for site_id in nearest_sites:
            val = search_table.get((site_id,date),None)   #如果key值不在會回傳預設None,預防key值不在會報錯
            #如果尋找到的鄰近點有數值(非Nan)則補值,透過pandas的函示notna來判斷 val 是否不是缺失值 (NaN 或 None)
            if pd.notna(val):
                df.at[idx,feature] = val
                #補植完成
                filled = True
                break
       
        #如果前三鄰近資料補完依舊有缺失值，就使用前後三天的同站資料
        if not filled:
            series_dict = site_table.get(row['siteid'],None)
                
            #如果尋找到的資料不是none,就補植
            if series_dict is not None:
                series = pd.Series(series_dict)
                series.index = pd.to_datetime(series.index)   #將索引值改為datatime格式
                #選出前後三天的資料
                mask = (series.index >= date - timedelta(days=3)) & (series.index <= date + timedelta(days=3))
                near_vals = series[mask].dropna()
                    
                #若臨近日存在有效值,取前後三天平均值
                if not near_vals.empty:
                    df.at[idx,feature] = near_vals.mean()
          
    return df

#  先抓出數值行欄位
filled_columns = ['AMB_TEMP', 'CO', 'NO', 'NO2','NOx', 'O3', 'PM10', 'PM25', 'SO2']

#  為了保護原始資料不被污染，因此建立副本
pollution_filled_df = pollution_final.copy()

#  因為所有空氣污染的特徵欄位均有缺失值，因此會寫一個函式來一次呼叫
for feature in filled_columns:
    pollution_filled_df = pollution_filled(pollution_filled_df , feature)

#  確認是否還有有哪些特徵欄位有缺失值
print(pollution_filled_df.isna().sum())
print(pollution_filled_df.columns)

#  將剩下能然補不了缺失值的資料刪除
pollution_missing_rate = pollution_filled_df[filled_columns].isna().mean()
print(pollution_missing_rate)
pollution_filled_df = pollution_filled_df.dropna(subset = filled_columns)

print(pollution_filled_df.isna().sum())

#  只留下有需要的欄位，第幾監測站補值那些特徵欄位不需要了
pollution_filled_final = pollution_filled_df.drop(columns=['lon','lat', 'nearest_station_1','nearest_station_1_id','nearest_station_1_distance',
                                                            'nearest_station_2','nearest_station_2_id', 'nearest_station_2_distance','nearest_station_3',
                                                            'nearest_station_3_id','nearest_station_3_distance'],axis = 1)



#                                                        =====  計算最接近的監測站一年平均污染物質(將問卷個案跟監測站對應（取有效站點ID）)  =====

#  Step1:將日期格式轉為datetime格式-需要先把SURVEY_DATE(年/月/日)跟monitordate(年-月-日)的日期格式改成一樣的
twb_final2['SURVEY_DATE'] = pd.to_datetime(twb_final2['SURVEY_DATE'])
#  再把monitordate資料改為datetime資料,並只保留年月日，時間不要
pollution_filled_final['monitordate'] = pd.to_datetime(pollution_filled_final['monitordate']).dt.normalize()
#  確認兩者的資料型態:datetime64[ns]
#print(twb_final2['SURVEY_DATE'].dtype)
#print(pollution_filled_final['monitordate'].dtype)

#  Step2:找出有效的監測站
#  建立一個包含所有污染資料的監測站ID的集合Set,為了後續方便查找,因資料型態為str,需轉成數字才能跟twb_final2的nearest_station_id匹配
valid_sites = set(pollution_filled_final['siteid'].astype(int).unique())

#  定義函式：如果沒有第一近的監測站，就取第二近的以此類推
def valid_pollution_station(row):

    if row['nearest_station_1_id'] in valid_sites:
        return row['nearest_station_1_id']
    elif row['nearest_station_2_id'] in valid_sites:
        return row['nearest_station_2_id']
    elif row['nearest_station_3_id'] in valid_sites:
        return row['nearest_station_3_id']
    else:
        return np.nan   #都沒有時就回傳Nan


#  apply(..., axis=1)：表示會對每一列（每個 row）呼叫valid_pollution_station(row)
twb_final2['final_station_id'] = twb_final2.apply(valid_pollution_station,axis = 1)
#  將不需要的欄位刪除
twb_final2.drop(columns=['nearest_station_1_id','nearest_station_2_id','nearest_station_3_id'],inplace = True)

print(twb_final2.columns)
print(pollution_filled_final.columns)
#  Step3: 合併問卷與污染資料（依個案/日期關聯一年平均):合併問卷資料跟空氣污染值,根據監測站siteid
Avg_pollution = """
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
        AVG(sub.SO2) AS avg_SO2
        
    FROM twb_final2 AS main
    LEFT JOIN pollution_filled_final AS sub
        ON main.final_station_id = sub.siteid
        AND julianday(main.SURVEY_DATE) - julianday(sub.monitordate) BETWEEN 0 AND 365
    GROUP BY main.Release_No
    """
#  將合併好的地點資料放在Avg_pollution_1220中
Avg_pollution = pysqldf(Avg_pollution)
#  把result0加入到全域命名空間中
globals()['Avg_pollution'] = Avg_pollution

#  把不需要的欄位刪除，因已經合併平均值了，所以監測站經緯度可刪除
Avg_pollution = Avg_pollution.drop(columns=['PLACE_CURR', 'lon', 'lat', 'num4_location', 'nearest_station_1',
                                                      'nearest_station_1_distance','nearest_station_2', 
                                                      'nearest_station_2_distance', 'nearest_station_3',
                                                       'nearest_station_3_distance'] , axis = 1)

#  確認缺失值
Avg_pollution.isna().sum()
#  會發現有559筆空氣污染值資料為nan,其siteid=85,經過核對發現空氣污染監測站的資料日期為(2021-03-24-2021~12-31),而問卷資料日期為(2009-02-05~2020-02-05)，所以根本對不上一年平均
#nan_rows = Avg_pollution[Avg_pollution[['avg_AMB_TEMP','avg_CO','avg_NO','avg_NO2','avg_NOx','avg_O3','avg_PM10','avg_PM25','avg_SO2']].isna().any(axis = 1)]
#pollution_filled_final[pollution_filled_final['siteid'] == 85].shape
#pollution_filled_final[pollution_filled_final['siteid'] == 85]['monitordate'].min(), pollution_filled_final[pollution_filled_final['siteid'] == 85]['monitordate'].max()
#twb_dates = twb_final2[twb_final2['final_station_id'] == 85]['SURVEY_DATE']

#所以應該將559筆資料刪除
result_twb_pollution = Avg_pollution.dropna(subset=['avg_AMB_TEMP','avg_CO','avg_NO','avg_NO2','avg_NOx','avg_O3','avg_PM10','avg_PM25','avg_SO2']).reset_index(drop=True)

#  將最終資料存為一個csv檔案
#result_twb_pollution.to_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/final_result.csv")

result_twb_pollution.columns

#                                                            =====  整理問卷基本資料(非空污資料)  =====
#  載入MMSE資料
df_MMSE = pd.read_excel(r"/Users/anyuchen/Desktop/空氣污染/dataset/MMSE.xlsx")
df_MMSE['MMSE'].value_counts()
df_MMSE['HighRisk'].value_counts()

#  確認df_MMSE的資料欄位沒有重複
df_MMSE = df_MMSE.loc[:,~df_MMSE.columns.duplicated()]
#  針對Release_No欄位中每個數值只保留第一筆出現的資料
df_MMSE_unique = df_MMSE.drop_duplicates(subset=['Release_No'])
df_MMSE_unique['MMSE'].value_counts()

#  為了預防final_result也有重複，一併drop掉
result_twb_pollution_unique = result_twb_pollution.drop_duplicates(subset = ['Release_No'])

#  將result_twb_pollution跟MMSE合併
final_result = result_twb_pollution_unique.merge(df_MMSE_unique,on='Release_No',how='left')

final_result.columns

#  基本資料欄位也只保留缺失值小於10%
final_result = low_miss_col(final_result)

#  將資料做變數分類：空氣污染欄位+基本欄位+串接欄位
base_col =['AGE', 'SEX', 'EDUCATION', 'MARRIAGE','SMK_EXPERIENCE',
            'DRK', 'SPO_HABIT', 'BODY_HEIGHT', 'BODY_WEIGHT', 'BODY_FAT_RATE',
            'BODY_WAISTLINE', 'avg_SYSTOLIC_PRESSURE','avg_DIASTOLIC_PRESSURE',
            'FASTING_GLUCOSE', 'HBA1C', 'T_CHO', 'TG', 'HDL_C', 'LDL_C',
            'CREATININE', 'URIC_ACID', 'MICROALB', 'BUN', 'SGOT', 'SGPT',
            'GAMMA_GT', 'ALBUMIN', 'WBC', 'RBC', 'PLATELET', 'HB', 'HCT']

pollution_col = ['avg_AMB_TEMP','avg_CO', 'avg_NO', 'avg_NO2', 'avg_NOx',
                 'avg_O3', 'avg_PM10','avg_PM25', 'avg_SO2']

meta_col = ['SURVEY_DATE', 'Release_No', 'final_station_id', 'FOLLOW']

#final_result.isna().sum()

#  刪除nan的資料
final_result = final_result.dropna(subset=base_col)

#  因婚姻狀況有2筆是R,不知道意義是什麼故刪掉
final_result = final_result[final_result['MARRIAGE'] != 'R']
'''
#  計算平均'SIT_1_SYSTOLIC_PRESSURE', 'SIT_1_DIASTOLIC_PRESSURE','SIT_2_SYSTOLIC_PRESSURE', 'SIT_2_DIASTOLIC_PRESSURE'，並將其刪除
final_result.loc[:,'avg_SYSTOLIC_PRESSURE'] = (final_result[['SIT_1_SYSTOLIC_PRESSURE','SIT_2_SYSTOLIC_PRESSURE']].mean(axis = 1))
final_result.loc[:,'avg_DIASTOLIC_PRESSURE'] = (final_result[['SIT_1_DIASTOLIC_PRESSURE','SIT_2_DIASTOLIC_PRESSURE']].mean(axis =1))
#  刪除原始欄位
final_result.drop(columns=['SIT_1_SYSTOLIC_PRESSURE','SIT_2_SYSTOLIC_PRESSURE','SIT_1_DIASTOLIC_PRESSURE','SIT_2_DIASTOLIC_PRESSURE'],inplace = True)
'''

#  確認所有缺失值，及其數值欄位，並建立對照表
education_map = {'1': '未受教育','2': '小學','3': '國中','4': '高中/高職','5': '專科','6': '大學','7': '研究所以上'}
final_result['EDUCATION'].value_counts()
#  將拒答或其他非正常答覆(=R/N)刪除
final_result = final_result[~final_result['EDUCATION'].isin(['R','N'])]

#  婚姻狀況
marriage_map = {'1': '未婚','2': '已婚','3': '離婚','4': '喪偶','R': '拒答/不明'}
final_result['MARRIAGE'].value_counts()
#  將資料.0拿掉：先將字串1.0轉為浮點數再轉為int
final_result.loc[:,'MARRIAGE'] = final_result['MARRIAGE'].astype(float).astype(int)

#  喝酒習慣
drk_map = {'1': '不喝','2': '偶爾喝','3': '經常喝',None: '缺漏'}
final_result['DRK'].value_counts()
#  將拒答或其他非正常答覆(=R/N)的資料刪除,共有11筆
final_result = final_result[~final_result['DRK'].isin(['R','N'])]
#  將資料.0拿掉：先將字串1.0轉為浮點數再轉為int
final_result.loc[:,'DRK'] = final_result['DRK'].astype(float).astype(int)

#  運動狀況
spo_habit_map = {'1': '規律運動','2': '不規律或不運動',None: '缺漏'}
final_result['SPO_HABIT'].value_counts()
#  將答覆為R的資料刪除，共有2筆
final_result = final_result[~final_result['SPO_HABIT'].isin(['R'])]
final_result.loc[:,'SPO_HABIT'] = final_result['SPO_HABIT'].astype(float).astype(int)

#  抽菸狀況
SMK_map = {'1':'否','2':'是'}
final_result['SMK_EXPERIENCE'].value_counts()

#  將拒答或其他非正常答覆(=R/N)的資料刪除,共有11筆
final_result = final_result[~final_result['SMK_EXPERIENCE'].isin(['R','N'])]
#  將資料.0拿掉：先將字串1.0轉為浮點數再轉為int
final_result.loc[:,'SMK_EXPERIENCE'] = final_result['SMK_EXPERIENCE'].astype(float).astype(int)

#  清理有包含<符號的資料
def clean_less_sign(df,col):
    df = df.copy()
    mask = df[col].astype(str).str.startswith('<')
    #從<3.9類型中擷取數字部分,ex:3.9
    extracted = df.loc[mask,col].str.extract(r'<\s*(\d+\.?\d*)')[0]
    #將擷取下來的字串轉為數字，因為是小於可以直接使用floor往低值抓
    df[col] = np.floor(pd.to_numeric(df[col],errors = 'coerce'))
    #回填到有包含'<'的資料中
    df.loc[mask,col] = extracted.astype(float)
    
    return df
    
#  找出需要清理的欄位
col_with_less_sign = [
    col for col in final_result.columns 
    if final_result[col].astype(str).str.contains('<').any()
    ]
print("含有'<'的欄位：",col_with_less_sign)   #  含有'<'的欄位： ['HBA1C', 'MICROALB', 'SGPT', 'GAMMA_GT']

for col in col_with_less_sign:
    final_result = clean_less_sign(final_result,col)



#                                                                     =====  繪圖  =====
#  空氣污染監測站資料：pollution_filled_final
#  問卷依據日期合併資料：final_result
#  畫出各污染物的平均圖：要確認問卷抓的資料跟空氣污染監測站的資料是否吻合

#這個會生成一個Series(污染物不是單獨一欄特徵欄位，而是索引值)，因此要轉成DataFrame
avg_twb_pollutant = final_result[pollution_col].mean().reset_index()
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
plt.savefig("/Users/anyuchen/Desktop/空氣污染/graph/TWB mean value of pollutant.png",dpi = 300,bbox_inches='tight')
plt.show()

'''
數值資料：
'AGE','BODY_WAISTLINE', 'FASTING_GLUCOSE-空腹血糖', 'HBA1C-糖化血紅素', 'T_CHO-總膽固醇','TG-三酸甘油脂', 'HDL_C-高密度脂蛋白膽固醇',
'LDL_C-低密度脂蛋白膽固醇', 'CREATININE-肌酸酐', 'URIC_ACID-尿酸', 'MICROALB-微量白蛋白', 'BUN-尿素氮', 'SGOT-天門冬氨酸轉氨酶',
'SGPT-丙胺酸轉氨酶','GAMMA_GT-γ-穀胱甘肽轉移酶', 'ALBUMIN-白蛋白', 'WBC-白血球數', 'RBC-紅血球數', 'PLATELET-血小板數', 'HB-血紅素', 'HCT-血球比容',
'''

#                                                            =====  觀察每個特徵欄位對於y的影響:數值/類別資料統計檢定  =====

#                                                                -----數值型資料與類別型資料-----

#  建立訓練模型資料，將不需要的欄位drop掉
model_data = final_result.drop(columns=meta_col).copy()

#  1.數值型態資料：'HBA1C', 'FASTING_GLUCOSE', 'T_CHO','LDL_C', 'HDL_C', 'TG', 'MICROALB', 'URIC_ACID','avg_SYSTOLIC_PRESSURE', 'avg_DIASTOLIC_PRESSURE', 'final_station_id','avg_AMB_TEMP', 'avg_CO', 'avg_NO', 'avg_NO2', 'avg_NOx', 'avg_O3','avg_PM10', 'avg_PM25', 'avg_SO2',
#  因為EDUCATION跟DRK屬於連續型類別資料，因此不適用ONE HOT ENCODING
num_col=['AGE', 'EDUCATION', 'DRK','BODY_HEIGHT', 'BODY_WEIGHT', 'BODY_FAT_RATE',
       'BODY_WAISTLINE', 'FASTING_GLUCOSE', 'HBA1C', 'T_CHO', 'TG', 'HDL_C',
       'LDL_C', 'CREATININE', 'URIC_ACID', 'MICROALB', 'BUN', 'SGOT', 'SGPT',
       'GAMMA_GT', 'ALBUMIN', 'WBC', 'RBC', 'PLATELET', 'HB', 'HCT',
       'avg_AMB_TEMP', 'avg_CO', 'avg_NO', 'avg_NO2', 'avg_NOx', 'avg_O3',
       'avg_PM10', 'avg_PM25', 'avg_SO2', 'avg_SYSTOLIC_PRESSURE',
       'avg_DIASTOLIC_PRESSURE']

#  2.類別型資料：'SEX', 'MARRIAGE_x', 'SPO_HABIT','SMK_EXPERIENCE'
#  'EDUCATION' ,'DRK'為連續型類別資料，歸在num_data一起做標準化
#  'SEX','SPO_HABIT','SMK_EXPERIENCE' 為二元類別，不需要one hot,'MARRIAGE'則需要one hot
category_col=[ 'SEX', 'MARRIAGE', 'SPO_HABIT','SMK_EXPERIENCE' ]

#  3.偏態資料需要np.log轉換
#  需要轉換的欄位
# 顯示偏態程度前幾高的欄位
print(model_data[num_col].skew().sort_values(ascending=False).head())

skewed_features = ['WBC', 'MICROALB', 'GAMMA_GT', 'CREATININE', 'SGOT']
for col in skewed_features:
    model_data[col] = np.log1p(model_data[col])

#  4.因變數資料：base_y = df_MMSE[['MMSEgroup']]

#  -----統計檢定：Shapiro-Wilk常態分佈,mannwhitneyu檢定,卡方檢定(多元類別型vs類別型變數)-----

#  1.Shapiro-Wilk 檢定常態分佈：先觀察所有特徵的資料分佈,原先用kstest檢定(假設輸入是連續變數,故效果不佳),改用Shapiro-Wilk 檢定   
#   會得到所有數值特徵都不符合常態分佈
from scipy.stats import shapiro
for feature in num_col:
    #dropna()是為了確保沒有nan值
    data = model_data[feature].dropna()
    
    if len(data) >= 3:  # Shapiro-Wilk 最少需要3筆資料    
        stat,p=shapiro(data)  
        #print(f"Feature:{feature} stat:{stat:.4f} p:{p:.4f}")
        if p >= 0.05:
            print(f"{feature}符合常態分佈")
        else:
            print(f"{feature}不符合常態分佈")
    else:
        print(f"{feature} 資料不足，無法進行Shapiro-Wilk 檢定")


#  2.mannwhitneyu檢定(數值型自變數vs類別型因變數)，來衡量變數與目標變數MMSE的關係，且因資料為不常態分佈，應使用Mann-Whitney U檢定
#  mannwhitneyu檢定:自變數值是否在不同因變數組別中顯著不同
from scipy.stats import mannwhitneyu
def manu_test(df,target_col,features):
    manu_results=[]
    
    #先將所有數值欄位轉成數值型態
    for feature in features:
        df[feature] = pd.to_numeric(df[feature], errors='coerce')
        
    for feature in features:
        #將因變數分成兩組，來看自變數在兩組中分佈是否相同
        #根據target的數值資料切出對應數值
        group0 = df[df[target_col] == 0][feature].dropna()
        group1 = df[df[target_col] == 1][feature].dropna()
        
        #需確認兩組均有資料
        if len(group0) > 0 and len(group1) >0:
            #alternative='two-sided': 檢定假設是雙尾檢定，意思是：「兩組數值分布不同」不管是哪邊大
            u_stat , p_value = mannwhitneyu(group0 ,group1 ,alternative = 'two-sided')
            print(f"{feature} : U = {u_stat:.2f} ,p = {p_value:.4f}, {'顯著' if p_value<0.05 else '不顯著'}")
            manu_results.append({
                'Feature':feature,
                'U_stat':u_stat,
                'P-Value':p_value,
                'Significance':'顯著性關係' if p_value <= 0.05 else '不顯著關係'
             })
        
        else:
             print(f"{feature} : 無法檢定(某組樣本數為0)")
             manu_results.append({
                 'Feature':feature,
                 'U_stat':None,
                 'P-Value':None,
                 'Significance':'無法檢定'
             })
    return pd.DataFrame(manu_results)
print("\n------數值型資料對 MMSE 的 mannwhitneyu 檢定------")
manu_test_num_data=manu_test(model_data,target_col = 'MMSE', features = num_col)


#  3.卡方檢定(多元類別型vs類別型變數)：觀察兩變數之間是否有顯著性關係，p-value <= 0.05為顯著性關係
multiple_category_col = ['MARRIAGE']
binary_category_col = ['SEX','SPO_HABIT','SMK_EXPERIENCE']

chi2_multiple_data=model_data[multiple_category_col]
chi2_binary_data=model_data[binary_category_col]
#print(chi2_data.dtypes) #可以得到所有的特徵欄位均為數值型態int64
from scipy.stats import chi2_contingency

#新增一個Cramer'sV計算函數，避免過度依賴P值
def cramers_v(confusion_matrix):
    chi2=chi2_contingency(confusion_matrix)[0]
    n=confusion_matrix.sum().sum()
    min_dim=min(confusion_matrix.shape)-1
    return np.sqrt(chi2/(n*min_dim))    

#將卡方檢定用一個函數包裝起來，那卡方檢定最終需要輸入的資料為輸入變數及輸出變數
def chi2_test(target,features):
    """
    target:目標變數Ｉ=ICUstayingdays_14,ICUstayingdays_8,需為一維資料
    features:所有考量的特徵資料，包含多個特徵
    """
    chi2_results=[]
    for feature in features.columns:
        #建立列連表,需確認輸入數據為一維
        contingency_table=pd.crosstab(target,features[feature])
        #執行卡方檢定
        chi2,p,dof,expected=chi2_contingency(contingency_table) 
        #計算Cramer's V
        cramers_v_value=cramers_v(contingency_table)
        #輸出結果為
        result={
            'Feature':feature,
            'Chi2_stat':chi2,
            'P-value':p,
            'Cramers_V':cramers_v_value,
            'Significance':'顯著性關係' if p <= 0.05 else '不顯著關係'}
        chi2_results.append(result)
        #顯示結果
        print(f"Feature:{feature} 與 目標變數間的關係為{'顯著性關係' if p <= 0.05 else '不顯著關係'} (P-value:{p:.4f}) | Cramer's V:{cramers_v_value:.4f} | {result['Significance']}")
        #print("\n---------ICUstayingdays_14卡方檢定的列連表----------\n")
        print(contingency_table)
    return pd.DataFrame(chi2_results) 

#使用所有多元類別特徵資料與MMSE做檢定
print("\n---------多元類別變數 對 MMSE 卡方檢定----------\n")

chi2_multiple_col=chi2_test(model_data['MMSE'],chi2_multiple_data)

chi2_binary_col=chi2_test(model_data['MMSE'],chi2_binary_data)


#   可視化數值特徵的常態分佈:長條圖
import matplotlib.pyplot as plt
import seaborn as sns
for feature in num_col:
    plt.figure()
    sns.histplot(model_data[feature],kde=True,stat='count',bins=30,color="#D3D3D3")
    plt.title(f"Distribution of {feature}")
    plt.xlabel(f"Value of {feature}")   
    plt.ylabel('Count')
    plt.savefig(f"/Users/anyuchen/Desktop/空氣污染/graph/{feature}_normal_distribution.png",dpi = 300) #並將所有圖形存下來
    plt.show()
    plt.close()  #加 plt.close() 釋放記憶體，避免多圖時堆積

#   可視化所有類別資料的分佈圖，但因為數值資料過多就不一一繪製
for feature in category_col:
    #如果是類別資料or變數唯一值的數量小於20，使用countplot,不支援kde
    if model_data[feature].dtype=='object' or model_data[feature].nunique() < 20:
        
        ax=sns.countplot(data=model_data,x=model_data[feature],width=0.3,color="#D3D3D3")
        ax.bar_label(ax.containers[0])
    
        plt.title(f"Distribution of {feature}")
        plt.xlabel(f"Value of {feature}")
        plt.ylabel('Count')
        plt.xticks(rotation=45)  #避險標籤重疊
        plt.savefig(f"/Users/anyuchen/Desktop/空氣污染/graph/{feature}_distribution.png",dpi = 300)
        plt.show()
        plt.close()



#                                                     =====  計算每個空氣污染監測站的統計資料，並繪出圖  =====

#  定義繪圖計算函式-根據每個監測站、每年、每個污染物繪圖
def avg_pollution_station_for_a_year(df):
    
    #抓出日期中的年份
    df['year'] = df['monitordate'].dt.year
    
    #先設定所有需要繪圖的污染物欄位
    columns = ['AMB_TEMP', 'CO', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM25', 'SO2']
    
    #並計算有所污染物的平均
    avg_pollution_sitename = df.groupby(['year','sitename'])[columns].mean().reset_index()
    
    #  將資料欄位轉為長格式，在繪圖上才方便繪圖
    mean_melt_sitename = avg_pollution_sitename.melt(
        id_vars = ['year','sitename'],
        value_vars = columns,
        var_name = 'Pollutant' ,
        value_name = 'Meanvalue')
 
    #  根據不同污染物繪圖來比較不同的監測站的差異
    pollutants = mean_melt_sitename['Pollutant'].unique()
    
    #  繪製個監測站的統計圖表
    # 指定字體檔案路徑
    font_path = "/System/Library/Fonts/STHeiti Medium.ttc"
    my_font = fm.FontProperties(fname=font_path)

    #  遞迴所有污染物質
    output_dir = "/Users/anyuchen/Desktop/空氣污染/graph"
    os.makedirs(output_dir, exist_ok=True)

    for pollutant in pollutants:
        plt.figure(figsize=(14,6))
        data_subset = mean_melt_sitename[mean_melt_sitename['Pollutant'] == pollutant]
        
        ax = sns.barplot(data=data_subset,x='sitename',y='Meanvalue',palette='muted',hue = 'sitename', legend = False)
            
        plt.xticks(rotation=90,ha='right', fontproperties=my_font)
        plt.title(f"{pollutant} 年平均趨勢- Mean value by Monitoring Station ", fontproperties=my_font)
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
    return avg_pollution_sitename , mean_melt_sitename
        
avg_pollution_year = avg_pollution_station_for_a_year(pollution_filled_final)


#                                                                ====  資料標準化:數值/類別  =====

#  1.數值型態資料：'HBA1C', 'FASTING_GLUCOSE', 'T_CHO','LDL_C', 'HDL_C', 'TG', 'MICROALB', 'URIC_ACID','avg_SYSTOLIC_PRESSURE', 'avg_DIASTOLIC_PRESSURE', ,'avg_AMB_TEMP', 'avg_CO', 'avg_NO', 'avg_NO2', 'avg_NOx', 'avg_O3','avg_PM10', 'avg_PM25', 'avg_SO2',
#  因為EDUCATION跟DRK屬於連續型類別資料，因此不適用ONE HOT ENCODING

#  3.因變數資料：base_y = df_MMSE[['MMSEgroup']]
'''
num_col=['AGE', 'EDUCATION', 'DRK','BODY_HEIGHT', 'BODY_WEIGHT', 'BODY_FAT_RATE',
       'BODY_WAISTLINE', 'FASTING_GLUCOSE', 'HBA1C', 'T_CHO', 'TG', 'HDL_C',
       'LDL_C', 'CREATININE', 'URIC_ACID', 'MICROALB', 'BUN', 'SGOT', 'SGPT',
       'GAMMA_GT', 'ALBUMIN', 'WBC', 'RBC', 'PLATELET', 'HB', 'HCT',
       'avg_AMB_TEMP', 'avg_CO', 'avg_NO', 'avg_NO2', 'avg_NOx', 'avg_O3',
       'avg_PM10', 'avg_PM25', 'avg_SO2', 'avg_SYSTOLIC_PRESSURE',
       'avg_DIASTOLIC_PRESSURE']
multiple_category_col = ['MARRIAGE']
binary_category_col = ['SEX','SPO_HABIT','SMK_EXPERIENCE']
'''

#  建立數值與類別資料
binary_data = model_data[binary_category_col].astype(int)
num_data = model_data[num_col]
category_data = model_data[multiple_category_col]

#  數值資料需要標準化：
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#  標準化完以後為numpy array,需要轉成df
num_data_scaled = scaler.fit_transform(num_data)
#  將標準化完的資料轉為df,並保持原本num_data的索引,以便後續concat
num_data_scaled = pd.DataFrame(num_data_scaled, columns = num_col, index = num_data.index)
#  把標準化後的資料放回原本的data
#model_data[num_col] = scaler_data


#  2.類別型資料：'SEX', 'MARRIAGE_x', 'SPO_HABIT','SMK_EXPERIENCE'
#  類別資料需要One Hot Encoding/Label Encoding,，二元類別不需要OneHot,one hot encoding是針對沒有次序性的多類別欄位
#  'EDUCATION' ,'DRK'為連續型類別資料，歸在num_data一起做標準化
#  'SEX','SPO_HABIT','SMK_EXPERIENCE' 為二元類別，不需要one hot,'MARRIAGE'則需要one hot
#  將多元類別資料做one hot encoding,因為在python中做dummies會產生布林值，如果想要呈現0/1要再轉回數值
#  需要先確認資料是否為數字類別:均為數字print(category_data.dtypes)，需要將資料型態轉為字串，才不會被誤認為是連續數字
for col in category_data.columns:
    print(f'{col} unique values : {category_data[col].unique()}')
category_data = category_data.astype(str)
category_data_encode = pd.get_dummies(category_data, drop_first=True)
#  其結果會顯示True/False,因此要再轉回數值
category_data_encode = category_data_encode.astype(int)

#  將二元/多元類別跟數值資料合併
x = pd.concat([num_data_scaled,category_data_encode,binary_data],axis = 1)

# 因變數資料
y = model_data[['MMSE','HighRisk']]

#  將最終訓練資料存為檔案csv
modet_train_data = pd.concat([x,y],axis = 1)
modet_train_data.to_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/整理好的資料輸出/modet_train_data_final.csv",index = False)



#                                                            =====  模型訓練(要使用平衡後的資料)  =====
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
import lightgbm as lgb
from xgboost import XGBClassifier
from itertools import cycle  #用於生成色彩循環
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_curve
from imblearn.over_sampling import SMOTE
from sklearn.metrics import make_scorer, fbeta_score
from catboost import CatBoostClassifier


import sys
print(sys.executable)

def model_train_evaluate(x,y):
    #  切割自變數、因變數：是否確診失智的資料model_data
    #model_data = pd.concat([num_data_scaled,category_data_encode,binary_data],axis = 1)
    #y = Avg_pollution_1220_final['MMSE']

    #  切割訓練測試資料
    X_train,X_test,y_train,y_test = train_test_split(x ,y, test_size = 0.3,random_state=3)
    print(y_test.value_counts())

    #  針對訓練資料做SMOTE資料平衡：因變數y存在資料不平衡
    smote = SMOTE(sampling_strategy = 0.3 , random_state=3)  # 代表少數類別將增生到多數類別樣本數的 50%
    X_train_sm , y_train_sm = smote.fit_resample(X_train, y_train)
    print("Before SMOTE:" , y_train.value_counts())
    print("After SMOTE:" , y_train_sm.value_counts())
    
    #  先計算正負樣本比例:負類樣本/正類樣本
    scale = (y == 0).sum() / (y == 1).sum()
    
    #  建立模型及設定超參數(邏輯回歸Logistic Regression做baseline)
    models_param_grids = {
        #邏輯回歸模型
        'Logistic Regression':{
            'model':LogisticRegression(class_weight = 'balanced'),
            'params':{
                'C':[0.01,0.1,1],
              }
            },
        #隨機森林模型
        'Random Forest':{
            'model':RandomForestClassifier(),
            'params':{
              'max_depth':[None,10,20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
                }
           },
        #新增一個XGBoost:處理極度不平衡資料
        #use_label_encoder = False  防止警告訊息
        #eval_metric = 'logloss'  必填，不然會跳警告
        'XGBoost':{
            'model':XGBClassifier(eval_metric = 'aucpr' ,random_state = 3),
            'params':{
                'max_depth': [3, 5, 7],   #決策數的最大深度
                'learning_rate': [0.01, 0.1, 0.3],   #學習率
                'n_estimators': [50, 100],   #森林裡數的數量
                #'subsample': [0.6, 0.8, 1.0],   #每棵樹訓練時隨機抽樣的樣本比例(可防止過擬)
                #'colsample_bytree': [0.6, 0.8, 1.0]   #每棵樹訓練時隨機抽樣的特徵比例
                #'scale_pos_weight': [scale]  #因有做SMOTE資料平衡，不需要再平衡
                }
            },
        #CatBoost模型
        'CatBoost':{
            'model':CatBoostClassifier(eval_metric='PRC', class_weights=[1, 100], verbose=0, random_seed=3),
            'params':{
                'iterations':[100,200],
                'learning_rate':[0.01,0.05,1],
                'depth':[4,6]
                }
            },
        
        #LightBoost模型
        'LightGBM':{
            'model':lgb.LGBMClassifier(class_weight='balanced', objective='binary', random_state=3),
            'params':{
                'n_estimators':[50,100],
                'learning_rate':[0.01,0.05,0.1]
                }
            }
        }

    #  4.Grid_Search自動搜尋最佳參數:需要將結果儲存在results中，因為繪製ROC只需要y_prob的結果，因此需要將所有模型的結果儲存在一起，在繪圖時一次叫出來即可
    results={}
    for name,config in models_param_grids.items():
        print(f"Performing GridSearchCV for {name}...")
        grid_search=GridSearchCV(estimator=config["model"],
                                 param_grid=config["params"],
                                 cv=3,
                                 scoring='recall',  #主要關心是「能抓出失智者」，你應該以：Recall（靈敏度）或 F1-score 為主要指標，而不是 Accuracy。
                                 verbose=2,
                                 n_jobs=1)
        grid_search.fit(X_train_sm,y_train_sm)   #訓練時使用SMOTE後的資料
        #取得最佳模型
        best_model=grid_search.best_estimator_
        y_prob=best_model.predict_proba(X_test)[:,1]  #獲取正類別概率
        y_pred=best_model.predict(X_test)
        results[name]=(best_model,y_prob)
        
        #輸出分類報告
        print(f"Best parameters for {name} : {grid_search.best_params_}")
        print(f"====Classification Report for {name}====\n",classification_report(y_test,y_pred))
        
        # -- f1_score調整決策閥值 --
        #調整決策閥值Threshold Tuning：計算精確率,召回率及其對應的閥值
        precisions , recalls , thresholds = precision_recall_curve(y_test,y_prob)
        #找最佳的閥值,因precisions和recalls第一個數值是
        f1_scores = 2 * (precisions[1:] * recalls[1:]) / (precisions[1:] + recalls[1:] + 1e-10)
        optimal_threshold_f1 = thresholds[np.argmax(f1_scores)]
        
        print(f"Optimal thresholds: {optimal_threshold_f1}")
        
        #  根據最佳閥值進行預測
        y_pred_new_f1 = (y_prob >= optimal_threshold_f1).astype(int)
        print(f"====Classification Report for {name} with new pred f1====\n",classification_report(y_test,y_pred_new_f1))
    
        #PR_AUC
        from sklearn.metrics import average_precision_score
        pr_auc = average_precision_score(y_test, y_prob)
        print(f"PR-AUC: {pr_auc:.4f}")
        
    return X_train,X_test,y_train,y_test, X_train_sm , y_train_sm ,results

X_train ,X_test ,y_train ,y_test ,X_train_sm ,y_train_sm ,results = model_train_evaluate(x, y)

        

#                                                             =====  建立降低負類比例的資料集  =====
#  原先的資料：X_train_sm , y_train_sm,X_test,y_test
#  先把y_test切分為正/負類資料
y_test_pos = y_test[y_test == 1]
y_test_neg = y_test[y_test == 0]

#  對負類下採樣：負類資料調整為約正類的10倍,sample為隨機抽樣n筆資料
y_test_neg_undersample = y_test_neg.sample(n=len(y_test_pos)*10,random_state = 3)

#  將test資料合併，並打亂:sample(frac=1)表示：抽樣一個「比例」的資料，frac=1 意味著抽取「全部資料的 100%」，但順序被隨機打亂
y_test_undersample = pd.concat([y_test_pos,y_test_neg_undersample]).sample(frac = 1,random_state=3)

#  抓對應y_test索引抓出x_test
x_test_undersample = X_test.loc[y_test_undersample.index]

#  透過下採樣的測試資料觀察預測結果
#  假設要使用的模型是XGBoost
best_model = results['XGBoost'][0]   # 取出模型本身
y_prob=best_model.predict_proba(x_test_undersample)[:,1]  #獲取正類別概率
y_pred=best_model.predict(x_test_undersample)


#  -- f1 score --
#  調整決策閥值Threshold Tuning：計算精確率,召回率及其對應的閥值
precisions , recalls , thresholds = precision_recall_curve(y_test_undersample,y_prob)
#  找最佳的閥值,因precisions和recalls第一個數值是
f1_scores = 2 * (precisions[1:] * recalls[1:]) / (precisions[1:] + recalls[1:] + 1e-10)
optimal_threshold = thresholds[np.argmax(f1_scores)]
print(f"Optimal thresholds: {optimal_threshold}")

#  根據f1最佳閥值進行預測
y_pred_new = (y_prob >= optimal_threshold).astype(int)

print(f"====Classification Report for XGBoost====\n",classification_report(y_test_undersample,y_pred))

print(f"====Classification Report for XGBoost for new y_pred ====\n",classification_report(y_test_undersample,y_pred_new))


#  畫出precision-recall curv，找出在precision跟recall的平衡點:Precision 至少 0.5同時 Recall 不小於 0.4
plt.figure()
plt.plot(thresholds,precisions[1:], label = 'Precision')
plt.plot(thresholds,recalls[1:], label = 'Recall')
plt.axvline(optimal_threshold , color='red' ,linestyle='--' , label = f'Threshold = {optimal_threshold:.2f}')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title(f'Precision & Recall vs Threshold : test_indersample')
plt.legend()
plt.grid()
plt.show()

#                                                                  =====  特徵重要性評估  =====
#  特徵重要性評估
#  抓出特徵欄位
feature_names = X_test.columns
#  取出特徵重要性
importances = best_model.feature_importances_
#  放進df中排序
feature_importance_df = pd.DataFrame({
    'feature':feature_names,
    'importance':importances}).sort_values(by='importance', ascending = False)

#  繪出前20個重要的特徵重要性
top_feature = 20
plt.figure(figsize=(10,6))
plt.barh(feature_importance_df['feature'][:top_feature][::-1],
         feature_importance_df['importance'][:top_feature][::-1])
plt.xlabel('Importance')
plt.title(f"Top {top_feature} Feature Importance (XGBoost)")
plt.grid()
plt.tight_layout()
plt.show()






#  再透過Random Forest or XGBoost提升準確率,並分析特徵重要性

#可視化PR圖
import matplotlib.pyplot as plt
plt.figure()
plt.plot(thresholds, precisions[1:], label='Precision')
plt.plot(thresholds, recalls[1:], label='Recall')
plt.axvline(optimal_threshold, color='red', linestyle='--', label=f'Threshold = {optimal_threshold:.2f}')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title(f'Precision & Recall vs Threshold: {name}')
plt.legend()
plt.grid()
plt.show()





from sklearn.metrics import average_precision_score




#  想先看一下失智症的分布狀況，資料為base_y
ax = sns.countplot(data=y , x='MMSE' , palette = {0:'skyblue',1:'salmon'},alpha = 0.7)
#  在每個柱子上加數字
for p in ax.patches:
    count = int(p.get_height())
    ax.annotate(f'{count}',
                (p.get_x() + p.get_width() / 2., p.get_height()),   # x, y 位置
                ha = 'center' , va='bottom' , fontsize=10 , color = 'black')
plt.title('失智症分佈圖')
plt.xlabel('失智症 (0=無, 1=有)')
plt.ylabel('樣本數')
plt.show()



#  利用Random Forest來看哪些特徵具有重要性
if 'Random Forest' in results and hasattr(results['Random Forest'][0], 'feature_importances_'):    #提取訓練好的隨機森林模型
    #提取訓練好的隨機森林模型
    rf_model = results['Random Forest'][0]
    feature_importance = rf_model.feature_importances_
    print('Feature Importance:\n',feature_importance)
    
    #繪製特徵重要性圖表
    plt.bar(base_x.columns, feature_importance, color = 'steelblue' ,align = 'center')
    plt.xlabel('Feature')
    plt.ylabel('Feature Importance')
    plt.title('Feature Importance of Randomforest')
    plt.xticks(rotation = 90,ha='right' ,fontsize = 8)
    plt.tight_layout()
    #plt.savefig("/Users/anyuchen/Desktop/bar_chart_feature_importance.jpg",dpi=500)
    plt.show()



















from sklearn.metrics import f1_score, precision_recall_curve

# 先用模型預測測試集的機率 (正類別機率)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# 計算 precision, recall 與 thresholds
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_prob)

f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)  # 加小常數避免除0

# 找到最大 F1 的閾值 index
best_idx = f1_scores.argmax()
best_threshold = thresholds[best_idx]

print(f"最佳閾值（F1 最大）: {best_threshold:.4f}")
print(f"該閾值下的 F1-score: {f1_scores[best_idx]:.4f}")

# 你可以用此閾值做最後分類
y_pred_best = (y_pred_prob >= best_threshold).astype(int)













import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_prob)

plt.figure(figsize=(8,6))
plt.plot(recalls, precisions, label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid(True)
plt.legend()
plt.show()






