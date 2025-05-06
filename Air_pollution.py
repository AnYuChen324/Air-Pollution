#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 21:07:20 2025

@author: anyuchen
"""
from pandasql import sqldf
import os
import sqlite3 
import pandas as pd

#                                                                  =====  1.讀取CSV  =====

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
print(df_2014.columns.tolist())


#                                                                  =====  2.定義函式  =====
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

df_2014=renamerow(df_2014)
df_2013=renamerow(df_2013)

#  2.轉換欄位，把污染物變成欄(長格式變寬格式)
def pivot_col(df):
    
    #將concentration轉為數值資料,並將無法轉成數值資料nan顯示
    df['concentration'] = pd.to_numeric(df['concentration'],errors='coerce')
    
    #建立寬表格
    df_wide = df.pivot_table(
        index = ['monitordate','siteid','sitename'],
        columns = 'itemname',
        values = 'concentration').reset_index()
    
    #回傳資料表
    return df_wide

df_2014_wide = pivot_col(df_2014)
df_2013_wide = pivot_col(df_2013)


#                                                      =====  3.將問卷的資料中地點欄位與台灣區域資料整合  =====
#  定義SQL查詢，定義一個幫你「用SQL查詢Pandas DataFrame」的工具函式
pysqldf = lambda q:sqldf(q,globals())

#  將TWBR10907-07_調查問卷.csv的PLACE_CURR欄位與(代碼表.xlsx)的num欄位合併
twb = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/TWBR10907-07_調查問卷.csv")
twlocation = pd.read_excel(r"/Users/anyuchen/Desktop/空氣污染/台灣區域代碼表.xlsx")

#  將空氣污染佔的sitename,siteid表格載入，並與問卷資料合併
site_info = pd.read_excel(r"/Users/anyuchen/Desktop/空氣污染/sitename_class.xlsx")

#  針對問卷先抓暫時覺得有需要的資料
twb_selection = twb[['SURVEY_DATE','ID_BIRTH','AGE', 'SEX', 'EDUCATION', 'MARRIAGE',
                     'INCOME_SELF', 'INCOME_FAMILY', 'DRK', 'SMK_CURR', 'DRUG_USE',
                     'SLEEP_QUALITY', 'ILL_ACT', 'SPO_HABIT', 'SEVERE_DYSMENORRHEA_SELF',
                     'SEVERE_DYSMENORRHEA_MOM', 'SEVERE_DYSMENORRHEA_SIS', 'MYOMA_SELF',
                     'MYOMA_MOM', 'MYOMA_SIS', 'COOK_FREQ', 'WET_HVY', 'WATER_BOILED',
                     'VEGE_YR', 'HORMOME_MED', 'HERBAL_MED', 'DRUG_KIND_A', 'DRUG_NAME_A',
                     'JOB_EXPERIENCE','PLACE_CURR']]

#  SQL指令：將問卷與台灣地方合併，並且只抓2014年的問卷
query1="""
    SELECT * FROM twb_selection
    LEFT JOIN twlocation
    ON twb_selection.PLACE_CURR = twlocation.num
    WHERE SURVEY_DATE BETWEEN '2014/01/01' and '2014/12/31'
    ORDER BY twlocation.area
    """
#  將合併好的地點資料放在twb_location中
twb_location = pysqldf(query1)
#  把result0加入到全域命名空間中
globals()['twb_location'] = twb_location

#  把重複欄位PLACE_CURR or num擇一刪除
twb_location = twb_location.drop(['PLACE_CURR'],axis=1)


#                                                      =====  4.將合併好地點的問卷與空氣污染資料合併  =====

#兩個表格的名字並不同，一個為縣市+鄉鎮,一個為鄉鎮，因此需要把twlocation的欄位做切割,我想保留city跟District
twb_location['city'] = twb_location['area'].str.extract(r'\[(.+?)\]')
twb_location['District'] = twb_location['area'].str.extract(r'\](.+?)[鄉鎮區市]')

#清除欄位District的一些空白字元
twb_location['District'] = twb_location['District'].str.replace(r'\s+','',regex=True)

#清除不需要的欄位area
twb_location = twb_location.drop(['area'],axis=1)

print(twb_location['District'].unique())
print('======================')
print(site_info['sitename'].unique())

#  SQL指令：將合併好地點的問卷與空氣污染資料合併
#  Step1:透過問卷的District與污染監測站的sitename合併，這樣才能有siteid
query2="""
    SELECT * FROM twb_location
    LEFT JOIN site_info
    ON twb_location.District = site_info.sitename
    ORDER BY sitename;
    """

#  將地點資料與twb_location合併，才能接續後面與空氣監測站資料串接
twb_location_01 = pysqldf(query2)
#  把twb_location_01加入到全域命名空間中
globals()['twb_location_01'] = twb_location_01
#  確認sitename有沒有對上
#  print(twb_location_01['sitename'].value_counts())
print(twb_location_01['siteid'].value_counts())

#  Step2:將有siteid的問卷資料與空氣污染資料合併,透過siteid合併，污染監測站資料只抓2013/06/30-2014/06/30
#  先將空氣污染監測資料合併並刪減成2013/06/30-2014/06/30區間內的資料
query3="""
    SELECT *
    FROM (
        SELECT * FROM df_2013_wide
        UNION ALL
        SELECT * FROM df_2014_wide
        ) AS merge_data
    WHERE monitordate BETWEEN '2013-06-30' AND '2014-06-30'
    ORDER BY monitordate;
    """
pollution_1314 = pysqldf(query3)
#  把pollutiom_1314加入到全域命名空間中
globals()['pollution_1314'] = pollution_1314

#  將有siteid的問卷資料與空氣污染資料合併,透過siteid合併
df_twb_pollution_2014 = pd.merge(twb_location_01,pollution_1314,on='siteid',how='left')

#把重複欄位sitename_x,sitename_y,刪掉一個，並改成sitename
df_twb_pollution_2014 = df_twb_pollution_2014.drop(['sitename_y'],axis = 1)
df_twb_pollution_2014 = df_twb_pollution_2014.rename(columns={'sitename_x':'sitename'})

#  SQL語法合併:缺點會有重複siteid欄位，除非要再SELECT教選你要的欄位
query4="""
    SELECT * FROM twb_location_01
    LEFT JOIN pollution_1314
    ON twb_location_01.siteid = pollution_1314.siteid
    ORDER BY twb_location_01.SURVEY_DATE;
"""

pollution_1314_1 = pysqldf(query4)
#  把pollutiom_1314加入到全域命名空間中
globals()['pollutiom_1314_1'] = pollution_1314_1


#                                      =====  6.利用調查問卷中的SURVEY_DATE作為起始日期，往回推半年，計算空氣汙染的平均  =====
#  SQL指令，往回推算半年的平均使用DATEDIFF(MySQL用法),但在python要使用julianday()

#  Step1:先把空氣污染監測站資料與問卷合併
#  先抓三年的空氣污染資料，怕資料量太大

#  Step2:透過self join自連結動態計算污染物半年平均
#  需要先把SURVEY_DATE(年/月/日)跟monitordate(年-月-日)的日期格式改成一樣的
query5 = """
    SELECT 
        main.*,
        #空氣污染值欄位
        
        AVG(sub.AMB_TEMP) AS avg_AMB_TEMP,
        AVG(sub.CH4) AS avg_CH4,
        AVG(sub.CO) AS avg_CO,
        AVG(sub.CO2) AS avg_CO2,
        AVG(sub.NMHC) AS avg_NMHC,
        AVG(sub.NO) AS avg_NO,
        AVG(sub.NO2) AS avg_NO2,
        AVG(sub.NOx) AS avg_NOx,
        AVG(sub.O3) AS avg_O3,
        AVG(sub.PH_RAIN) AS avg_PH_RAIN,
        AVG(sub.PM10) AS avg_PM10,
        AVG(sub.PM25) AS avg_PM25,
        AVG(sub.RAIN_COND) AS avg_RAIN_COND,
        AVG(sub.RAIN_INT) AS avg_RAIN_INT,
        AVG(sub.RH) AS avg_RH,
        AVG(sub.SO2) AS avg_SO2,
        AVG(sub.THC) AS avg_THC,
        AVG(sub.WIND_SPEED) AS avg_WIND_SPEED,
        AVG(sub.WS_HR) AS avg_WS_HR

    FROM df_twb_pollution_2014 AS main
    LEFT JOIN df_twb_pollution_2014 AS sub
        ON main.ID_BIRTH = sub.ID_BIRTH
        AND sub.monitordate BETWEEN DATE(main.SURVEY_DATE, INTERVAL 6 MONTH) AND main.SURVEY_DATE
    GROUP BY main.ID_BIRTH, main.SURVEY_DATE
"""
data_1314 = pysqldf(query5)
#  把pollutiom_1314加入到全域命名空間中
globals()['data_1314'] = data_1314





























#  =====  SQL結果轉乘DataFrame


"""
#  =====  2.針對污染物依據不同地區分類  =====

     地點有：['基隆' '汐止' '萬里' '新店' '土城' '板橋' '新莊' '菜寮' '林口' '淡水' '士林' '中山' 
             '萬華' '古亭' '松山' '大同' '桃園' '大園' '觀音' '平鎮' '龍潭' '湖口' '竹東' '新竹' 
             '頭份' '苗栗' '三義' '豐原' '沙鹿' '大里' '忠明' '西屯' '彰化' '線西' '二林' '南投' 
             '斗六' '崙背' '新港' '朴子' '臺西' '嘉義' '新營' '善化' '安南' '臺南' '美濃' '橋頭' 
             '仁武' '鳳山' '大寮' '林園' '楠梓' '左營' '前金' '前鎮' '小港' '屏東' '潮州' '恆春' 
             '臺東' '花蓮' '陽明' '宜蘭' '冬山' '三重' '中壢' '竹山' '永和' '復興' '埔里' '馬祖' 
             '金門' '馬公' '關山']

#定義一個韓式建立不同地區的資料，可為一個地區(字串)或多個地區list
def area_data(areas):
    
    #判斷是否為字串，若是轉換成list，like['基隆']
    if isinstance(areas,str):
        areas = [areas]
        
    #把每個值加上引號做區隔，並用逗號串接，例如：'土城','淡水'，就可以直接放在SQL的IN(...)條件中
    area_str = ','.join([f"'{area}'" for area in areas])
        
    query=f"""
        SELECT
            *
        FROM result0
        WHERE sitename IN ({area_str});""" 
    #print(f"=====  2005資料 - {areas} =====")
    return pysqldf(query)

#基隆(基隆)
df_Keelung = area_data('基隆')

#新北(汐止,永和,新店,三重,土城,淡水,萬里,林口,板橋,菜寮,新莊)沒有石門
NewTaipei = ['汐止','永和','新店','三重','土城','淡水','萬里','林口','板橋','菜寮','新莊']
df_NewTaipei = area_data(NewTaipei)

#台北(古亭,萬華,中山,士林,陽明,大同,松山)
Taipei = ['古亭','萬華','中山','士林','陽明','大同','松山']
df_Taipei = area_data(Taipei)

#桃園(中壢,龍潭,平鎮,觀音,大園,桃園)
Taoyuan = ['中壢','龍潭','平鎮','觀音','大園','桃園']
df_Taoyuan = area_data(Taoyuan)

#新竹(湖口,竹東,新竹)
Hsinchu = ['湖口','竹東','新竹']
df_Hsinchu = area_data(Hsinchu)

#苗栗(頭份,苗栗,三義)
Miaoli = ['頭份','苗栗','三義']
df_Miaoli = area_data(Miaoli)

#台中(豐原,沙鹿,大里,西屯,忠明)
Taichung = ['豐原','沙鹿','大里','西屯','忠明']
df_Taichung = area_data(Taichung)

#彰化(彰化,二林,線西)沒有大城
Changhua = ['彰化','二林','線西']
df_Changhua = area_data(Changhua)

#南投(南投,竹山,埔里)
Nantou = ['南投','竹山','埔里']
df_Nantou = area_data(Nantou)

#雲林(崙背,斗六,臺西)居然沒有麥寮
Yunlin = ['崙背','斗六','臺西']
df_Yunlin = area_data(Yunlin)

#嘉義(新港,朴子,嘉義)
Chiayi = ['新港','朴子','嘉義']
df_Chiayi = area_data(Chiayi)

#台南(安南,新營,臺南,善化)
Tainan = ['安南','新營','臺南','善化']
df_Tainan = area_data(Tainan)

#高雄(前金,仁武,橋頭,楠梓,左營,前鎮,美濃,鳳山,大寮,小港,林園,復興)
Kaohsiung = ['前金','仁武','橋頭','楠梓','左營','前鎮','美濃','鳳山','大寮','小港','林園','復興']
df_Kaohsiung = area_data(Kaohsiung)

#屏東(恆春,屏東,潮州)
Pingtung = ['恆春','屏東','潮州']
df_Pingtung = area_data(Pingtung)

#台東(臺東,關山)
Taitung = ['臺東','關山']
df_Taitung = area_data(Taitung)

#花蓮(花蓮)
Hualien = ['花蓮']
df_Hualien = area_data(Hualien)

#宜蘭(宜蘭,冬山)
Yilan = ['宜蘭','冬山']
df_Yilan = area_data(Yilan)

#外島(馬祖,金門,馬公)
Island = ['馬祖','金門','馬公']
df_Island = area_data(Island)
"""

