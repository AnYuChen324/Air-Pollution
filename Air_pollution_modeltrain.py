#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 20:20:00 2025

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
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
import numpy as np


#  讀取空氣污染站的資料
model_data = pd.read_csv(r"/Users/anyuchen/Desktop/空氣污染/dataset/整理好的資料輸出/modet_train_data_final.csv")

x = model_data.drop(columns = ['MMSE','HighRisk'],axis = 1)
y_MMSE = model_data['MMSE']
y_highrisk = model_data['HighRisk']

y_highrisk.value_counts()

y_MMSE.value_counts()
y_highrisk.value_counts()


#                                                               =====  變數之間共線性VIF問題  =====
VIF_x = x.drop(columns=['avg_NO','avg_NOx','BODY_WEIGHT','T_CHO','BODY_WAISTLINE'],axis = 1)
#輸入的x為數值型資料(標準化後的數值變數,one hot encoding後的類別變數)
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
def calculate_vif(x):
    vif_data=pd.DataFrame()
    vif_data["Feature"]=x.columns   #變數名稱
    vif_data['VIF']=[variance_inflation_factor(x.values,i)for i in range(x.shape[1])]
    print(f"=== Variance Inflation Factor ===")
    print(vif_data)
#  1.計算VIF所有特徵X之間的貢獻性
calculate_vif(VIF_x)


#                                                          =====  利用PCA/TSNE降成2維，觀察所有資料分布  =====
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

#  畫出資料分佈圖
def draw_pca_tsne(x,y,method='PCA',title='Feature Space'):
    #執行降維
    if method.upper() == 'PCA':
        reducer = PCA(n_components=2)
        x_reduced = reducer.fit_transform(x)
        comp_names = ['PCA1','PCA2']
        y_plot = y   #全部y
    elif method.upper() == 'TSNE':
        #為了降低計算成本，隨機取50000個樣本
        x_tsne = x.sample(n=50000, random_state=42)
        y_tsne = y.loc[x_tsne.index]  # 抽樣相同的 index
        reducer = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=250)
        x_reduced = reducer.fit_transform(x_tsne)
        comp_names = ['TSNE1','TSNE2']
        y_plot = y_tsne
    else:
        raise ValueError("method 必須是 'PCA' 或 'TSNE'")
        
    # 繪圖資料
    df = pd.DataFrame(data=x_reduced , columns = comp_names)
    df['label'] = y_plot.reset_index(drop=True)  #在降維資料中創建新欄位label,並填入原本的y
    
    #繪圖
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=df , x=comp_names[0] , y=comp_names[1] , hue = 'label' ,palette = {0:'brown',1:'blue'} , alpha = 0.6)
    plt.title(f'{title} ({method.upper()})')
    plt.xlabel(comp_names[0])
    plt.ylabel(comp_names[1])
    plt.legend(title='Label')
    plt.savefig(f"/Users/anyuchen/Desktop/空氣污染/graph/{title} ({method.upper()}).png",dpi = 300) #並將所有圖形存下來
    plt.grid(True)
    plt.show()

draw_pca_tsne(VIF_x ,y_MMSE ,title = 'PCA_MMSE')
draw_pca_tsne(VIF_x ,y_highrisk ,title = 'PCA_HighRisk')
draw_pca_tsne(VIF_x ,y_MMSE ,method='TSNE' ,title = 'TSNE_MMSE')
draw_pca_tsne(VIF_x ,y_highrisk ,method='TSNE' ,title = 'TSNE_HighRisk')



#                                                            =====  模型訓練(要使用平衡後的資料)  =====
#  需要丟進模型訓練的欄位
'''
['AGE', 'SEX', 'EDUCATION', 'MARRIAGE', 'DRK', 'SPO_HABIT','avg_AMB_TEMP', 'avg_CO', 'avg_NO', 'avg_NO2', 'avg_NOx', 'avg_O3',
 'avg_PM10', 'avg_PM25', 'avg_RH', 'avg_SO2', 'avg_WIND_SPEED','avg_WS_HR', 'MMSE']
'''

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import lightgbm as lgb
from itertools import cycle  #用於生成色彩循環
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import make_scorer, fbeta_score
from catboost import CatBoostClassifier


def model_train_evaluate(x,y):
    #  切割自變數、因變數：是否確診失智的資料model_data
    #model_data = pd.concat([num_data_scaled,category_data_encode,binary_data],axis = 1)
    #y = Avg_pollution_1220_final['MMSE']

    #  切割訓練測試及驗證資料(因為要找最佳決策罰值，所以會需要驗證資料),stratify=y：確保三個資料集中的失智比例跟原本一樣
    #  先將測試資料切出來,將剩餘資料暫存於X_temp,y_temp中
    #  最終會得到train 60%, Valid 20%, test 20%
    X_temp,X_test,y_temp,y_test = train_test_split(x ,y, test_size = 0.2, stratify=y,random_state=3)
    X_train,X_val,y_train,y_val = train_test_split(X_temp,y_temp, test_size = 0.25, stratify=y_temp, random_state=3)
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
            #'Logistic Regression':{
            #'model':LogisticRegression(class_weight = 'balanced'),
            #'params':{
                #'C':[0.01,0.1,1],
                #'penalty':['l1'],
                #'solver': ['liblinear'],  # 'liblinear' 支援 l1/l2
                #'class_weight': [None, 'balanced']
              #}
            #},
        #隨機森林模型
        #'Random Forest':{
            #'model':RandomForestClassifier(),
            #'params':{
              #'max_depth':[None,10,20],
                #'min_samples_split': [2, 5, 10],
                #'min_samples_leaf': [1, 2, 4],
                #'max_features': ['sqrt', 'log2']
                #}
           #},
        #新增一個XGBoost:處理極度不平衡資料
        #use_label_encoder = False  防止警告訊息
        #eval_metric = 'logloss'  必填，不然會跳警告
        #'XGBoost':{
            #'model':XGBClassifier(eval_metric = 'aucpr' ,random_state = 3),
            #'params':{
                #'max_depth': [3, 5, 7],   #決策數的最大深度
                #'learning_rate': [0.01, 0.1, 0.3],   #學習率
                #'n_estimators': [50, 100],   #森林裡數的數量
                #'subsample': [0.6, 0.8, 1.0],   #每棵樹訓練時隨機抽樣的樣本比例(可防止過擬)
                #'colsample_bytree': [0.6, 0.8, 1.0] ,  #每棵樹訓練時隨機抽樣的特徵比例
                #'scale_pos_weight': [scale]  #因有做SMOTE資料平衡，不需要再平衡
                #}
           # },
        #CatBoost模型
        'CatBoost':{
            'model':CatBoostClassifier(loss_function='Logloss',eval_metric='PRAUC', verbose=0, random_seed=3,auto_class_weights='Balanced'),
            'params':{
                'iterations':[100,200],
                'learning_rate':[0.01,0.05,1],
                'depth':[4,6],
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
        y_prob=best_model.predict_proba(X_test)[:,1]  #針對最終的測試資料，獲取正類別概率
        y_pred=best_model.predict(X_test)
        
        #輸出分類報告
        print(f"Best parameters for {name} : {grid_search.best_params_}")
        print(f"====Classification Report for {name}====\n",classification_report(y_test,y_pred))
        
        
        # -- f1_score調整決策閥值 --
        #利用驗證資料及找正類別機率：目的找出「最佳 decision threshold」，例如最大化 F1-score
        y_val_prob = best_model.predict_proba(X_val)[:,1]
        #調整決策閥值Threshold Tuning：計算精確率,召回率及其對應的閥值
        precisions , recalls , thresholds = precision_recall_curve(y_val,y_val_prob)
        #找最佳的閥值,因precisions和recalls第一個數值是
        f1_scores = 2 * (precisions[1:] * recalls[1:]) / (precisions[1:] + recalls[1:] + 1e-10)
        optimal_threshold_f1 = thresholds[np.argmax(f1_scores)]
        
        print(f"Optimal thresholds: {optimal_threshold_f1}")
        
        #  將結果存放在result中
        results[name]={
            'model':best_model,
            'y_prob':y_prob,
            'optimal_threshold':optimal_threshold_f1
            }
        
        #  根據驗證資料找到最佳閥值進行預測
        y_pred_new_f1 = (y_prob >= optimal_threshold_f1).astype(int)
        print(f"====Classification Report for {name} with new pred f1====\n",classification_report(y_test,y_pred_new_f1))
    
        #PR_AUC
        from sklearn.metrics import average_precision_score
        pr_auc = average_precision_score(y_test, y_prob)
        print(f"PR-AUC: {pr_auc:.4f}")
        
        
    return X_train,X_test,y_train,y_test, X_train_sm , y_train_sm ,results

X_train ,X_test ,y_train ,y_test ,X_train_sm ,y_train_sm ,results = model_train_evaluate(VIF_x, y_highrisk)


#                                                             =====  建立降低負類比例的資料集  =====
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score,roc_curve

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

#訓練下採樣的資料模型，並畫出ROC,PRC
def models_on_undersample(results,x_test_undersample,y_test_undersample,title_suffix="下採樣測試集"):
    for model_name,info in results.items():
        print(f"\n====== Evaluating {model_name} on Undersampled Test Data ======")
        
        model = info['model']
        threshold = info['optimal_threshold']
        y_prob = model.predict_proba(x_test_undersample)[:,1]  #獲取正類別概率
        y_pred = (y_prob >= threshold).astype(int)
        
        #儲存下採樣的預測機率於results中
        info['y_prob_undersample'] = y_prob
        info['y_pred_undersample'] = y_pred

        
        #分類報告
        print(f"=== Classification Report with Optimal Threshold for {model_name} === \n",classification_report(y_test_undersample,y_pred))
        #PR_AUC
        pr_auc = average_precision_score(y_test_undersample,y_prob)
        print(f"PR_AUC: {pr_auc:.4f}")
        
        #ROC_AUC
        auc_score = roc_auc_score(y_test_undersample,y_prob)
        print(f"ROC_AUC: {auc_score:.4f}")
        
    #因為後續需要繪製ROC曲線，因此需要回傳X_test,y_test
    return model,y_pred,y_prob
        
models_on_undersample(results,x_test_undersample,y_test_undersample)


#                                                                  =====  繪製ROC曲線  =====
from itertools import cycle   #用於生成色彩循環

def plot_roc_curve(results ,y_test ,prob_key = 'y_prob' ,title_suffix=""):
    """
    通用版 ROC 繪圖函式。
    - results: 儲存模型資訊與預測機率的字典
    - y_test: 對應的測試資料（原始或下採樣）
    - prob_key: 指定使用哪種預測機率欄位，預設為 'y_prob'
    """
    
    #使用matplotlib的色彩循環
    colors=cycle(["blue","green",'red','purple','orange','brown'])
    plt.figure(figsize = (10,8))
    
    for (model_name,info),color in zip(results.items(),colors):
        if prob_key not in info:
            print(f"跳過 {model_name} 沒有 {prob_key},要先執行模型預測")
            continue
        y_prob = info[prob_key]
        fpr,tpr,_ = roc_curve(y_test,y_prob)
        auc_score = roc_auc_score(y_test,y_prob)
        plt.plot(fpr,tpr,color=color,label=f"{model_name}' (AUC = {auc_score:.3f})")
            
    #添加基線(可有可無)
    plt.plot([0,1],[0,1],color='gray',linestyle='--',label="Random Base Line")
    plt.title("ROC Curve",fontsize=16)
    plt.xlabel('False Positive Rate(FPR)',fontsize=14)
    plt.ylabel('True Positive Rate(TPR)',fontsize=14)
    plt.legend(loc='lower right',fontsize=12) 
    plt.grid(True)
    plt.savefig(f"/Users/anyuchen/Desktop/空氣污染/graph/{title_suffix} ROC curve.jpg",dpi=500)
    plt.tight_layout()
    plt.show()

plot_roc_curve(results ,y_test ,title_suffix="_SMOTE")    
plot_roc_curve(results,y_test_undersample ,prob_key ='y_prob_undersample' ,title_suffix="_SMOTE+Undersample")


#                                                                  =====  繪製PRC曲線  =====
def plot_prc_curve(results ,y_test ,prob_key = 'y_prob' ,title_suffix=""):
    """
    通用版 PRC 繪圖函式。
    - results: 儲存模型資訊與預測機率的字典
    - y_test: 對應的測試資料（原始或下採樣）
    - prob_key: 指定使用哪種預測機率欄位，預設為 'y_prob'
    """
    
    #  畫出precision-recall curv，找出在precision跟recall的平衡點:Precision 至少 0.5同時 Recall 不小於 0.4
    plt.figure(figsize=(10,8))
    colors=cycle(["blue","green",'red','purple','orange','brown'])
    
    for (model_name,info), color  in zip(results.items(),colors):
        if prob_key not in info:
            print(f"跳過 {model_name} 沒有 {prob_key},要先執行模型預測")
            continue
        
        y_prob = info[prob_key]
        optimal_threshold = info['optimal_threshold']
        
        precision,recall,thresholds = precision_recall_curve(y_test,y_prob)
        pr_auc = average_precision_score(y_test,y_prob)
        
        plt.plot(thresholds,precision[1:] ,linestyle='-' ,color = color ,label = f"{model_name} - Precision (AUC={pr_auc:.3f})")
        plt.plot(thresholds,recall[1:] ,linestyle='--' ,color = color, label = f"{model_name} - Recall")
        plt.axvline(optimal_threshold , color=color ,linestyle=':' , label = f'{model_name} - Threshold = {optimal_threshold:.2f}')
        
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(f'Precision & Recall vs Threshold : {title_suffix}')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(f"/Users/anyuchen/Desktop/空氣污染/graph/{title_suffix} Precision & Recall vs Threshold curve.jpg",dpi=500)
    plt.tight_layout()
    plt.show()

plot_prc_curve(results ,y_test ,title_suffix="SMOTE")   
plot_prc_curve(results ,y_test_undersample ,prob_key ='y_prob_undersample' ,title_suffix="SMOTE_Undersample")


#  -----Shap Analysis,並繪製Beeswarm plot-----

import shap
def plot_shap(results,model_name = 'CatBoost' ,X=None ,title_suffix = ""):
    #  1.SHAP Analysis,只需要result中模型的本體results[model][0]
    #  從results中獲取訓練好的XGBoost模型本體
    model=results[model_name]['model']
    
    #  創建SHAP解釋氣，對於隨機森林可以使用TreeExplainer
    explainer=shap.TreeExplainer(model)
    
    #  取得SHAP值
    shap_values=explainer.shap_values(X)
    
    #處理二酚類情況下的shape_values結構(多數結構都為list,但LightBoost為np.array，不需要另外取正類shap_values[1])
    if isinstance(shap_values, list):
        #多數模型(CatBoost,XGBoost)在二分類會回傳[負類,正類],需要選正類的值
        #shap_values[0].shape == shap_values[1].shape 主要是一種保險機制，用來確保你拿到的是「標準的二分類 SHAP 結構」，而不是多分類或其他奇怪情況
        if len(shap_values) == 2 and shap_values[0].shape == shap_values[1].shape:
            shap_values_used = shap_values[1]
        else:
            raise ValueError("Unexpected SHAP output structure for multi-class model.")
    
    else:
        # LightGBM 直接回傳 ndarray，直接用即可
        shap_values_used = shap_values
        
    #  顯示shape值
    print(f"shap_values:{shap_values_used}")
    
    #檢查SHAP值的形狀
    print(f"X shape:{X.shape}")
    print(f"shap_values shape:{shap_values_used.shape}")
    #因為是目標變數是二分類問題，shap_values 是一個包含兩個元素的列表：其中shap_values[0]是負類（類別0）的 SHAP 值，shap_values[1] 是正類（類別1）的 
    #但因為這次使用的是LightGBM他只會為傳ndarray:shape = (n_samples,n_features)，所以不需要特別抓正類1
    #如果是Random Forest/CatBoost就會是回傳list,需要特別抓正類1
    #shap_values_class_1=shap_values[1]  #只關心正類
    
    #  2.繪製Beeswarmplot
    shap.summary_plot(shap_values_used,X,show=False)
    plt.savefig(f"/Users/anyuchen/Desktop/空氣污染/graph/{title_suffix} Beeswarm plot.jpg",dpi=500)
    plt.show()
    
    #  3.印出特徵重要性(SHAP值的平均絕對值)
    importance_df = pd.DataFrame({
        'feature':X.columns,
        'mean_abs_shap':np.abs(shap_values_used).mean(axis = 0)}).sort_values(by='mean_abs_shap',ascending=False)
    print("\n Top 10 重要特徵(by mean|SHAP|):")
    print(importance_df.head(10))
    
plot_shap(results ,model_name = 'LightGBM' ,X=x_test_undersample ,title_suffix = 'LightGBM')
plot_shap(results ,model_name = 'CatBoost' ,X=x_test_undersample ,title_suffix = 'CatBoost')



#                                                      * * * * * * * * * * * * * * * * * * * * * * 
#                                                 //////   =====  單純做一個空氣污染的預測  =====   //////
air_pollution_col = ['avg_AMB_TEMP', 'avg_CO', 'avg_NO', 'avg_NO2', 'avg_NOx','avg_O3', 'avg_PM10', 'avg_PM25', 'avg_SO2']
x_air_pollution = model_data[air_pollution_col]
y_air_pollution = model_data['HighRisk']

#  空氣污染值模型訓練
X_train_air ,X_test_air ,y_train_air ,y_test_air ,X_train_sm_air ,y_train_sm_air ,results_air = model_train_evaluate(x_air_pollution,y_air_pollution)

#  原先的資料：X_train_sm_air , y_train_sm_air ,X_test_air ,y_test_air
#  先把y_test切分為正/負類資料
y_test_pos_air = y_test_air[y_test_air == 1]
y_test_neg_air = y_test_air[y_test_air == 0]

#  對負類下採樣：負類資料調整為約正類的10倍,sample為隨機抽樣n筆資料
y_test_neg_undersample_air = y_test_neg_air.sample(n=len(y_test_pos_air)*10,random_state = 3)

#  將test資料合併，並打亂:sample(frac=1)表示：抽樣一個「比例」的資料，frac=1 意味著抽取「全部資料的 100%」，但順序被隨機打亂
y_test_undersample_air = pd.concat([y_test_pos_air,y_test_neg_undersample_air]).sample(frac = 1,random_state=3)

#  抓對應y_test索引抓出x_test
x_test_undersample_air = X_test_air.loc[y_test_undersample_air.index]

#  根據下採樣以後做模型訓練
models_on_undersample(results_air,x_test_undersample_air,y_test_undersample_air)

#  繪製ROC Curve
plot_roc_curve(results_air ,y_test_undersample_air ,prob_key ='y_prob_undersample' ,title_suffix="SMOTE_Undersample_air")

#  繪製PRC Curve
plot_prc_curve(results_air ,y_test_undersample_air ,prob_key ='y_prob_undersample' ,title_suffix="SMOTE_Undersample_air")

#  繪製Shape
plot_shap(results_air ,model_name = 'CatBoost' ,X=x_test_undersample_air ,title_suffix = 'CatBoost_air')


#                                                      * * * * * * * * * * * * * * * * * * * * * * 
#                                                 //////   =====  單純做個人基本資料的預測  =====   //////
x_personal = model_data.drop(columns = ['avg_AMB_TEMP', 'avg_CO', 'avg_NO', 'avg_NO2', 'avg_NOx','avg_O3', 'avg_PM10', 'avg_PM25', 'avg_SO2','HighRisk','MMSE'],axis=1)
y_personal = model_data['HighRisk']

#  空氣污染值模型訓練
X_train_per ,X_test_per ,y_train_per ,y_test_per ,X_train_sm_per ,y_train_sm_per ,results_per = model_train_evaluate(x_personal,y_personal)

#  原先的資料：X_train_sm_air , y_train_sm_air ,X_test_air ,y_test_air
#  先把y_test切分為正/負類資料
y_test_pos_per = y_test_per[y_test_per == 1]
y_test_neg_per = y_test_per[y_test_per == 0]

#  對負類下採樣：負類資料調整為約正類的10倍,sample為隨機抽樣n筆資料
y_test_neg_undersample_per = y_test_neg_per.sample(n=len(y_test_pos_per)*10,random_state = 3)

#  將test資料合併，並打亂:sample(frac=1)表示：抽樣一個「比例」的資料，frac=1 意味著抽取「全部資料的 100%」，但順序被隨機打亂
y_test_undersample_per = pd.concat([y_test_pos_per,y_test_neg_undersample_per]).sample(frac = 1,random_state=3)

#  抓對應y_test索引抓出x_test
x_test_undersample_per = X_test_per.loc[y_test_undersample_per.index]

#  根據下採樣以後做模型訓練
models_on_undersample(results_per,x_test_undersample_per,y_test_undersample_per)

#  繪製ROC Curve
plot_roc_curve(results_per ,y_test_undersample_per ,prob_key ='y_prob_undersample' ,title_suffix="SMOTE_Undersample_per")

#  繪製PRC Curve
plot_prc_curve(results_per ,y_test_undersample_per ,prob_key ='y_prob_undersample' ,title_suffix="SMOTE_Undersample_per")

#  繪製Shape
plot_shap(results_per ,model_name = 'CatBoost' ,X=x_test_undersample_per ,title_suffix = "CatBoost_per")







