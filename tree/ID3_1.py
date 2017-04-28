
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from pandas import DataFrame,Series
from IPython.display import display

from math import log
from collections import Counter


# In[3]:

'''1、信息熵的计算
   2、默认用DataFrame去处理
   3、给定数据集、标签columns名称'''
def calcShannonEnt(dataSet, label_col_name):
    num_data = dataSet.shape[0]
    labels = dataSet[label_col_name]
    labels_counter = Counter(labels)
    labels_keys = labels_counter.keys()
    shannonEnt = 0
    for k,v in labels_counter.items():
        p = 1.0*v/num_data
        shannonEnt -= p*log(p,2)
    return shannonEnt


# In[4]:

'''1、划分数据集
   2、给定数据集、划分feature、feature值
   3、dataSet默认类型为DataFrame'''
def splitDateSet(dataSet,feature_name,feature_value):
    filter_bool = (dataSet[feature_name] == feature_value)
    splited_data_set = dataSet[filter_bool].drop([feature_name],axis=1,inplace=False)
    return splited_data_set


# In[5]:

'''1、选择最好的划分feature
   2、通过信息增益标准来划分'''
def chooseBestFeatureToSplit(dataSet,label_col_name):
    #整个数据集的信息熵
    base_entropy = calcShannonEnt(dataSet,label_col_name)    
    #选取所有要参加选择的feature，label不参与选择
    choose_features = list(dataSet.columns)
    choose_features.remove(label_col_name)
    #定义一个信息增益的字典用来存储
    feature_entropy_change = dict.fromkeys(choose_features)
    for feature in choose_features:
        feature_values = np.unique(dataSet[feature])
        new_entropy = 0
        #逐个算当前feature不同取值下的信息熵，最后算出当前feature的信息熵
        for feature_value in feature_values:
            split_data = splitDateSet(dataSet, feature, feature_value)
            p =1.0*split_data.shape[0] / dataSet.shape[0]
            new_entropy += p * calcShannonEnt(split_data, label_col_name)
        #feature的信息增益字典
        feature_entropy_change[feature] = base_entropy - new_entropy
    s = Series(feature_entropy_change)
    #获取信息增益最大的feature，如果最大信息增益最大的feature有多个，则选择第一个    
    return list(s[s==s.max()].index)[0]
    #return sorted(feature_entropy_change.items(), key=lambda item:item[1] )[0][0]        


# In[6]:

'''1、如果所有的属性都被划分完，但还是没有能够将数据完全划分开，这时候利用投票来决定该数据集属于哪一类'''
def majority(labels_list):
    #这里的labels_list为Series结构    
    diff_label = labels_list.value_counts()
    max_label = list(diff_label[diff_label==diff_label.max()].index)[0]
    return max_label  


# In[7]:

'''1、用递归来创建树
   2、递归的停止条件有2个：
       1)、程序遍历完所有划分数据集的属性
           如果所有属性都遍历完之后，某一分支上的数据依然不属于同一类，则通过投票决定该数据属于那一类
       2）、每个分支下的所有数据都属于同一类
   3、这里的dataSet是带有label（标签）栏的
'''
def createTree(dataSet,labels_name):
    feature_num = dataSet.shape[1] - 1
    series_labels = dataSet[labels_name]
    
    #当前数据集label不相等的类别的个数
    diff_label = series_labels.unique()
    diff_label_num = len(diff_label)
    
    #类别（label）完全相同则停止继续划分
    if diff_label_num == 1:
        return diff_label[0]
    #遍历完所有特征时，数据集中的类别依然不是完全相同的，则返回出现次数最多的类别
    if (diff_label_num >1 and feature_num ==0):
        return majority(series_labels)
    
    best_feature = chooseBestFeatureToSplit(dataSet,labels_name)
    myTree = {best_feature:{}}
    best_feature_values = dataSet[best_feature].unique()
    for value in best_feature_values:
        myTree[best_feature][value] = createTree(
            splitDateSet(dataSet, best_feature, value),labels_name)
    return myTree 


# In[11]:

'''根据决策树，对没有标签的数据进行预测'''
def classify(myTree,features,test_data):
    #以根节点和其子节点为例，进行简单分析
    parent = myTree.keys()[0]
    childs = myTree[parent]
    #获取根节点的feature在数据上的索引位置
    parent_index = features.index(parent)
    for k in childs.keys():
        #找到test_data数据在根节点属性上的取值，从而进入其子节点的相应分支
        if test_data[parent_index] == k:
            if type(childs[k]) == dict:
                #进入子节点之后，判断子节点是否还是字典，如果不是，那就是叶子节点了
                class_label = classify(childs[k], features,test_data)
            else:
                class_label = childs[k]
    return class_label
#调用实例：
#classify(createTree(data,'label'), list(data.columns), [1,0])

