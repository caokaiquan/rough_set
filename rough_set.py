# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 09:08:09 2018

@author: SiE0009
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 基本集
def basic_set(df):
  basic = {}
  for i in df.drop_duplicates().values.tolist():
    basic[str(i)] = []
    for j,k in enumerate(df.values.tolist()):
        if k == i:
            basic[str(i)].append(j)
            
  return basic

   
# 求子集
def issubset(data):
  if len(data) == 0:
    return [[]]
  
  subsets = []
  first_element = data[0]
  rest_element = data[1:]
  
  for s in issubset(rest_element):
    subsets.append(s)
    next_subset = s[:] + [first_element]
    subsets.append(next_subset)
    
  return subsets
    




  




# 下近似集，用来做逻辑分析
# 下近似集的目标是用来找到一个什么样的条件下一定会得出什么样的决策条件
# index就是在怎么样的决策条件下的索引，columns是决策属性的列
def lower_approximation(data,index,*columns):
  # 求出条件属性的基本集
  basicset = basic_set(data.drop(list(columns),axis = 1))
  # 循环迭代基本集的每个子集，如果该子集也是这个索引列的子集，则输出  
  basicset = [v for i,v in basicset.items()]
  lower_app = []
  for nu in basicset:
    if set(nu).issubset(index):
      lower_app.append(nu)
  
  # 把列表解嵌套
  try:
    lower_app = [k for i in lower_app for k in i]
    return lower_app
  
  except:
    return lower_app    

  
  
  
  
# 上近似集，用来做逻辑分析  
# 上近似集的目标是找到一个什么样的条件下可能得出什么样的决策条件
# index就是在该条件下的索引，columns是决策属性的列
def upper_approximation(data,index,*columns):
  basicset = basic_set(data.drop(list(columns),axis = 1))  
  
  basicset = [v for i,v in basicset.items()]
  upper_app = []
  for nu in basicset:
    for ind in index:
      if (set([ind]).issubset(nu)) and (nu not in upper_app):
        upper_app.append(nu)
  
  try:
    upper_app = [k for i in upper_app for k in i]
    return upper_app
  
  except:
    return upper_app 









# 边界域
def bound(data,index,*columns):
  upper_app = upper_approximation(data,index,*columns)
  lower_app = lower_approximation(data,index,*columns)
  bound = set(upper_app) - set(lower_app)
  return bound











# 粗糙度
def rough_rate(data,index,*columns):
  upper_app = upper_approximation(data,index,*columns)
  lower_app = lower_approximation(data,index,*columns)
  alpha = locals()
  alpha['alpha_R_%s' % index] = len(lower_app)/len(upper_app)
  print('样本子集在该条件属性下的粗糙度为:',alpha['alpha_R_%s' % index])




  
  


  

# 绝对约简，用得较少
def absolute_attributes_reduction(data):
  r = sorted([v for k,v in basic_set(data).items()])
  u = locals()
  columns_num = list(range(len(data.columns)))
  imp = []
  r_x = basic_set(data)
  for i in columns_num:
    c = columns_num.copy()
    c.remove(i)
    u['U_Ind(R-%d)' % i] = data.iloc[:,c]
    u['U_Ind(R-%d)' % i] = sorted([v for k,v in basic_set(u['U_Ind(R-%s)' % i]).items()])
    if u['U_Ind(R-%d)' % i] != r:
      imp.append(i)
    
  return imp
      
    
#  for i in columns_num:
#   uu['U_Ind(R%s) % i] = basic_set(data.iloc,i)
    
  








# 相对约简
# 只能对多属性进行相对约简
def relative_attributes_reduction(data,*y):
  # 把数据集划分为条件属性集与决策属性集
  x_data = data.drop(list(y),axis = 1,inplace = False)
  y_data = data.loc[:,y]
  # 决策属性基本集
  y_basic_set = sorted([v for k,v in basic_set(y_data).items()])
  # 条件属性基本集
  x_basic_set = sorted([v for k,v in basic_set(x_data).items()])
  # 求y的x正域pos_x(y)
  pos = []
  for i in x_basic_set:
    for j in y_basic_set:
      if set(i).issubset(j):
        pos.append(i)
  pos.sort()
  print('y的x正域Pos_x(y): ',pos)     
  r_x_y = len(pos)/len(data)
  print('依赖度r_x_(y):',r_x_y)
  
  # 探索条件属性中不可省关系
  u = locals()
  pos_va = locals()
  r = locals()
  columns_num = list(range(len(x_data.columns)))
  # 收集核
  imp_core = []
  # 收集属性重要性
  imp_attr = []
  for i in columns_num:
    c = columns_num.copy()
    c.remove(i)
    u['U_Ind(R-%d)' % i] = data.iloc[:,c]
    u['U_Ind(R-%d)' % i] = sorted([v for k,v in basic_set(u['U_Ind(R-%s)' % i]).items()])
    pos_va['Pos_p-%d(y)' % i] = []
    for k in u['U_Ind(R-%d)' % i]:
      for j in y_basic_set:
        if set(k).issubset(j):
          pos_va['Pos_p-%d(y)' % i].append(k)
    if sorted(pos_va['Pos_p-%d(y)' % i]) != pos:
      imp_core.append(i)
    r['r_x-%d(y)' % i] = len(sorted(pos_va['Pos_p-%d(y)' % i]))/len(data)
    r_diff = r_x_y - r['r_x-%d(y)' % i] 
    imp_attr.append(r_diff)
  
  dict_imp = {}
  for o,p in enumerate(imp_attr):
    dict_imp[data.columns[o]] = p
    
  sorted_dict_imp = sorted(dict_imp,key = lambda x:dict_imp[x],reverse = True)
  sorted_dict_imp = list(map(lambda x: {x:dict_imp[x]},sorted_dict_imp))
  
  imp_core = [data.columns[i] for i in imp_core]
     
  print('属性重要度为:',sorted_dict_imp)
      
          
    

  
  
  
  
  
# 修改后，解决单变量不能分析的问题  
# 使用之前一定要对数据集进行离散化处理，包括决策属性
# 相对约简
def relative_attributes_reduction2(data,*y):
  # 把数据集划分为条件属性集与决策属性集
  x_data = data.drop(list(y),axis = 1,inplace = False)
  y_data = data.loc[:,y]
  # 决策属性基本集
  y_basic_set = sorted([v for k,v in basic_set(y_data).items()])
  # 条件属性基本集
  x_basic_set = sorted([v for k,v in basic_set(x_data).items()])
  # 求y的x正域pos_x(y)
  pos = []
  for i in x_basic_set:
    for j in y_basic_set:
      if set(i).issubset(j):
        pos.append(i)
  pos.sort()
  print('y的x正域Pos_x(y): ',pos)     
  r_x_y = len([k for i in pos for k in i])/len(data)
  print('依赖度r_x_(y):',r_x_y)
  
  # 探索条件属性中不可省关系
  u = locals()
  pos_va = locals()
  r = locals()
  columns_num = list(range(len(x_data.columns)))
  # 收集核
  imp_core = []
  # 收集属性重要性
  imp_attr = []
  for i in columns_num:
    if len(columns_num) > 1:
      
      c = columns_num.copy()
      c.remove(i)
      u['U_Ind(R-%d)' % i] = data.iloc[:,c]
      u['U_Ind(R-%d)' % i] = sorted([v for k,v in basic_set(u['U_Ind(R-%s)' % i]).items()])
      pos_va['Pos_p-%d(y)' % i] = []
      for k in u['U_Ind(R-%d)' % i]:
        for j in y_basic_set:
          if set(k).issubset(j):
            pos_va['Pos_p-%d(y)' % i].append(k)
      if sorted(pos_va['Pos_p-%d(y)' % i]) != pos:
        imp_core.append(i)
      r['r_x-%d(y)' % i] = len(sorted(pos_va['Pos_p-%d(y)' % i]))/len(data)
      r_diff = r_x_y - r['r_x-%d(y)' % i] 
      imp_attr.append(r_diff)
    else:
      c = columns_num.copy()
      u['U_Ind(R%d)' % i] = data.iloc[:,c]
      u['U_Ind(R%d)' % i] = sorted([v for k,v in basic_set(u['U_Ind(R%d)' % i]).items()])
      pos_va['Pos_p%d(y)' % i] = []
      for k in u['U_Ind(R%d)' % i]:
        for j in y_basic_set:
          if set(k).issubset(j):
            pos_va['Pos_p%d(y)' % i].append(k)
      if sorted(pos_va['Pos_p%d(y)' % i]) != pos:
        imp_core.append(i)
      r['r_x%d(y)' % i] = len(sorted(pos_va['Pos_p%d(y)' % i]))/len(data)
      r_diff = r_x_y - r['r_x%d(y)' % i] 
      imp_attr.append(r_diff)
      
  
  dict_imp = {}
  for o,p in enumerate(imp_attr):
    dict_imp[data.columns[o]] = p
    
  sorted_dict_imp = sorted(dict_imp,key = lambda x:dict_imp[x],reverse = True)
  sorted_dict_imp = list(map(lambda x: {x:dict_imp[x]},sorted_dict_imp))
  
  imp_core = [data.columns[i] for i in imp_core]
     
  print('属性重要度为:',sorted_dict_imp)  
  
  
  
  
  
  
  
  
  
  
  
  

# 把属性与决策都相同的数据只留一个；把决策不同，但是条件属性都一样的数据全部删去
# 产生一致决策表   
def prepro(data,*y):
  data_new = data.copy()
  n,m = data.shape
  for i in range(0,n-1):
    for k in range(i+1,n):
      
      if (data.drop(list(y),axis = 1,inplace = False).iloc[i] == data.drop(list(y),axis = 1,inplace = False).iloc[k]).all() and ((data[list(y)].iloc[i] != data[list(y)].iloc[k]).any()):
        data_new.drop(i,inplace = True)
        data_new.drop(k,inplace = True)
      
      if (data.iloc[i] == data.iloc[k]).all():
         data_new.drop(k,inplace = True)
  return data_new
      
    
  

    
    
  
  
# 分辨矩阵
def distinguish_matrix(data,*y):
  # m为行数，n为列数
  m,n = data.shape
  # 构造分辨矩阵dataframe结构
  distinguish_matrix = pd.DataFrame(index = data.index,columns = data.index)
  # 保存剔除决策属性后的条件属性
  new_data = data.drop(list(y),axis = 1,inplace = False)
  # 循环迭代给矩阵上三角赋值
  for i in range(0,len(distinguish_matrix)-1):
    for j in range(i+1,len(distinguish_matrix.columns)):
      # 如果两条数据不一样，则把不同的条件属性保存起来
      if (np.array(new_data.iloc[i].tolist()) != np.array(new_data.iloc[j].tolist())).any():
        list_difference = np.array(new_data.iloc[i].tolist()) != np.array(new_data.iloc[j].tolist())
        distinguish_matrix.iloc[i,j] = new_data.columns[list_difference].tolist()
      # 如果两行数据决策相同的话，则不用保存各自不同的条件属性，直接赋值0
      if ((data[list(y)].iloc[i] == data[list(y)].iloc[j]).all()):
        distinguish_matrix.iloc[i,j] = 0
  # 把分辨矩阵的NULL值用0填充
  distinguish_matrix.fillna(0,inplace = True)    
  return distinguish_matrix
  
  
  
  
  

# 属性重要度排序
# 需要对DELTA离散化，不然会对结果产生影响
def importance_sort(data,*y):
  # 把数据转换成分辨矩阵
  Distinguish_matrix = distinguish_matrix(data,*y)
  # 条件属性
  new_columns = data.columns.drop(list(y))
  # 条件属性的数量
  new_columns_num = len(new_columns)
  # 通过字典结构收集属性重要度排序
  importance_list = {}
  # 迭代每个属性
  for column in new_columns:
    importance_all = 0
    for i in range(0,len(Distinguish_matrix)-1):
      for j in range(i+1,len(Distinguish_matrix)):
        if Distinguish_matrix.iloc[i,j] != 0:   # 有问题
          
          # 如果属性存在于分辨矩阵其中的Mij上时候，则a=1，反之a=0
          if column in Distinguish_matrix.iloc[i,j]:
            a = 1
          else:
            a = 0
          # 根据分辨矩阵中的项大小，赋予不同的权重
          weight = new_columns_num/len(Distinguish_matrix.iloc[i,j])  
          # 计算重要性
          importance = a*weight/len(Distinguish_matrix.iloc[i,j])
          # 根据属性，累加重要性
          importance_all += importance
    if importance_all > 0:
      # 把重要度收集在数据字典上      
      importance_list[column] = importance_all      
  # 把字典根据值进行排序
  sorted_importance_list = sorted(importance_list,key = lambda x:importance_list[x],reverse = True)
  sorted_importance_list = list(map(lambda x: {x:importance_list[x]},sorted_importance_list))
  return sorted_importance_list
    
  







  
  
      
#离散化算法
# naive_sclar算法
def Naive_Scaler(data,y):
  n = len(data)
  C = []
  for i in data.drop(y,axis = 1,inplace = False).columns:
    col_sort = data.sort_values(by = i)
    c = []
    for j in range(0,n):
      if j == n-1:
        break
      if (col_sort[i][j] == col_sort[i][j+1]) and (col_sort[y][j] == col_sort[y][j+1]):
        continue
      else:
        c.append((col_sort[i][j] + col_sort[i][j+1]) / 2)
    C.append(c)
  return C
        
      
    

# 等距离划分离散化  用于工况离散化
def discretization(data,k,*y):
  new_data = data.copy()
  # 总属性断点集
  breakingpoint_all = []
  # 对除决策属性以外的条件属性进行迭代循环
  for i in new_data.drop(list(y),axis = 1,inplace = False).columns:
    # 各属性断点集
    breakingpoint = []
    # 根据属性值从小到大排序
    col_sort = new_data.sort_values(by = i)
    # 属性最大、最小值
    x_max = col_sort[i].max() +0.01
    x_min = col_sort[i].min() 
    # 断点间隔
    diff = (x_max - x_min)/k
    # 求各个断点
    for k in range(0,k+1):
      breakingpoint.append(x_min + k * diff)
    breakingpoint_all.append(breakingpoint)
  # 根据断点区间切分属性
  for num in range(len(breakingpoint_all)):
    new_data.iloc[:,num] = pd.cut(new_data.iloc[:,num],breakingpoint_all[num],right = False,labels = list(range(k)))
  print('断点集合：',breakingpoint_all)
  return new_data





# LASERVALUE = 2/9,MFC_CLAD1_OO_ACTVALUE = 1.5/38,MFC_CLAD1_IS_ACTVALUE = 1/14,MFC_CLAD1_IO_ACTVALUE = 1/14,MFC_CLAD1_OS_ACTVALUE = 1/8,
# MFC_CORE_OO_ACTVALUE = 1/70,MFC_CORE_IO_ACTVALUE = 1/30,MFC_CORE_IH_ACTVALUE = 1/42,MFC_CORE_OH_ACTVALUE = 0.05/10.2,MFC_CLAD1_IH_ACTVALUE = 0.0001/44.99936,
# MFC_CLAD1_OH_ACTVALUE = 4/151,MFC_1A_ACTVALUE = 0.16,MFC_1C_ACTVALUE = 0.005/0.1814,MFC_2A_ACTVALUE = 0.5/43,MFC_2C_ACTVALUE = 0.5/10.6,MFC_2D_ACTVALUE = 0.01/0.572
 
  
# MFC_CLAD1_OO_ACTVALUE = 2,MFC_CLAD1_IS_ACTVALUE,MFC_2A_ACTVALUE = 0.5,MFC_2D_ACTVALUE = 0.008538




# 配方离散化（不适用）  应用下面的函数
def recipe_discret(data,*y,**rate):
  new_data = data.copy()
  # 总属性断点集
  breakingpoint_all = []
  k_all = []
  # 对除决策属性以外的条件属性进行迭代循环
  for i in new_data.drop(list(y),axis = 1,inplace = False).columns:
    # 各属性断点集
    breakingpoint = []
    # 根据属性值从小到大排序
    col_sort = new_data.sort_values(by = i)
    # 属性最小值、变化幅度
    x_min = col_sort[i].min() - 0.00001
    # rate为调整幅度，多数为1/40
    if i in rate.keys():
      diff = abs(col_sort[i].mean() * rate[i])
    else:
      rate_ = 1/40
      diff = abs(col_sort[i].mean() * rate_)
    k = 0
    while pd.cut(new_data[i],breakingpoint).isnull().any():
      breakingpoint.append(x_min + k * diff)
      k += 1
    
    breakingpoint_all.append(breakingpoint)
    k_all.append(k)
  # 根据断点区间切分属性
  for num in range(len(breakingpoint_all)):
    new_data.iloc[:,num] = pd.cut(new_data.iloc[:,num],breakingpoint_all[num],right = False,labels = list(range(1,k_all[num])))
  print('断点集合：',breakingpoint_all)
  print('k为',k_all)
  return new_data









# 使用函数之前需要reindex，重排索引，不然会报错
# 修改后的配方离散化  
# 离散化的幅度还需要多加斟酌，IH划分的区间太多
def recipediscret(data):
  # 针对配方进行离散化，工况直接用上面的discretization等距离离散化函数来划分
  recipe = ['MFC_CLAD1_OO','MFC_CLAD1_IS','MFC_CLAD1_IO','MFC_CLAD1_OS','MFC_CORE_OO','MFC_CORE_IO',
   'MFC_CORE_IH','MFC_CORE_OH','MFC_CLAD1_IH','MFC_CLAD1_OH','MFC_1A','MFC_1C','MFC_2A','MFC_2C','MFC_2D']
  
  new_data = data.copy()
  # 总属性断点集

  # 对除决策属性以外的条件属性进行迭代循环
  for i in recipe:
    # 各属性断点集
    breakingpoint = []
    # 根据属性值从小到大排序
    #col_sort = new_data.sort_values(by = i+'_ACTVALUE')
    # 属性最小值、变化幅度，把属性最小值减去一个小值以免后面存在最小值分配不了区间的情况
    x_min = new_data[i+'_ACTVALUE'].min() - 0.00001
    
    # 因为MFC_CLAD1_OO与MFC_2D的设定值格式不一样（MFC_CLAD1_OO_SET,MFC_2D_SETVALUE）,所以需要分开处理
    
    try:
      # 根据该配方大小范围来决定保留多少位小数
      if new_data[i+'_SETVALUE'].max() >= 40:
        new_data[i+'_SETVALUE'] = new_data[i+'_SETVALUE'].apply(lambda x: float('%.1d' % x))
      
      
      elif (4 <= new_data[i+'_SETVALUE'].max())  and (new_data[i+'_SETVALUE'].max() < 40):                   #(10,40)
        new_data[i+'_SETVALUE'] = new_data[i+'_SETVALUE'].apply(lambda x: float('%.1f' % x))
      
      elif (0 <= new_data[i+'_SETVALUE'].max()) and (new_data[i+'_SETVALUE'].mean() < 4):                    #(0,10)
        new_data[i+'_SETVALUE'] = new_data[i+'_SETVALUE'].apply(lambda x: float('%.2f' % x))
#      elif (0 < new_data[i+'_SETVALUE'].max()) and (new_data[i+'_SETVALUE'].mean() < 5):
#        new_data[i+'_SETVALUE'] = new_data[i+'_SETVALUE'].apply(lambda x: float('%.3f' % x))
      else:
        new_data[i+'_SETVALUE'] = new_data[i+'_SETVALUE'].apply(lambda x: float('%.3f' % x))
        
        
      # 如果该配方最小值等于最大值，则该配方没有变化，都属于同一类别
      if new_data[i+'_SETVALUE'].min() == new_data[i+'_SETVALUE'].max():
        new_data[i+'_ACTVALUE'] = 0
      else:
        nn = []
        # 找出每个属性除了0以外最小差值
        for j in range(0,len(new_data[i+'_SETVALUE'])-1):
          for n in range(j+1,len(new_data[i+'_SETVALUE'])):
            nn.append(abs(new_data[i+'_SETVALUE'][j] - new_data[i+'_SETVALUE'][n]))
        diff = np.min([i for i in nn if i != 0])
        
        # 如果存在null值就继续划分区间知道所有值都赋予相应区间
        k = 0
        while pd.cut(new_data[i+'_ACTVALUE'],breakingpoint).isnull().any():
          breakingpoint.append(x_min + k * diff)
          k += 1
        # 打印每个属性的划分区间
        print(i,breakingpoint)  
          
        new_data.loc[:,i+'_ACTVALUE'] = pd.cut(new_data[i+'_ACTVALUE'],breakingpoint,right = False,labels = list(range(0,k-1)))
      
    except:
      if new_data[i+'_SET'].max() >= 40:
        new_data[i+'_SET'] = new_data[i+'_SET'].apply(lambda x: float('%.1d' % x))
      
      
      elif (10 <= new_data[i+'_SET'].max())  and (new_data[i+'_SET'].max() < 40):
        new_data[i+'_SET'] = new_data[i+'_SET'].apply(lambda x: float('%.1f' % x))
      
      elif (0 <= new_data[i+'_SET'].max()) and (new_data[i+'_SET'].mean() < 10):
        new_data[i+'_SET'] = new_data[i+'_SET'].apply(lambda x: float('%.2f' % x))
#      elif (0 < new_data[i+'_SETVALUE'].max()) and (new_data[i+'_SETVALUE'].mean() < 5):
#        new_data[i+'_SETVALUE'] = new_data[i+'_SETVALUE'].apply(lambda x: float('%.3f' % x))
      else:
        new_data[i+'_SET'] = new_data[i+'_SET'].apply(lambda x: float('%.3f' % x))
      
      
      
      
      if new_data[i+'_SET'].min() == new_data[i+'_SET'].max():
        new_data[i+'_ACTVALUE'] = 0
      else:
        nn = []
        for j in range(0,len(new_data[i+'_SET'])-1):
          for n in range(j+1,len(new_data[i+'_SET'])):
            nn.append(abs(new_data[i+'_SET'][j] - new_data[i+'_SET'][n]))
        diff = np.min([i for i in nn if i != 0])
        
        
        k = 0
        while pd.cut(new_data[i+'_ACTVALUE'],breakingpoint).isnull().any():
          breakingpoint.append(x_min + k * diff)
          k += 1
        print(i,breakingpoint)  
        
        new_data.loc[:,i+'_ACTVALUE'] = pd.cut(new_data[i+'_ACTVALUE'],breakingpoint,right = False,labels = list(range(0,k-1)))
        
  
  return new_data









 
    
# 把超过或者低于平均值50%的离群点去除
def outliers(data):
  outliers_all = []
  for i in data.columns:
    mean_x = data[i].mean()
    upper_x = mean_x + mean_x * 0.5
    lowwer_x = mean_x - mean_x * 0.5
    outlier = []
    for j,v in enumerate(data[i]):
      if (v > upper_x) or (v < lowwer_x):
        outlier.append(j)
    outliers_all.append(outlier)
  return outliers_all
        

        
      





# 对每根棒子进行重要度排序，对每根棒子进行一次粗糙集算法程序
# 需要对DELTA离散化，不然会对结果产生影响
def importance_sort_barcode(data):
  barcode = data.BARCODE.value_counts().index
  barname = []
  importance = []
  for bar in barcode:
    i = recipediscret(data[data.BARCODE == bar].reset_index(drop = True))
    i = i[['MFC_CLAD1_OO_ACTVALUE','MFC_CLAD1_IS_ACTVALUE','MFC_CLAD1_IO_ACTVALUE',
           'MFC_CLAD1_OS_ACTVALUE','MFC_CORE_OO_ACTVALUE','MFC_CORE_IO_ACTVALUE',
           'MFC_CORE_IH_ACTVALUE','MFC_CORE_OH_ACTVALUE','MFC_CLAD1_IH_ACTVALUE',
           'MFC_CLAD1_OH_ACTVALUE','MFC_1A_ACTVALUE','MFC_1C_ACTVALUE','MFC_2A_ACTVALUE',
           'MFC_2C_ACTVALUE','MFC_2D_ACTVALUE','COREDELTA']]
    importance_rate = importance_sort(i,'COREDELTA')
    
    if importance_rate:
      barname.append(bar)
      importance.append(importance_rate)
      print(bar,'的配方重要度排序为：',importance_rate)
  return pd.DataFrame({'BARCODE':barname,'IMPORTANCE':importance})
  








  

# 根据设备划分数据集，用来进行配方上下限分析
def data_yrp(data,yrp):
  #根据设备划分数据集
  yrp_data = data[data.EQUIPID == yrp]
  #取数据集配方实际值与设定值
  #工况也可以用同样方式操作
  yrp_data = yrp_data[['MFC_CLAD1_OO_ACTVALUE', 'MFC_CLAD1_IS_ACTVALUE',
       'MFC_CLAD1_IO_ACTVALUE', 'MFC_CLAD1_OS_ACTVALUE',
       'MFC_CORE_OO_ACTVALUE', 'MFC_CORE_IO_ACTVALUE', 'MFC_CORE_IH_ACTVALUE',
       'MFC_CORE_OH_ACTVALUE', 'MFC_CLAD1_IH_ACTVALUE','MFC_CLAD1_OH_ACTVALUE', 
       'MFC_1A_ACTVALUE', 'MFC_1C_ACTVALUE',
       'MFC_2A_ACTVALUE', 'MFC_2C_ACTVALUE', 'MFC_2D_ACTVALUE',
       'MFC_CLAD1_OO_SET', 'MFC_CLAD1_IS_SET', 'MFC_CLAD1_IO_SET',
       'MFC_CLAD1_OS_SET', 'MFC_CORE_OO_SET', 'MFC_CORE_IO_SET',
       'MFC_CORE_IH_SET', 'MFC_CORE_OH_SET', 'MFC_CLAD1_IH_SET',
       'MFC_CLAD1_OH_SET', 'MFC_1A_SETVALUE', 'MFC_1C_SETVALUE',
       'MFC_2A_SETVALUE', 'MFC_2C_SETVALUE', 'MFC_2D_SETVALUE', 
       'GROWTH_RATE','SECTIONTYPE','COREDELTA']]
  #把划分后的数据集索引重排
  yrp_data.reset_index(drop = True, inplace = True)
  #数据集进行配方离散化
  #也可以用discretization进行工况离散化
  yrp_data_xiugai = recipediscret(yrp_data)
  #取离散化后的属性值
  yrp_data_xiugai = yrp_data_xiugai[['MFC_CLAD1_OO_ACTVALUE','MFC_CORE_OH_ACTVALUE','MFC_CORE_IH_ACTVALUE',
                   'MFC_2A_ACTVALUE','MFC_2D_ACTVALUE']]
  #对DELTA进行分类
  # standard类
  yrp_data_xiugai.loc[(yrp_data.COREDELTA > 0.325) & (yrp_data.COREDELTA < 0.36),'DELTAQUALITY'] = 1
  # tight类，如果数据中剖面类型全都是U型，则不用区分直接划分tight类，否则需要根据剖面划分tight，因为tight的都是U型剖面
  if (yrp_data.SECTIONTYPE == 'U').all():
    yrp_data_xiugai.loc[(yrp_data.COREDELTA >= 0.34) & (yrp_data.COREDELTA <= 0.35),'DELTAQUALITY'] = 0
  else:
    yrp_data_xiugai.loc[(yrp_data.COREDELTA >= 0.34) & (yrp_data.COREDELTA <= 0.35) &
                     (yrp_data.SECTIONTYPE == 'U'),'DELTAQUALITY'] = 0
  # 报废类，只要不是standard类、tight类，那就是报废类
  yrp_data_xiugai.loc[(yrp_data_xiugai.DELTAQUALITY != 0) & (yrp_data_xiugai.DELTAQUALITY != 1),'DELTAQUALITY'] = 2
  
  # 用下限找出导致delta为tight的属性值
  #lower_approximation(yrp_data_xiugai,yrp_data_xiugai.loc[yrp_data_xiugai['DELTAQUALITY'] == 0].index,'DELTAQUALITY')
  # 用下限找出导致delta为报废的属性值
  #lower_approximation(yrp_data_xiugai,yrp_data_xiugai.loc[yrp_data_xiugai['DELTAQUALITY'] == 2].index,'DELTAQUALITY')
  
  # 用上限找出导致delta为tight的属性值
  #upper_approximation(yrp_data_xiugai,yrp_data_xiugai.loc[yrp_data_xiugai['DELTAQUALITY'] == 0].index,'DELTAQUALITY')
  # 用上限找出导致delta为报废的属性值
  #upper_approximation(yrp_data_xiugai,yrp_data_xiugai.loc[yrp_data_xiugai['DELTAQUALITY'] == 2].index,'DELTAQUALITY')
  
  
  return yrp_data_xiugai








# 单变量变化对DELTA的影响，固定其他变量的值，找出目标值与DELTA之间的联系
def corr_recipe(data,yrp):
  # 先划分相应yrp的数据集
  yrp_data = data[data.EQUIPID == yrp]
  # 因为考虑到再多数情况下OH与IH都是同向且同时变动，所以把两者相加创建一个新变量
  yrp_data['OH_IH_SETVALUE'] = yrp_data['MFC_CORE_OH_SET'] + yrp_data['MFC_CORE_IH_SET']
  yrp_data['OH_IH_ACTVALUE'] = yrp_data['MFC_CORE_OH_ACTVALUE'] + yrp_data['MFC_CORE_IH_ACTVALUE']
  
  # 观察OH_IH与DELTA的关系变化，这里不考虑2A等属性，默认只有2D与OH_IH两个属性变化
  # 定义locals作为动态变量
  f = locals()
  for i in yrp_data.MFC_2D_SETVALUE.value_counts()[yrp_data.MFC_2D_SETVALUE.value_counts()>1].index:
    f['2D_EQ_%f' % i] = yrp_data[(yrp_data.MFC_2D_SETVALUE == i)][['MFC_2D_SETVALUE','OH_IH_SETVALUE','MFC_CORE_OH_ACTVALUE',
     'MFC_CORE_IH_ACTVALUE','OH_IH_ACTVALUE','COREDELTA']]
    # 过滤空的数据集，不存进csv文件
    if len(f['2D_EQ_%f' % i]) < 0:
      f['2D_EQ_%f' % i].to_csv('2D_EQ_%f.csv')
  
  # 观察2D与DELTA的关系变化，这里不考虑2A等属性，默认只有2D与OH_IH两个属性变化  
  e = locals()
  for i in yrp_data.OH_IH_SETVALUE.value_counts()[yrp_data.OH_IH_SETVALUE.value_counts()>1].index:
    e['OHIH_EQ_%f' % i] = yrp_data[(yrp_data.OH_IH_SETVALUE == i)][['OH_IH_SETVALUE','MFC_CORE_OH_ACTVALUE',
     'MFC_CORE_IH_ACTVALUE','OH_IH_ACTVALUE','MFC_2D_ACTVALUE','COREDELTA']]
    if  len(e['OHIH_EQ_%f' % i]) >0:

        e['OHIH_EQ_%f' % i].to_csv('OHIH_EQ_%f.csv' % i)
  # 取当2D设定值相等，2A设定值相等时，OH_IH与DELTA之间的关系
  g = locals()
  for i in yrp_data.MFC_2D_SETVALUE.value_counts()[yrp_data.MFC_2D_SETVALUE.value_counts()>1].index:
    for j in yrp_data.MFC_2A_SETVALUE.value_counts()[yrp_data.MFC_2A_SETVALUE.value_counts() > 3].index:
        g['2D_EQ_%f_2A_EQ_%f' % (i,j)] = yrp_data[(yrp_data.MFC_2D_SETVALUE == i) & (yrp_data.MFC_2A_SETVALUE == j)][['MFC_2D_SETVALUE','MFC_2A_SETVALUE','MFC_CORE_OH_ACTVALUE','MFC_CORE_IH_ACTVALUE','OH_IH_SETVALUE','OH_IH_ACTVALUE','COREDELTA']]
        if len(g['2D_EQ_%f_2A_EQ_%f' % (i,j)])>0:

            g['2D_EQ_%f_2A_EQ_%f' % (i,j)].to_csv('2D_EQ_%f_2A_EQ_%f.csv' % (i,j))
  
  # 取当2D设定值相等，OH_IH设定值相等，2A与DELTA之间的关系
  # 因OH_IH设定值与实际值相差太大，所以应把OH_IH实际值限定在特定区间
  # 观察2A与DELTA的关系变化
  b = locals()
  for i in yrp_data.MFC_2D_SETVALUE.value_counts()[yrp_data.MFC_2D_SETVALUE.value_counts()>1].index:
    for j in yrp_data.OH_IH_SETVALUE.value_counts()[yrp_data.OH_IH_SETVALUE.value_counts() > 1].index:
        b['2D_EQ_%f_OHIH_EQ_%f' % (i,j)] = yrp_data[(yrp_data.MFC_2D_SETVALUE == i) & (yrp_data.OH_IH_SETVALUE == j) & ((yrp_data.OH_IH_ACTVALUE>14.45) & (yrp_data.OH_IH_ACTVALUE <14.55))][['MFC_2D_SETVALUE','OH_IH_SETVALUE','MFC_CORE_OH_ACTVALUE','MFC_CORE_IH_ACTVALUE','MFC_2A_ACTVALUE','OH_IH_ACTVALUE','COREDELTA']]
        if  len(b['2D_EQ_%f_OHIH_EQ_%f' % (i,j)]) >0:

            b['2D_EQ_%f_OHIH_EQ_%f' % (i,j)].to_csv('2D_EQ_%f_OHIH_EQ_%f.csv' % (i,j))

  # 取当2A、OH_IH相等时，2D与DELTA之间的关系
  a = locals()
  for i in yrp_data.MFC_2A_SETVALUE.value_counts()[yrp_data.MFC_2A_SETVALUE.value_counts()>3].index:
    for j in yrp_data.OH_IH_SETVALUE.value_counts()[yrp_data.OH_IH_SETVALUE.value_counts() > 1].index:
        a['2A_EQ_%f_OHIH_EQ_%f' % (i,j)] = yrp_data[(yrp_data.MFC_2A_SETVALUE == i) & (yrp_data.OH_IH_SETVALUE == j) & ((yrp_data.OH_IH_ACTVALUE>14.45) & (yrp_data.OH_IH_ACTVALUE <14.55))][['MFC_2A_SETVALUE','OH_IH_SETVALUE','MFC_CORE_OH_ACTVALUE','MFC_CORE_IH_ACTVALUE','MFC_2A_ACTVALUE','OH_IH_ACTVALUE','MFC_2D_ACTVALUE','COREDELTA']]
        if len(a['2A_EQ_%f_OHIH_EQ_%f' % (i,j)])>0:

            a['2A_EQ_%f_OHIH_EQ_%f' % (i,j)].to_csv('2A_EQ_%f_OHIH_EQ_%f.csv' % (i,j))
            
  print('done')










