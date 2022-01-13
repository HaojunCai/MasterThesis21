# -*- coding: utf-8 -*-
"""
Created on Jun 2021 
@author: Haojun Cai
"""

import numpy as np
import pandas as pd

def cal_inbound_plus(quan_list, userlist, model_name, RESULT_PATH):
    """
    Caculate overesmatied and underestimated ratio for each level.
    
    Paramaters
    ----------
    quan_list : list, given quantile lists
    userlist : list, users to be evaluated
    model_name : model type
    RESULT_PATH : path to save results
    
    Returns
    ----------
    inbound_stat : dataframe, updated inbound statistics plus overestimation and underestimation ratio
    """
    
    inbound_stat = pd.DataFrame()
    y_pred_folder = RESULT_PATH + '/prediction/' + model_name
        
    for user in userlist:
        inbound = pd.DataFrame()
        y_pred_range = pd.DataFrame()
           
        y_pred_path = y_pred_folder + '/' + str(int(user)) + '_result.csv'
        y_pred = pd.read_csv(y_pred_path)
        
        sig_levels = []
        for i in range(0,int(len(quan_list)/2)): 
            sig_levels.append(round(quan_list[-(i+1)] - quan_list[i],3))
        
        column_names = [str(sig_level)+'_outbound' for sig_level in sig_levels] + [str(sig_level)+'_inbound_range' for sig_level in sig_levels]
        inbound_stat_user = pd.DataFrame(columns = column_names)
        
        for i in range(0,int(len(quan_list)/2)):
            lower_quan = quan_list[i]
            upper_quan = quan_list[-(i+1)]
            if lower_quan + upper_quan != 1:
                print('Error: wrong match of upper and lower quantile.')
            sig_level = round((upper_quan-lower_quan), 3)
            
            inbound[str(sig_level)+'_inbound'] = y_pred['true'].between(left=y_pred[str(lower_quan)], right=y_pred[str(upper_quan)])
            inbound[str(sig_level)+'_under'] = y_pred['true'].between(left=0, right=y_pred[str(lower_quan)], inclusive='left')
            inbound[str(sig_level)+'_over'] = y_pred['true'].between(left=y_pred[str(upper_quan)], right=100, inclusive='right')
            inbound.loc[inbound[str(sig_level)+'_inbound']==True,str(sig_level)+'_under'] = False
            inbound.loc[inbound[str(sig_level)+'_inbound']==True,str(sig_level)+'_over'] = False
            
            y_pred_range[str(sig_level)+'_range'] = y_pred[str(upper_quan)] - y_pred[str(lower_quan)]
            
            inbound_ratio = inbound.loc[inbound[str(sig_level)+'_inbound']==True, str(sig_level)+'_inbound'].sum()/len(inbound)
            under_ratio = inbound.loc[inbound[str(sig_level)+'_under']==True, str(sig_level)+'_under'].sum()/len(inbound)
            over_ratio = inbound.loc[inbound[str(sig_level)+'_over']==True, str(sig_level)+'_over'].sum()/len(inbound)
            outbound_ratio = 1 - inbound_ratio
            
            inbound_range = y_pred_range.loc[inbound[str(sig_level)+'_inbound']==True, str(sig_level)+'_range'].mean()
            inbound_stat_user.loc[0,str(sig_level)+'_outbound'] = outbound_ratio
            inbound_stat_user.loc[0,str(sig_level)+'_inbound_range'] = inbound_range
            inbound_stat_user.loc[0,str(sig_level)+'_under'] = under_ratio            
            inbound_stat_user.loc[0,str(sig_level)+'_over'] = over_ratio
            
        inbound_stat = inbound_stat.append(inbound_stat_user,ignore_index=True)
    
    inbound_stat['user_id'] = userlist
     
    return inbound_stat

                
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    