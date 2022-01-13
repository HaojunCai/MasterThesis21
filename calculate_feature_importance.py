# -*- coding: utf-8 -*-
"""
Created on Jun 2021 
@author: Haojun Cai
"""

import pandas as pd
import numpy as np

def compare_feat_importance(model, mob_flags, EVAL_PATH):
    """
    Return mean feature importance and the ranks of the importance for QRF model.
    
    Paramaters
    ----------
    model : str, model type (QRF)
    mob_flags : boolean, flag indicating whether to consider mobility features
    EVAL_PATH : str, path to save results
    
    Returns
    ----------
    N/A
    """
    
    for mob_flag in mob_flags:
        if mob_flag == False:
            model_name = model
        else:
            model_name = model + '_mob'
            
        print(model_name)
        eval_folder = EVAL_PATH + '/evaluation/' + model_name
        
        feat_importance_model = pd.read_csv(eval_folder+'/'+'importances.csv')
        if feat_importance_model[feat_importance_model<0].sum().sum() > 0:
            print('Feature importance smaller than 0:\n', feat_importance_model[feat_importance_model<0].sum())
            
        # calculate mean importance for each feature across all users
        feat_mean_importance_model = feat_importance_model.drop(columns=['user_id']).mean(axis=0)
        feat_mean_importance_model = pd.DataFrame(feat_mean_importance_model)
        feat_mean_importance_model.columns = ['mean_importance']
        feat_mean_importance_model = feat_mean_importance_model.sort_values(by=['mean_importance'], ascending=False)
        
        # calculate when it ranks in the first third of all features
        feats = list(set(feat_importance_model.columns))
        feats.remove('user_id')

        model_rank = {'user_id':[]}
        for feat in feats:
            model_rank[feat] = []

        for i in range(0,len(feat_importance_model)):
            # print(i)
            feat_row = list(feat_importance_model.loc[i,feats])
            feat_row_sort = sorted(feat_row, reverse=True)
            
            for feat in feat_row_sort:
                feat_df = feat_importance_model.loc[i,feat_importance_model.loc[i,:]==feat].index[0]
                model_rank[str(feat_df)].append(feat_row_sort.index(feat) + 1)
            
            user_id = feat_importance_model.loc[i, 'user_id']
            model_rank['user_id'].append(user_id)
        
        # calculate mean importance for each feature by different temporal resolution across all users if applied
        model_rank = pd.DataFrame(model_rank)
        
        timefeat = ['1day', '2day', '3day', '3dayavr', '7day', '1weekday', '2weekday', '3weekday', '4weekday']
        mean_importance_attr = feat_mean_importance_model.copy()
        mean_importance_attr['time'] = np.nan
        mean_importance_attr['feat'] = np.nan
        for idx in feat_mean_importance_model.index:
            attrs = idx.split("_")
            mean_importance_attr.loc[idx,'time'] = attrs[-1]
            mean_importance_attr.loc[idx,'feat'] = idx.replace('_'+attrs[-1],'')
        
        mean_importance_attr_part = mean_importance_attr[mean_importance_attr['time'].isin(timefeat)]
        mean_importance_attr_left = mean_importance_attr.drop(mean_importance_attr_part.index)
        mean_importance_attr_left = pd.DataFrame(mean_importance_attr_left['mean_importance'])
        mean_importance_attr_save = {}
        for time in mean_importance_attr_part['time']:
            mean_importance_attr_save[time] = mean_importance_attr_part.loc[mean_importance_attr_part['time']==time,'mean_importance'].sum()
        for feat in mean_importance_attr_part['feat']:
            mean_importance_attr_save[feat] = mean_importance_attr_part.loc[mean_importance_attr_part['feat']==feat,'mean_importance'].sum()            
        mean_importance_attr_save = pd.DataFrame(mean_importance_attr_save, index=[0]).T
        mean_importance_attr_save.columns = ['mean_importance']

        mean_importance_attr_save_new1 = mean_importance_attr_save.loc[mean_importance_attr_part['time'].unique()].sort_values(by='mean_importance',ascending=False)
        mean_importance_attr_save_new2 = mean_importance_attr_save.loc[mean_importance_attr_part['feat'].unique()].sort_values(by='mean_importance',ascending=False)
        mean_importance_attr_save = pd.concat([mean_importance_attr_save_new1,mean_importance_attr_save_new2,mean_importance_attr_left])
                
        feat_mean_importance_model.to_csv(eval_folder+'/'+'mean_importance.csv', index=True)
        mean_importance_attr_save.to_csv(eval_folder+'/'+'mean_importance_byattr.csv', index=True)
        model_rank.to_csv(eval_folder+'/'+'rank_importance.csv', index=False)

def calculate_rank_stat(model, mob_flags, EVAL_PATH):
    """
    Return the total times of one feature ranking top.
    
    Paramaters
    ----------
    model : str, model type (QRF)
    mob_flags : boolean, flag indicating whether to consider mobility features
    EVAL_PATH : str, path to save results
    
    Returns
    ----------
    N/A
    """
    
    for mob_flag in mob_flags:
        
        if mob_flag == False:
            model_name = model
        else: 
            model_name = model + '_mob'
            
        print(model_name)
        eval_folder = EVAL_PATH + '/evaluation' + '/' + model_name
        
        rank_importance = pd.read_csv(eval_folder+'/'+'rank_importance.csv')
        
        feats = list(set(rank_importance.columns))
        feats.remove('user_id')
        
        top_rank = np.ceil(len(feats)/3)
        print('The number of top fetaures:',top_rank)
        rank_stat = {}
            
        for feat in feats:
            rank_sum = (rank_importance[feat]<=top_rank).sum()
            rank_stat[feat] = [rank_sum]
        
        rank_stat = pd.DataFrame(rank_stat).T
        rank_stat.columns = ['Importance']
        rank_stat = rank_stat.sort_values(by=['Importance'], ascending=False)
        rank_stat.to_csv(eval_folder+'/'+'rank_importance_stat.csv', index=True)          

        


    
    
    