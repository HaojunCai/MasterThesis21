# -*- coding: utf-8 -*-
"""
Created on Jun 2021
@author: Haojun Cai 
"""

import pandas as pd

def compare_deter_eval(models, mob_flags, quan_list, EVAL_PATH):
    """
    Calculate mean values over all users of one model
    
    Paramaters
    ----------
    models : list, models to be evaluted
    mob_flags : boolean, flag indicating whether to consider mobility features
    quan_list : list, given quantiles to be evaluated
    EVAL_PATH : str, path to save results
    
    Returns
    ----------
    N/A
    """

    deter_eval = pd.DataFrame()
    quanloss_quan = pd.DataFrame()
    inbound_stat = pd.DataFrame()
    model_cols = []
    
    for model in models:
        for mob_flag in mob_flags:
            if mob_flag == False:
                model_name = model
            else:
                model_name = model + '_mob'
            
            model_cols.append(model_name)  
            eval_folder = EVAL_PATH + '/evaluation/' + model_name
            
            deter_eval_model = pd.read_csv(eval_folder+'/'+'deter_eval.csv')
            quanloss_model = pd.read_csv(eval_folder+'/'+'prob_quanloss.csv')
            inbound_stat_model = pd.read_csv(eval_folder+'/'+'prob_inbound_underover.csv')
    
            deter_eval_model = pd.DataFrame(deter_eval_model.drop(columns=['user_id']).mean(axis=0))
            quanloss_model = pd.DataFrame(quanloss_model.drop(columns=['user_id']).mean(axis=0))
            inbound_stat_model = pd.DataFrame(inbound_stat_model.drop(columns=['user_id']).mean(axis=0)[:])
            
            deter_eval = pd.concat([deter_eval, deter_eval_model], axis=1)
            quanloss_quan = pd.concat([quanloss_quan, quanloss_model], axis=1)
            inbound_stat = pd.concat([inbound_stat, inbound_stat_model], axis=1)
    
    # calculate mean outbound ratio and average inbound range over all users
    sig_levels = []
    for i in range(0,int(len(quan_list)/2)): 
        sig_levels.append(round(quan_list[-(i+1)] - quan_list[i],3))
    range_index = [str(sig_level)+'_inbound_range' for sig_level in sig_levels]
    ratio_index = list(set((inbound_stat.index))-set(range_index))
    inbound_stat.loc[ratio_index] = inbound_stat.loc[ratio_index] * 100
            
    deter_eval.columns = model_cols
    quanloss_quan.columns = model_cols 
    inbound_stat = inbound_stat.T
    
    outbound_cols = [str(sig_level)+'_outbound' for sig_level in sig_levels]
    over_cols = [str(sig_level)+'_over' for sig_level in sig_levels]
    under_cols = [str(sig_level)+'_under' for sig_level in sig_levels]
    inbound_cols = outbound_cols + range_index + over_cols + under_cols
    
    inbound_stat = inbound_stat[inbound_cols]
    inbound_stat.index = model_cols
    inbound_stat = inbound_stat.T

    # calculate mean quantile loss over all users
    quanloss_level = pd.DataFrame()
    for i in range(0,len(quanloss_quan.index)//2):
        lower = quanloss_quan.index[i]
        upper = quanloss_quan.index[-(i+1)]
        alpha = round(float(upper)-float(lower), 2)
        quanloss_level[alpha] = (quanloss_quan.loc[upper]+quanloss_quan.loc[lower]) / 2
    
    quanloss_quan = quanloss_quan.T
    quanloss_quan['mean'] = quanloss_quan.mean(axis=1)
    deter_eval = round(deter_eval, 4)
    quanloss_quan = round(quanloss_quan, 4)
    quanloss_level = round(quanloss_level, 4)
    inbound_stat = round(inbound_stat, 4)
    
    return deter_eval, quanloss_quan, quanloss_level, inbound_stat

