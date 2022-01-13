# -*- coding: utf-8 -*-
"""
Created on Jul 2021
@author: Haojun Cai 
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def calculate_cost_user(userlist, save_flag, model_type, quan, BASELINE_PATH, SMARTCHARGE_PATH, RESULT_PATH):
    """
    Calculate financial measures over all users.
    
    Paramaters
    ---------- 
    userlist : list, users to be evaluted
    save_flag : boolean, flag indicating whether to save results
    model_type : str, model type to be evaluated (QRF)
    BASELINE_PATH : str, path of baseline results
    SMARTCHARGE_PATH : str, path of bidiectional smart charging results
    RESULT_PATH : str, path to save results
    
    Returns
    ----------
    N/A
    """
    
    cost = {'user_id':[],'total_days':[],'finan_base':[],'finan_smart':[],'finan_cost':[],'v2g_gain':[],'tech_base':[],'tech_smart':[]}
    
    print('Model type:', model_type)
    print('Quantile prediction of soc:', quan)
    print('------------------------------')
    
    for user in userlist:
        cost['user_id'].append(user)
        
        # read data
        baseline_path = BASELINE_PATH + '/cost/' + str(int(user)) + '_result.csv'
        baseline = pd.read_csv(baseline_path)
        smartcharge_path = SMARTCHARGE_PATH + '/cost/' + str(int(user)) + '_result.csv'
        smartcharge = pd.read_csv(smartcharge_path)
        
        base_dates = list(baseline['date'].unique()[:])
        smart_dates = list(smartcharge['date'].unique()[:])
        common_dates = list(set(base_dates).intersection(smart_dates))
        
        len(common_dates)
        baseline = baseline[baseline['date'].isin(common_dates)]
        smartcharge = smartcharge[smartcharge['date'].isin(common_dates)]
        cost['total_days'].append(len(common_dates))
        
        cost['finan_base'].append(baseline['money_cost'].sum())
        smart_cost = smartcharge['money_cost'].sum()
        v2g_gain = smartcharge['money_gain_v2g'].sum()
        finan_smart_sum = smart_cost - v2g_gain
        cost['finan_cost'].append(smart_cost)
        cost['v2g_gain'].append(v2g_gain)
        cost['finan_smart'].append(finan_smart_sum)
        
        cost['tech_base'].append(baseline['tech_cost'].sum())
        cost['tech_smart'].append(smartcharge['tech_cost'].sum())        
        
    cost = pd.DataFrame(cost)

    cost['finan_diff'] = cost['finan_base'] - cost['finan_smart']
    cost['tech_diff'] = cost['tech_base'] - cost['tech_smart']

    # average for each user over all days
    cost['finan_base_mean'] = cost['finan_base'] / cost['total_days']
    cost['finan_smart_mean'] = cost['finan_smart'] / cost['total_days']
    cost['tech_base_mean'] = cost['tech_base'] / cost['total_days']
    cost['tech_smart_mean'] = cost['tech_smart'] / cost['total_days']
    cost['finan_diff_mean'] = cost['finan_diff'] / cost['total_days']
    cost['tech_diff_mean'] = cost['tech_diff'] / cost['total_days']
        
    print('------------------------------')
    if save_flag == True:
        cost_folder = RESULT_PATH + '/cost_by_model' 
        if not os.path.exists(cost_folder):
            os.makedirs(cost_folder)
        cost_path = cost_folder + '/' + model_type + '_soc' + str(quan) + '_cost.csv'
        cost.to_csv(cost_path, index=False)
    
def evaluate_cost_model(model_type, mob_flags, soc_quan_list, save_flag, RESULT_PATH):
    """
    Evaluate financial costs between baseline and bidirectional smart charging。
    
    Paramaters
    ---------- 
    model_type : str, model type to be evaluated (QRF)
    mob_flags : boolean, flag indicating whether to include mobility features
    soc_quan_list : list, given quantiles of soc prediction
    save_flag : boolean, flag indicating whether to save results
    RESULT_PATH : str, path to save evaluation results
    
    Returns
    ----------
    N/A
    """

    cost_model = {'model':[],'finan_base_sum':[],'finan_smart_sum':[],
                  'smart_cost_sum':[],'smart_gain_sum':[],'finan_diff_sum':[],
                  'tech_base_sum':[],'tech_smart_sum':[],'tech_diff_sum':[],
                  'finan_base_mean_user':[],'finan_smart_mean_user':[],'tech_base_mean_user':[],
                  'tech_smart_mean_user':[],'finan_diff_mean_user':[],'tech_diff_mean_user':[]}
    
    for mob_flag in mob_flags:
        for quan in soc_quan_list:
            if mob_flag == False:
                model_type_name = model_type + '_soc' + str(quan)
            else:
                model_type_name = model_type + '_mob' + '_soc' + str(quan)
                
            cost_model['model'].append(model_type_name)
            cost_path = RESULT_PATH + '/cost_by_model/' + model_type_name + '_cost.csv'
            cost = pd.read_csv(cost_path)
                
            # calculate sum over all days and all users
            finan_base_sum = cost['finan_base'].sum()
            finan_smart_sum = cost['finan_smart'].sum()
            smart_cost_sum = cost['finan_cost'].sum()
            smart_gain_sum = cost['v2g_gain'].sum()
            finan_diff_sum = cost['finan_diff'].sum()
            tech_base_sum = cost['tech_base'].sum()
            tech_smart_sum = cost['tech_smart'].sum()
            tech_diff_sum = cost['tech_diff'].sum()               
            
            # calculate average over all days and all users
            total_users = len(cost)
            finan_base_mean_user = cost['finan_base_mean'].sum() / total_users
            finan_smart_mean_user = cost['finan_smart_mean'].sum() / total_users            
            tech_base_mean_user = cost['tech_base_mean'].sum() / total_users
            tech_smart_mean_user = cost['tech_smart_mean'].sum() / total_users
            finan_diff_mean_user = cost['finan_diff_mean'].sum() / total_users
            tech_diff_mean_user = cost['tech_diff_mean'].sum() / total_users
            
            cost_model['finan_base_sum'].append(finan_base_sum)
            cost_model['finan_smart_sum'].append(finan_smart_sum)
            cost_model['smart_cost_sum'].append(smart_cost_sum)
            cost_model['smart_gain_sum'].append(smart_gain_sum)
            cost_model['finan_diff_sum'].append(finan_diff_sum)
            cost_model['tech_base_sum'].append(tech_base_sum)
            cost_model['tech_smart_sum'].append(tech_smart_sum)
            cost_model['tech_diff_sum'].append(tech_diff_sum)
            
            cost_model['finan_base_mean_user'].append(finan_base_mean_user)
            cost_model['finan_smart_mean_user'].append(finan_smart_mean_user)
            cost_model['tech_base_mean_user'].append(tech_base_mean_user)
            cost_model['tech_smart_mean_user'].append(tech_smart_mean_user)
            cost_model['finan_diff_mean_user'].append(finan_diff_mean_user)
            cost_model['tech_diff_mean_user'].append(tech_diff_mean_user)

    # save results
    cost_model = pd.DataFrame(cost_model)
    
    if save_flag == True:
        if not os.path.exists(RESULT_PATH):
            os.makedirs(RESULT_PATH)
        cost_path = RESULT_PATH + '/' + 'cost_over_model.csv'
        cost_model.to_csv(cost_path, index=False)
            

def evaluate_peakshaving_way1(model_type, mob_flags, soc_quan_list, BASELINE_PATH, SMARTCHARGE_BASE_PATH, RESULT_PATH):
    """
    Evaluate technical effects between unidirectional smart charging and baseline,
    .. using on-peak definition 1: on-peak is during [8,20], off-peak is during [0-8] & [20,24],
    .. based on the documentation of day-ahead electricity market price
    
    Paramaters
    ---------- 
    model_type : str, model type to be evaluated (QRF)
    mob_flags : boolean, flag indicating whether to include mobility features
    soc_quan_list : list, given quantiles of soc prediction
    BASELINE_PATH : str, path of baseline results
    SMARTCHARGE_PATH : str, path of bidirectional smart charging results
    RESULT_PATH : str, path to save results
    
    Returns
    ----------
    N/A
    """

    peakshaving_method1 = {
        'model':[],'base_sum':[],'smart_sum':[],'base_onpeak_sum':[],'smart_onpeak_sum':[],
        'base_offpeak_sum':[],'smart_offpeak_sum':[],
        'base_onpeak_ratio':[],'smart_onpeak_ratio':[],
        'base_offpeak_ratio':[],'smart_offpeak_ratio':[],
        'base_smart_onpeak_diff':[]
        }
    
    for mob_flag in mob_flags:
        for quan in soc_quan_list:
            if mob_flag == False:
                model_type_name = model_type + '_soc' + str(quan)
            else:
                model_type_name = model_type + '_mob' + '_soc' + str(quan)
            print(model_type_name)
            
            peakshaving_method1['model'].append(model_type_name)
            base_charge_time_path = BASELINE_PATH + '/' + 'hourly_charge_profile.csv'
            base_charge_time = pd.read_csv(base_charge_time_path)  
        
            smart_charge_time_path = SMARTCHARGE_BASE_PATH + '/' + model_type_name +'_onpeakdef2/'+ 'hourly_charge_profile.csv'
            smart_charge_time = pd.read_csv(smart_charge_time_path)     
            
            onpeak_hrs = [str(n) for n in range(8,20)]
            offpeak_hrs = [str(n) for n in list(range(0,8)) + list(range(20,24))]
            
            base_onpeak = base_charge_time[onpeak_hrs]
            base_offpeak = base_charge_time[offpeak_hrs]
        
            smart_onpeak = smart_charge_time[onpeak_hrs]
            smart_offpeak = smart_charge_time[offpeak_hrs]
            
            # calculate relevant measures
            base_sum = base_charge_time.sum(axis=1)[0]
            base_onpeak_sum = base_onpeak.sum(axis=1)[0]
            base_offpeak_sum = base_offpeak.sum(axis=1)[0]
            base_onpeak_ratio = base_onpeak_sum / base_sum * 100
            base_offpeak_ratio = base_offpeak_sum / base_sum * 100
        
            smart_sum = smart_charge_time.sum(axis=1)[0]
            smart_onpeak_sum = smart_onpeak.sum(axis=1)[0]
            smart_offpeak_sum = smart_offpeak.sum(axis=1)[0]
            smart_onpeak_ratio = smart_onpeak_sum / smart_sum * 100
            smart_offpeak_ratio = smart_offpeak_sum / smart_sum * 100         

            base_smart_onpeak_diff = base_onpeak_ratio - smart_onpeak_ratio
            
            peakshaving_method1['base_sum'].append(base_sum)
            peakshaving_method1['smart_sum'].append(smart_sum)
            peakshaving_method1['base_onpeak_sum'].append(base_onpeak_sum)
            peakshaving_method1['smart_onpeak_sum'].append(smart_onpeak_sum)
            peakshaving_method1['base_offpeak_sum'].append(base_offpeak_sum)
            peakshaving_method1['smart_offpeak_sum'].append(smart_offpeak_sum)
            peakshaving_method1['base_onpeak_ratio'].append(base_onpeak_ratio)
            peakshaving_method1['smart_onpeak_ratio'].append(smart_onpeak_ratio)
            peakshaving_method1['base_offpeak_ratio'].append(base_offpeak_ratio)
            peakshaving_method1['smart_offpeak_ratio'].append(smart_offpeak_ratio)   
            peakshaving_method1['base_smart_onpeak_diff'].append(base_smart_onpeak_diff)
            
            # plot the line graph of respective baseline and smart charging load profiles 
            base_charge_time['24'] = base_charge_time['0']
            smart_charge_time['24'] = smart_charge_time['0']
               
            plt.figure(figsize=(8, 4), dpi=80)
            plt.plot(base_charge_time.loc[0], '-b', label='EV Uncontrolled Charging')
            plt.plot(smart_charge_time.loc[0], '-r', label='EV Bidirectional Smart Charging')
            plt.xlabel("Hour")
            plt.ylabel("EV Electricity Load (kWh)")
            plt.legend(loc="upper right")
            plt.grid()
            plt.margins(0) # remove default margins (matplotlib verision 2+)
            axes = plt.gca()
            y_min, y_max = axes.get_ylim()
            
            # set background color for on-peak and off-peak periods
            plt.axhspan(ymin=0, ymax=15000, xmin=8/24, xmax=20/24, facecolor='red', alpha=0.10, label='On-peak')
            plt.axhspan(ymin=0, ymax=15000, xmin=0/24, xmax=8/24, facecolor='green', alpha=0.10, label='Off-peak')
            plt.axhspan(ymin=0, ymax=15000, xmin=20/24, xmax=24/24, facecolor='green', alpha=0.10, label='Off-peak')
            
            fig_folder = RESULT_PATH+'/peakshaving_method1'
            if not os.path.exists(fig_folder):
                os.makedirs(fig_folder)
            fig_name = fig_folder+'/'+model_type_name+'_peakshaving_medthod1.png'
            plt.savefig(fig_name, dpi=300)
            
            # plt.show()
        
    # save plots
    peakshaving_method1 = pd.DataFrame(peakshaving_method1)
    peakshaving_method1_path = RESULT_PATH + '/peakshaving_method1/' + 'peakshaving_method1_over_model.csv'
    peakshaving_method1.to_csv(peakshaving_method1_path, index=False)
        

def evaluate_peakshaving_way2(model_type, mob_flags, soc_quan_list, LOADPROFILE_PATH, BASELINE_PATH, SMARTCHARGE_PATH, RESULT_PATH):
    """
    Evaluate technical effects between unidirectional smart charging and baseline,
    .. using on-peak definition 2: on-peak is the period whose household loads are above 75th percentile.
    
    Paramaters
    ---------- 
    model_type : str, model type to be evaluated (QRF)
    mob_flags : boolean, flag indicating whether to include mobility features
    soc_quan_list : list, given quantiles of soc prediction
    LOADPROFILE_PATH : str, path of load profile data
    BASELINE_PATH : str, path of baseline results
    SMARTCHARGE_PATH : str, path of bidirectional smart charging results
    RESULT_PATH : str, path to save evaluation results
    
    Returns
    ----------
    N/A
    """

    peakshaving_method2 = {
        'model':[],'house_sum':[],'base_sum':[],'smart_sum':[],
        'base_onpeak_sum':[],'smart_onpeak_sum':[],
        'base_offpeak_sum':[],'smart_offpeak_sum':[],
        'base_onpeak_house_ratio':[],'smart_onpeak_house_ratio':[],
        'base_offpeak_house_ratio':[],'smart_offpeak_house_ratio':[],
        'base_smart_onpeak_diff':[]}
    
    for mob_flag in mob_flags:
        for quan in soc_quan_list:
            if mob_flag == False:
                model_type_name = model_type + '_soc' + str(quan)
            else:
                model_type_name = model_type + '_mob' + '_soc' + str(quan)
            print(model_type_name)
            
            peakshaving_method2['model'].append(model_type_name)
            
            house_load = pd.read_csv(LOADPROFILE_PATH)
    
            # calculate load by hour of one day
            house_load_hour = {'load':[]}
            for i in range(0,len(house_load),4):
                # print(i)
                load_mean_hour = house_load.loc[i:(i+3),'H00 [kWh]'].sum()
                house_load_hour['load'].append(load_mean_hour)
            house_load_hour = pd.DataFrame(house_load_hour) 
            
            # define peak load that is over 75th percentile
            load_75per = np.percentile(house_load_hour['load'], 75)
            # house_load_hour.describe()
            house_onpeak = house_load_hour[house_load_hour['load']>=load_75per]
            onpeak_hrs = [n for n in list(house_onpeak.index)]
            offpeak_hrs = list(set(range(0,24)) - set(onpeak_hrs))
            print('On-peak hours',onpeak_hrs)
            
            # read load profile of baseline and smart charging strategy
            base_charge_time_path = BASELINE_PATH + '/' + 'hourly_charge_profile.csv'
            base_charge_time = pd.read_csv(base_charge_time_path)  
        
            smart_charge_time_path = SMARTCHARGE_PATH + '/' + model_type_name +'_onpeakdef2/' +'hourly_charge_profile.csv'
            smart_charge_time = pd.read_csv(smart_charge_time_path)     
            
            # calculate daily load profile of one user by dividing by total sum of days (sum of all days for all users)
            cost_path = RESULT_PATH + '/cost_by_model/qrf_mob_soc0.6_cost.csv'
            cost = pd.read_csv(cost_path)
            day_sum = cost['total_days'].sum() # calculate sum of days of all users
            user_sum = len(cost)
            
            # correct the size of household of household load profile by the number of ev uer
            ratio_house_ev = user_sum / 247
            house_load_hour['load'] = house_load_hour['load'] * ratio_house_ev

            # calculate daily load profile for all users by baseline
            base_charge_time_t = base_charge_time.T
            base_charge_time_t.index = range(0,24)
            base_charge_time_t.columns = ['sum']
            base_charge_time_t['mean_perday_peruser'] = base_charge_time_t['sum'] / day_sum # calculate daily mean for each user
            base_charge_time_t['mean_perday_alluser'] = base_charge_time_t['mean_perday_peruser'] * user_sum # calculate daily mean for all users
            
            # repeat above calculation: calculate daily load profile for all users by smart charging
            smart_charge_time_t = smart_charge_time.T
            smart_charge_time_t.index = range(0,24)
            smart_charge_time_t.columns = ['sum']
            smart_charge_time_t['mean_perday_peruser'] = smart_charge_time_t['sum'] / day_sum
            smart_charge_time_t['mean_perday_alluser'] = smart_charge_time_t['mean_perday_peruser'] * user_sum
                        
            # find load on on-peak off-peak hours for household, baseline, and smart charging data            
            house_onpeak = house_load_hour.loc[onpeak_hrs]
            base_onpeak = base_charge_time_t.loc[onpeak_hrs,'mean_perday_alluser']
            smart_onpeak = smart_charge_time_t.loc[onpeak_hrs,'mean_perday_alluser']
            house_offpeak = house_load_hour.loc[offpeak_hrs]
            base_offpeak = base_charge_time_t.loc[offpeak_hrs,'mean_perday_alluser']
            smart_offpeak = smart_charge_time_t.loc[offpeak_hrs,'mean_perday_alluser']
            
            # calculate relevant measures
            house_sum = house_load_hour['load'].sum()
            base_sum = base_charge_time_t['mean_perday_alluser'].sum()
            smart_sum = smart_charge_time_t['mean_perday_alluser'].sum()
            
            house_onpeak_sum = house_onpeak.sum()[0]
            base_onpeak_sum = base_onpeak.sum()
            smart_onpeak_sum = smart_onpeak.sum()
            
            house_offpeak_sum = house_offpeak.sum()[0]
            base_offpeak_sum = base_offpeak.sum()
            smart_offpeak_sum = smart_offpeak.sum()
            
            base_onpeak_house_ratio = base_onpeak_sum / (house_onpeak_sum + base_onpeak_sum) * 100
            smart_onpeak_house_ratio = smart_onpeak_sum / (house_onpeak_sum + smart_onpeak_sum) * 100
            base_offpeak_house_ratio = base_offpeak_sum / (house_offpeak_sum + base_offpeak_sum) * 100
            smart_offpeak_house_ratio = smart_offpeak_sum / (house_offpeak_sum + smart_offpeak_sum) * 100
            
            base_smart_onpeak_diff = base_onpeak_house_ratio - smart_onpeak_house_ratio
            
            peakshaving_method2['house_sum'].append(house_sum)
            peakshaving_method2['base_sum'].append(base_sum)
            peakshaving_method2['smart_sum'].append(smart_sum)
            peakshaving_method2['base_onpeak_sum'].append(base_onpeak_sum)
            peakshaving_method2['smart_onpeak_sum'].append(smart_onpeak_sum)
            peakshaving_method2['base_offpeak_sum'].append(base_offpeak_sum)
            peakshaving_method2['smart_offpeak_sum'].append(smart_offpeak_sum)
            peakshaving_method2['base_onpeak_house_ratio'].append(base_onpeak_house_ratio)
            peakshaving_method2['smart_onpeak_house_ratio'].append(smart_onpeak_house_ratio)
            peakshaving_method2['base_offpeak_house_ratio'].append(base_offpeak_house_ratio)
            peakshaving_method2['smart_offpeak_house_ratio'].append(smart_offpeak_house_ratio) 
            peakshaving_method2['base_smart_onpeak_diff'].append(base_smart_onpeak_diff) 
                        
            # plot the line graph of respective baseline and smart charging load profiles 
            # add load of e-car to basic household load profile
            base_load_hour = house_load_hour['load'] + base_charge_time_t['mean_perday_alluser']
            smart_load_hour = house_load_hour['load'] + smart_charge_time_t['mean_perday_alluser']

            hrs_24h = [str(n) for n in range(0,25)]
            
            house_load_hour.loc[24] = house_load_hour.loc[0]
            base_load_hour.loc[24] = base_load_hour.loc[0]
            base_load_hour = pd.DataFrame(base_load_hour)
            smart_load_hour.loc[24] = smart_load_hour.loc[0]
            smart_load_hour = pd.DataFrame(smart_load_hour)
            
            house_load_hour_t = house_load_hour.T
            house_load_hour_t.index = range(0,1)
            house_load_hour_t.columns = hrs_24h
            base_load_hour_t = base_load_hour.T
            base_load_hour_t.columns = hrs_24h
            smart_load_hour_t = smart_load_hour.T
            smart_load_hour_t.columns = hrs_24h
            
            plt.figure(figsize=(8, 4), dpi=80)
            plt.plot(house_load_hour_t.loc[0], '-ok', label='Household')
            plt.plot(base_load_hour_t.loc[0], '-b', label='Household + EV Uncontrolled Charging')
            plt.plot(smart_load_hour_t.loc[0], '-r', label='Household + EV Bidirectional Smart Charging')
            plt.xlabel("Hour")
            plt.ylabel("Household & E-car Electricity Load (kWh)")
            plt.legend(loc="lower right")
            plt.grid()
            plt.margins(0) # remove default margins (matplotlib verision 2+)
            axes = plt.gca()
            y_min, y_max = axes.get_ylim()
            x_max = max(list(house_onpeak.index))
            
            # set background color for on-peak and off-peak periods
            plt.axhspan(ymin=0, ymax=y_max*1.1, xmin=12/24, xmax=14/24, facecolor='red', alpha=0.10, label='On-peak')
            plt.axhspan(ymin=0, ymax=y_max*1.1, xmin=19/24, xmax=23/24, facecolor='red', alpha=0.10, label='On-peak')
            plt.axhspan(ymin=0, ymax=y_max*1.1, xmin=0/24, xmax=12/24, facecolor='green', alpha=0.10, label='Off-peak')
            plt.axhspan(ymin=0, ymax=y_max*1.1, xmin=14/24, xmax=19/24, facecolor='green', alpha=0.10, label='Off-peak')
            plt.axhspan(ymin=0, ymax=y_max*1.1, xmin=23/24, xmax=24/24, facecolor='green', alpha=0.10, label='Off-peak')
            
            fig_folder = RESULT_PATH+'/peakshaving_method2'
            if not os.path.exists(fig_folder):
                os.makedirs(fig_folder)
            fig_name = fig_folder+'/'+model_type_name+'_peakshaving_medthod2.png'
            
            plt.savefig(fig_name, dpi=300)
            
            # plt.show()
 
    # save plots
    peakshaving_method2 = pd.DataFrame(peakshaving_method2)
    peakshaving_method2_path = RESULT_PATH + '/peakshaving_method2/' + 'peakshaving_method2_over_model.csv'
    peakshaving_method2.to_csv(peakshaving_method2_path, index=False)



