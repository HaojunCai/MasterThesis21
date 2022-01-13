# -*- coding: utf-8 -*-
"""
Created on Jul 2021
@author: Haojun Cai  
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def compare_three_charging(model_type, mob_flags, soc_quan_list, LOADPROFILE_PATH, BASELINE_PATH, UNISMARTCHARGE_BASE_PATH, SMARTCHARGE_PATH, RESULT_PATH):
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
            print('On-peak hours',offpeak_hrs)
                            
            # calculate daily load profile of one user by dividing by total sum of days (sum of all days for all users)
            cost_path = RESULT_PATH + '/baseline_bidirectional/cost_by_model/' + model_type_name + '_cost.csv'
            cost = pd.read_csv(cost_path)
            day_sum = cost['total_days'].sum() # calculate sum of days of all users
            user_sum = len(cost)
            
            # correct the size of household of household load profile by the number of ev uer
            ratio_house_ev = user_sum / 247
            house_load_hour['load'] = house_load_hour['load'] * ratio_house_ev
            print('On-peak hours',house_load_hour.loc[onpeak_hrs])
                        
            # read load profile of baseline and smart charging strategy
            base_charge_time_path = BASELINE_PATH + '/' + 'hourly_charge_profile.csv'
            base_charge_time = pd.read_csv(base_charge_time_path)  

            # calculate daily load profile for all users by baseline
            base_charge_time_t = base_charge_time.T
            base_charge_time_t.index = range(0,24)
            base_charge_time_t.columns = ['sum']
            base_charge_time_t['mean_perday_peruser'] = base_charge_time_t['sum'] / day_sum # calculate daily mean for each user
            base_charge_time_t['mean_perday_alluser'] = base_charge_time_t['mean_perday_peruser'] * user_sum # calculate daily mean for all users
            
            unismart_charge_time_path = UNISMARTCHARGE_BASE_PATH + '/' + model_type_name + '/hourly_charge_profile.csv'
            unismart_charge_time = pd.read_csv(unismart_charge_time_path) 
            
            # repeat above calculation: calculate daily load profile for all users by unismart charging
            unismart_charge_time_t = unismart_charge_time.T
            unismart_charge_time_t.index = range(0,24)
            unismart_charge_time_t.columns = ['sum']
            unismart_charge_time_t['mean_perday_peruser'] = unismart_charge_time_t['sum'] / day_sum
            unismart_charge_time_t['mean_perday_alluser'] = unismart_charge_time_t['mean_perday_peruser'] * user_sum
                        
            # find load on on-peak off-peak hours for household, baseline, and unismart charging data            
            house_onpeak = house_load_hour.loc[onpeak_hrs]
            base_onpeak = base_charge_time_t.loc[onpeak_hrs,'mean_perday_alluser']
            unismart_onpeak = unismart_charge_time_t.loc[onpeak_hrs,'mean_perday_alluser']
            house_offpeak = house_load_hour.loc[offpeak_hrs]
            base_offpeak = base_charge_time_t.loc[offpeak_hrs,'mean_perday_alluser']
            unismart_offpeak = unismart_charge_time_t.loc[offpeak_hrs,'mean_perday_alluser']
            
            # output statistics
            house_sum = house_load_hour['load'].sum()
            base_sum = base_charge_time_t['mean_perday_alluser'].sum()
            unismart_sum = unismart_charge_time_t['mean_perday_alluser'].sum()
            
            house_onpeak_sum = house_onpeak.sum()[0]
            base_onpeak_sum = base_onpeak.sum()
            unismart_onpeak_sum = unismart_onpeak.sum()
            
            house_offpeak_sum = house_offpeak.sum()[0]
            base_offpeak_sum = base_offpeak.sum()
            unismart_offpeak_sum = unismart_offpeak.sum()
            
            base_onpeak_house_ratio = base_onpeak_sum / (house_onpeak_sum + base_onpeak_sum) * 100
            unismart_onpeak_house_ratio = unismart_onpeak_sum / (house_onpeak_sum + unismart_onpeak_sum) * 100
            base_offpeak_house_ratio = base_offpeak_sum / (house_offpeak_sum + base_offpeak_sum) * 100
            unismart_offpeak_house_ratio = unismart_offpeak_sum / (house_offpeak_sum + unismart_offpeak_sum) * 100
            
            base_unismart_onpeak_diff = base_onpeak_house_ratio - unismart_onpeak_house_ratio
            
            # plot the line graph of respective baseline and unismart charging load profiles 
            # add load of e-car to basic household load profile
            base_load_hour = house_load_hour['load'] + base_charge_time_t['mean_perday_alluser']
            unismart_load_hour = house_load_hour['load'] + unismart_charge_time_t['mean_perday_alluser']

            hrs_24h = [str(n) for n in range(0,25)]
            house_load_hour.loc[24] = house_load_hour.loc[0]
            base_load_hour.loc[24] = base_load_hour.loc[0]
            base_load_hour = pd.DataFrame(base_load_hour)
            unismart_load_hour.loc[24] = unismart_load_hour.loc[0]
            unismart_load_hour = pd.DataFrame(unismart_load_hour)
            
            house_load_hour_t = house_load_hour.T
            house_load_hour_t.index = range(0,1)
            house_load_hour_t.columns = hrs_24h
            base_load_hour_t = base_load_hour.T
            base_load_hour_t.columns = hrs_24h
            unismart_load_hour_t = unismart_load_hour.T
            unismart_load_hour_t.columns = hrs_24h

            smart_charge_time_path = SMARTCHARGE_PATH + '/' + model_type_name +'_onpeakdef2/' +'hourly_charge_profile.csv'
            smart_charge_time = pd.read_csv(smart_charge_time_path)     
            
            # repeat above calculation: calculate daily load profile for all users by smart charging
            smart_charge_time_t = smart_charge_time.T
            smart_charge_time_t.index = range(0,24)
            smart_charge_time_t.columns = ['sum']
            smart_charge_time_t['mean_perday_peruser'] = smart_charge_time_t['sum'] / day_sum
            smart_charge_time_t['mean_perday_alluser'] = smart_charge_time_t['mean_perday_peruser'] * user_sum
                        
            # find load on on-peak off-peak hours for household, baseline, and smart charging data            
            smart_onpeak = smart_charge_time_t.loc[onpeak_hrs,'mean_perday_alluser']
            smart_offpeak = smart_charge_time_t.loc[offpeak_hrs,'mean_perday_alluser']
            
            # plot the line graph of respective baseline and smart charging load profiles 
            # add load of e-car to basic household load profile
            smart_load_hour = house_load_hour['load'] + smart_charge_time_t['mean_perday_alluser']

            hrs_24h = [str(n) for n in range(0,25)]
            smart_load_hour.loc[24] = smart_load_hour.loc[0]
            smart_load_hour = pd.DataFrame(smart_load_hour)
            smart_load_hour_t = smart_load_hour.T
            smart_load_hour_t.columns = hrs_24h
            
            plt.figure(figsize=(8, 4), dpi=80)
            plt.plot(house_load_hour_t.loc[0], '-.k', label='Household')
            plt.plot(base_load_hour_t.loc[0], '-b', label='Household + EV Uncontrolled Charging')            
            plt.plot(unismart_load_hour_t.loc[0], '-Xg', label='Household + EV Unidirectional Smart Charging')
            plt.plot(smart_load_hour_t.loc[0], '-r', label='Household + EV Bidirectional Smart Charging')
            plt.hlines(56.262398, 0, 24, colors=None, linestyles='dashed', label='On-peak Division')
            plt.xlabel("Hour")
            plt.ylabel("Household Electricity Load (kWh)")
            plt.legend(loc="lower right")
            plt.grid()
            plt.margins(0) # remove default margins (matplotlib verision 2+)
        
            axes = plt.gca()
            y_min, y_max = axes.get_ylim()
            x_max = max(list(house_onpeak.index))
            # print(y_max)
            # y_max = 64.85867766732792
            y_max = 70.96108932405713
            # set background color for on-peak and off-peak periods
            plt.axhspan(ymin=0, ymax=y_max*1.1, xmin=12/24, xmax=14/24, facecolor='red', alpha=0.10, label='On-peak')
            plt.axhspan(ymin=0, ymax=y_max*1.1, xmin=19/24, xmax=24/24, facecolor='red', alpha=0.10, label='On-peak')
            plt.axhspan(ymin=0, ymax=y_max*1.1, xmin=0/24, xmax=12/24, facecolor='green', alpha=0.10, label='Off-peak')
            plt.axhspan(ymin=0, ymax=y_max*1.1, xmin=14/24, xmax=19/24, facecolor='green', alpha=0.10, label='Off-peak')
            
            fig_folder = RESULT_PATH
            if not os.path.exists(fig_folder):
                os.makedirs(fig_folder)
            fig_name = fig_folder+'/'+model_type_name+'_comparison.png'
            
            plt.savefig(fig_name, dpi=300)
            
            plt.show()

