# -*- coding: utf-8 -*-
"""
Created on Jun 2021
@author: Haojun Cai
""" 

import pandas as pd
import numpy as np
import math
import os
import datetime

def charge_price_first(charge_hr, price, price_avail, charge_time, charge_time_avail, finan_cost):
    """
    Simulate charging processes after user arrives home to record charging periods and corresponding charging prices.
    
    Paramaters
    ----------
    charge_hr : float, needed charging time 
    price : dict, price sorted by hour after user arrives home
    price_avail : dict, prices during the period that is available for charging by hour
    charge_time : dict, initialzied charged time by hour
    charge_time_avail : dict, available charged time by hour
    finan_cost : float, initialzied financial cost
    
    Returns
    ----------
    finan_cost : float, total financial costs
    charge_time : dict, actual charged time by hour
    charge_time_avail : dict, available charged time by hour
    """
    
    for price_first in price:
        # find date for price_first
        for key, value in price_avail.items():
            if price_first in value:
                price_first_hr = key
                break
        
        # check if there are left available charging time
        if charge_time_avail[price_first_hr] >= charge_hr:
            charge_energy_temp = charge_hr * 11
            finan_cost += charge_energy_temp/1000 * price_first # price data is in the unit of EUR/mWh
            charge_time[price_first_hr] += charge_hr
            charge_time_avail[price_first_hr] += -charge_hr
            break
        
        # in case that parking duration within price_first hour is not enough for charging
        if charge_time_avail[price_first_hr] < charge_hr and charge_time_avail[price_first_hr] != 0: 
            charge_hr_part = charge_time_avail[price_first_hr]
            charge_energy_part = charge_hr_part * 11
            finan_cost += charge_energy_part/1000 * price_first
            charge_time[price_first_hr] += charge_hr_part
            charge_time_avail[price_first_hr] += -charge_hr_part
            charge_hr += -charge_hr_part
    
    return finan_cost, charge_time, charge_time_avail

def calculate_finan_cost(charge_hrs_ceil, charge_hrs, price, price_avail, charge_time, charge_time_avail, finan_cost):
    """
    Calculate total prices.
    
    Paramaters
    ----------
    charge_hrs_ceil : float, ceiled total needed charging hours 
    charge_hrs : float, total needed charging hours 
    price_sort : dict, price sorted from lowest to highest by hour
    price_avail : dict, prices during the period that is available for charging by hour
    charge_time : dict, initialzied actual charged time by hour
    charge_time_avail : dict, available charged time by hour
    finan_cost : float, initialzied financial cost
    
    Returns
    ----------
    finan_cost : float, total financial costs
    charge_time : dict, actual charged time by hour
    charge_time_avail : dict, available charged time by hour
    """
    
    ## caculate financial cost: in the unit of EUR
    if charge_hrs_ceil <=1 and charge_hrs_ceil!=0:
        [finan_cost, charge_time, charge_time_avail] = charge_price_first(charge_hrs, price, price_avail, charge_time, charge_time_avail, finan_cost)
        
    elif charge_hrs_ceil > 1:
        [charge_hrs_frac, charge_hrs_whole] = math.modf(charge_hrs)
        # calculate for the whole part of charging hours
        for hour in range(0,int(charge_hrs_whole)):
            charge_hr = 1
            [finan_cost, charge_time, charge_time_avail] = charge_price_first(charge_hr, price, price_avail, charge_time, charge_time_avail, finan_cost)                                                       
        # calculate for the fraction part of charging hours
        charge_hr = charge_hrs_frac
        [finan_cost, charge_time, charge_time_avail] = charge_price_first(charge_hr, price, price_avail, charge_time, charge_time_avail, finan_cost)
    
    # when the parking duration is 0, the car cannot be charged, thus no energy was traded from the market
    elif charge_hrs_ceil == 0:
        finan_cost = 0
    
    else:
        print('Error: charge_hrs_int CANNOT be negative.')
             
    return finan_cost, charge_time, charge_time_avail

def evaluate_baseline(price_data, userlist, save_flag, SMARTCHARGE_PATH, ARRIVAL_PATH, DEPART_PATH, SOC_PATH, RESULT_PATH):
    """
    Simluate the uncontrolled charging.
    
    Paramaters
    ---------- 
    price_data : dataframe, price data over all days
    userlist : list, users to be evaluted
    save_flag : boolean, flag indicating whether to save results
    ARRIVAL_PATH : str, path of true values of arrival time
    DEPART_PATH : str, path of true values of departure time
    SOC_PATH : str, path of true values of soc
    RESULT_PATH : str, path to save uncotrolled charging results
    
    Returns
    ----------
    N/A
    """

    # create ditionary to store charging electricity by hour
    charge_time = {}
    time_24hr = [str(n) for n in range(0,24)]
    for time in time_24hr:
        charge_time[time] = 0
        
    for user in userlist:
        print(user)
        print('-------------START-----------------')
        
        # read data       
        smartcharge_path = SMARTCHARGE_PATH + '/qrf_mob_soc0.5/cost/' + str(int(user)) + '_result.csv'
        smartcharge_cost = pd.read_csv(smartcharge_path)
        arrival_path = ARRIVAL_PATH + 'prediction/qrf/' + str(int(user)) + '_result.csv'
        arrival_pred = pd.read_csv(arrival_path)
        depart_path = DEPART_PATH + 'prediction/qrf/' + str(int(user)) + '_result.csv'
        depart_pred = pd.read_csv(depart_path)
        soc_path = SOC_PATH + 'prediction/qrf/' + str(int(user)) + '_result.csv'
        soc_pred = pd.read_csv(soc_path)

        # filter out daily soc consumption that is over 100
        soc_pred = soc_pred[soc_pred['true']<=100]

        smartcharge_dates = list(smartcharge_cost['date'].unique()[:])
        arr_dates = list(arrival_pred['date'].unique()[:])
        dep_dates = list(depart_pred['date'].unique()[:])
        delta = datetime.timedelta(days=1)
        
        soc_end_prev = 0 # on day 1, starting soc is set as 0
        cost_user = {'date':[], 'money_cost':[], 'tech_cost':[]}
        soc_user = {'date':[], 'soc_start':[], 'soc_end':[], 'soc_charge':[]}
        dep_true_date = -99
        arr_true_date = -99        
        
        # for soc on certain day
        for date in smartcharge_dates: 

            parking_flag = False
            date_prev = str((pd.to_datetime(date)-delta).date())
            
            # extract soc on that day
            soc_true_date = list(soc_pred.loc[soc_pred['date']==date,'true'])[0]           
            
            # extract departure on the same day
            if date in dep_dates:
                dep_true_date = list(depart_pred.loc[depart_pred['date']==date,'true'])[0]          
            
            # extract arrival on the last day
            if date_prev in arr_dates:
                arr_true_date = list(arrival_pred.loc[arrival_pred['date']==date_prev,'true'])[0]
            
            parking_duration = np.floor(dep_true_date) + 24 - np.ceil(arr_true_date) 
            invalid_values = [-99, -1]
            
            # check if there are valid values for arrival and departure time
            # also filter out dates which do not have valid parking duration true values
            if arr_true_date not in invalid_values and dep_true_date not in invalid_values and parking_duration>0:
                parking_flag = True
                
            if parking_flag == True:
                cost_user['date'].append(date)
                soc_user['date'].append(date)
                
                # find prices following arrival time during available charging slots
                # read price data on that day and on previous day
                price_prev_date = price_data.loc[price_data['date']==date_prev]
                price_prev_date.index = range(0,len(price_prev_date))
                price_date = price_data.loc[price_data['date']==date]
                price_date.index = range(0,len(price_date))
                
                # correct invalid prediction for parking duration
                if arr_true_date > 24 and dep_true_date < 0:
                    print(user, date, ' Invalid prediction')
                if arr_true_date > 24:
                    arr_true_date = 24
                if dep_true_date < 0:
                    dep_true_date = 0
                                  
                # find prices for available charging slots and prices from arrival time (previous day) to depature time
                charge_time_avail = {}
                time_24hr = [str(n) for n in range(0,24)]
                for time in time_24hr:
                    charge_time_avail[time] = 0
                
                price_avail = {}
                
                period_prev_date = list(range(int(np.floor(arr_true_date)),24))
                period_prev_date = [str(hour) for hour in period_prev_date]
                for time in period_prev_date:
                    price_avail[time] = [price_prev_date.loc[0,time]]
                    if time != period_prev_date[0]:
                        charge_time_avail[time] += 1
                    else:
                        charge_time_avail[time] += 1 - (arr_true_date%1)
                
                period_date = list(range(0,int(np.ceil(dep_true_date))))
                period_date = [str(hour) for hour in period_date]
                for time in period_date:
                    if time in list(price_avail.keys()):
                        price_avail[time].append(price_date.loc[0,time])
                    else:
                        price_avail[time] = [price_date.loc[0,time]]
                    if time != period_date[-1]:
                        charge_time_avail[time] += 1
                    else:
                        charge_time_avail[time] += (dep_true_date%1)
                
                price_prev_date = list(price_prev_date.loc[0,period_prev_date])
                price_date = list(price_date.loc[0,period_date])
                price = price_prev_date + price_date                
                
                # calculate charging hours needed 
                soc_charge = 100 - soc_end_prev
                charge_energy = 27.2 * (soc_charge/100) # in the unit of kWh
                charge_hrs = charge_energy / 11 # 11kW is the rated power of charging voltbox installed at home
                charge_hrs_ceil = int(np.ceil(charge_hrs))
                
                # for the case that parking duration is not enough to charge to 100, charge as much as possible
                if parking_duration < charge_hrs:
                    charge_hrs = parking_duration
                    charge_energy = charge_hrs * 11
                    soc_charge = charge_energy / 27.2 * 100
                    charge_hrs_ceil = int(np.ceil(charge_hrs))
                                                                       
                ## Step 1: calculate soc_start on that day to always 100
                # on day 1, starting soc is set as soc lower threshold
                soc_start = soc_end_prev + soc_charge
                soc_user['soc_charge'].append(soc_charge)
                soc_user['soc_start'].append(soc_start)
                
                # calculate financial cost: in the unit of Euro
                finan_cost = 0
                [finan_cost, charge_time, charge_time_avail] = calculate_finan_cost(charge_hrs_ceil, charge_hrs, price, price_avail, charge_time, charge_time_avail, finan_cost)          
              
                # calculate technical cost: in the unit of kWh
                tech_cost = charge_energy
                
                ## Step 2: calculate soc_end
                soc_end = soc_start - soc_true_date
                
                # check soc_end range
                if soc_end < 0:
                    print('Warning: soc_end is BELOW 0 due to limited parking duration.')
                    soc_end = 0
                if soc_end > 100:
                    print('Error: soc_end is ABOVE 100.')
                
                soc_user['soc_end'].append(soc_end)
                soc_end_prev = soc_end
                cost_user['money_cost'].append(finan_cost)
                cost_user['tech_cost'].append(tech_cost)
        
        # save results 
        cost_user = pd.DataFrame(cost_user)
        soc_user = pd.DataFrame(soc_user)
        
        if save_flag == True:
            cost_res_folder = RESULT_PATH + '/cost'
            if not os.path.exists(cost_res_folder):
                os.makedirs(cost_res_folder)
            cost_res_path = cost_res_folder + '/' + str(int(user)) + '_result.csv'
            cost_user.to_csv(cost_res_path, index=False)        
            
            soc_res_folder = RESULT_PATH + '/soc_state'
            if not os.path.exists(soc_res_folder):
                os.makedirs(soc_res_folder)            
            soc_res_path = soc_res_folder + '/' + str(int(user)) + '_result.csv'
            soc_user.to_csv(soc_res_path, index=False)  

    charge_time = pd.DataFrame(charge_time, index=[1])
    
    if save_flag == True:
        charge_time_folder = RESULT_PATH
        if not os.path.exists(charge_time_folder):
            os.makedirs(charge_time_folder)
        chagre_time_path = charge_time_folder + '/' + 'hourly_charge_profile.csv'
        charge_time.to_csv(chagre_time_path, index=False) 
                
