# -*- coding: utf-8 -*-
"""
Created on Jun 2021
@author: Haojun Cai
""" 

import os
import numpy as np
import pandas as pd
import datetime

def preprocess_bmw(bmwdf):
    """
    Preprocess staypoints data in preparation to extract mobility features
    
    Paramaters
    ----------
    bmwdf : dataframe, bmw data
    
    Returns
    ----------
    bmwdf_negsoc: dataframe, preprocessed bmw data
    """
    
    # filter out incomplete data without valid user id
    bmwdf = bmwdf[bmwdf['user_id'].notnull()]
    
    # modify the one wrong labels of zustand
    bmwdf['zustand'].value_counts()
    bmwdf.loc[bmwdf['zustand']=='fah,t','zustand'] = 'fahrt'

    # calculate useful attributes
    bmwdf['soc_diff'] = bmwdf['soc_customer_end'] - bmwdf['soc_customer_start']
    bmwdf = bmwdf.sort_values(by='timestamp_start_utc', ascending=True)
    bmwdf['a_temp_mean'] = (bmwdf['a_temp_start']+bmwdf['a_temp_end'])/2
    bmwdf['duration'] = bmwdf['timestamp_end_utc'] - bmwdf['timestamp_start_utc']
    bmwdf['timestamp_utc_mean'] = (bmwdf['timestamp_end_utc']-bmwdf['timestamp_start_utc'])/2 + bmwdf['timestamp_start_utc']
    bmwdf['start_ymd'] = pd.to_datetime(bmwdf['timestamp_start_utc']).dt.date
    bmwdf['end_ymd'] = pd.to_datetime(bmwdf['timestamp_end_utc']).dt.date
    bmwdf['day_diff'] = bmwdf['end_ymd'] - bmwdf['start_ymd']

    # only keep items with soc consumption
    bmwdf_negsoc = bmwdf[bmwdf['soc_diff']<0].copy()
    
    return bmwdf_negsoc

def extract_soc_target(userlist, bmwdf_negsoc, saveflag, PREPROCESS_PATH):
    """
    Extract daily soc relevant features.
    
    Paramaters
    ----------
    userlist : list, userlist to extract daily mobility features
    bmwdf_negsoc : dataframe, preprocessed bmw data
    saveflag : boolean, flag to indicate if save results
    PREPROCESS_PATH : str, path to save results
    
    Returns
    ----------
    soc_above100: dataframe, statistics with daily soc over 100
    """
    
    soc_above100 = {'user_id':[], 'sum':[]}
    
    # iterate over all users
    for user in userlist:
        print(user)
        
        above100_sum = 0
        soc_above100['user_id'].append(user)
        bmwdf_negsoc_user = bmwdf_negsoc[bmwdf_negsoc['user_id']==user]
        
        date = list(set(bmwdf_negsoc_user['start_ymd']))
        
        start_date = min(date)
        end_date = max(date)
        delta = datetime.timedelta(days=1)
        
        soc_user = {'start_date':[],'soc':[],'out_temp':[],'day_of_week':[],'weekend_flag':[],'first_time_of_day':[],'last_time_of_day':[],'mean_time_of_day':[],'day_of_year':[]}
        
        # iterate over all days
        while start_date <= end_date:
            
            bmwdf_negsoc_user_date = bmwdf_negsoc_user[bmwdf_negsoc_user['start_ymd']==start_date]
            bmwdf_negsoc_user_date.index = range(0,len(bmwdf_negsoc_user_date))
            exist_flag = len(bmwdf_negsoc_user_date)
            
            if exist_flag != 0:
                soc_sum_date = bmwdf_negsoc_user_date['soc_diff'].sum()
                
                soc_sum_date = 0 - soc_sum_date
                if soc_sum_date > 100:
                    print('SOC is above 100. Assert it to 100.')
                    above100_sum += 1
                    soc_sum_date = 100
                if soc_sum_date < 0:
                    print('Error: soc cannot be negative.')
                
                out_temp_date = bmwdf_negsoc_user_date['a_temp_mean'].mean()
                first_time_of_day_date = bmwdf_negsoc_user_date.loc[0, 'timestamp_start_utc'].time()
                first_time_of_day_float_date = first_time_of_day_date.hour + first_time_of_day_date.minute/60.0
                if exist_flag >= 2:
                    last_time_of_day_date = bmwdf_negsoc_user_date.loc[len(bmwdf_negsoc_user_date)-1, 'timestamp_start_utc'].time()
                    last_time_of_day_float_date = last_time_of_day_date.hour + last_time_of_day_date.minute/60.0
                else:
                    last_time_of_day_float_date = -1
                mean_time_of_day_date = bmwdf_negsoc_user_date['timestamp_utc_mean'].mean().time()
                mean_time_of_day_float_date = mean_time_of_day_date.hour + mean_time_of_day_date.minute/60.0
                
                if start_date.weekday() >=5:
                    weekend_flag = 1
                else:
                    weekend_flag = 0                
                
                soc_user['start_date'].append(start_date)
                soc_user['soc'].append(soc_sum_date)
                soc_user['out_temp'].append(out_temp_date)
                soc_user['day_of_week'].append(start_date.weekday())
                soc_user['weekend_flag'].append(weekend_flag)
                soc_user['first_time_of_day'].append(first_time_of_day_float_date)
                soc_user['last_time_of_day'].append(last_time_of_day_float_date)
                soc_user['mean_time_of_day'].append(mean_time_of_day_float_date)
                soc_user['day_of_year'].append(start_date.timetuple().tm_yday)
            
            else:
                soc_user['start_date'].append(start_date)
                soc_user['soc'].append(np.nan)
                soc_user['out_temp'].append(np.nan)
                soc_user['day_of_week'].append(start_date.weekday())
                soc_user['weekend_flag'].append(weekend_flag)
                soc_user['first_time_of_day'].append(np.nan)
                soc_user['last_time_of_day'].append(np.nan)
                soc_user['mean_time_of_day'].append(np.nan)
                soc_user['day_of_year'].append(start_date.timetuple().tm_yday)
                
            start_date += delta
        
        soc_above100['sum'].append(above100_sum)
        soc_user = pd.DataFrame(soc_user)
        soc_user['user_id'] = user
        
        if(len(bmwdf_negsoc_user['vin'].unique())==1):
            soc_user['vin'] = bmwdf_negsoc_user['vin'].unique()[0]
        else:
            print(user, "has more than on vin.")
        
        # save results
        if saveflag == True:
            if not os.path.exists(PREPROCESS_PATH):
                os.makedirs(PREPROCESS_PATH)
            soc_path = PREPROCESS_PATH + '/' + str(int(user)) + '_soc.csv'
            soc_user.to_csv(soc_path, index=False)

    soc_above100 = pd.DataFrame(soc_above100)
    return soc_above100
 
def add_soc_hhindex(userlist, SOC_PATH, HHINDEX_PATH, SAVEDATA_PATH):
    """
    Add past hhindex features on soc features.
    
    Paramaters
    ----------
    userlist : list, userlist to extract daily mobility features.
    SOC_PATH : str, path of soc relevant features.
    HHINDEX_PATH : str, path of soc ecarhhindex features.
    SAVEDATA_PATH : str, path to save temporary results.
    
    Returns
    ----------
    N/A
    """
    
    delta = datetime.timedelta(days=1)
    
    # iterate over all users
    for user in userlist:
        print(user)
        print('-----------------------')
        
        soc_path = SOC_PATH + '/' + str(int(user)) + '_soc.csv'
        soc_user = pd.read_csv(soc_path)
            
        # read hhindex features
        hhindex_path = HHINDEX_PATH + '/' + str(int(user)) + '_hhindex.csv'
        hhindex_user = pd.read_csv(hhindex_path)
        hhindex_user['ecar_hhindex'] = hhindex_user['ecar_hhindex'].fillna(0) 

        soc_user['ecar_hhindex_1day'] = np.nan
        soc_user['ecar_hhindex_2day'] = np.nan
        soc_user['ecar_hhindex_3day'] = np.nan
        soc_user['ecar_hhindex_3dayavr'] = np.nan
        soc_user['ecar_hhindex_7day'] = np.nan
        soc_user['ecar_hhindex_1weekday'] = np.nan
        soc_user['ecar_hhindex_2weekday'] = np.nan
        soc_user['ecar_hhindex_3weekday'] = np.nan
        soc_user['ecar_hhindex_4weekday'] = np.nan
        
        # iterate over all days
        period = soc_user['start_date'].unique()[:]
        for start_date in period:

            # add last day's ecarhhindex features
            start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            prev_date_obj = start_date_obj - delta
            prev_date_str = str(prev_date_obj.date())
            hhindex_item = hhindex_user[hhindex_user['date']==prev_date_str]
            hhindex_item.index = range(0,len(hhindex_item))
            if len(hhindex_item)!=0:
                soc_user.loc[soc_user['start_date']==start_date,'ecar_hhindex_1day'] = hhindex_item.loc[0,'ecar_hhindex']

            # add last second day's ecarhhindex features
            start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            prev_date_obj = start_date_obj - delta*2
            prev_date_str = str(prev_date_obj.date())
            hhindex_item = hhindex_user[hhindex_user['date']==prev_date_str]
            hhindex_item.index = range(0,len(hhindex_item))
            if len(hhindex_item)!=0:
                soc_user.loc[soc_user['start_date']==start_date,'ecar_hhindex_2day'] = hhindex_item.loc[0,'ecar_hhindex']
            
            # add last third day's ecarhhindex features
            start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            prev_date_obj = start_date_obj - delta*3
            prev_date_str = str(prev_date_obj.date())
            hhindex_item = hhindex_user[hhindex_user['date']==prev_date_str]
            hhindex_item.index = range(0,len(hhindex_item))
            if len(hhindex_item)!=0:
                soc_user.loc[soc_user['start_date']==start_date,'ecar_hhindex_3day'] = hhindex_item.loc[0,'ecar_hhindex']
                
            # add past three days' mean ecarhhindex features
            start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            prev_date_str_3days = []
            for i in range(1,4):
                prev_date_obj = start_date_obj - delta*i
                prev_date_str = str(prev_date_obj.date())
                prev_date_str_3days.append(prev_date_str)
            hhindex_item = hhindex_user[hhindex_user['date'].isin(prev_date_str_3days)]
            hhindex_item.index = range(0,len(hhindex_item))
            if len(hhindex_item)!=0:
                if len(hhindex_item)!=3:
                    print('no 3 last days')
                hhindex_item = hhindex_item.mean(axis=0)
                soc_user.loc[soc_user['start_date']==start_date,'ecar_hhindex_3dayavr'] = hhindex_item['ecar_hhindex']
        
            # add past seven days' mean ecarhhindex features
            start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            prev_date_str_7days = []
            for i in range(1,8):
                prev_date_obj = start_date_obj - delta*i
                prev_date_str = str(prev_date_obj.date())
                prev_date_str_7days.append(prev_date_str)
            hhindex_item = hhindex_user[hhindex_user['date'].isin(prev_date_str_7days)]
            hhindex_item.index = range(0,len(hhindex_item))
            if len(hhindex_item)!=0:
                if len(hhindex_item)!=7:
                    print('no 7 last days')
                hhindex_item = hhindex_item.mean(axis=0)
                soc_user.loc[soc_user['start_date']==start_date,'ecar_hhindex_7day'] = hhindex_item['ecar_hhindex']            
            
            # add last same weekday's ecarhhindex features
            prev_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            prev_date_str_1weekdays = []
            for i in range(1,2):
                prev_date_obj = prev_date_obj - delta*7*1
                prev_date_str = str(prev_date_obj.date())
                prev_date_str_1weekdays.append(prev_date_str)
            hhindex_item = hhindex_user[hhindex_user['date'].isin(prev_date_str_1weekdays)]
            hhindex_item.index = range(0,len(hhindex_item))
            if len(hhindex_item)!=0:
                if len(hhindex_item)!=1:
                    print('no 1 last weekdays')
                hhindex_item = hhindex_item.mean(axis=0)
                soc_user.loc[soc_user['start_date']==start_date,'ecar_hhindex_1weekday'] = hhindex_item['ecar_hhindex']             

            # add past two same weekdays' mean ecarhhindex features
            prev_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            prev_date_str_2weekdays = []
            for i in range(1,3):
                prev_date_obj = prev_date_obj - delta*7*1
                prev_date_str = str(prev_date_obj.date())
                prev_date_str_2weekdays.append(prev_date_str)
            hhindex_item = hhindex_user[hhindex_user['date'].isin(prev_date_str_2weekdays)]
            hhindex_item.index = range(0,len(hhindex_item))
            if len(hhindex_item)!=0:
                if len(hhindex_item)!=2:
                    print('no 2 last weekdays')
                hhindex_item = hhindex_item.mean(axis=0)
                soc_user.loc[soc_user['start_date']==start_date,'ecar_hhindex_2weekday'] = hhindex_item['ecar_hhindex']             

            # add past three same weekdays' mean ecarhhindex features
            prev_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            prev_date_str_3weekdays = []
            for i in range(1,4):
                prev_date_obj = prev_date_obj - delta*7*1
                prev_date_str = str(prev_date_obj.date())
                prev_date_str_3weekdays.append(prev_date_str)
            hhindex_item = hhindex_user[hhindex_user['date'].isin(prev_date_str_3weekdays)]
            hhindex_item.index = range(0,len(hhindex_item))
            if len(hhindex_item)!=0:
                if len(hhindex_item)!=3:
                    print('no 3 last weekdays')
                hhindex_item = hhindex_item.mean(axis=0)
                soc_user.loc[soc_user['start_date']==start_date,'ecar_hhindex_3weekday'] = hhindex_item['ecar_hhindex']             

            # add past four same weekdays' mean ecarhhindex features
            prev_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            prev_date_str_4weekdays = []
            for i in range(1,5):
                prev_date_obj = prev_date_obj - delta*7*1
                prev_date_str = str(prev_date_obj.date())
                prev_date_str_4weekdays.append(prev_date_str)
            hhindex_item = hhindex_user[hhindex_user['date'].isin(prev_date_str_4weekdays)]
            hhindex_item.index = range(0,len(hhindex_item))
            if len(hhindex_item)!=0:
                if len(hhindex_item)!=4:
                    print('no 4 last weekdays')
                hhindex_item = hhindex_item.mean(axis=0)
                soc_user.loc[soc_user['start_date']==start_date,'ecar_hhindex_4weekday'] = hhindex_item['ecar_hhindex']             
            
        # save results
        if not os.path.exists(SAVEDATA_PATH):
            os.makedirs(SAVEDATA_PATH) 
        input_path = SAVEDATA_PATH + '/' + str(int(user)) + '_input.csv'
        soc_user.to_csv(input_path, index=False)
 

def add_sochhindex_evstat(userlist, SOC_PATH, EVSTAT_PATH, SAVEDATA_PATH):
    """
    Add past ev duration and disatnce features on soc and ecarhhindex features.
    
    Paramaters
    ----------
    userlist : list, userlist to extract daily mobility features
    SOC_PATH : str, path of soc+ecarhhindex features
    EVSTAT_PATH : str, path of ev duration and distance features.
    SAVEDATA_PATH : str, path to save temporary results.
    
    Returns
    ----------
    N/A
    """

    delta = datetime.timedelta(days=1)
    
    # iterate over all users
    for user in userlist:
        print(user)
        print('-----------------------')
        
        soc_path = SOC_PATH + '/' + str(int(user)) + '_input.csv'
        soc_user = pd.read_csv(soc_path)
            
        # read ev distance and duration features
        evstat_path = EVSTAT_PATH + '/' + str(int(user)) + '_EVStat.csv'
        evstat_user = pd.read_csv(evstat_path)
        evstat_user = evstat_user.fillna(0)

        soc_user['ev_duration_1day'] = np.nan
        soc_user['ev_duration_2day'] = np.nan
        soc_user['ev_duration_3day'] = np.nan
        soc_user['ev_duration_3dayavr'] = np.nan
        soc_user['ev_duration_7day'] = np.nan
        soc_user['ev_duration_1weekday'] = np.nan
        soc_user['ev_duration_2weekday'] = np.nan
        soc_user['ev_duration_3weekday'] = np.nan
        soc_user['ev_duration_4weekday'] = np.nan
        
        soc_user['ev_dist_1day'] = np.nan
        soc_user['ev_dist_2day'] = np.nan
        soc_user['ev_dist_3day'] = np.nan
        soc_user['ev_dist_3dayavr'] = np.nan
        soc_user['ev_dist_7day'] = np.nan
        soc_user['ev_dist_1weekday'] = np.nan
        soc_user['ev_dist_2weekday'] = np.nan
        soc_user['ev_dist_3weekday'] = np.nan
        soc_user['ev_dist_4weekday'] = np.nan
        
        # iterate over all days
        period = soc_user['start_date'].unique()[:]
        for start_date in period:
            
            # add last day's ev distance and duration features
            start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            prev_date_obj = start_date_obj - delta
            prev_date_str = str(prev_date_obj.date())
            evstat_item = evstat_user[evstat_user['date']==prev_date_str]
            evstat_item.index = range(0,len(evstat_item))
            if len(evstat_item)!=0:
                soc_user.loc[soc_user['start_date']==start_date,'ev_duration_1day'] = evstat_item.loc[0,'duration']
                soc_user.loc[soc_user['start_date']==start_date,'ev_dist_1day'] = evstat_item.loc[0,'dist']

            # add last second day's ev distance and duration features
            start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            prev_date_obj = start_date_obj - delta*2
            prev_date_str = str(prev_date_obj.date())
            evstat_item = evstat_user[evstat_user['date']==prev_date_str]
            evstat_item.index = range(0,len(evstat_item))
            if len(evstat_item)!=0:
                soc_user.loc[soc_user['start_date']==start_date,'ev_duration_2day'] = evstat_item.loc[0,'duration']
                soc_user.loc[soc_user['start_date']==start_date,'ev_dist_2day'] = evstat_item.loc[0,'dist']

            # add last third day's ev distance and duration features              
            start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            prev_date_obj = start_date_obj - delta*3
            prev_date_str = str(prev_date_obj.date())
            evstat_item = evstat_user[evstat_user['date']==prev_date_str]
            evstat_item.index = range(0,len(evstat_item))
            if len(evstat_item)!=0:
                soc_user.loc[soc_user['start_date']==start_date,'ev_duration_3day'] = evstat_item.loc[0,'duration']
                soc_user.loc[soc_user['start_date']==start_date,'ev_dist_3day'] = evstat_item.loc[0,'dist']
            
            # add past three days' mean ev distance and duration features
            start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            prev_date_str_3days = []
            for i in range(1,4):
                prev_date_obj = start_date_obj - delta*i
                prev_date_str = str(prev_date_obj.date())
                prev_date_str_3days.append(prev_date_str)
            evstat_item = evstat_user[evstat_user['date'].isin(prev_date_str_3days)]
            evstat_item.index = range(0,len(evstat_item))
            if len(evstat_item)!=0:
                if len(evstat_item)!=3:
                    print('no 3 last days')
                evstat_item = evstat_item.mean(axis=0)
                soc_user.loc[soc_user['start_date']==start_date,'ev_duration_3dayavr'] = evstat_item['duration']
                soc_user.loc[soc_user['start_date']==start_date,'ev_dist_3dayavr'] = evstat_item['dist']
                
            # add past seven days' mean ev distance and duration features
            start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            prev_date_str_7days = []
            for i in range(1,8):
                prev_date_obj = start_date_obj - delta*i
                prev_date_str = str(prev_date_obj.date())
                prev_date_str_7days.append(prev_date_str)
            evstat_item = evstat_user[evstat_user['date'].isin(prev_date_str_7days)]
            evstat_item.index = range(0,len(evstat_item))
            if len(evstat_item)!=0:
                if len(evstat_item)!=7:
                    print('no 7 last days')
                evstat_item = evstat_item.mean(axis=0)
                soc_user.loc[soc_user['start_date']==start_date,'ev_duration_7day'] = evstat_item['duration']            
                soc_user.loc[soc_user['start_date']==start_date,'ev_dist_7day'] = evstat_item['dist']
                
            # add last same weekday's ev distance and duration features
            prev_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            prev_date_str_1weekdays = []
            for i in range(1,2):
                prev_date_obj = prev_date_obj - delta*7*1
                prev_date_str = str(prev_date_obj.date())
                prev_date_str_1weekdays.append(prev_date_str)
            evstat_item = evstat_user[evstat_user['date'].isin(prev_date_str_1weekdays)]
            evstat_item.index = range(0,len(evstat_item))
            if len(evstat_item)!=0:
                if len(evstat_item)!=1:
                    print('no 1 last weekdays')
                evstat_item = evstat_item.mean(axis=0)
                soc_user.loc[soc_user['start_date']==start_date,'ev_duration_1weekday'] = evstat_item['duration']             
                soc_user.loc[soc_user['start_date']==start_date,'ev_dist_1weekday'] = evstat_item['dist']
                
            # add past two same weekdays' mean ev distance and duration features
            prev_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            prev_date_str_2weekdays = []
            for i in range(1,3):
                prev_date_obj = prev_date_obj - delta*7*1
                prev_date_str = str(prev_date_obj.date())
                prev_date_str_2weekdays.append(prev_date_str)
            evstat_item = evstat_user[evstat_user['date'].isin(prev_date_str_2weekdays)]
            evstat_item.index = range(0,len(evstat_item))
            if len(evstat_item)!=0:
                if len(evstat_item)!=2:
                    print('no 2 last weekdays')
                evstat_item = evstat_item.mean(axis=0)
                soc_user.loc[soc_user['start_date']==start_date,'ev_duration_2weekday'] = evstat_item['duration']             
                soc_user.loc[soc_user['start_date']==start_date,'ev_dist_2weekday'] = evstat_item['dist']
                
            # add last three same weekday's ev distance and duration features
            prev_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            prev_date_str_3weekdays = []
            for i in range(1,4):
                prev_date_obj = prev_date_obj - delta*7*1
                prev_date_str = str(prev_date_obj.date())
                prev_date_str_3weekdays.append(prev_date_str)
            evstat_item = evstat_user[evstat_user['date'].isin(prev_date_str_3weekdays)]
            evstat_item.index = range(0,len(evstat_item))
            if len(evstat_item)!=0:
                if len(evstat_item)!=3:
                    print('no 3 last weekdays')
                evstat_item = evstat_item.mean(axis=0)
                soc_user.loc[soc_user['start_date']==start_date,'ev_duration_3weekday'] = evstat_item['duration']             
                soc_user.loc[soc_user['start_date']==start_date,'ev_dist_3weekday'] = evstat_item['dist']
                
            # add last four same weekday's ev distance and duration features
            prev_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            prev_date_str_4weekdays = []
            for i in range(1,5):
                prev_date_obj = prev_date_obj - delta*7*1
                prev_date_str = str(prev_date_obj.date())
                prev_date_str_4weekdays.append(prev_date_str)
            evstat_item = evstat_user[evstat_user['date'].isin(prev_date_str_4weekdays)]
            evstat_item.index = range(0,len(evstat_item))
            if len(evstat_item)!=0:
                if len(evstat_item)!=4:
                    print('no 4 last weekdays')
                evstat_item = evstat_item.mean(axis=0)
                soc_user.loc[soc_user['start_date']==start_date,'ev_duration_4weekday'] = evstat_item['duration']             
                soc_user.loc[soc_user['start_date']==start_date,'ev_dist_4weekday'] = evstat_item['dist']
                 
        # save results
        if not os.path.exists(SAVEDATA_PATH):
            os.makedirs(SAVEDATA_PATH) 
        input_path = SAVEDATA_PATH + '/' + str(int(user)) + '_input.csv'
        soc_user.to_csv(input_path, index=False)

def add_soc_mob(userlist, SOC_PATH, MOB_PATH, SAVEDATA_PATH):
    """
    Add past mobility features on soc+ecarhhindex+evstat features.
    
    Paramaters
    ----------
    userlist : list, userlist to extract daily mobility features.
    SOC_PATH : str, path of soc+ecarhhindex+evstat features.
    MOB_PATH : str, path of mobility features.
    SAVEDATA_PATH : str, path to save final inputs.
    
    Returns
    ----------
    N/A
    """

    delta = datetime.timedelta(days=1)
    
    # iterate over all users
    for user in userlist:
        print(user)
        
        soc_path = SOC_PATH + '/' + str(int(user)) + '_input.csv'
        soc_user = pd.read_csv(soc_path)
            
        # read mobility features
        mob_path = MOB_PATH + '/' + str(int(user)) + '_mob.csv'
        mob_user = pd.read_csv(mob_path)
        mob_user = mob_user.fillna(0)

        soc_user['top10locfre_1day'] = np.nan
        soc_user['top10locfre_2day'] = np.nan
        soc_user['top10locfre_3day'] = np.nan
        soc_user['top10locfre_3dayavr'] = np.nan
        soc_user['top10locfre_7day'] = np.nan
        soc_user['top10locfre_1weekday'] = np.nan
        soc_user['top10locfre_2weekday'] = np.nan
        soc_user['top10locfre_3weekday'] = np.nan
        soc_user['top10locfre_4weekday'] = np.nan

        soc_user['radgyr_1day'] = np.nan
        soc_user['radgyr_2day'] = np.nan
        soc_user['radgyr_3day'] = np.nan
        soc_user['radgyr_3dayavr'] = np.nan
        soc_user['radgyr_7day'] = np.nan
        soc_user['radgyr_1weekday'] = np.nan
        soc_user['radgyr_2weekday'] = np.nan
        soc_user['radgyr_3weekday'] = np.nan
        soc_user['radgyr_4weekday'] = np.nan

        soc_user['avrjumplen_1day'] = np.nan
        soc_user['avrjumplen_2day'] = np.nan
        soc_user['avrjumplen_3day'] = np.nan
        soc_user['avrjumplen_3dayavr'] = np.nan
        soc_user['avrjumplen_7day'] = np.nan
        soc_user['avrjumplen_1weekday'] = np.nan
        soc_user['avrjumplen_2weekday'] = np.nan
        soc_user['avrjumplen_3weekday'] = np.nan
        soc_user['avrjumplen_4weekday'] = np.nan

        soc_user['uncorentro_1day'] = np.nan
        soc_user['uncorentro_2day'] = np.nan
        soc_user['uncorentro_3day'] = np.nan
        soc_user['uncorentro_3dayavr'] = np.nan
        soc_user['uncorentro_7day'] = np.nan
        soc_user['uncorentro_1weekday'] = np.nan
        soc_user['uncorentro_2weekday'] = np.nan
        soc_user['uncorentro_3weekday'] = np.nan
        soc_user['uncorentro_4weekday'] = np.nan

        soc_user['realentro_1day'] = np.nan
        soc_user['realentro_2day'] = np.nan
        soc_user['realentro_3day'] = np.nan
        soc_user['realentro_3dayavr'] = np.nan
        soc_user['realentro_7day'] = np.nan
        soc_user['realentro_1weekday'] = np.nan
        soc_user['realentro_2weekday'] = np.nan
        soc_user['realentro_3weekday'] = np.nan
        soc_user['realentro_4weekday'] = np.nan
        
        # iterate over all days    
        period = soc_user['start_date'].unique()[:]
        for start_date in period:

            # add last day's mobility features
            start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            prev_date_obj = start_date_obj - delta
            prev_date_str = str(prev_date_obj.date())
            mob_item = mob_user[mob_user['start_date']==prev_date_str]
            mob_item.index = range(0,len(mob_item))
            if len(mob_item)!=0:
                soc_user.loc[soc_user['start_date']==start_date,'top10locfre_1day'] = mob_item.loc[0,'locfre_top10']
                soc_user.loc[soc_user['start_date']==start_date,'radgyr_1day'] = mob_item.loc[0,'rad_gyr']
                soc_user.loc[soc_user['start_date']==start_date,'avrjumplen_1day'] = mob_item.loc[0,'jump_len']
                soc_user.loc[soc_user['start_date']==start_date,'uncorentro_1day'] = mob_item.loc[0,'uncor_entro']
                soc_user.loc[soc_user['start_date']==start_date,'realentro_1day'] = mob_item.loc[0,'real_entro']

            # add last second day's mobility features
            start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            prev_date_obj = start_date_obj - delta*2
            prev_date_str = str(prev_date_obj.date())
            mob_item = mob_user[mob_user['start_date']==prev_date_str]
            mob_item.index = range(0,len(mob_item))
            if len(mob_item)!=0:
                soc_user.loc[soc_user['start_date']==start_date,'top10locfre_2day'] = mob_item.loc[0,'locfre_top10']
                soc_user.loc[soc_user['start_date']==start_date,'radgyr_2day'] = mob_item.loc[0,'rad_gyr']
                soc_user.loc[soc_user['start_date']==start_date,'avrjumplen_2day'] = mob_item.loc[0,'jump_len']
                soc_user.loc[soc_user['start_date']==start_date,'uncorentro_2day'] = mob_item.loc[0,'uncor_entro']
                soc_user.loc[soc_user['start_date']==start_date,'realentro_2day'] = mob_item.loc[0,'real_entro']
                
            # add last third day's mobility features           
            start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            prev_date_obj = start_date_obj - delta*3
            prev_date_str = str(prev_date_obj.date())
            mob_item = mob_user[mob_user['start_date']==prev_date_str]
            mob_item.index = range(0,len(mob_item))
            if len(mob_item)!=0:
                soc_user.loc[soc_user['start_date']==start_date,'top10locfre_3day'] = mob_item.loc[0,'locfre_top10']
                soc_user.loc[soc_user['start_date']==start_date,'radgyr_3day'] = mob_item.loc[0,'rad_gyr']
                soc_user.loc[soc_user['start_date']==start_date,'avrjumplen_3day'] = mob_item.loc[0,'jump_len']
                soc_user.loc[soc_user['start_date']==start_date,'uncorentro_3day'] = mob_item.loc[0,'uncor_entro']
                soc_user.loc[soc_user['start_date']==start_date,'realentro_3day'] = mob_item.loc[0,'real_entro']
                
            # add past three days' mean mobility features           
            start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            prev_date_str_3dayavrs = []
            for i in range(1,4):
                prev_date_obj = start_date_obj - delta*i
                prev_date_str = str(prev_date_obj.date())
                prev_date_str_3dayavrs.append(prev_date_str)
            mob_item = mob_user[mob_user['start_date'].isin(prev_date_str_3dayavrs)]
            mob_item.index = range(0,len(mob_item))
            if len(mob_item)!=0:
                if len(mob_item)!=3:
                    print('no 3 last days')
                mob_item = mob_item.mean(axis=0)
                soc_user.loc[soc_user['start_date']==start_date,'top10locfre_3dayavr'] = mob_item['locfre_top10']
                soc_user.loc[soc_user['start_date']==start_date,'radgyr_3dayavr'] = mob_item['rad_gyr']
                soc_user.loc[soc_user['start_date']==start_date,'avrjumplen_3dayavr'] = mob_item['jump_len']
                soc_user.loc[soc_user['start_date']==start_date,'uncorentro_3dayavr'] = mob_item['uncor_entro']
                soc_user.loc[soc_user['start_date']==start_date,'realentro_3dayavr'] = mob_item['real_entro']
                
            # add past seven days' mean mobility features              
            start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            prev_date_str_7days = []
            for i in range(1,8):
                prev_date_obj = start_date_obj - delta*i
                prev_date_str = str(prev_date_obj.date())
                prev_date_str_7days.append(prev_date_str)
            mob_item = mob_user[mob_user['start_date'].isin(prev_date_str_7days)]
            mob_item.index = range(0,len(mob_item))
            if len(mob_item)!=0:
                if len(mob_item)!=7:
                    print('no 7 last days')
                mob_item = mob_item.mean(axis=0)
                soc_user.loc[soc_user['start_date']==start_date,'top10locfre_7day'] = mob_item['locfre_top10']
                soc_user.loc[soc_user['start_date']==start_date,'radgyr_7day'] = mob_item['rad_gyr']
                soc_user.loc[soc_user['start_date']==start_date,'avrjumplen_7day'] = mob_item['jump_len']
                soc_user.loc[soc_user['start_date']==start_date,'uncorentro_7day'] = mob_item['uncor_entro']
                soc_user.loc[soc_user['start_date']==start_date,'realentro_7day'] = mob_item['real_entro']           

            # add last same weekday's mobility features             
            prev_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            prev_date_str_1weekdays = []
            for i in range(1,2):
                prev_date_obj = prev_date_obj - delta*7*1
                prev_date_str = str(prev_date_obj.date())
                prev_date_str_1weekdays.append(prev_date_str)
            mob_item = mob_user[mob_user['start_date'].isin(prev_date_str_1weekdays)]
            mob_item.index = range(0,len(mob_item))
            if len(mob_item)!=0:
                if len(mob_item)!=1:
                    print('no last 1 weekdays')
                mob_item = mob_item.mean(axis=0)
                soc_user.loc[soc_user['start_date']==start_date,'top10locfre_1weekday'] = mob_item['locfre_top10']
                soc_user.loc[soc_user['start_date']==start_date,'radgyr_1weekday'] = mob_item['rad_gyr']
                soc_user.loc[soc_user['start_date']==start_date,'avrjumplen_1weekday'] = mob_item['jump_len']
                soc_user.loc[soc_user['start_date']==start_date,'uncorentro_1weekday'] = mob_item['uncor_entro']
                soc_user.loc[soc_user['start_date']==start_date,'realentro_1weekday'] = mob_item['real_entro']
                
            # add past two same weekdays' mean mobility features            
            prev_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            prev_date_str_2weekdays = []
            for i in range(1,3):
                prev_date_obj = prev_date_obj - delta*7*1
                prev_date_str = str(prev_date_obj.date())
                prev_date_str_2weekdays.append(prev_date_str)
            mob_item = mob_user[mob_user['start_date'].isin(prev_date_str_2weekdays)]
            mob_item.index = range(0,len(mob_item))
            if len(mob_item)!=0:
                if len(mob_item)!=2:
                    print('no last 2 weekdays')
                mob_item = mob_item.mean(axis=0)
                soc_user.loc[soc_user['start_date']==start_date,'top10locfre_2weekday'] = mob_item['locfre_top10']
                soc_user.loc[soc_user['start_date']==start_date,'radgyr_2weekday'] = mob_item['rad_gyr']
                soc_user.loc[soc_user['start_date']==start_date,'avrjumplen_2weekday'] = mob_item['jump_len']
                soc_user.loc[soc_user['start_date']==start_date,'uncorentro_2weekday'] = mob_item['uncor_entro']
                soc_user.loc[soc_user['start_date']==start_date,'realentro_2weekday'] = mob_item['real_entro']
                
            # add past three same weekdays' mean mobility features          
            prev_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            prev_date_str_3weekdays = []
            for i in range(1,4):
                prev_date_obj = prev_date_obj - delta*7*1
                prev_date_str = str(prev_date_obj.date())
                prev_date_str_3weekdays.append(prev_date_str)
            mob_item = mob_user[mob_user['start_date'].isin(prev_date_str_3weekdays)]
            mob_item.index = range(0,len(mob_item))
            if len(mob_item)!=0:
                if len(mob_item)!=3: 
                    print('no last 3 weekdays')
                mob_item = mob_item.mean(axis=0)
                soc_user.loc[soc_user['start_date']==start_date,'top10locfre_3weekday'] = mob_item['locfre_top10']
                soc_user.loc[soc_user['start_date']==start_date,'radgyr_3weekday'] = mob_item['rad_gyr']
                soc_user.loc[soc_user['start_date']==start_date,'avrjumplen_3weekday'] = mob_item['jump_len']
                soc_user.loc[soc_user['start_date']==start_date,'uncorentro_3weekday'] = mob_item['uncor_entro']
                soc_user.loc[soc_user['start_date']==start_date,'realentro_3weekday'] = mob_item['real_entro']
                
            # add past four same weekdays' mean mobility features          
            prev_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            prev_date_str_4weekdays = []
            for i in range(1,5):
                prev_date_obj = prev_date_obj - delta*7*1
                prev_date_str = str(prev_date_obj.date())
                prev_date_str_4weekdays.append(prev_date_str)
            mob_item = mob_user[mob_user['start_date'].isin(prev_date_str_4weekdays)]
            mob_item.index = range(0,len(mob_item))
            if len(mob_item)!=0:
                if len(mob_item)!=4:
                    print('no last 4 weekdays')
                mob_item = mob_item.mean(axis=0)
                soc_user.loc[soc_user['start_date']==start_date,'top10locfre_4weekday'] = mob_item['locfre_top10']
                soc_user.loc[soc_user['start_date']==start_date,'radgyr_4weekday'] = mob_item['rad_gyr']
                soc_user.loc[soc_user['start_date']==start_date,'avrjumplen_4weekday'] = mob_item['jump_len']
                soc_user.loc[soc_user['start_date']==start_date,'uncorentro_4weekday'] = mob_item['uncor_entro']
                soc_user.loc[soc_user['start_date']==start_date,'realentro_4weekday'] = mob_item['real_entro']  
                
        # keep valid data from first valid soc item
        soc_user.index = range(0,len(soc_user))        
        valid_soc_idx = soc_user['top10locfre_4weekday'].first_valid_index()
        soc_user = soc_user.iloc[valid_soc_idx:,:]
        soc_user.index = range(0,len(soc_user)) 
        
        # fill nan
        soc_user['soc'] = soc_user['soc'].fillna(0)
        soc_user['out_temp'] = soc_user['out_temp'].interpolate(method ='linear', limit_direction ='forward')
        soc_user['first_time_of_day'] = soc_user['first_time_of_day'].fillna(soc_user['first_time_of_day'].mean())
        soc_user['last_time_of_day'] = soc_user['last_time_of_day'].fillna(soc_user['last_time_of_day'].mean())
        soc_user['mean_time_of_day'] = soc_user['mean_time_of_day'].fillna(soc_user['mean_time_of_day'].mean())
        
        # save results
        dates = soc_user['start_date']
        soc_user = soc_user.drop(columns=['user_id','vin','start_date'])
        soc_user = soc_user.astype('float32')
        soc_user['date'] = dates
    
        if not os.path.exists(SAVEDATA_PATH):
            os.makedirs(SAVEDATA_PATH)         
        input_path = SAVEDATA_PATH + '/' + str(int(user)) + '_input.csv'
        soc_user.to_csv(input_path, index=False)

