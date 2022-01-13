# -*- coding: utf-8 -*-
"""
Created on Jun 2021
@author: Haojun Cai
""" 

import datetime
import geopandas as gpd
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import csv
from sqlalchemy import create_engine

def check_home_labels(stp_cls, userlist):
    """
    Check number and percentage of home labels for each user.
    
    Paramaters
    ----------
    stp_cls : dataframe, clustered staypoins data
    userlist : list, userlist to extract targets
    
    Returns
    ----------
    home_stat: dataframe, statistics of home labels
    """

    home_stat = {'user_id':[], 'number':[], 'ratio':[]}
    
    for user in userlist:
        stp_cls_user = stp_cls[stp_cls['user_id']==user].sort_values(by='started_at',ascending=True)
        home_num = len(stp_cls_user[stp_cls_user['purpose_validated']=='home'])
        total_num = len(stp_cls_user)
        home_ratio = home_num/total_num
        
        home_stat['user_id'].append(user)
        home_stat['number'].append(home_num)
        home_stat['ratio'].append(home_ratio)
    
    home_stat = pd.DataFrame(home_stat)
    
    return home_stat

def extract_arrival_target(engine, userlist, savefile_flag, PREPROCESS_PATH):
    """
    Extract daily arrival time for each user.
    
    Paramaters
    ----------
    engine
    userlist : list, userlist to extract targets
    savefile_flag : boolean, flag indicating whether to save results
    PREPROCESS_PATH : str, path to save results
    
    Returns
    ----------
    arrival_stat: dataframe, statistics of arrival time
    """

    pandas_query = """SELECT * FROM caihao.stp_cls""" 
    stp_cls = gpd.read_postgis(pandas_query, engine, geom_col='geometry')
    stp_cls['started_at'] = pd.to_datetime(stp_cls['started_at'], utc=True)
    stp_cls['finished_at'] = pd.to_datetime(stp_cls['finished_at'], utc=True)
    stp_cls['started_at_ymd'] = pd.to_datetime(stp_cls['started_at']).dt.date
    stp_cls['finished_at_ymd'] = pd.to_datetime(stp_cls['finished_at']).dt.date

    special_dates = {'user_id':[], 'date':[]}
    arrival_stat = {'user_id':[], 'lastnothome_days':[], 'valid_days':[], 'nonexist_days':[], 'total':[]}   
    
    # iterate over all users
    for i in range(0,len(userlist)):
        
        arr_user = {'date_id':[], 'start_ymd':[], 'arrival':[], 'day_of_week':[], 'weekend_flag':[], 'day_of_year':[]}
        user = userlist[i]    
        print(user)
        print('-------------START-----------------')
        stp_cls_user = stp_cls[stp_cls['user_id']==user].sort_values(by='started_at',ascending=True)
        stp_cls_user.index = range(0,len(stp_cls_user))    
        
        start_ymd = sorted(set(list(stp_cls_user['started_at_ymd'])))
        start_date = min(start_ymd)
        end_date = max(start_ymd)
        delta = datetime.timedelta(days=1)
        
        arrival_stat['total'].append((end_date-start_date).days+1)
        
        last_not_home_user = []
        date_non_exist_user = []
        j = 0

        # iterate over all days        
        while start_date <= end_date:
            
            stp_cls_user_date = stp_cls_user[stp_cls_user['started_at_ymd']==start_date]       
            date_flag = len(stp_cls_user_date)
            
            ## CASE 1: the last item of the day is labeled as home
            if date_flag!=0 and stp_cls_user_date['purpose_validated'].iloc[-1]=='home': 
                k = -1
                # find last consecutive items labeled as home
                if (date_flag>=2):
                    home_flag = stp_cls_user_date['purpose_validated'].iloc[-2]=='home' 
                    while home_flag==True:
                        k = k-1
                        if abs(k)<date_flag:
                            home_flag = stp_cls_user_date['purpose_validated'].iloc[k-1]=='home'
                        else:
                            break
                if start_date.weekday() >=5:
                    weekend_flag = 1
                else:
                    weekend_flag = 0
                arr_user['date_id'].append(j)
                arr_user['start_ymd'].append(start_date)
                arr_user['arrival'].append(stp_cls_user_date['started_at'].iloc[k]) 
                arr_user['day_of_week'].append(start_date.weekday())
                arr_user['weekend_flag'].append(weekend_flag)
                arr_user['day_of_year'].append(start_date.timetuple().tm_yday)
            
                ## CASE 2: user stayed at home for a weekend
                date_diff = stp_cls_user_date['finished_at_ymd'].iloc[-1] - stp_cls_user_date['started_at_ymd'].iloc[-1]
                temp_end_date = stp_cls_user_date['finished_at_ymd'].iloc[-1] - delta
                if date_diff.days >= 2:
                    while start_date < temp_end_date:
                        start_date += delta
                        j = j+1
                        # print(start_date, " 24h at home")
                        if start_date.weekday() >=5:
                            weekend_flag = 1
                        else:
                            weekend_flag = 0
                        arr_user['date_id'].append(j)
                        arr_user['start_ymd'].append(start_date)
                        arr_user['arrival'].append('24h_at_home') 
                        arr_user['day_of_week'].append(start_date.weekday())
                        arr_user['weekend_flag'].append(weekend_flag)
                        arr_user['day_of_year'].append(start_date.timetuple().tm_yday)
                        
                start_date += delta
                j = j+1
                                            
            ## CASE 3: the last item of the day is not labeled as home, which was treated as invalid cases
            elif date_flag!=0 and stp_cls_user_date['purpose_validated'].iloc[-1]!='home':
                date_diff = stp_cls_user_date['finished_at_ymd'].iloc[-1] - stp_cls_user_date['started_at_ymd'].iloc[-1]
                if date_diff.days >= 1:
                    k = -1
                    # find last consecutive items labeled as home
                    if (date_flag>=2):
                        home_flag = stp_cls_user_date['purpose_validated'].iloc[-2]=='home' 
                        while home_flag==True:
                            k = k-1
                            if abs(k)<date_flag:
                                home_flag = stp_cls_user_date['purpose_validated'].iloc[k-1]=='home'
                            else:
                                break
                    if start_date.weekday() >=5:
                        weekend_flag = 1
                    else:
                        weekend_flag = 0
                    arr_user['date_id'].append(j)
                    arr_user['start_ymd'].append(start_date)
                    arr_user['arrival'].append(stp_cls_user_date['started_at'].iloc[k]) 
                    arr_user['day_of_week'].append(start_date.weekday())
                    arr_user['weekend_flag'].append(weekend_flag)
                    arr_user['day_of_year'].append(start_date.timetuple().tm_yday)
                
                    # user stayed at home for a weekend
                    temp_end_date = stp_cls_user_date['finished_at_ymd'].iloc[-1] - delta
                    if date_diff.days >= 2:
                        while start_date < temp_end_date:
                            start_date += delta
                            j = j+1
                            # print(start_date, " 24h at home")
                            if start_date.weekday() >=5:
                                weekend_flag = 1
                            else:
                                weekend_flag = 0                            
                            arr_user['date_id'].append(j)
                            arr_user['start_ymd'].append(start_date)
                            arr_user['arrival'].append('24h_at_home') 
                            arr_user['day_of_week'].append(start_date.weekday())
                            arr_user['weekend_flag'].append(weekend_flag)
                            arr_user['day_of_year'].append(start_date.timetuple().tm_yday)                
                else:
                    last_not_home_user.append(stp_cls_user_date.iloc[-1])    
                    print(start_date, " last item not at home")
                    if start_date.weekday() >=5:
                        weekend_flag = 1
                    else:
                        weekend_flag = 0 
                    arr_user['date_id'].append(j)
                    arr_user['start_ymd'].append(start_date)
                    arr_user['arrival'].append('last_item_not_at_home') 
                    arr_user['day_of_week'].append(start_date.weekday())
                    arr_user['weekend_flag'].append(weekend_flag)
                    arr_user['day_of_year'].append(start_date.timetuple().tm_yday)                
                start_date += delta
                j = j+1
                
            ## CASE 4: no data on that day
            elif date_flag==0:
                date_non_exist_user.append(start_date)
                if start_date.weekday() >=5:
                    weekend_flag = 1
                else:
                    weekend_flag = 0 
                arr_user['date_id'].append(j)
                arr_user['start_ymd'].append(start_date)
                arr_user['arrival'].append('date_not_exist') 
                arr_user['day_of_week'].append(start_date.weekday())
                arr_user['weekend_flag'].append(weekend_flag)
                arr_user['day_of_year'].append(start_date.timetuple().tm_yday) 
                # print(start_date, " do not exist")
                start_date += delta
                j = j+1
                
            else:
                special_dates['user_id'].append(user)
                special_dates['date'].append(start_date)
                print(start_date, " special things happen")
                start_date += delta
                j = j+1
        
        # save results
        arr_user = pd.DataFrame(arr_user)
        arr_user['user_id'] = user
        last_not_home_user = pd.DataFrame(last_not_home_user) 
        date_non_exist_user = pd.DataFrame(date_non_exist_user)
        date_non_exist_user['user_id'] = user
        
        arrival_stat['user_id'].append(user)
        arrival_stat['lastnothome_days'].append(len(last_not_home_user))
        arrival_stat['valid_days'].append(len(arr_user)-len(last_not_home_user)-len(date_non_exist_user))
        arrival_stat['nonexist_days'].append(len(date_non_exist_user))

        if savefile_flag == True:  
            arr_path = PREPROCESS_PATH + '/' + str(int(user)) + '_arrival.csv'
            arr_user.to_csv(arr_path, index=False)
            # last_not_at_home_path = PREPROCESS_PATH + '/' + str(int(user)) + '_arrival_lastnotathome.csv'
            # last_not_home_user.to_csv(last_not_at_home_path, index=False)
            # date_non_exist_path = PREPROCESS_PATH + '/' + str(int(user)) + '_arrival_datenonexist.csv'
            # date_non_exist_user.to_csv(date_non_exist_path, index=False)        
        
        print('---------------END------------------')
        print('------------------------------------')
        
    arrival_stat = pd.DataFrame(arrival_stat)
    special_dates = pd.DataFrame(special_dates)
    
    return arrival_stat
                    
def add_arrival_mob(userlist, ARRIVAL_PATH, MOB_PATH, SAVEDATA_PATH):
    """
    Add past mobility features on arrival features.
    
    Paramaters
    ----------
    userlist : list, userlist to add daily mobility features
    ARRIVAL_PATH : str, path of arrival features
    MOB_PATH : str, path of mobility features
    SAVEDATA_PATH : str, path to save final inputs
    
    Returns
    ----------
    N/A
    """

    delta = datetime.timedelta(days=1)
    
    # iterate over all users
    for user in userlist:
        print(user)
        
        arrival_path = ARRIVAL_PATH + '/' + str(int(user)) + '_arrival.csv'
        arrival_user = pd.read_csv(arrival_path)
            
        # read mobility features
        mob_path = MOB_PATH + '/' + str(int(user)) + '_mob.csv'
        mob_user = pd.read_csv(mob_path)
        mob_user = mob_user.drop(columns='user_id')
        mob_user = mob_user.fillna(0)
        
        arrival_user['top10locfre_1day'] = np.nan
        arrival_user['top10locfre_2day'] = np.nan
        arrival_user['top10locfre_3day'] = np.nan
        arrival_user['top10locfre_3dayavr'] = np.nan
        arrival_user['top10locfre_7day'] = np.nan
        arrival_user['top10locfre_1weekday'] = np.nan
        arrival_user['top10locfre_2weekday'] = np.nan
        arrival_user['top10locfre_3weekday'] = np.nan
        arrival_user['top10locfre_4weekday'] = np.nan

        arrival_user['radgyr_1day'] = np.nan
        arrival_user['radgyr_2day'] = np.nan
        arrival_user['radgyr_3day'] = np.nan
        arrival_user['radgyr_3dayavr'] = np.nan
        arrival_user['radgyr_7day'] = np.nan
        arrival_user['radgyr_1weekday'] = np.nan
        arrival_user['radgyr_2weekday'] = np.nan
        arrival_user['radgyr_3weekday'] = np.nan
        arrival_user['radgyr_4weekday'] = np.nan

        arrival_user['avrjumplen_1day'] = np.nan
        arrival_user['avrjumplen_2day'] = np.nan
        arrival_user['avrjumplen_3day'] = np.nan
        arrival_user['avrjumplen_3dayavr'] = np.nan
        arrival_user['avrjumplen_7day'] = np.nan
        arrival_user['avrjumplen_1weekday'] = np.nan
        arrival_user['avrjumplen_2weekday'] = np.nan
        arrival_user['avrjumplen_3weekday'] = np.nan
        arrival_user['avrjumplen_4weekday'] = np.nan

        arrival_user['uncorentro_1day'] = np.nan
        arrival_user['uncorentro_2day'] = np.nan
        arrival_user['uncorentro_3day'] = np.nan
        arrival_user['uncorentro_3dayavr'] = np.nan
        arrival_user['uncorentro_7day'] = np.nan
        arrival_user['uncorentro_1weekday'] = np.nan
        arrival_user['uncorentro_2weekday'] = np.nan
        arrival_user['uncorentro_3weekday'] = np.nan
        arrival_user['uncorentro_4weekday'] = np.nan

        arrival_user['realentro_1day'] = np.nan
        arrival_user['realentro_2day'] = np.nan
        arrival_user['realentro_3day'] = np.nan
        arrival_user['realentro_3dayavr'] = np.nan
        arrival_user['realentro_7day'] = np.nan
        arrival_user['realentro_1weekday'] = np.nan
        arrival_user['realentro_2weekday'] = np.nan
        arrival_user['realentro_3weekday'] = np.nan
        arrival_user['realentro_4weekday'] = np.nan

        # iterate over all days 
        period = arrival_user['start_ymd'].unique()[:]
        for start_date in period:
            
            # add last day's mobility features
            start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            prev_date_obj = start_date_obj - delta
            prev_date_str = str(prev_date_obj.date())
            mob_item = mob_user[mob_user['start_date']==prev_date_str]
            mob_item.index = range(0,len(mob_item))
            if len(mob_item)!=0:
                arrival_user.loc[arrival_user['start_ymd']==start_date,'top10locfre_1day'] = mob_item.loc[0,'locfre_top10']
                arrival_user.loc[arrival_user['start_ymd']==start_date,'radgyr_1day'] = mob_item.loc[0,'rad_gyr']
                arrival_user.loc[arrival_user['start_ymd']==start_date,'avrjumplen_1day'] = mob_item.loc[0,'jump_len']
                arrival_user.loc[arrival_user['start_ymd']==start_date,'uncorentro_1day'] = mob_item.loc[0,'uncor_entro']
                arrival_user.loc[arrival_user['start_ymd']==start_date,'realentro_1day'] = mob_item.loc[0,'real_entro']

            # add last second day's mobility features
            start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            prev_date_obj = start_date_obj - delta*2
            prev_date_str = str(prev_date_obj.date())
            mob_item = mob_user[mob_user['start_date']==prev_date_str]
            mob_item.index = range(0,len(mob_item))
            if len(mob_item)!=0:
                arrival_user.loc[arrival_user['start_ymd']==start_date,'top10locfre_2day'] = mob_item.loc[0,'locfre_top10']
                arrival_user.loc[arrival_user['start_ymd']==start_date,'radgyr_2day'] = mob_item.loc[0,'rad_gyr']
                arrival_user.loc[arrival_user['start_ymd']==start_date,'avrjumplen_2day'] = mob_item.loc[0,'jump_len']
                arrival_user.loc[arrival_user['start_ymd']==start_date,'uncorentro_2day'] = mob_item.loc[0,'uncor_entro']
                arrival_user.loc[arrival_user['start_ymd']==start_date,'realentro_2day'] = mob_item.loc[0,'real_entro']

            # add last third day's mobility features
            start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            prev_date_obj = start_date_obj - delta*3
            prev_date_str = str(prev_date_obj.date())
            mob_item = mob_user[mob_user['start_date']==prev_date_str]
            mob_item.index = range(0,len(mob_item))
            if len(mob_item)!=0:
                arrival_user.loc[arrival_user['start_ymd']==start_date,'top10locfre_3day'] = mob_item.loc[0,'locfre_top10']
                arrival_user.loc[arrival_user['start_ymd']==start_date,'radgyr_3day'] = mob_item.loc[0,'rad_gyr']
                arrival_user.loc[arrival_user['start_ymd']==start_date,'avrjumplen_3day'] = mob_item.loc[0,'jump_len']
                arrival_user.loc[arrival_user['start_ymd']==start_date,'uncorentro_3day'] = mob_item.loc[0,'uncor_entro']
                arrival_user.loc[arrival_user['start_ymd']==start_date,'realentro_3day'] = mob_item.loc[0,'real_entro']

            # add past three days' mean mobility features            
            start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            prev_date_str_3days = []
            for i in range(1,4):
                prev_date_obj = start_date_obj - delta*i
                prev_date_str = str(prev_date_obj.date())
                prev_date_str_3days.append(prev_date_str)
            mob_item = mob_user[mob_user['start_date'].isin(prev_date_str_3days)]
            mob_item.index = range(0,len(mob_item))
            if len(mob_item)!=0:
                if len(mob_item)!=3:
                    print('no 3 last days')
                mob_item = mob_item.mean(axis=0)
                arrival_user.loc[arrival_user['start_ymd']==start_date,'top10locfre_3dayavr'] = mob_item['locfre_top10']
                arrival_user.loc[arrival_user['start_ymd']==start_date,'radgyr_3dayavr'] = mob_item['rad_gyr']
                arrival_user.loc[arrival_user['start_ymd']==start_date,'avrjumplen_3dayavr'] = mob_item['jump_len']
                arrival_user.loc[arrival_user['start_ymd']==start_date,'uncorentro_3dayavr'] = mob_item['uncor_entro']
                arrival_user.loc[arrival_user['start_ymd']==start_date,'realentro_3dayavr'] = mob_item['real_entro']

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
                arrival_user.loc[arrival_user['start_ymd']==start_date,'top10locfre_7day'] = mob_item['locfre_top10']
                arrival_user.loc[arrival_user['start_ymd']==start_date,'radgyr_7day'] = mob_item['rad_gyr']
                arrival_user.loc[arrival_user['start_ymd']==start_date,'avrjumplen_7day'] = mob_item['jump_len']
                arrival_user.loc[arrival_user['start_ymd']==start_date,'uncorentro_7day'] = mob_item['uncor_entro']
                arrival_user.loc[arrival_user['start_ymd']==start_date,'realentro_7day'] = mob_item['real_entro']           

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
                arrival_user.loc[arrival_user['start_ymd']==start_date,'top10locfre_1weekday'] = mob_item['locfre_top10']
                arrival_user.loc[arrival_user['start_ymd']==start_date,'radgyr_1weekday'] = mob_item['rad_gyr']
                arrival_user.loc[arrival_user['start_ymd']==start_date,'avrjumplen_1weekday'] = mob_item['jump_len']
                arrival_user.loc[arrival_user['start_ymd']==start_date,'uncorentro_1weekday'] = mob_item['uncor_entro']
                arrival_user.loc[arrival_user['start_ymd']==start_date,'realentro_1weekday'] = mob_item['real_entro']
                
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
                arrival_user.loc[arrival_user['start_ymd']==start_date,'top10locfre_2weekday'] = mob_item['locfre_top10']
                arrival_user.loc[arrival_user['start_ymd']==start_date,'radgyr_2weekday'] = mob_item['rad_gyr']
                arrival_user.loc[arrival_user['start_ymd']==start_date,'avrjumplen_2weekday'] = mob_item['jump_len']
                arrival_user.loc[arrival_user['start_ymd']==start_date,'uncorentro_2weekday'] = mob_item['uncor_entro']
                arrival_user.loc[arrival_user['start_ymd']==start_date,'realentro_2weekday'] = mob_item['real_entro']
                
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
                arrival_user.loc[arrival_user['start_ymd']==start_date,'top10locfre_3weekday'] = mob_item['locfre_top10']
                arrival_user.loc[arrival_user['start_ymd']==start_date,'radgyr_3weekday'] = mob_item['rad_gyr']
                arrival_user.loc[arrival_user['start_ymd']==start_date,'avrjumplen_3weekday'] = mob_item['jump_len']
                arrival_user.loc[arrival_user['start_ymd']==start_date,'uncorentro_3weekday'] = mob_item['uncor_entro']
                arrival_user.loc[arrival_user['start_ymd']==start_date,'realentro_3weekday'] = mob_item['real_entro']
                
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
                arrival_user.loc[arrival_user['start_ymd']==start_date,'top10locfre_4weekday'] = mob_item['locfre_top10']
                arrival_user.loc[arrival_user['start_ymd']==start_date,'radgyr_4weekday'] = mob_item['rad_gyr']
                arrival_user.loc[arrival_user['start_ymd']==start_date,'avrjumplen_4weekday'] = mob_item['jump_len']
                arrival_user.loc[arrival_user['start_ymd']==start_date,'uncorentro_4weekday'] = mob_item['uncor_entro']
                arrival_user.loc[arrival_user['start_ymd']==start_date,'realentro_4weekday'] = mob_item['real_entro']  
                
        # only keep days with valid mobility features
        arrival_user.index = range(0,len(arrival_user))        
        valid_arr_idx = arrival_user['top10locfre_4weekday'].first_valid_index()
        arrival_user = arrival_user.iloc[valid_arr_idx:,:]
        arrival_user.index = range(0,len(arrival_user))
        
        # save results
        if not os.path.exists(SAVEDATA_PATH):
            os.makedirs(SAVEDATA_PATH) 
        res_path = SAVEDATA_PATH + '/' + str(int(user)) + '_arrival.csv'
        arrival_user.to_csv(res_path, index=False)
        
def construct_arrival_input(userlist, ARRIVAL_PATH, RESULT_PATH):
    """
    Convert arrival features to float numbers in [0, 24].
    
    Paramaters
    ----------
    userlist : list, userlist to extract final inputs
    ARRIVAL_PATH : str, path of arrival+mobility features
    RESULT_PATH : str, path to save final inputs
    
    Returns
    ----------
    N/A
    """
    
    # iterate over all users
    for user in userlist:
        print(user)
        
        arrival_path = ARRIVAL_PATH + '/' + str(int(user)) + '_arrival.csv'
        arrival_user = pd.read_csv(arrival_path)    
        arrival_user['arrival_float'] = ''
        
        string_list = ['24h_at_home', 'last_item_not_at_home', 'date_not_exist']
        for row in range(0,len(arrival_user)): 
            if arrival_user.loc[row, 'arrival'] not in string_list:
                arrival_user.loc[row, 'arrival'] = pd.to_datetime(arrival_user.loc[row, 'arrival'])
                arrival_time = arrival_user.loc[row, 'arrival'].time()
                arrival_time_float = arrival_time.hour + arrival_time.minute/60 + arrival_time.second/3600 + arrival_time.microsecond/(1000*60*3600)
                arrival_user.loc[row, 'arrival_float'] = arrival_time_float
            elif arrival_user.loc[row, 'arrival'] == '24h_at_home':
                arrival_user.loc[row, 'arrival_float'] = 0
            else:
                arrival_user.loc[row, 'arrival_float'] = np.nan
        
        if (arrival_user['arrival_float']>24).sum().sum() != 0:
            print('Error:', (arrival_user['arrival_float']>24).sum(), 'is over 24')
            arrival_user.loc[arrival_user['arrival_float']>24, 'arrival_float'] = 24
                
        if (arrival_user['arrival_float']<0).sum().sum() != 0:
            print('Error:', (arrival_user['arrival_float']<0).sum(), 'is below 0')
            arrival_user.loc[arrival_user['arrival_float']<0, 'arrival_float'] = 0
            
        # save results
        if not os.path.exists(RESULT_PATH):
            os.makedirs(RESULT_PATH)             
        res_path = RESULT_PATH + '/' + str(int(user)) + '_input.csv'
        arrival_user.to_csv(res_path, index=False)        











