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

def extract_depart_target(engine, userlist, savefile_flag, PREPROCESS_PATH, ARRIVAL_PATH):
    """
    Extract daily departrue time for each user.
    
    Paramaters
    ----------
    engine
    userlist : list, userlist to extract targets
    savefile_flag : boolean, flag about whether to save results
    PREPROCESS_PATH : str, path to save results
    ARRIVAL_PATH : str, path of arrival results
    
    Returns
    ----------
    depart_stat : dataframe, statistics of departure time
    """
    
    pandas_query = """SELECT * FROM caihao.stp_cls""" 
    stp_cls = gpd.read_postgis(pandas_query, engine, geom_col='geometry')
    stp_cls['started_at'] = pd.to_datetime(stp_cls['started_at'], utc=True)
    stp_cls['finished_at'] = pd.to_datetime(stp_cls['finished_at'], utc=True)
    stp_cls['started_at_ymd'] = pd.to_datetime(stp_cls['started_at']).dt.date
    stp_cls['finished_at_ymd'] = pd.to_datetime(stp_cls['finished_at']).dt.date
    
    special_dates = {'user_id':[], 'date':[]}
    wrong_dates = {'user_id':[], 'date':[]}
    depart_stat = {'user_id':[], 'valid_days':[], 'nonexist_days':[], 'total':[]} 
        
    # iterate over all users
    for i in range(0,len(userlist)):
        
        dep_user = {'date_id':[], 'finish_ymd':[], 'departure':[], 'day_of_week':[], 'weekend_flag':[], 'day_of_year':[]}
        user = userlist[i]    
        print(user)
        print('-------------START-----------------')
        
        stp_cls_user = stp_cls[stp_cls['user_id']==user].sort_values(by='finished_at',ascending=True)  
        stp_cls_user.index = range(0,len(stp_cls_user))
        
        finish_ymd = sorted(set(list(stp_cls_user['finished_at_ymd'])))
        start_date = min(finish_ymd)
        end_date = max(finish_ymd)
        delta = datetime.timedelta(days=1)
        
        depart_stat['total'].append((end_date-start_date).days+1)
        
        first_not_home_user = []
        date_non_exist_user = []
        j = 0
    
        # iterate over all days
        while start_date <= end_date:
            # print(start_date)
            stp_cls_user_date = stp_cls_user[stp_cls_user['finished_at_ymd']==start_date]       
            date_flag = len(stp_cls_user_date)
            
            ## CASE 1: the fist item of the day is labeled as home
            if date_flag!=0 and stp_cls_user_date['purpose_validated'].iloc[0]=='home': 
                k = 0
                # find next consecutive items labeled as home
                if (date_flag>=2):
                    home_flag = stp_cls_user_date['purpose_validated'].iloc[1]=='home' 
                    while home_flag==True:
                        k = k+1
                        if abs(k+1)<date_flag:
                            home_flag = stp_cls_user_date['purpose_validated'].iloc[k+1]=='home'
                        else:
                            break
                if stp_cls_user_date.index[k] == len(stp_cls_user)-1:
                    dep_index = stp_cls_user_date.index[k]
                else:
                    dep_index = stp_cls_user_date.index[k]+1
                dep_item = stp_cls_user[stp_cls_user.index==dep_index]
                
                if (dep_item['started_at_ymd'].iloc[0]==start_date):   
                    if start_date.weekday() >=5:
                        weekend_flag = 1
                    else:
                        weekend_flag = 0 
                    dep_user['date_id'].append(j)
                    dep_user['finish_ymd'].append(start_date)
                    dep_user['departure'].append(dep_item['started_at'].iloc[0])          
                    dep_user['day_of_week'].append(start_date.weekday())
                    dep_user['weekend_flag'].append(weekend_flag)
                    dep_user['day_of_year'].append(start_date.timetuple().tm_yday)  
                else:
                    if start_date.weekday() >=5:
                        weekend_flag = 1
                    else:
                        weekend_flag = 0 
                    dep_user['date_id'].append(j)
                    dep_user['finish_ymd'].append(start_date)
                    dep_user['departure'].append('next_not_same_day')          
                    dep_user['day_of_week'].append(start_date.weekday())
                    dep_user['weekend_flag'].append(weekend_flag)
                    dep_user['day_of_year'].append(start_date.timetuple().tm_yday)                    
                start_date += delta
                j = j+1
                                            
            ## CASE 2: the first item of the day is not labeled as home, which was treated as invalid cases
            elif date_flag!=0 and stp_cls_user_date['purpose_validated'].iloc[0]!='home':
                date_diff = stp_cls_user_date['finished_at_ymd'].iloc[0] - stp_cls_user_date['started_at_ymd'].iloc[0]
                if date_diff.days == 1:
                    k = 0
                    # find next consecutive items labeled as home
                    if (date_flag>=2):
                        home_flag = stp_cls_user_date['purpose_validated'].iloc[1]=='home' 
                        while home_flag==True:
                            k = k+1
                            if abs(k+1)<date_flag:
                                home_flag = stp_cls_user_date['purpose_validated'].iloc[k+1]=='home'
                            else:
                                break
                    if stp_cls_user_date.index[k] == len(stp_cls_user)-1:
                        dep_index = stp_cls_user_date.index[k]
                    else:
                        dep_index = stp_cls_user_date.index[k]+1
                    dep_item = stp_cls_user[stp_cls_user.index==dep_index]
                                
                    if (dep_item['started_at_ymd'].iloc[0]==start_date):   
                        if start_date.weekday() >=5:
                            weekend_flag = 1
                        else:
                            weekend_flag = 0 
                        dep_user['date_id'].append(j)
                        dep_user['finish_ymd'].append(start_date)
                        dep_user['departure'].append(dep_item['started_at'].iloc[0])  
                        dep_user['day_of_week'].append(start_date.weekday())
                        dep_user['weekend_flag'].append(weekend_flag)
                        dep_user['day_of_year'].append(start_date.timetuple().tm_yday) 
                    
                    else:
                        if start_date.weekday() >=5:
                            weekend_flag = 1
                        else:
                            weekend_flag = 0 
                        dep_user['date_id'].append(j)
                        dep_user['finish_ymd'].append(start_date)
                        dep_user['departure'].append('next_not_same_day')          
                        dep_user['day_of_week'].append(start_date.weekday())
                        dep_user['weekend_flag'].append(weekend_flag)
                        dep_user['day_of_year'].append(start_date.timetuple().tm_yday) 

                else:
                    first_not_home_user.append(stp_cls_user_date.iloc[0])
                    if start_date.weekday() >=5:
                        weekend_flag = 1
                    else:
                        weekend_flag = 0 
                    dep_user['date_id'].append(j)
                    dep_user['finish_ymd'].append(start_date)
                    if stp_cls_user_date['started_at_ymd'].iloc[0]==start_date:
                        dep_user['departure'].append(stp_cls_user_date['started_at'].iloc[0]) 
                    else:
                        dep_user['departure'].append(stp_cls_user_date['finished_at'].iloc[0]) 
                    dep_user['day_of_week'].append(start_date.weekday())
                    dep_user['weekend_flag'].append(weekend_flag)
                    dep_user['day_of_year'].append(start_date.timetuple().tm_yday) 
                        
                start_date += delta
                j = j+1        
            
            elif date_flag==0:
                date_non_exist_user.append(start_date)
                # print(start_date, " do not exist")
                if start_date.weekday() >=5:
                    weekend_flag = 1
                else:
                    weekend_flag = 0 
                dep_user['date_id'].append(j)
                dep_user['finish_ymd'].append(start_date)
                dep_user['departure'].append('date_not_exist')          
                dep_user['day_of_week'].append(start_date.weekday())
                dep_user['weekend_flag'].append(weekend_flag)
                dep_user['day_of_year'].append(start_date.timetuple().tm_yday) 
                start_date += delta
                j = j+1
                
            else:
                special_dates['user_id'].append(user)
                special_dates['date'].append(start_date)
                print('!!! ', start_date, " special things happen")
                start_date += delta
                j = j+1
        
        dep_user = pd.DataFrame(dep_user) 
        dep_user['user_id'] = user 

        ## CASE 3: the user went outside for more than one day
        # combine 24h_at_home information from arrival attributes
        arr_path = ARRIVAL_PATH + '/' + str(int(user)) + '_arrival.csv'
        arr_user = pd.read_csv(arr_path)
        arr_user_24h = arr_user[arr_user['arrival']=='24h_at_home']
        arr_user_24h.index = range(0,len(arr_user_24h))
        arr_user_24h = arr_user_24h.rename(columns={'start_ymd':'finish_ymd', 'arrival':'departure'})
        dep_user = dep_user.astype({'finish_ymd': 'str'})
        valid_dep_user = dep_user[dep_user['departure']!='date_not_exist']
        dep_user_date = list(set(valid_dep_user['finish_ymd']))
        date_exist_24h = []
        
        for r in range(0,len(arr_user_24h)):
            date_24h = arr_user_24h['finish_ymd'].iloc[r]
            if (date_24h in dep_user_date) and (date_24h != str(datetime.date(2017,5,8))):
                wrong_dates['user_id'].append(user)
                wrong_dates['date'].append(date_24h)
                print(date_24h, 'something wrong happens')
            else:
                date_exist_24h.append(date_24h)
                dep_user.loc[dep_user['finish_ymd']==date_24h, 'departure'] = arr_user_24h['departure'].iloc[r]
        
        # save results
        date_non_exist_user = pd.DataFrame(date_non_exist_user)
        if len(date_non_exist_user) > 0:
            date_non_exist_user = date_non_exist_user.astype('str')
            date_non_exist_user = list(set(date_non_exist_user.loc[:,0]) - set(date_exist_24h))
        dep_user = dep_user.sort_values(by='date_id', ascending=True)
        dep_user.index = range(0,len(dep_user))    
        
        first_not_home_user = pd.DataFrame(first_not_home_user)
        date_non_exist_user = pd.DataFrame(date_non_exist_user)
        
        depart_stat['user_id'].append(user)
        depart_stat['valid_days'].append(len(dep_user)-len(date_non_exist_user))    
        depart_stat['nonexist_days'].append(len(date_non_exist_user))

        if savefile_flag == True:
            dep_path = PREPROCESS_PATH + '/' + str(int(user)) + '_depart.csv'
            dep_user.to_csv(dep_path, index=False)
            # first_not_home_path = PREPROCESS_PATH + '/' + str(int(user)) + '_depart_firstnotathome.csv'
            # first_not_home_user.to_csv(first_not_home_path, index=False)
            # date_non_exist_path = PREPROCESS_PATH + '/' + str(int(user)) + '_depart_datenonexist.csv'
            # date_non_exist_user.to_csv(date_non_exist_path, index=False)  
            # next_not_sameday_path = PREPROCESS_PATH + '/' + str(int(user)) + '_depart_nextnotsameday.csv'
            # next_not_sameday_user.to_csv(next_not_sameday_path, index=False)  

        print('---------------END------------------')
        print('------------------------------------')   
    
    depart_stat = pd.DataFrame(depart_stat)
    wrong_dates = pd.DataFrame(wrong_dates)
    special_dates = pd.DataFrame(special_dates)
    
    return depart_stat

def add_depart_mob(userlist, DEPART_PATH, MOB_PATH, SAVEDATA_PATH):
    """
    Add past mobility features on departure features.
    
    Paramaters
    ----------
    userlist : list, userlist to add daily mobility features
    DEPART_PATH : str, path of departure features
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
        
        depart_path = DEPART_PATH + '/' + str(int(user)) + '_depart.csv'
        depart_user = pd.read_csv(depart_path)
            
        # read mobility features
        mob_path = MOB_PATH + '/' + str(int(user)) + '_mob.csv'
        mob_user = pd.read_csv(mob_path)
        mob_user = mob_user.drop(columns='user_id')
        mob_user = mob_user.fillna(0)
        
        depart_user['top10locfre_1day'] = np.nan
        depart_user['top10locfre_2day'] = np.nan
        depart_user['top10locfre_3day'] = np.nan
        depart_user['top10locfre_3dayavr'] = np.nan
        depart_user['top10locfre_7day'] = np.nan
        depart_user['top10locfre_1weekday'] = np.nan
        depart_user['top10locfre_2weekday'] = np.nan
        depart_user['top10locfre_3weekday'] = np.nan
        depart_user['top10locfre_4weekday'] = np.nan

        depart_user['radgyr_1day'] = np.nan
        depart_user['radgyr_2day'] = np.nan
        depart_user['radgyr_3day'] = np.nan
        depart_user['radgyr_3dayavr'] = np.nan
        depart_user['radgyr_7day'] = np.nan
        depart_user['radgyr_1weekday'] = np.nan
        depart_user['radgyr_2weekday'] = np.nan
        depart_user['radgyr_3weekday'] = np.nan
        depart_user['radgyr_4weekday'] = np.nan

        depart_user['avrjumplen_1day'] = np.nan
        depart_user['avrjumplen_2day'] = np.nan
        depart_user['avrjumplen_3day'] = np.nan
        depart_user['avrjumplen_3dayavr'] = np.nan
        depart_user['avrjumplen_7day'] = np.nan
        depart_user['avrjumplen_1weekday'] = np.nan
        depart_user['avrjumplen_2weekday'] = np.nan
        depart_user['avrjumplen_3weekday'] = np.nan
        depart_user['avrjumplen_4weekday'] = np.nan

        depart_user['uncorentro_1day'] = np.nan
        depart_user['uncorentro_2day'] = np.nan
        depart_user['uncorentro_3day'] = np.nan
        depart_user['uncorentro_3dayavr'] = np.nan
        depart_user['uncorentro_7day'] = np.nan
        depart_user['uncorentro_1weekday'] = np.nan
        depart_user['uncorentro_2weekday'] = np.nan
        depart_user['uncorentro_3weekday'] = np.nan
        depart_user['uncorentro_4weekday'] = np.nan

        depart_user['realentro_1day'] = np.nan
        depart_user['realentro_2day'] = np.nan
        depart_user['realentro_3day'] = np.nan
        depart_user['realentro_3dayavr'] = np.nan
        depart_user['realentro_7day'] = np.nan
        depart_user['realentro_1weekday'] = np.nan
        depart_user['realentro_2weekday'] = np.nan
        depart_user['realentro_3weekday'] = np.nan
        depart_user['realentro_4weekday'] = np.nan

        # iterate over all days         
        period = depart_user['finish_ymd'].unique()[:]
        for start_date in period:
            
            # add last day's mobility features
            start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            prev_date_obj = start_date_obj - delta
            prev_date_str = str(prev_date_obj.date())
            mob_item = mob_user[mob_user['start_date']==prev_date_str]
            mob_item.index = range(0,len(mob_item))
            if len(mob_item)!=0:
                depart_user.loc[depart_user['finish_ymd']==start_date,'top10locfre_1day'] = mob_item.loc[0,'locfre_top10']
                depart_user.loc[depart_user['finish_ymd']==start_date,'radgyr_1day'] = mob_item.loc[0,'rad_gyr']
                depart_user.loc[depart_user['finish_ymd']==start_date,'avrjumplen_1day'] = mob_item.loc[0,'jump_len']
                depart_user.loc[depart_user['finish_ymd']==start_date,'uncorentro_1day'] = mob_item.loc[0,'uncor_entro']
                depart_user.loc[depart_user['finish_ymd']==start_date,'realentro_1day'] = mob_item.loc[0,'real_entro']

            start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            prev_date_obj = start_date_obj - delta
            prev_date_str = str(prev_date_obj.date())

            # add last second day's mobility features
            start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            prev_date_obj = start_date_obj - delta*2
            prev_date_str = str(prev_date_obj.date())
            mob_item = mob_user[mob_user['start_date']==prev_date_str]
            mob_item.index = range(0,len(mob_item))
            if len(mob_item)!=0:
                depart_user.loc[depart_user['finish_ymd']==start_date,'top10locfre_2day'] = mob_item.loc[0,'locfre_top10']
                depart_user.loc[depart_user['finish_ymd']==start_date,'radgyr_2day'] = mob_item.loc[0,'rad_gyr']
                depart_user.loc[depart_user['finish_ymd']==start_date,'avrjumplen_2day'] = mob_item.loc[0,'jump_len']
                depart_user.loc[depart_user['finish_ymd']==start_date,'uncorentro_2day'] = mob_item.loc[0,'uncor_entro']
                depart_user.loc[depart_user['finish_ymd']==start_date,'realentro_2day'] = mob_item.loc[0,'real_entro']
                
            # add last third day's mobility features          
            start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            prev_date_obj = start_date_obj - delta*3
            prev_date_str = str(prev_date_obj.date())
            mob_item = mob_user[mob_user['start_date']==prev_date_str]
            mob_item.index = range(0,len(mob_item))
            if len(mob_item)!=0:
                depart_user.loc[depart_user['finish_ymd']==start_date,'top10locfre_3day'] = mob_item.loc[0,'locfre_top10']
                depart_user.loc[depart_user['finish_ymd']==start_date,'radgyr_3day'] = mob_item.loc[0,'rad_gyr']
                depart_user.loc[depart_user['finish_ymd']==start_date,'avrjumplen_3day'] = mob_item.loc[0,'jump_len']
                depart_user.loc[depart_user['finish_ymd']==start_date,'uncorentro_3day'] = mob_item.loc[0,'uncor_entro']
                depart_user.loc[depart_user['finish_ymd']==start_date,'realentro_3day'] = mob_item.loc[0,'real_entro']
                
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
                depart_user.loc[depart_user['finish_ymd']==start_date,'top10locfre_3dayavr'] = mob_item['locfre_top10']
                depart_user.loc[depart_user['finish_ymd']==start_date,'radgyr_3dayavr'] = mob_item['rad_gyr']
                depart_user.loc[depart_user['finish_ymd']==start_date,'avrjumplen_3dayavr'] = mob_item['jump_len']
                depart_user.loc[depart_user['finish_ymd']==start_date,'uncorentro_3dayavr'] = mob_item['uncor_entro']
                depart_user.loc[depart_user['finish_ymd']==start_date,'realentro_3dayavr'] = mob_item['real_entro']

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
                depart_user.loc[depart_user['finish_ymd']==start_date,'top10locfre_7day'] = mob_item['locfre_top10']
                depart_user.loc[depart_user['finish_ymd']==start_date,'radgyr_7day'] = mob_item['rad_gyr']
                depart_user.loc[depart_user['finish_ymd']==start_date,'avrjumplen_7day'] = mob_item['jump_len']
                depart_user.loc[depart_user['finish_ymd']==start_date,'uncorentro_7day'] = mob_item['uncor_entro']
                depart_user.loc[depart_user['finish_ymd']==start_date,'realentro_7day'] = mob_item['real_entro']           

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
                depart_user.loc[depart_user['finish_ymd']==start_date,'top10locfre_1weekday'] = mob_item['locfre_top10']
                depart_user.loc[depart_user['finish_ymd']==start_date,'radgyr_1weekday'] = mob_item['rad_gyr']
                depart_user.loc[depart_user['finish_ymd']==start_date,'avrjumplen_1weekday'] = mob_item['jump_len']
                depart_user.loc[depart_user['finish_ymd']==start_date,'uncorentro_1weekday'] = mob_item['uncor_entro']
                depart_user.loc[depart_user['finish_ymd']==start_date,'realentro_1weekday'] = mob_item['real_entro']
                
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
                depart_user.loc[depart_user['finish_ymd']==start_date,'top10locfre_2weekday'] = mob_item['locfre_top10']
                depart_user.loc[depart_user['finish_ymd']==start_date,'radgyr_2weekday'] = mob_item['rad_gyr']
                depart_user.loc[depart_user['finish_ymd']==start_date,'avrjumplen_2weekday'] = mob_item['jump_len']
                depart_user.loc[depart_user['finish_ymd']==start_date,'uncorentro_2weekday'] = mob_item['uncor_entro']
                depart_user.loc[depart_user['finish_ymd']==start_date,'realentro_2weekday'] = mob_item['real_entro']
                
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
                depart_user.loc[depart_user['finish_ymd']==start_date,'top10locfre_3weekday'] = mob_item['locfre_top10']
                depart_user.loc[depart_user['finish_ymd']==start_date,'radgyr_3weekday'] = mob_item['rad_gyr']
                depart_user.loc[depart_user['finish_ymd']==start_date,'avrjumplen_3weekday'] = mob_item['jump_len']
                depart_user.loc[depart_user['finish_ymd']==start_date,'uncorentro_3weekday'] = mob_item['uncor_entro']
                depart_user.loc[depart_user['finish_ymd']==start_date,'realentro_3weekday'] = mob_item['real_entro']
                
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
                depart_user.loc[depart_user['finish_ymd']==start_date,'top10locfre_4weekday'] = mob_item['locfre_top10']
                depart_user.loc[depart_user['finish_ymd']==start_date,'radgyr_4weekday'] = mob_item['rad_gyr']
                depart_user.loc[depart_user['finish_ymd']==start_date,'avrjumplen_4weekday'] = mob_item['jump_len']
                depart_user.loc[depart_user['finish_ymd']==start_date,'uncorentro_4weekday'] = mob_item['uncor_entro']
                depart_user.loc[depart_user['finish_ymd']==start_date,'realentro_4weekday'] = mob_item['real_entro']  
                
        # only keep days with valid mobility features
        depart_user.index = range(0,len(depart_user))        
        valid_dep_idx = depart_user['top10locfre_4weekday'].first_valid_index()
        depart_user = depart_user.iloc[valid_dep_idx:,:]
        depart_user.index = range(0,len(depart_user))
        
        # save results
        if not os.path.exists(SAVEDATA_PATH):
            os.makedirs(SAVEDATA_PATH)                 
        res_path = SAVEDATA_PATH + '/' + str(int(user)) + '_depart.csv'
        depart_user.to_csv(res_path, index=False)


def construct_depart_input(userlist, DEPART_PATH, RESULT_PATH):  
    """
    Convert departure features to float numbers in [0, 24].
    
    Paramaters
    ----------
    userlist : list, userlist to extract final inputs
    DEPART_PATH : str, path of departure+mobility features
    RESULT_PATH : str, path to save final inputs
    
    Returns
    ----------
    N/A 
    """
    
    # iterate over all users
    for user in userlist:
        print(user)
        
        depart_path = DEPART_PATH + '/' + str(int(user)) + '_depart.csv'
        depart_user = pd.read_csv(depart_path)    
        depart_user['depart_float'] = ''
        depart_user = depart_user.rename(columns={'departure':'depart'})
        
        string_list = ['24h_at_home', 'next_not_same_day', 'date_not_exist']
        for row in range(0,len(depart_user)):
            if depart_user.loc[row, 'depart'] not in string_list:
                depart_user.loc[row, 'depart'] = pd.to_datetime(depart_user.loc[row, 'depart'])
                depart_time = depart_user.loc[row, 'depart'].time()
                depart_time_float = depart_time.hour + depart_time.minute/60 + depart_time.second/3600 + depart_time.microsecond/(1000*60*3600)
                depart_user.loc[row, 'depart_float'] = depart_time_float
            elif depart_user.loc[row, 'depart'] in ['24h_at_home', 'next_not_same_day']:
                depart_user.loc[row, 'depart_float'] = 24
            else:
                depart_user.loc[row, 'depart_float'] = np.nan
        
        if (depart_user['depart_float']>24).sum().sum() != 0:
            print('Error:', (depart_user['depart_float']>24).sum(), 'is over 24')
            depart_user.loc[depart_user['depart_float']>24, 'depart_float'] = 24
                
        if (depart_user['depart_float']<0).sum().sum() != 0:
            print('Error:', (depart_user['depart_float']<0).sum(), 'is below 0')
            depart_user.loc[depart_user['depart_float']<0, 'depart_float'] = 0
        
        if not os.path.exists(RESULT_PATH):
            os.makedirs(RESULT_PATH)             
        res_path = RESULT_PATH + '/' + str(int(user)) + '_input.csv'
        depart_user.to_csv(res_path, index=False)        
 