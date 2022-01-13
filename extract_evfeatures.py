# -*- coding: utf-8 -*-
"""
Created on Tue Jun 2021 
@author: Haojun Cai
"""

import os 
import pandas as pd
import geopandas as gpd
import haversine as hs
import datetime
import numpy as np
from haversine import Unit

def extract_hhindex_daily(triplegdf, userlist, RESULT_PATH):
    """
    Extract daily hhindex and evhhidex features.
    
    Paramaters
    ----------
    triplegdf : dataframe, tripleg data
    userlist : list, userlist to extract daily mobility features
    RESULT_PATH : str, path to save hhindex results
    
    Returns
    ----------
    N/A
    """
    
    # convert timezone to utc ones
    triplegdf['started_at'] = pd.to_datetime(triplegdf['started_at'], utc=True)
    triplegdf['finished_at'] = pd.to_datetime(triplegdf['finished_at'], utc=True)
    
    triplegdf['duration'] = triplegdf['finished_at'] - triplegdf['started_at']
    triplegdf['started_at_ymd'] = pd.to_datetime(triplegdf['started_at']).dt.date
    triplegdf['finished_at_ymd'] = pd.to_datetime(triplegdf['finished_at']).dt.date
    
    # iterate over all users
    for user in userlist:
        print(user)
    
        triplegdf_user = triplegdf[triplegdf['user_id']==user].sort_values(by='started_at', ascending=True)
        triplegdf_user.index = range(0,len(triplegdf_user))
        
        date = list(set(triplegdf_user['started_at_ymd']))  
        start_date = min(date)
        end_date = max(date)
        delta = datetime.timedelta(days=1)
        
        hhindex_stat = {'date':[], 'hhindex':[], 'ecar_hhindex':[]}
        
        # iterate over all days
        while start_date <= end_date:
            hhindex_stat['date'].append(start_date)
            triplegdf_user_date = triplegdf_user[triplegdf_user['started_at_ymd']==start_date]
            triplegdf_user_date.index = range(0,len(triplegdf_user_date))
            exist_flag = len(triplegdf_user_date)
            
            all_mode_seconds = []
            if exist_flag != 0:
                total_secs = triplegdf_user_date['duration'].sum().total_seconds()
                all_modes = list(set(triplegdf_user_date['mode_validated']))
                ecar_seconds = 0
                
                for mode in all_modes:
                    triplegdf_user_date_mode = triplegdf_user_date[triplegdf_user_date['mode_validated']==mode]
                    mode_seconds = triplegdf_user_date_mode['duration'].sum().total_seconds()
                    all_mode_seconds.append(mode_seconds)
                    
                    if mode == 'Mode::Car':
                        ecar_seconds = mode_seconds
                
                all_mode_seconds = np.array(all_mode_seconds) / total_secs
                hhindex_date = (all_mode_seconds ** 2).sum()
                ecar_hhindex_date = (ecar_seconds/total_secs)** 2 / hhindex_date
                if hhindex_date>1 or hhindex_date<0 or ecar_hhindex_date>1 or ecar_hhindex_date<0:
                    print('Error: hhindex cannot be above 1 or lower 0.')
                hhindex_stat['hhindex'].append(hhindex_date)
                hhindex_stat['ecar_hhindex'].append(ecar_hhindex_date)

            else:
                hhindex_stat['hhindex'].append(np.nan)
                hhindex_stat['ecar_hhindex'].append(np.nan)
        
            start_date += delta    
        
        # save results
        hhindex_stat = pd.DataFrame(hhindex_stat)
        
        if not os.path.exists(RESULT_PATH):
            os.makedirs(RESULT_PATH)
            
        hhindex_folder = RESULT_PATH + '/hhindex'
        if not os.path.exists(hhindex_folder):
            os.makedirs(hhindex_folder)   
        hhindex_path = hhindex_folder + '/' + str(int(user)) + '_hhindex.csv'
        hhindex_stat.to_csv(hhindex_path, index=False)

def preprocess_bmw(save_folder, engine):
    """
    Preprocess bmw data to calculate distance and duration.
    
    Paramaters
    ----------
    save_folder : str, folder to save processed bmw features
    engine
        
    Returns
    ----------
    N/A
    """
    
    # preprocess bmw data
    pandas_query = """SELECT * FROM version_20181213.bmw""" 
    bmwgdf_start = gpd.read_postgis(pandas_query, engine, geom_col='geom_start')
    bmwgdf_end = gpd.read_postgis(pandas_query, engine, geom_col='geom_end')
    bmwgdf = bmwgdf_start.copy()
    bmwgdf['geom_end'] = bmwgdf_end['geom_end']
    bmwgdf['geom_start'] = bmwgdf_start['geom_start']
    bmwgdf['timestamp_start_utc'] = pd.to_datetime(bmwgdf['timestamp_start_utc'], utc=True)
    bmwgdf['timestamp_end_utc'] = pd.to_datetime(bmwgdf['timestamp_end_utc'], utc=True)
    
    # calculate distance
    lon1 = np.array(bmwgdf['geom_start'].x)
    lat1 = np.array(bmwgdf['geom_start'].y)
    lon2 = np.array(bmwgdf['geom_end'].x)
    lat2 = np.array(bmwgdf['geom_end'].y)
    for i in range(0,len(bmwgdf)):
        dist = hs.haversine((lat1[i],lon1[i]), (lat2[i],lon2[i]), unit=Unit.METERS)
        bmwgdf.loc[i,'dist'] = dist
    
    # save data
    bmwgdf.to_csv(save_folder+'/bmw_process.csv',index=False)
    
def extract_evstat_daily(bmwdf, userlist, RESULT_PATH):  
    """
    Extract daily ecarduration and ecardistance features.
    
    Paramaters
    ----------
    bmwdf : dataframe, bmw data
    userlist : list, userlist to extract daily mobility features
    RESULT_PATH : str, path to save hhindex results
    
    Returns
    ----------
    N/A
    """
     
    # convert timezone to utc ones
    bmwdf['started_at_ymd'] = pd.to_datetime(bmwdf['timestamp_start_utc']).dt.date
    bmwdf['finished_at_ymd'] = pd.to_datetime(bmwdf['timestamp_end_utc']).dt.date
    bmwdf['timestamp_start_utc'] = pd.to_datetime(bmwdf['timestamp_start_utc'], utc=True)
    bmwdf['timestamp_end_utc'] = pd.to_datetime(bmwdf['timestamp_end_utc'], utc=True)
    
    # calculate relevant features
    bmwdf['soc_diff'] = bmwdf['soc_customer_start'] - bmwdf['soc_customer_end']
    bmwdf['duration'] = bmwdf['timestamp_end_utc'] - bmwdf['timestamp_start_utc']
    bmwdf = bmwdf[bmwdf['soc_diff']>0]
    
    # iterate over all users
    for user in userlist:
        print(user)
        evstat = {'date':[], 'duration':[], 'dist':[]}
        bmwdf_user = bmwdf[bmwdf['user_id']==user].sort_values(by='timestamp_start_utc', ascending=True)
        bmwdf_user.index = range(0,len(bmwdf_user))
        
        date = list(set(bmwdf_user['started_at_ymd']))  
        start_date = min(date)
        end_date = max(date)
        delta = datetime.timedelta(days=1)
        
        # iterate over all days
        while start_date <= end_date:
            evstat['date'].append(start_date)
            bmwdf_user_date = bmwdf_user[bmwdf_user['started_at_ymd']==start_date]
            bmwdf_user_date.index = range(0,len(bmwdf_user_date))
            exist_flag = len(bmwdf_user_date)
            
            if exist_flag != 0:
                total_secs = bmwdf_user_date['duration'].sum().total_seconds()
                total_dist = bmwdf_user_date['dist'].sum()
                evstat['duration'].append(total_secs)
                evstat['dist'].append(total_dist)

            else:
                evstat['duration'].append(np.nan)
                evstat['dist'].append(np.nan)
        
            start_date += delta    
        
        # save results
        evstat = pd.DataFrame(evstat)
        evstat_folder = RESULT_PATH + '/evstat'
        if not os.path.exists(evstat_folder):
            os.makedirs(evstat_folder)   
        evstat_path = evstat_folder + '/' + str(int(user)) + '_EVStat.csv'
        evstat.to_csv(evstat_path, index=False)


















