# -*- coding: utf-8 -*-
"""
Created on Jun 2021
@author: Haojun Cai  
"""

import os
import numpy as np
import pandas as pd
import datetime
import trackintel
from trackintel.preprocessing.staypoints import generate_locations
from trackintel_downloaded import read_staypoints_gpd
import skmob
from skmob.measures.individual import location_frequency, jump_lengths, radius_of_gyration, k_radius_of_gyration, random_entropy, uncorrelated_entropy, real_entropy

def preprocess_staypoints(bmwdf, stpgdf, mob_folder):
    """
    Preprocess staypoints data in preparation to extract mobility features
    
    Paramaters
    ----------
    bmwdf : dataframe, bmw data
    stpgdf : geodataframe, staypoints data
    mob_folder : str, folder to save mobility features
    
    Returns
    ----------
    N/A
    """
    
    # only keep common users
    bmwdf_userid = bmwdf['user_id'].value_counts()[:].index.tolist()
    stpgdf = stpgdf[stpgdf['user_id'].isin(bmwdf_userid)]

    # DBSCAN clustering on user level
    stp_trackintel = read_staypoints_gpd(stpgdf, started_at='started_at', finished_at='finished_at', user_id="user_id", geom="geometry_raw")
    [stp_cls, loc_cls] = generate_locations(stp_trackintel, method='dbscan', epsilon=100, num_samples=1, agg_level='user')

    stp_cls = stp_cls.drop(columns=['geometry']) 
    stp_cls.rename(columns={'geom':'geometry'}, inplace=True)
    
    # join two returned datasets
    stp_cls = stp_cls.join(loc_cls, on="location_id", lsuffix='user_id')
    stp_cls = stp_cls.drop(columns=['user_id'])
    stp_cls.rename(columns={'user_iduser_id':'user_id'}, inplace=True)
    stp_cls['lat_loc'] = stp_cls['center'].y
    stp_cls['lon_loc'] = stp_cls['center'].x
    stp_cls['started_at_ymd'] = pd.to_datetime(stp_cls['started_at']).dt.date
    stp_cls['finished_at_ymd'] = pd.to_datetime(stp_cls['finished_at']).dt.date
    
    # save results
    if not os.path.exists(mob_folder):
        os.makedirs(mob_folder) 
    stp_cls.to_csv(mob_folder+'/stp_cls.csv', index=False)
    loc_cls.to_csv(mob_folder+'/loc_cls.csv', index=False)
    

def extract_mobility_daily(stp_cls, userlist, PREPROCESS_PATH):
    """
    Extract daily mobility features for each user on clustered staypoints data.
    
    Paramaters
    ----------
    stp_cls : dataframe, clustered staypoints data
    userlist : list, userlist to extract daily mobility features
    PREPROCESS_PATH : str, path to save daily mobility features for each user
    
    Returns
    ----------
    N/A
    """
    
    for user in userlist:
        print(user)
        
        # find clustered staypoints for the user
        stp_cls_user = stp_cls[stp_cls['user_id']==user].sort_values(by='started_at', ascending=True)
        stp_cls_user.index = range(0,len(stp_cls_user))
        # find top 10 locations for the user
        topk_user = stp_cls_user['location_id'].value_counts()[:10].index.tolist()
        
        date = list(set(stp_cls_user['started_at_ymd']))   
        start_date = min(date)
        end_date = max(date)
        start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        
        delta = datetime.timedelta(days=1)
        
        mob_user = {'start_date':[],'locfre_top10':[],'rad_gyr':[],'jump_len':[],'rand_entro':[],'uncor_entro':[],'real_entro':[]}
        
        # add start_date by two days for temporal resolution of three days
        start_date = start_date + delta*2
        
        # iterature dates to extract daily mobility features for the user
        while start_date <= end_date:
            temp_res_period = [datetime.datetime.strftime(start_date,'%Y-%m-%d'), 
                               datetime.datetime.strftime(start_date-delta,'%Y-%m-%d'), 
                               datetime.datetime.strftime(start_date-delta*2,'%Y-%m-%d')]
            stp_cls_user_date = stp_cls_user[stp_cls_user['started_at_ymd'].isin(temp_res_period)]
            
            exist_flag = len(stp_cls_user_date)
            
            if exist_flag != 0:
                # calculate top-10 location frequency
                num_topk_user_date = len(stp_cls_user_date[stp_cls_user_date['location_id'].isin(topk_user)])
                num_user_date = len(stp_cls_user_date)
                fre_topk_user_date = num_topk_user_date / num_user_date
                
                # prepare dataset in skmob package format to calculate other mobility features
                stp_cls_skmob = skmob.TrajDataFrame(stp_cls_user_date, 
                                            latitude='lat_loc', 
                                            longitude='lon_loc', 
                                            datetime='started_at', 
                                            user_id='user_id')
                
                # calculate average daily jump distance
                jump_len = jump_lengths(stp_cls_skmob, merge=False, show_progress=False) 
                if len(jump_len)!=0:
                    for i in jump_len.index:
                        jump_len.loc[i,'mean_jumplen'] = np.mean(jump_len.loc[i,'jump_lengths'])        
                
                # calculate radius of gyration
                rad_gyr = radius_of_gyration(stp_cls_skmob, show_progress=False)        
                
                # calculate daily random entropy
                rand_entro = random_entropy(stp_cls_skmob, show_progress=False)
                
                # calculate daily uncorrelated entropy
                uncor_entro = uncorrelated_entropy(stp_cls_skmob, show_progress=False)    
                
                # calculate daily real entropy
                real_entro = real_entropy(stp_cls_skmob, show_progress=False)
                
                mob_user['start_date'].append(start_date)
                mob_user['locfre_top10'].append(fre_topk_user_date)
                mob_user['jump_len'].append(jump_len.loc[0,'mean_jumplen'])
                mob_user['rad_gyr'].append(rad_gyr.loc[0,'radius_of_gyration'])
                mob_user['rand_entro'].append(rand_entro.loc[0,'random_entropy'])
                mob_user['uncor_entro'].append(uncor_entro.loc[0,'uncorrelated_entropy'])
                mob_user['real_entro'].append(real_entro.loc[0,'real_entropy'])
            
            else:
                mob_user['start_date'].append(start_date)
                mob_user['locfre_top10'].append(np.nan)
                mob_user['jump_len'].append(np.nan)
                mob_user['rad_gyr'].append(np.nan)
                mob_user['rand_entro'].append(np.nan)
                mob_user['uncor_entro'].append(np.nan)
                mob_user['real_entro'].append(np.nan)
                
            start_date += delta
        
        # save data
        mob_user = pd.DataFrame(mob_user)
        mob_user['user_id'] = user
        if not os.path.exists(PREPROCESS_PATH):
            os.makedirs(PREPROCESS_PATH) 
        mob_path = PREPROCESS_PATH + '/' + str(int(user)) + '_mob.csv'
        mob_user.to_csv(mob_path, index=False)



