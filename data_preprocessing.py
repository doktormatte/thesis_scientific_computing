import json
import datetime
import pandas as pd
import numpy as np
from scipy import stats
from workalendar.usa import California
from workalendar.usa import Colorado
import sys

# round timestamp to nearest 15-minute slot
def conv_timestamp(ts):
    time_arr = str(ts)[-8:].split(':')
    hours = int(time_arr[0])
    mins = int(time_arr[1])
    return (hours*60+mins)//15

def get_day_of_week(ts):
    return ts.weekday()

def get_day_of_month(ts):
    return ts.day

def get_day_of_year(ts):
    return ts.timetuple().tm_yday

def get_weekend(ts):
    if ts.weekday() > 4:
        return 1
    return 0

# create California holiday dummy 
def get_holiday_cal(ts):
    year = ts.year
    cal = California()
    holidays = cal.holidays(year)
    for holiday in holidays:
        if holiday[0] == datetime.date.fromtimestamp(datetime.datetime.timestamp(ts)):
            return 1
    return 0    

# create Colorado holiday dummy 
def get_holiday_col(ts):
    year = ts.year
    cal = Colorado()
    holidays = cal.holidays(year)
    for holiday in holidays:
        if holiday[0] == datetime.date.fromtimestamp(datetime.datetime.timestamp(ts)):
            return 1
    return 0  


# functions for cyclic encoding
def get_sin(x, x_max):
    return np.sin(2.0*np.pi*x/x_max)
def get_cos(x, x_max):
    return np.cos(2.0*np.pi*x/x_max)

# central function that generates time series data
def add_to_backbones(row, dataset, stat_name):    
    iters_energy = int(row['charging_duration']//15)  
    delta_occup = row['disconnect_time'] - row['connect_time']
    iters_occup = int(round(delta_occup.total_seconds()/60.0)/15.0)        
    backbone_energy = meta_dict[dataset][stat_name]['backbones'][0]
    for i in range(iters_energy):
        backbone_energy.loc[backbone_energy['date_time'] == row['connect_time'] + datetime.timedelta(minutes=15*i), 'value'] += row['energy_per_quarter']                 
    backbone_occup = meta_dict[dataset][stat_name]['backbones'][1]
    for i in range(iters_occup):
        backbone_occup.loc[backbone_occup['date_time'] == row['disconnect_time'] + datetime.timedelta(minutes=15*i), 'value'] = 1.0   


# define import and export paths
import_path = 'path/to/original_event_based_datasets'
export_path = 'path/to/export_generated_time_series'


datasets = ['acn_caltech', 'acn_jpl', 'boulder', 'palo_alto']
# datasets = ['boulder', 'palo_alto']
# datasets = ['acn_caltech','acn_jpl']

meta_data = ['start', 'end', 'calendar']
val_splits = [4]



meta_dict = dict.fromkeys(datasets)

for dataset in datasets:
    meta_dict[dataset] = dict.fromkeys(meta_data)
    
if 'acn_caltech' in datasets:
    meta_dict['acn_caltech']['start'] = datetime.datetime.strptime('4/10/2018 00:00', '%m/%d/%Y %H:%M')
    meta_dict['acn_caltech']['end'] = datetime.datetime.strptime('3/12/2019 15:45', '%m/%d/%Y %H:%M')
    meta_dict['acn_caltech']['calendar'] = California()
    meta_dict['acn_caltech']['max_test'] = 0.0
    meta_dict['acn_caltech']['min_test'] = 0.0    

if 'acn_jpl' in datasets:
    meta_dict['acn_jpl']['start'] = datetime.datetime.strptime('10/07/2018 00:00', '%m/%d/%Y %H:%M')
    meta_dict['acn_jpl']['end'] = datetime.datetime.strptime('8/13/2019 11:45', '%m/%d/%Y %H:%M')
    meta_dict['acn_jpl']['calendar'] = California()
    meta_dict['acn_jpl']['max_test'] = 0.0
    meta_dict['acn_jpl']['min_test'] = 0.0

if 'palo_alto' in datasets:
    meta_dict['palo_alto']['start'] = datetime.datetime.strptime('8/1/2017 00:00', '%m/%d/%Y %H:%M')
    meta_dict['palo_alto']['end'] = datetime.datetime.strptime('3/1/2020 07:45', '%m/%d/%Y %H:%M')
    meta_dict['palo_alto']['calendar'] = California()
    meta_dict['palo_alto']['max_test'] = 0.0
    meta_dict['palo_alto']['min_test'] = 0.0

if 'boulder' in datasets:
    meta_dict['boulder']['start'] = datetime.datetime.strptime('2019/01/01 00:00:00', '%Y/%m/%d %H:%M:%S')
    meta_dict['boulder']['end'] = datetime.datetime.strptime('2020/03/01 19:45:00', '%Y/%m/%d %H:%M:%S')
    meta_dict['boulder']['calendar'] = Colorado()
    meta_dict['boulder']['max_test'] = 0.0
    meta_dict['boulder']['min_test'] = 0.0


for dataset in datasets:
    df = pd.read_csv(import_path + dataset + '.csv')
    stations = list(set(list(df['station_id'])))    
    for stat_name in stations:
        meta_dict[dataset][stat_name] = {}

for dataset in datasets:
    df = pd.read_csv(import_path + dataset + '.csv')    
    
    # remove outliers
    df['z_score_tot'] = np.abs(stats.zscore(df['total_duration']))
    df = df[df.z_score_tot <= 3.0]
    df['z_score_ene_quart'] = np.abs(stats.zscore(df['energy_per_quarter']))
    df = df[df.z_score_ene_quart <= 3.0]
    
    # set up dictionary of empty time series dataframes    
    df.connect_time = df.connect_time.map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    df.disconnect_time = df.disconnect_time.map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    stations = list(set(list(df['station_id'])))    
    stat_backbones = dict.fromkeys(stations)        
    start = meta_dict[dataset]['start']
    end = meta_dict[dataset]['end']
    
    
    for stat_name in stations:        
        
        # add exogenous date-related variables to energy and occupancy time series
        backbone_energy = pd.DataFrame({'date_time': pd.date_range(start, end, freq='15min')})
        backbone_energy.set_index('date_time')            
        backbone_energy.to_csv(export_path + dataset + '/energy/backbone.csv', encoding='utf-8')
        backbone_energy.to_csv(export_path + dataset + '/occup/backbone.csv', encoding='utf-8')
        backbone_energy['timeslot'] = backbone_energy['date_time'].apply(conv_timestamp)
        max_timeslot = max(backbone_energy['timeslot'])
        backbone_energy['day_of_week'] = backbone_energy['date_time'].apply(get_day_of_week)
        max_day_of_week = max(backbone_energy['day_of_week'])
        backbone_energy['day_of_month'] = backbone_energy['date_time'].apply(get_day_of_month)
        max_day_of_month = max(backbone_energy['day_of_month'])
        backbone_energy['day_of_year'] = backbone_energy['date_time'].apply(get_day_of_year)
        max_day_of_year = max(backbone_energy['day_of_year'])
        backbone_energy['weekend'] = backbone_energy['date_time'].apply(get_weekend)        
        if dataset == 'boulder':            
            backbone_energy['holiday'] = backbone_energy['date_time'].apply(get_holiday_col)
        else:
            backbone_energy['holiday'] = backbone_energy['date_time'].apply(get_holiday_cal)          
        backbone_energy['timeslot_sin'] = backbone_energy.apply(lambda x: get_sin(x['timeslot'], max_timeslot),axis=1)
        backbone_energy['timeslot_cos'] = backbone_energy.apply(lambda x: get_cos(x['timeslot'], max_timeslot),axis=1)
        backbone_energy['day_of_week_sin'] = backbone_energy.apply(lambda x: get_sin(x['day_of_week'], max_day_of_week),axis=1)
        backbone_energy['day_of_week_cos'] = backbone_energy.apply(lambda x: get_cos(x['day_of_week'], max_day_of_week),axis=1)
        backbone_energy['day_of_month_sin'] = backbone_energy.apply(lambda x: get_sin(x['day_of_month'], max_day_of_month),axis=1)
        backbone_energy['day_of_month_cos'] = backbone_energy.apply(lambda x: get_cos(x['day_of_month'], max_day_of_month),axis=1)    
        backbone_energy['day_of_year_sin'] = backbone_energy.apply(lambda x: get_sin(x['day_of_year'], max_day_of_year),axis=1)
        backbone_energy['day_of_year_cos'] = backbone_energy.apply(lambda x: get_cos(x['day_of_year'], max_day_of_year),axis=1)        
        backbone_energy['value'] = 0.0          
        
        backbone_occup = pd.DataFrame({'date_time': pd.date_range(start, end, freq='15min')})
        backbone_occup.set_index('date_time')        
        backbone_occup['timeslot'] = backbone_occup['date_time'].apply(conv_timestamp)
        max_timeslot = max(backbone_occup['timeslot'])
        backbone_occup['day_of_week'] = backbone_occup['date_time'].apply(get_day_of_week)
        max_day_of_week = max(backbone_occup['day_of_week'])
        backbone_occup['day_of_month'] = backbone_occup['date_time'].apply(get_day_of_month)
        max_day_of_month = max(backbone_occup['day_of_month'])
        backbone_occup['day_of_year'] = backbone_occup['date_time'].apply(get_day_of_year)
        max_day_of_year = max(backbone_occup['day_of_year'])
        backbone_occup['weekend'] = backbone_occup['date_time'].apply(get_weekend) 
        if dataset == 'boulder':            
            backbone_occup['holiday'] = backbone_occup['date_time'].apply(get_holiday_col)
        else:
            backbone_occup['holiday'] = backbone_occup['date_time'].apply(get_holiday_cal)         
        backbone_occup['timeslot_sin'] = backbone_occup.apply(lambda x: get_sin(x['timeslot'], max_timeslot),axis=1)
        backbone_occup['timeslot_cos'] = backbone_occup.apply(lambda x: get_cos(x['timeslot'], max_timeslot),axis=1)
        backbone_occup['day_of_week_sin'] = backbone_occup.apply(lambda x: get_sin(x['day_of_week'], max_day_of_week),axis=1)
        backbone_occup['day_of_week_cos'] = backbone_occup.apply(lambda x: get_cos(x['day_of_week'], max_day_of_week),axis=1)
        backbone_occup['day_of_month_sin'] = backbone_occup.apply(lambda x: get_sin(x['day_of_month'], max_day_of_month),axis=1)
        backbone_occup['day_of_month_cos'] = backbone_occup.apply(lambda x: get_cos(x['day_of_month'], max_day_of_month),axis=1)
        backbone_occup['day_of_year_sin'] = backbone_occup.apply(lambda x: get_sin(x['day_of_year'], max_day_of_year),axis=1)
        backbone_occup['day_of_year_cos'] = backbone_occup.apply(lambda x: get_cos(x['day_of_year'], max_day_of_year),axis=1)
        backbone_occup['value'] = 0             
        
        meta_dict[dataset][stat_name]['backbones'] = [backbone_energy, backbone_occup]   
        print('Done ' + dataset + ': ' + stat_name)    

    print('\n')
    
    backbone_num = 1
    for stat_name in stations:        
        train_test_split = 0.75
        df_stat = df[df['station_id'] == stat_name]         
        df_stat.apply(lambda row: add_to_backbones(row, dataset, stat_name), axis=1)                
        backbone_energy = meta_dict[dataset][stat_name]['backbones'][0]              
        n_train = round(train_test_split*len(backbone_energy)) 
        backbone_energy_train = backbone_energy[:n_train]
        backbone_energy_test = backbone_energy[n_train:]
        meta_dict[dataset][stat_name]['backbone_energy_train'] = backbone_energy_train
        meta_dict[dataset][stat_name]['backbone_energy_test'] = backbone_energy_test        
        backbone_energy_train['value_shifted'] = np.roll(backbone_energy_train['value'],-1)        
        backbone_energy_train.drop(labels=['date_time','timeslot','day_of_week','day_of_month','day_of_year'],axis=1).to_csv(export_path + dataset + '/energy/' + str(backbone_num) + '_train.csv', encoding='utf-8', index=False)
        backbone_energy_test['value_shifted'] = np.roll(backbone_energy_test['value'],-1)        
        backbone_energy_test.drop(labels=['date_time','timeslot','day_of_week','day_of_month','day_of_year'],axis=1).to_csv(export_path + dataset + '/energy/' + str(backbone_num) + '_test.csv', encoding='utf-8', index=False)        

        for split in val_splits:
            slice_len = round(len(backbone_energy_train)/split)            
            for i in range(1, split+1):                
                val_slice = backbone_energy_train[:(i*slice_len)]
                
                train_slice = val_slice[:-round(slice_len/2)]
                test_slice = val_slice[-round(slice_len/2):]
                meta_dict[dataset][stat_name]['backbone_energy_train_' + str(split) + '_' + str(i) + '_train'] = train_slice
                meta_dict[dataset][stat_name]['backbone_energy_train_' + str(split) + '_' + str(i) + '_test'] = test_slice
         
                train_slice['value_shifted'] = np.roll(train_slice['value'],-1)  
                train_slice.drop(labels=['date_time','timeslot','day_of_week','day_of_month','day_of_year'],axis=1).to_csv(export_path + dataset + '/energy/' + str(backbone_num) + '_' + str(split) + '_' + str(i) +'_train.csv', encoding='utf-8', index=False)
                test_slice['value_shifted'] = np.roll(test_slice['value'],-1)  
                test_slice.drop(labels=['date_time','timeslot','day_of_week','day_of_month','day_of_year'],axis=1).to_csv(export_path + dataset + '/energy/' + str(backbone_num) + '_' + str(split) + '_' + str(i) +'_test.csv', encoding='utf-8', index=False)               
                            
        backbone_occup = meta_dict[dataset][stat_name]['backbones'][1]     
        n_train = round(train_test_split*len(backbone_occup)) 
        backbone_occup_train = backbone_occup[:n_train]
        backbone_occup_test = backbone_occup[n_train:]
        meta_dict[dataset][stat_name]['backbone_occup_train'] = backbone_occup_train
        meta_dict[dataset][stat_name]['backbone_occup_test'] = backbone_occup_test
        
        backbone_occup_train['value_shifted'] = np.roll(backbone_occup_train['value'],-1)        
        backbone_occup_train.drop(labels=['date_time','timeslot','day_of_week','day_of_month','day_of_year'],axis=1).to_csv(export_path + dataset + '/occup/' + str(backbone_num) + '_train.csv', encoding='utf-8', index=False)
        backbone_occup_test['value_shifted'] = np.roll(backbone_occup_test['value'],-1)        
        backbone_occup_test.drop(labels=['date_time','timeslot','day_of_week','day_of_month','day_of_year'],axis=1).to_csv(export_path + dataset + '/occup/' + str(backbone_num) + '_test.csv', encoding='utf-8', index=False)       
        
        for split in val_splits:
            slice_len = round(len(backbone_occup_train)/split)            
            for i in range(1, split+1):
                val_slice = backbone_occup_train[:(i*slice_len)]
                train_slice = val_slice[:-round(slice_len/2)]
                test_slice = val_slice[-round(slice_len/2):]
                
                meta_dict[dataset][stat_name]['backbone_occup_train_' + str(split) + '_' + str(i) + '_train'] = train_slice
                meta_dict[dataset][stat_name]['backbone_occup_train_' + str(split) + '_' + str(i) + '_test'] = test_slice
         
                train_slice['value_shifted'] = np.roll(train_slice['value'],-1)  
                train_slice.drop(labels=['date_time','timeslot','day_of_week','day_of_month','day_of_year'],axis=1).to_csv(export_path + dataset + '/occup/' + str(backbone_num) + '_' + str(split) + '_' + str(i) +'_train.csv', encoding='utf-8', index=False)
                test_slice['value_shifted'] = np.roll(test_slice['value'],-1)  
                test_slice.drop(labels=['date_time','timeslot','day_of_week','day_of_month','day_of_year'],axis=1).to_csv(export_path + dataset + '/occup/' + str(backbone_num) + '_' + str(split) + '_' + str(i) +'_test.csv', encoding='utf-8', index=False)
                
        
        # calculate average energy demand
        energy_avg_weekday = pd.DataFrame({'timeslot': list(range(96))})
        energy_avg_weekday['avg_value'] = 0.0      
        energy_avg_weekend = pd.DataFrame({'timeslot': list(range(96))})
        energy_avg_weekend['avg_value'] = 0.0            
        for i in range(96): 
            avg_value_energy = backbone_energy_train[(backbone_energy_train.timeslot == i) & (backbone_energy_train.weekend == 0)].value.sum()
            energy_avg_weekday.loc[energy_avg_weekday['timeslot'] == i, 'avg_value']  = avg_value_energy
        for i in range(96): 
            energy_avg_weekday.loc[energy_avg_weekday['timeslot'] == i, 'avg_value'] /= len(backbone_energy_train[(backbone_energy_train.timeslot == i) & (backbone_energy_train.weekend == 0)])
        meta_dict[dataset][stat_name]['backbone_energy_train_weekday_avg'] = energy_avg_weekday        
        for i in range(96): 
            avg_value_energy = backbone_energy_train[(backbone_energy_train.timeslot == i) & (backbone_energy_train.weekend == 1)].value.sum()
            energy_avg_weekend.loc[energy_avg_weekend['timeslot'] == i, 'avg_value']  = avg_value_energy
        for i in range(96): 
            energy_avg_weekend.loc[energy_avg_weekend['timeslot'] == i, 'avg_value'] /= len(backbone_energy_train[(backbone_energy_train.timeslot == i) & (backbone_energy_train.weekend == 1)])        
        meta_dict[dataset][stat_name]['backbone_energy_train_weekend_avg'] = energy_avg_weekend   
        
        averages = pd.DataFrame(columns=['weekday', 'weekend'])
        averages.weekday = energy_avg_weekday.drop(labels='timeslot',axis=1)
        averages.weekend = energy_avg_weekend.drop(labels='timeslot',axis=1)        
        averages.to_csv(export_path + dataset + '/energy/' + str(backbone_num) + '_averages.csv', encoding='utf-8', index=False)
        
        # calculate average occupancy rates
        occup_avg_weekday = pd.DataFrame({'timeslot': list(range(96))})
        occup_avg_weekday['avg_value'] = 0.0      
        occup_avg_weekend = pd.DataFrame({'timeslot': list(range(96))})
        occup_avg_weekend['avg_value'] = 0.0          
        for i in range(96):    
            avg_value_occup = len(backbone_occup_train[(backbone_occup_train.timeslot == i) & (backbone_occup_train.value == 1) & (backbone_occup_train.weekend == 0)]) / len(backbone_occup_train[(backbone_occup_train.timeslot == i) & (backbone_occup_train.weekend == 0)])
            occup_avg_weekday.loc[occup_avg_weekday['timeslot'] == i, 'avg_value'] = avg_value_occup
        meta_dict[dataset][stat_name]['backbone_occup_train_weekday_avg'] = occup_avg_weekday            
        for i in range(96):    
            avg_value_occup = len(backbone_occup_train[(backbone_occup_train.timeslot == i) & (backbone_occup_train.value == 1) & (backbone_occup_train.weekend == 1)]) / len(backbone_occup_train[(backbone_occup_train.timeslot == i) & (backbone_occup_train.weekend == 1)])        
            occup_avg_weekend.loc[occup_avg_weekend['timeslot'] == i, 'avg_value'] = avg_value_occup
        meta_dict[dataset][stat_name]['backbone_occup_train_weekend_avg'] = occup_avg_weekend
        
        averages = pd.DataFrame(columns=['weekday', 'weekend'])
        averages.weekday = occup_avg_weekday.drop(labels='timeslot',axis=1)
        averages.weekend = occup_avg_weekend.drop(labels='timeslot',axis=1)        
        averages.to_csv(export_path + dataset + '/occup/' + str(backbone_num) + '_averages.csv', encoding='utf-8', index=False)
                
        # do the same for validation splits
        for split in val_splits:
            for i in range(1, split+1):
                backbone_energy_train = meta_dict[dataset][stat_name]['backbone_energy_train_' + str(split) + '_' + str(i) + '_train']
                energy_avg_weekday = pd.DataFrame({'timeslot': list(range(96))})
                energy_avg_weekday['avg_value'] = 0.0      
                energy_avg_weekend = pd.DataFrame({'timeslot': list(range(96))})
                energy_avg_weekend['avg_value'] = 0.0            
                for j in range(96): 
                    avg_value_energy = backbone_energy_train[(backbone_energy_train.timeslot == j) & (backbone_energy_train.weekend == 0)].value.sum()
                    energy_avg_weekday.loc[energy_avg_weekday['timeslot'] == j, 'avg_value']  = avg_value_energy
                for j in range(96): 
                    energy_avg_weekday.loc[energy_avg_weekday['timeslot'] == j, 'avg_value'] /= len(backbone_energy_train[(backbone_energy_train.timeslot == j) & (backbone_energy_train.weekend == 0)])
                meta_dict[dataset][stat_name]['backbone_energy_train_' + str(split) + '_' + str(i) + '_weekday_avg'] = energy_avg_weekday        
                for j in range(96): 
                    avg_value_energy = backbone_energy_train[(backbone_energy_train.timeslot == j) & (backbone_energy_train.weekend == 1)].value.sum()
                    energy_avg_weekend.loc[energy_avg_weekend['timeslot'] == j, 'avg_value']  = avg_value_energy
                for j in range(96): 
                    energy_avg_weekend.loc[energy_avg_weekend['timeslot'] == j, 'avg_value'] /= len(backbone_energy_train[(backbone_energy_train.timeslot == j) & (backbone_energy_train.weekend == 1)]) 
                meta_dict[dataset][stat_name]['backbone_energy_train_' + str(split) + '_' + str(i) + '_weekend_avg'] = energy_avg_weekday        
                
                averages = pd.DataFrame(columns=['weekday', 'weekend'])
                averages.weekday = occup_avg_weekday.drop(labels='timeslot',axis=1)
                averages.weekend = occup_avg_weekend.drop(labels='timeslot',axis=1)                
                averages.to_csv(export_path + dataset + '/energy/' + str(backbone_num) + '_' + str(split) + '_' + str(i) + '_averages.csv', encoding='utf-8', index=False)
                
                
                backbone_occup_train = meta_dict[dataset][stat_name]['backbone_occup_train_' + str(split) + '_' + str(i) + '_train']
                occup_avg_weekday = pd.DataFrame({'timeslot': list(range(96))})
                occup_avg_weekday['avg_value'] = 0.0      
                occup_avg_weekend = pd.DataFrame({'timeslot': list(range(96))})
                occup_avg_weekend['avg_value'] = 0.0   
                for j in range(96):    
                    avg_value_occup = len(backbone_occup_train[(backbone_occup_train.timeslot == j) & (backbone_occup_train.value == 1) & (backbone_occup_train.weekend == 0)]) / len(backbone_occup_train[(backbone_occup_train.timeslot == j) & (backbone_occup_train.weekend == 0)])
                    occup_avg_weekday.loc[occup_avg_weekday['timeslot'] == j, 'avg_value'] = avg_value_occup                
                meta_dict[dataset][stat_name]['backbone_occup_train_' + str(split) + '_' + str(i) + '_weekday_avg'] = occup_avg_weekday        
                for j in range(96):    
                    avg_value_occup = len(backbone_occup_train[(backbone_occup_train.timeslot == j) & (backbone_occup_train.value == 1) & (backbone_occup_train.weekend == 1)]) / len(backbone_occup_train[(backbone_occup_train.timeslot == j) & (backbone_occup_train.weekend == 1)])        
                    occup_avg_weekend.loc[occup_avg_weekend['timeslot'] == j, 'avg_value'] = avg_value_occup
                meta_dict[dataset][stat_name]['backbone_occup_train_' + str(split) + '_' + str(i) + '_weekend_avg'] = occup_avg_weekend
                
                averages = pd.DataFrame(columns=['weekday', 'weekend'])
                averages.weekday = occup_avg_weekday.drop(labels='timeslot',axis=1)
                averages.weekend = occup_avg_weekend.drop(labels='timeslot',axis=1)                
                averages.to_csv(export_path + dataset + '/occup/' + str(backbone_num) + '_' + str(split) + '_' + str(i) + '_averages.csv', encoding='utf-8', index=False)
                
        backbone_num += 1
        print(stat_name)
        
    # generate and export aggregated energy time series  
    sum_averages = pd.read_csv(export_path + dataset + '/energy/1_averages.csv')
    sum_averages['weekday'] = 0.0
    sum_averages['weekend'] = 0.0
    for num in range(1, len(stations)+1):
        df_avg = pd.read_csv(export_path + dataset + '/energy/' + str(num) + '_averages.csv')
        for column in df_avg:
            if column == 'weekday':
                sum_averages[column] += df_avg[column]
            if column == 'weekend':
                sum_averages[column] += df_avg[column]
                
    sum_averages_norm = pd.DataFrame()
    for column in sum_averages:
        x = sum_averages[column]            
        x = (x-min(x))/(max(x)-min(x))   
        sum_averages_norm[column] = x   
    
    sum_averages_norm.to_csv(export_path + dataset + '/energy/averages_norm.csv', encoding='utf-8', index=False)
    
    for split in val_splits:                
        for i in range(1, split+1): 
            sum_averages = pd.read_csv(export_path + dataset + '/energy/1_' + str(split) + '_' + str(i) + '_averages.csv')
            sum_averages['weekday'] = 0.0
            sum_averages['weekend'] = 0.0
            for num in range(1, len(stations)+1):
                df_avg = pd.read_csv(export_path + dataset + '/energy/' + str(num) + '_' + str(split) + '_' + str(i) + '_averages.csv')
                for column in df_avg:
                    if column == 'weekday':
                        sum_averages[column] += df_avg[column]
                    if column == 'weekend':
                        sum_averages[column] += df_avg[column]
                        
            sum_averages_norm = pd.DataFrame()
            for column in sum_averages:
                x = sum_averages[column]            
                x = (x-min(x))/(max(x)-min(x))   
                sum_averages_norm[column] = x   
            
            sum_averages_norm.to_csv(export_path + dataset + '/energy/' + str(split) + '_' + str(i) + '_averages_norm.csv', encoding='utf-8', index=False)
    
    
    sum_energy_train = pd.read_csv(export_path + dataset + '/energy/1_train.csv')    
    for column in sum_energy_train:
        if column == 'value' or column == 'value_shifted':
            sum_energy_train[column] = 0.0     
    for num in range(1, len(stations)+1):
        df_train = pd.read_csv(export_path + dataset + '/energy/' + str(num) + '_train.csv')                            
        for column in df_train:
            if column == 'value' or column == 'value_shifted':
                sum_energy_train[column] += df_train[column] 
                
                
    df_train_norm = pd.DataFrame()
    for column in sum_energy_train:        
        if column == 'value' or column == 'value_shifted':          
            x = sum_energy_train[column]               
            x = (x-min(x))/(max(x)-min(x))   
            df_train_norm[column] = x            
        else:            
            y = sum_energy_train[column]
            df_train_norm[column] = y      

    file_name = export_path + dataset + '/energy/train_sum.csv'
    df_train_norm.to_csv(file_name, encoding='utf-8', index=False)
    
    sum_energy_test = pd.read_csv(export_path + dataset + '/energy/1_test.csv')    
    for column in sum_energy_test:
        if column == 'value' or column == 'value_shifted':
            sum_energy_test[column] = 0.0     
    for num in range(1, len(stations)+1):
        df_test = pd.read_csv(export_path + dataset + '/energy/' + str(num) + '_test.csv')                            
        for column in df_test:
            if column == 'value' or column == 'value_shifted':
                sum_energy_test[column] += df_test[column]                  
                
    df_test_norm = pd.DataFrame()
    for column in sum_energy_test:        
        if column == 'value' or column == 'value_shifted':          
            x = sum_energy_test[column]       
            meta_dict[dataset]['max_test'] = max(x)
            meta_dict[dataset]['min_test'] = min(x)              
            x = (x-min(x))/(max(x)-min(x))   
            df_test_norm[column] = x            
        else:            
            y = sum_energy_test[column]
            df_test_norm[column] = y      

    file_name = export_path + dataset + '/energy/test_sum.csv'
    df_test_norm.to_csv(file_name, encoding='utf-8', index=False)
    
    
    for split in val_splits:                
        for i in range(1, split+1): 
            sum_energy_train = pd.read_csv(export_path + dataset + '/energy/1' + '_' + str(split) + '_' + str(i) + '_train.csv')    
            for column in sum_energy_train:
                if column == 'value' or column == 'value_shifted':
                    sum_energy_train[column] = 0.0     
            for num in range(1, len(stations)+1):
                df_train = pd.read_csv(export_path + dataset + '/energy/' + str(num) + '_' + str(split) + '_' + str(i) + '_train.csv')                            
                for column in df_train:
                    if column == 'value' or column == 'value_shifted':
                        sum_energy_train[column] += df_train[column]                  
                        
            df_train_norm = pd.DataFrame()
            for column in sum_energy_train:        
                if column == 'value' or column == 'value_shifted':          
                    x = sum_energy_train[column]            
                    x = (x-min(x))/(max(x)-min(x))   
                    df_train_norm[column] = x            
                else:            
                    y = sum_energy_train[column]
                    df_train_norm[column] = y      

            file_name = export_path + dataset + '/energy/' + str(split) + '_' + str(i) + '_train_sum.csv'
            df_train_norm.to_csv(file_name, encoding='utf-8', index=False)
            
            sum_energy_test = pd.read_csv(export_path + dataset + '/energy/1' + '_' + str(split) + '_' + str(i) + '_test.csv')    
            for column in sum_energy_test:
                if column == 'value' or column == 'value_shifted':
                    sum_energy_test[column] = 0.0     
            for num in range(1, len(stations)+1):
                df_test = pd.read_csv(export_path + dataset + '/energy/' + str(num) + '_' + str(split) + '_' + str(i) + '_test.csv')                            
                for column in df_test:
                    if column == 'value' or column == 'value_shifted':
                        sum_energy_test[column] += df_test[column]                  
                        
            df_test_norm = pd.DataFrame()
            for column in sum_energy_test:        
                if column == 'value' or column == 'value_shifted':          
                    x = sum_energy_test[column]            
                    x = (x-min(x))/(max(x)-min(x))   
                    df_test_norm[column] = x            
                else:            
                    y = sum_energy_test[column]
                    df_test_norm[column] = y      

            file_name = export_path + dataset + '/energy/' + str(split) + '_' + str(i) + '_test_sum.csv'
            df_test_norm.to_csv(file_name, encoding='utf-8', index=False)

    # backbone_num = 1
    # for stat_name in stations:
        
    #     # backbone_energy = meta_dict[dataset][stat_name]['backbones'][0]
    #     backbone_energy_train = meta_dict[dataset][stat_name]['backbone_energy_train']
       
    #     backbone_energy_train_weekday = backbone_energy_train[backbone_energy_train['weekend'] == 0]        
    #     energy_avg_weekday = meta_dict[dataset][stat_name]['backbone_energy_train_weekday_avg'].T.drop(labels='timeslot', axis=0)
    #     energy_weekday_data = pd.concat([energy_avg_weekday]*len(backbone_energy_train_weekday), ignore_index=True)
    #     backbone_energy_train_weekday = pd.concat([backbone_energy_train_weekday.reset_index(drop=True), energy_weekday_data.reset_index(drop=True)], axis=1)
       
    #     backbone_energy_train_weekend = backbone_energy_train[backbone_energy_train['weekend'] == 1]
    #     energy_avg_weekend = meta_dict[dataset][stat_name]['backbone_energy_train_weekday_avg'].T.drop(labels='timeslot', axis=0)
    #     energy_weekend_data = pd.concat([energy_avg_weekend]*len(backbone_energy_train_weekend), ignore_index=True)
    #     backbone_energy_train_weekend = pd.concat([backbone_energy_train_weekend.reset_index(drop=True), energy_weekend_data.reset_index(drop=True)], axis=1)
       
    #     sorted_backbone_energy_train = pd.concat([backbone_energy_train_weekday, backbone_energy_train_weekend]).sort_values(by=['date_time'], ascending=True)
    #     sorted_backbone_energy_train['value_shifted'] = np.roll(sorted_backbone_energy_train['value'],-1)
    #     sorted_backbone_energy_train.drop('date_time', axis=1, inplace=True)
       
    #     file_name_energy_train = export_path + dataset + '/energy/' + str(backbone_num) + '_train.csv'
    #     # file_name_energy = '/home/doktormatte/MA_SciComp/Boul der/energys/' + stat_name.replace('/', '') + '.csv'
    #     sorted_backbone_energy_train.to_csv(file_name_energy_train, encoding='utf-8', index=False, header=False)
        
        
    #     backbone_energy_test = meta_dict[dataset][stat_name]['backbone_energy_test']
       
    #     backbone_energy_test_weekday = backbone_energy_test[backbone_energy_test['weekend'] == 0]        
    #     energy_avg_weekday = meta_dict[dataset][stat_name]['backbone_energy_train_weekday_avg'].T.drop(labels='timeslot', axis=0)
    #     energy_weekday_data = pd.concat([energy_avg_weekday]*len(backbone_energy_test_weekday), ignore_index=True)
    #     backbone_energy_test_weekday = pd.concat([backbone_energy_test_weekday.reset_index(drop=True), energy_weekday_data.reset_index(drop=True)], axis=1)
       
    #     backbone_energy_test_weekend = backbone_energy_test[backbone_energy_test['weekend'] == 1]
    #     energy_avg_weekend = meta_dict[dataset][stat_name]['backbone_energy_train_weekday_avg'].T.drop(labels='timeslot', axis=0)
    #     energy_weekend_data = pd.concat([energy_avg_weekend]*len(backbone_energy_test_weekend), ignore_index=True)
    #     backbone_energy_test_weekend = pd.concat([backbone_energy_test_weekend.reset_index(drop=True), energy_weekend_data.reset_index(drop=True)], axis=1)
       
    #     sorted_backbone_energy_test = pd.concat([backbone_energy_test_weekday, backbone_energy_test_weekend]).sort_values(by=['date_time'], ascending=True)
    #     sorted_backbone_energy_test['value_shifted'] = np.roll(sorted_backbone_energy_test['value'],-1)
    #     sorted_backbone_energy_test.drop('date_time', axis=1, inplace=True)
       
    #     file_name_energy_test = export_path + dataset + '/energy/' + str(backbone_num) + '_test.csv'
    #     # file_name_energy = '/home/doktormatte/MA_SciComp/Boul der/energys/' + stat_name.replace('/', '') + '.csv'
    #     sorted_backbone_energy_test.to_csv(file_name_energy_test, encoding='utf-8', index=False, header=False)
        
        
       
    #     for split in val_splits:
    #        for i in range(1, split+1):
    #            backbone_energy_train = meta_dict[dataset][stat_name]['backbone_energy_train_' + str(split) + '_' + str(i) + '_train']
    #            backbone_energy_train_weekday = backbone_energy_train[backbone_energy_train['weekend'] == 0]        
    #            energy_avg_weekday = meta_dict[dataset][stat_name]['backbone_energy_train_' + str(split) + '_' + str(i) + '_weekday_avg'].T.drop(labels='timeslot', axis=0)
    #            energy_weekday_data = pd.concat([energy_avg_weekday]*len(backbone_energy_train_weekday), ignore_index=True)
    #            backbone_energy_train_weekday = pd.concat([backbone_energy_train_weekday.reset_index(drop=True), energy_weekday_data.reset_index(drop=True)], axis=1)
              
    #            backbone_energy_train_weekend = backbone_energy_train[backbone_energy_train['weekend'] == 1]
    #            energy_avg_weekend = meta_dict[dataset][stat_name]['backbone_energy_train_' + str(split) + '_' + str(i) + '_weekend_avg'].T.drop(labels='timeslot', axis=0)
    #            energy_weekend_data = pd.concat([energy_avg_weekend]*len(backbone_energy_train_weekend), ignore_index=True)
    #            backbone_energy_train_weekend = pd.concat([backbone_energy_train_weekend.reset_index(drop=True), energy_weekend_data.reset_index(drop=True)], axis=1)
              
    #            sorted_backbone_energy_train = pd.concat([backbone_energy_train_weekday, backbone_energy_train_weekend]).sort_values(by=['date_time'], ascending=True)
    #            sorted_backbone_energy_train['value_shifted'] = np.roll(sorted_backbone_energy_train['value'],-1)
    #            sorted_backbone_energy_train.drop('date_time', axis=1, inplace=True)
              
    #            file_name_energy_train = export_path + dataset + '/energy/' + str(backbone_num) + '_' + str(split) + '_' + str(i) + '_train.csv'
    #            # file_name_energy = '/home/doktormatte/MA_SciComp/Boul der/energys/' + stat_name.replace('/', '') + '.csv'
    #            sorted_backbone_energy_train.to_csv(file_name_energy_train, encoding='utf-8', index=False, header=False)
               
               
    #            backbone_energy_test = meta_dict[dataset][stat_name]['backbone_energy_train_' + str(split) + '_' + str(i) + '_test']
    #            backbone_energy_test_weekday = backbone_energy_test[backbone_energy_test['weekend'] == 0]        
    #            energy_avg_weekday = meta_dict[dataset][stat_name]['backbone_energy_train_' + str(split) + '_' + str(i) + '_weekday_avg'].T.drop(labels='timeslot', axis=0)
    #            energy_weekday_data = pd.concat([energy_avg_weekday]*len(backbone_energy_test_weekday), ignore_index=True)
    #            backbone_energy_test_weekday = pd.concat([backbone_energy_test_weekday.reset_index(drop=True), energy_weekday_data.reset_index(drop=True)], axis=1)
              
    #            backbone_energy_test_weekend = backbone_energy_test[backbone_energy_test['weekend'] == 1]
    #            energy_avg_weekend = meta_dict[dataset][stat_name]['backbone_energy_train_' + str(split) + '_' + str(i) + '_weekend_avg'].T.drop(labels='timeslot', axis=0)
    #            energy_weekend_data = pd.concat([energy_avg_weekend]*len(backbone_energy_test_weekend), ignore_index=True)
    #            backbone_energy_test_weekend = pd.concat([backbone_energy_test_weekend.reset_index(drop=True), energy_weekend_data.reset_index(drop=True)], axis=1)
              
    #            sorted_backbone_energy_test = pd.concat([backbone_energy_test_weekday, backbone_energy_test_weekend]).sort_values(by=['date_time'], ascending=True)
    #            sorted_backbone_energy_test['value_shifted'] = np.roll(sorted_backbone_energy_test['value'],-1)
    #            sorted_backbone_energy_test.drop('date_time', axis=1, inplace=True)
              
    #            file_name_energy_test = export_path + dataset + '/energy/' + str(backbone_num) + '_' + str(split) + '_' + str(i) + '_test.csv'
    #            # file_name_energy = '/home/doktormatte/MA_SciComp/Boul der/energys/' + stat_name.replace('/', '') + '.csv'
    #            sorted_backbone_energy_test.to_csv(file_name_energy_test, encoding='utf-8', index=False, header=False)
               
        
        
    #     backbone_occup_train = meta_dict[dataset][stat_name]['backbone_occup_train']
       
    #     backbone_occup_train_weekday = backbone_occup_train[backbone_occup_train['weekend'] == 0]        
    #     occup_avg_weekday = meta_dict[dataset][stat_name]['backbone_occup_train_weekday_avg'].T.drop(labels='timeslot', axis=0)
    #     occup_weekday_data = pd.concat([occup_avg_weekday]*len(backbone_occup_train_weekday), ignore_index=True)
    #     backbone_occup_train_weekday = pd.concat([backbone_occup_train_weekday.reset_index(drop=True), occup_weekday_data.reset_index(drop=True)], axis=1)
       
    #     backbone_occup_train_weekend = backbone_occup_train[backbone_occup_train['weekend'] == 1]
    #     occup_avg_weekend = meta_dict[dataset][stat_name]['backbone_occup_train_weekday_avg'].T.drop(labels='timeslot', axis=0)
    #     occup_weekend_data = pd.concat([occup_avg_weekend]*len(backbone_occup_train_weekend), ignore_index=True)
    #     backbone_occup_train_weekend = pd.concat([backbone_occup_train_weekend.reset_index(drop=True), occup_weekend_data.reset_index(drop=True)], axis=1)
       
    #     sorted_backbone_occup_train = pd.concat([backbone_occup_train_weekday, backbone_occup_train_weekend]).sort_values(by=['date_time'], ascending=True)
    #     sorted_backbone_occup_train['value_shifted'] = np.roll(sorted_backbone_occup_train['value'],-1)
    #     sorted_backbone_occup_train.drop('date_time', axis=1, inplace=True)
       
    #     file_name_occup = export_path + dataset + '/occup/' + str(backbone_num) + '_train.csv'
    #     # file_name_occup = '/home/doktormatte/MA_SciComp/Boul der/occups/' + stat_name.replace('/', '') + '.csv'
    #     sorted_backbone_occup_train.to_csv(file_name_occup, encoding='utf-8', index=False, header=False)
        
    #     backbone_occup_test = meta_dict[dataset][stat_name]['backbone_occup_test']
       
    #     backbone_occup_test_weekday = backbone_occup_test[backbone_occup_test['weekend'] == 0]        
    #     occup_avg_weekday = meta_dict[dataset][stat_name]['backbone_occup_train_weekday_avg'].T.drop(labels='timeslot', axis=0)
    #     occup_weekday_data = pd.concat([occup_avg_weekday]*len(backbone_occup_test_weekday), ignore_index=True)
    #     backbone_occup_test_weekday = pd.concat([backbone_occup_test_weekday.reset_index(drop=True), occup_weekday_data.reset_index(drop=True)], axis=1)
       
    #     backbone_occup_test_weekend = backbone_occup_test[backbone_occup_test['weekend'] == 1]
    #     occup_avg_weekend = meta_dict[dataset][stat_name]['backbone_occup_train_weekday_avg'].T.drop(labels='timeslot', axis=0)
    #     occup_weekend_data = pd.concat([occup_avg_weekend]*len(backbone_occup_test_weekend), ignore_index=True)
    #     backbone_occup_test_weekend = pd.concat([backbone_occup_test_weekend.reset_index(drop=True), occup_weekend_data.reset_index(drop=True)], axis=1)
       
    #     sorted_backbone_occup_test = pd.concat([backbone_occup_test_weekday, backbone_occup_test_weekend]).sort_values(by=['date_time'], ascending=True)
    #     sorted_backbone_occup_test['value_shifted'] = np.roll(sorted_backbone_occup_test['value'],-1)
    #     sorted_backbone_occup_test.drop('date_time', axis=1, inplace=True)
       
    #     file_name_occup = export_path + dataset + '/occup/' + str(backbone_num) + '_test.csv'
    #     # file_name_occup = '/home/doktormatte/MA_SciComp/Boul der/occups/' + stat_name.replace('/', '') + '.csv'
    #     sorted_backbone_occup_test.to_csv(file_name_occup, encoding='utf-8', index=False, header=False)
        
        
        
       
    #     for split in val_splits:
    #        for i in range(1, split+1):
    #            backbone_occup_train = meta_dict[dataset][stat_name]['backbone_occup_train_' + str(split) + '_' + str(i) + '_train']
    #            backbone_occup_train_weekday = backbone_occup_train[backbone_occup_train['weekend'] == 0]        
    #            occup_avg_weekday = meta_dict[dataset][stat_name]['backbone_occup_train_' + str(split) + '_' + str(i) + '_weekday_avg'].T.drop(labels='timeslot', axis=0)
    #            occup_weekday_data = pd.concat([occup_avg_weekday]*len(backbone_occup_train_weekday), ignore_index=True)
    #            backbone_occup_train_weekday = pd.concat([backbone_occup_train_weekday.reset_index(drop=True), occup_weekday_data.reset_index(drop=True)], axis=1)
              
    #            backbone_occup_train_weekend = backbone_occup_train[backbone_occup_train['weekend'] == 1]
    #            occup_avg_weekend = meta_dict[dataset][stat_name]['backbone_occup_train_' + str(split) + '_' + str(i) + '_weekend_avg'].T.drop(labels='timeslot', axis=0)
    #            occup_weekend_data = pd.concat([occup_avg_weekend]*len(backbone_occup_train_weekend), ignore_index=True)
    #            backbone_occup_train_weekend = pd.concat([backbone_occup_train_weekend.reset_index(drop=True), occup_weekend_data.reset_index(drop=True)], axis=1)
              
    #            sorted_backbone_occup_train = pd.concat([backbone_occup_train_weekday, backbone_occup_train_weekend]).sort_values(by=['date_time'], ascending=True)
    #            sorted_backbone_occup_train['value_shifted'] = np.roll(sorted_backbone_occup_train['value'],-1)
    #            sorted_backbone_occup_train.drop('date_time', axis=1, inplace=True)
              
    #            file_name_occup = export_path + dataset + '/occup/' + str(backbone_num) + '_' + str(split) + '_' + str(i) + '_train.csv'
    #            # file_name_occup = '/home/doktormatte/MA_SciComp/Boul der/occups/' + stat_name.replace('/', '') + '.csv'
    #            sorted_backbone_occup_train.to_csv(file_name_occup, encoding='utf-8', index=False, header=False)
               
               
    #            backbone_occup_test = meta_dict[dataset][stat_name]['backbone_occup_train_' + str(split) + '_' + str(i) + '_test']
    #            backbone_occup_test_weekday = backbone_occup_test[backbone_occup_test['weekend'] == 0]        
    #            occup_avg_weekday = meta_dict[dataset][stat_name]['backbone_occup_train_' + str(split) + '_' + str(i) + '_weekday_avg'].T.drop(labels='timeslot', axis=0)
    #            occup_weekday_data = pd.concat([occup_avg_weekday]*len(backbone_occup_test_weekday), ignore_index=True)
    #            backbone_occup_test_weekday = pd.concat([backbone_occup_test_weekday.reset_index(drop=True), occup_weekday_data.reset_index(drop=True)], axis=1)
              
    #            backbone_occup_test_weekend = backbone_occup_test[backbone_occup_test['weekend'] == 1]
    #            occup_avg_weekend = meta_dict[dataset][stat_name]['backbone_occup_train_' + str(split) + '_' + str(i) + '_weekend_avg'].T.drop(labels='timeslot', axis=0)
    #            occup_weekend_data = pd.concat([occup_avg_weekend]*len(backbone_occup_test_weekend), ignore_index=True)
    #            backbone_occup_test_weekend = pd.concat([backbone_occup_test_weekend.reset_index(drop=True), occup_weekend_data.reset_index(drop=True)], axis=1)
              
    #            sorted_backbone_occup_test = pd.concat([backbone_occup_test_weekday, backbone_occup_test_weekend]).sort_values(by=['date_time'], ascending=True)
    #            sorted_backbone_occup_test['value_shifted'] = np.roll(sorted_backbone_occup_test['value'],-1)
    #            sorted_backbone_occup_test.drop('date_time', axis=1, inplace=True)
              
    #            file_name_occup = export_path + dataset + '/occup/' + str(backbone_num) + '_' + str(split) + '_' + str(i) + '_test.csv'
    #            # file_name_occup = '/home/doktormatte/MA_SciComp/Boul der/occups/' + stat_name.replace('/', '') + '.csv'
    #            sorted_backbone_occup_test.to_csv(file_name_occup, encoding='utf-8', index=False, header=False)
               
        
        
        
    #     backbone_num += 1  
       
    #     print(stat_name)