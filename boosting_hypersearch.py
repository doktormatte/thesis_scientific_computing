from datetime import datetime
import pandas as pd
import numpy  as np 
import random
import uuid
import pickle
from sklearn.metrics import mean_squared_error
from keras.losses import binary_crossentropy
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from file_operations import wait_for_files, check_progress_hyper
from data_generators import insert_lags, insert_avgs, read_data_ml
from set_env import set_paths, set_seeds


def main():
    
    # set number of trials for random hyperparameter search
    trials = 30
    
    # column names for global csv files that contain results of hyperparameter search and forecasting performance
    glob_res_cols = ['id','model','dataset','rmse','mae','r_squared','accuracy','precision','recall','f1', 'fp','tp']	
    
    # list entries are: [total number of charging stations in dataset, charging stations with minimum / maximum average occupancy rates, respectively]
    dir_ranges = {'acn_caltech': [51,4,22], 'acn_jpl': [52,13,20], 'boulder': [20,9,1], 'palo_alto': [27,12,23]}	
    
    # set paths for import and export, set random seed
    import_path, export_path = set_paths()    
    set_seeds(0)
    
    mode = random.choice(['energy', 'occup'])        
    dirname = random.choice(['acn_caltech','acn_jpl','palo_alto','boulder'])  
    if mode == 'occup':
        dir_range = list(range(1, dir_ranges[dirname][0]+1))
        trial_num = 10
        n_steps_out = random.choice([1,4,16])
    else:
        dir_range = []
        trial_num = 1
        n_steps_out = random.choice([4,16,96])
        
    # select model 
    models = ['AdaBoost', 'XGBoost']    
    architecture = random.choice(models)
    
    
    
    # define hyperparameter ranges; only one feature set is used here -> parameters can be modified to conduct experiments on several feature sets
    include_lags = 1
    include_avgs = 1    
    n_steps_in = random.choice([2,4,8,16])  
    
    model_prop_dict = {}
    model_prop_dict['mode'] = mode 
    model_prop_dict['architecture'] = architecture
    model_prop_dict['n_steps_in'] = n_steps_in
    model_prop_dict['n_steps_out'] = n_steps_out
    model_prop_dict['include_lags'] = include_lags
    model_prop_dict['include_avgs'] = include_avgs 
    
    hyper_dict = {'n_estimators': {'AdaBoost': [32,64,128], 'XGBoost': [32,128,512]}, 'max_depth': {'AdaBoost': [1,2,3], 'XGBoost': [3,6,10]}, 'learning_rate': {'AdaBoost': [0.1,0.5,1,2], 'XGBoost': [0.5,1,2]}}
    model_prop_dict['n_estimators'] = random.choice(hyper_dict['n_estimators'][architecture])
    model_prop_dict['max_depth'] = random.choice(hyper_dict['max_depth'][architecture])
    model_prop_dict['alpha_reg'] = random.choice([0,0.5,1])
    model_prop_dict['lambda_reg'] = random.choice([0.5,1,2])   
    
    # check for hyperparameter search progress
    wait_for_files([export_path + dirname + '_real.csv']) 
    glob_res_df = pd.read_csv(export_path + dirname + '_real.csv')       
    trial = check_progress_hyper(glob_res_df[glob_res_df.dataset == dirname], model_prop_dict, trials)
    if trial == 0:
        return    
    
    # lists for collecting results
    val_losses = []    
    durs_all = [] 
    
    
    # run experiments on ten randomly sampled time series for occupancy and one aggregated time series for energy demand
    for _ in range(trial_num):
        if len(dir_range) > 0:
            stat_num = random.choice(dir_range)
            min_stat = dir_ranges[dirname][1]
            max_stat = dir_ranges[dirname][2]
            if min_stat in dir_range:
                stat_num = min_stat
                dir_range.remove(stat_num)
            elif max_stat in dir_range:
                stat_num = max_stat
                dir_range.remove(stat_num)  
            train_df = pd.read_csv(import_path + dirname + '/occup/' + str(stat_num) + '_train.csv')
            averages = pd.read_csv(import_path + dirname + '/occup/' + str(stat_num) + '_averages.csv') 
        else:
            train_df = pd.read_csv(import_path + dirname + '/energy/train_sum.csv')
            averages = pd.read_csv(import_path + dirname + '/energy/averages_norm.csv')
            
        # split data in training and validation set
        split_ind_train = int(.75*len(train_df))
        val_df = train_df[:split_ind_train]
        train_df = train_df[:split_ind_train]
    
        # add lagged target variables
        if include_lags == 1:
            train_df = insert_lags(train_df, n_steps_in)
            val_df = insert_lags(val_df, n_steps_in)            
    
        # add average occupancy rates / average energy demand
        if include_avgs == 1:            
            train_df = insert_avgs(train_df, averages)
            val_df = insert_avgs(val_df, averages)
        
        
        # generate datasets
        X_train, y_train = read_data_ml(train_df)           
        X_val, y_val = read_data_ml(val_df)
        
        # setup model
        if architecture == 'AdaBoost':
            if mode == 'occup':
                base_model = DecisionTreeClassifier(max_depth=model_prop_dict['max_depth'])
                model = AdaBoostClassifier(base_estimator=base_model,n_estimators=model_prop_dict['n_estimators'],learning_rate=model_prop_dict['n_estimators'])
            else:
                base_model = DecisionTreeRegressor(max_depth=model_prop_dict['max_depth'])
                model = AdaBoostRegressor(base_estimator=base_model,n_estimators=model_prop_dict['n_estimators'],learning_rate=model_prop_dict['learning_rate'])
        else:
            if mode == 'occup':
                model = XGBClassifier(n_estimators = model_prop_dict['n_estimators'],max_depth=model_prop_dict['max_depth'],learning_rate=model_prop_dict['learning_rate'],reg_alpha=model_prop_dict['alpha_reg'],reg_lambda=model_prop_dict['lambda_reg'])
            else:
                model = XGBRegressor(objective ='reg:squarederror',n_estimators = model_prop_dict['n_estimators'],max_depth=model_prop_dict['max_depth'],learning_rate=model_prop_dict['learning_rate'],reg_alpha=model_prop_dict['alpha_reg'],reg_lambda=model_prop_dict['lambda_reg'])
                       
        # fit model
        start = datetime.now()
        model.fit(X_train, y_train)
        end = datetime.now()
        duration = end-start
        durs_all.append(duration.total_seconds()) 
        
        # export model
        model_name = uuid.uuid4().hex
        filename = export_path + 'models/' + str(model_name) + '.sav'
        pickle.dump(model, open(filename, 'wb'))
        
        # compute validation loss by performing recursive forecasting
        res_all = []        
        t_target = n_steps_out
        m,n=X_val.shape
        yhat=np.zeros([m,t_target])            
        y_obs=np.zeros([m,t_target])
        for kk in range(m-t_target) :
            y_obs[kk,:]=y_val[kk:kk+t_target]    
        n_sample=m-n_steps_out     
        for i in range(n_sample):     
            print(i)
            X_val_temp=X_val.copy()
            num_cols = X_val_temp.shape[1]
            for _ in range(n_steps_out):
                X_val_temp=np.append(X_val_temp,[[0]*num_cols],0)                
            for j in range(n_steps_out):   
                temp11=X_val_temp[i+j,:].reshape(1, -1)                    
                yhat[i,j] = model.predict(temp11) 
                rng1=[i+j+n_steps_out -_ii for _ii in range(0, n_steps_out)]                    
                rng2=[_jj for _jj in range(-n_steps_in,0)]
                for ii in rng1:
                    for jj in rng2:                            
                        X_val_temp[ii,jj]=yhat[i,j]                 
            rr = [s for s in range(n_steps_out)]
            if mode == 'occup':
                _loss = binary_crossentropy(y_obs[i,rr], yhat[i,rr])
            else:                
                _loss = mean_squared_error(y_obs[i,rr], yhat[i,rr])                
            res_all.append(_loss) 
        val_losses.append(np.mean(res_all))
    
    # collect average results     
    model_prop_dict['min_val_loss'] = np.mean(val_losses)
    model_prop_dict['dur'] = np.mean(durs_all)          
    model_prop_dict['trial'] = trial                  
    row = pd.DataFrame(columns=glob_res_cols)    
    row.dataset = [dirname]               
    row.model = [model_prop_dict]     
    row.id = [model_name]
        
    # export results 
    wait_for_files([export_path + dirname + '_real.csv'])             
    glob_res_table = pd.read_csv(export_path + dirname + '_real.csv')
    glob_res_table=pd.concat([glob_res_table,row])
    glob_res_table.to_csv(export_path + dirname + '_real.csv', encoding='utf-8',index=False) 

        

# main loop for hyperparameter search
if __name__ == '__main__':    
    while True:
        try:
            main()
        except KeyboardInterrupt:
            print('\n')
            print('Hyperparameter search interrupted...')
            break
        except Exception as e:
            print(e)
            continue

