import numpy as np
import pandas as pd
import tensorflow as tf
import os
import pickle
from keras import optimizers
import random
import uuid
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
from file_operations import wait_for_files, get_best_config, check_progress_forecast
from set_env import set_policy, set_paths, reset_tensorflow_keras_backend, set_seeds
from data_generators import insert_lags, insert_avgs, NHgenerator, Hgenerator, PLCgenerator
from DL_models import non_hybrid, hybrid, plc 
    
		

def main():    

    
    # column names for global csv files that contain results of hyperparameter search and forecasting performance
    glob_res_cols = ['id','model','dataset','rmse','mae','r_squared','accuracy','precision','recall','f1', 'fp','tp']	
    
    # list entries are: [total number of charging stations in dataset, charging stations with minimum / maximum average occupancy rates, respectively]
    dir_ranges = {'acn_caltech': [51,4,22], 'acn_jpl': [52,13,20], 'boulder': [20,9,1], 'palo_alto': [27,12,23]}	
    
    # set paths for import and export, set mixed precision policy, set random seed
    import_path, export_path = set_paths()
    set_policy(os.environ)
    set_seeds(0)    
    
    # select mode and define ranges for number of models trained per dataset and their forecasting horizons
    mode = random.choice(['energy', 'occup'])        
    dirname = random.choice(['acn_caltech','acn_jpl','palo_alto','boulder'])  
    if mode == 'occup':
        dir_range = list(range(1, dir_ranges[dirname][0]+1))        
        n_steps_out = 8
    else:
        dir_range = list(range(1, 6))        
        n_steps_out = random.choice([4,16,96])
                  
    # select architecture 
    architectures = ['LSTM', 'GRU', 'BiLSTM', 'Stacked', 'Conv1D', 'CNN_LSTM', 'CNN_BiLSTM', 'CNN_GRU', 'ConvLSTM', 'Hybrid', 'PLCnet']    
    architecture = random.choice(architectures)
    
      
    # get best hyperparameter config
    model_prop_dict = {}
    model_prop_dict['mode'] = mode
    model_prop_dict['architecture'] = architecture
    model_prop_dict['n_steps_out'] = n_steps_out
    wait_for_files([export_path + dirname + '_real.csv']) 
    glob_res_df = pd.read_csv(export_path + dirname + '_real.csv')       
    best_model_dict = get_best_config(glob_res_df[glob_res_df.dataset == dirname], model_prop_dict)
    if best_model_dict == 0:
        return
    
    # get model parameters from best hyperparameter config
    include_lags = best_model_dict['include_lags']
    include_avgs = best_model_dict['include_avgs']
    n_steps_in = best_model_dict['n_steps_in']
    n_features = best_model_dict['n_features']    
    
    bat_size = best_model_dict['bat_size']    
    stateful = best_model_dict['stateful']
    n_epoch = best_model_dict['n_epoch']  
    for i in range(4,9):
        if 2**i >= n_epoch:
            n_epoch = 2**i
            break
    best_model_dict['n_epoch']= n_epoch
    
    optimizer = best_model_dict['optimizer']
    grad_clip = best_model_dict['grad_clip']    
    
    
    for k in dir_range:
        best_model_dict['stat_num'] = k
        # check forecasting progress
        if mode == 'occup':
            n_steps_out = random.choice([1,4,16]) 
            best_model_dict['n_steps_out'] = n_steps_out
        if check_progress_forecast(glob_res_df[glob_res_df.dataset == dirname], best_model_dict) == 0:
            return
        
        if mode == 'occup':
            train_df = pd.read_csv(import_path + dirname + '/occup/' + str(k) + '_train.csv')
            test_df = pd.read_csv(import_path + dirname + '/occup/' + str(k) + '_test.csv')
            averages = pd.read_csv(import_path + dirname + '/occup/' + str(k) + '_averages.csv') 
        else:
            train_df = pd.read_csv(import_path + dirname + '/energy/train_sum.csv')
            test_df = pd.read_csv(import_path + dirname + '/energy/test_sum.csv')
            averages = pd.read_csv(import_path + dirname + '/energy/averages_norm.csv')         
       
    
        # add lagged target variables
        if include_lags == 1:
            train_df = insert_lags(train_df, n_steps_in)
            test_df = insert_lags(test_df, n_steps_in)            
    
        # add average occupancy rates / average energy demand
        if include_avgs == 1:
            if architecture != 'Hybrid':
                train_df = insert_avgs(train_df, averages)
                test_df = insert_avgs(test_df, averages)  
        
        # generate datasets
        if architecture == 'Hybrid':
            train_generator = Hgenerator(train_df, averages, n_steps_in, n_steps_out, n_features, batch_size=bat_size,stateful=stateful)
            test_generator = Hgenerator(test_df, averages, n_steps_in, n_steps_out, n_features, batch_size=bat_size,stateful=stateful)
            model = hybrid(best_model_dict, train_df, test_df)
        elif architecture == 'PLCnet':
            train_generator = PLCgenerator(train_df, n_steps_in, n_steps_out, batch_size=bat_size,stateful=stateful)
            test_generator = PLCgenerator(test_df, n_steps_in, n_steps_out, batch_size=bat_size,stateful=stateful)
            model = plc(best_model_dict, train_df, test_df)
        else:
            train_generator = NHgenerator(train_df, n_steps_in, n_steps_out, n_features, architecture, batch_size=bat_size,stateful=stateful)
            test_generator = NHgenerator(test_df, n_steps_in, n_steps_out, n_features, architecture, batch_size=bat_size,stateful=stateful)
            model = non_hybrid(best_model_dict, train_df, test_df)     
            
        # setup optimizer
        if optimizer == 1:
            opt = optimizers.Adam(clipnorm=grad_clip)
        if optimizer == 2:            
            opt = optimizers.SGD(clipnorm=grad_clip, momentum=0.9)
            
        # compile model
        if mode == 'occup':
            model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        else:
            model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
        
        # fit model
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=n_epoch//8,restore_best_weights=True)        
        if stateful:
            history = model.fit(
                train_generator,                
                epochs=n_epoch,
                batch_size=bat_size,
                shuffle=False,
                callbacks=[callback]
                )          
            temp = model.predict(test_generator,batch_size=bat_size)
        else:
            history = model.fit(
                train_generator,                
                epochs=n_epoch,                    
                shuffle=False,
                callbacks=[callback]
                )       
            temp = model.predict(test_generator)     
        
        # export model and learning curve
        model_name = uuid.uuid4().hex      
        train_hist = history.history['loss'] 
        filename = export_path + 'models/' + str(model_name) + '.sav'
        pickle.dump(model, open(filename, 'wb'))         
        train_hist = pd.DataFrame(train_hist)
        train_hist.to_csv(export_path + 'models/' + str(model_name) + '_train_hist.csv')        
        
        # reset tensorflow to avoid memory congestion
        reset_tensorflow_keras_backend(os.environ) 
        
        # setup structure for exporting forecast results
        row = pd.DataFrame(columns=glob_res_cols)
        row.id = [model_name]
        row.dataset = [dirname]
        row.model = [best_model_dict]
        
        # forecast evaluation        
        m,n=temp.shape 
        t_target = n_steps_out   
        y_obs = test_generator.get_data()    
          
        if mode == 'occup':      
            # occupancy forecast evaluation
            yhat = np.zeros((m,t_target))                       
            res_all = []
            for i in np.arange(m):  
                for j in np.arange(t_target):  
                        if temp[i][j] >= 0.5:
                            yhat[i][j] = 1    
                res_temp=[]
                _acc = 1.0 - sum(abs(yhat[i,]-y_obs[i,:]))/t_target    
                _pre = precision_score(y_obs[i,:], yhat[i,],zero_division=1)
                _recall = recall_score(y_obs[i,:], yhat[i,],zero_division=1)
                _f1 = f1_score(y_obs[i,:], yhat[i,],zero_division=1)                 
                CM = confusion_matrix(y_obs[i,:], yhat[i,],labels=[0,1])             
                res_temp=np.append(res_temp,[_acc, _pre,_recall,_f1,CM[1][1],CM[0][1]],0)                
                res_all.append(res_temp)
            row.accuracy = np.mean(res_all,axis=0)[0]
            row.precision = np.mean(res_all,axis=0)[1]
            row.recall = np.mean(res_all,axis=0)[2]
            row.f1 = np.mean(res_all,axis=0)[3]             
            row.tp = np.mean(res_all,axis=0)[4]
            row.fp = np.mean(res_all,axis=0)[5]                               

        
        else:
            # energy demand evaluation
            rmses = np.zeros(m)
            maes = np.zeros(m)            
            for j in np.arange(m):                        
                rmse = mean_squared_error(y_obs[j,:], temp[j,:], squared=False)
                mae = mean_absolute_error(y_obs[j,:], temp[j,:])                
                rmses[j] = 1.0 - rmse
                maes[j] = mae                
            row.rmse = np.mean(rmses)
            row.mae = np.mean(maes)
            row.r_squared = r2_score(y_obs[:,0], temp[:,0])
        
        # export results 
        wait_for_files([export_path + dirname + '_real.csv']) 
        glob_res_table = pd.read_csv(export_path + dirname + '_real.csv')
        glob_res_table=pd.concat([glob_res_table,row])
        glob_res_table.to_csv(export_path + dirname + '_real.csv', encoding='utf-8',index=False)
    
   

    
# main loop for forecast evaluation
if __name__ == '__main__':    
    while True:
        try:
            main()
        except KeyboardInterrupt:
            print('\n')
            print('Forecast evaluation interrupted...')
            break
        except Exception as e:
            print(e)
            continue        
        
        
    