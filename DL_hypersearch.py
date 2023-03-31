import numpy as np
import pandas as pd
import tensorflow as tf
import os
import pickle
from keras import optimizers
from datetime import datetime
import random
import uuid
from file_operations import wait_for_files, check_progress_hyper
from set_env import set_policy, set_paths, reset_tensorflow_keras_backend, set_seeds
from data_generators import insert_lags, insert_avgs, NHgenerator, Hgenerator, PLCgenerator
from DL_models import non_hybrid, hybrid, plc  


def main():
    
    # set number of trials for random hyperparameter search
    trials = 60
    
    # column names for global csv files that contain results of hyperparameter search and forecasting performance
    glob_res_cols = ['id','model','dataset','rmse','mae','r_squared','accuracy','precision','recall','f1', 'fp','tp']	
    
    # list entries are: [total number of charging stations in dataset, charging stations with minimum / maximum average occupancy rates, respectively]
    dir_ranges = {'acn_caltech': [51,4,22], 'acn_jpl': [52,13,20], 'boulder': [20,9,1], 'palo_alto': [27,12,23]}	
    
    # set paths for import and export, set mixed precision policy, set random seed
    import_path, export_path = set_paths()
    set_policy(os.environ)
    set_seeds(0)    
    
    # select mode and define ranges for number of models trained per dataset and their forecasting horizons during hyperparameter search
    mode = random.choice(['energy', 'occup'])        
    dirname = random.choice(['acn_caltech','acn_jpl','palo_alto','boulder'])  
    if mode == 'occup':
        dir_range = list(range(1, dir_ranges[dirname][0]+1))
        trial_num = 10
        n_steps_out = 8
    else:
        dir_range = []
        trial_num = 1
        n_steps_out = random.choice([4,16,96])
                  
    # select architecture 
    architectures = ['LSTM', 'GRU', 'BiLSTM', 'Stacked', 'Conv1D', 'CNN_LSTM', 'CNN_BiLSTM', 'CNN_GRU', 'ConvLSTM', 'Hybrid', 'PLCnet']    
    architecture = random.choice(architectures)
    
    # only one feature set is used here -> parameters can be modified to conduct experiments on several feature sets
    include_lags = 1
    include_avgs = 1    
    n_steps_in = random.choice([2,4,8,16])    
    
    # number of features depends on architecture and lookback window size
    if include_lags == 1:
        if include_avgs == 1:
            n_features = 107 + 2 + n_steps_in
        else:
            n_features = 11 + 2 + n_steps_in
    else:
        if include_avgs == 1:
            n_features = 107
        else:
            n_features = 11   
            
    # fixed pool size and kernel size, depending on architecture
    po_size = 2
    if n_steps_out == 1:
        po_size = 1
    ker_size = 4
    if architecture == 'Conv1D' or architecture == 'ConvLSTM':            
        ker_size=1 
    if architecture == 'Hybrid':
        n_features = 1  
    
    
    # define hyperparameter ranges    

    bat_size = random.choice([4, 8, 16, 32, 64, 128]) 
    n_epoch = random.choice([16, 32, 64, 128, 256])     

    grad_clip = random.choice([None, 1.1, 2.0, 4.0, 8.0])            
    optimizer = random.choice([1,2])   
    
    stateful = random.choice([False])     # models can optionally be trained in stateful mode
    
    model_prop_dict = {}
    model_prop_dict['mode'] = mode 
    model_prop_dict['bat_size'] = bat_size 
    model_prop_dict['n_epoch'] = n_epoch
    model_prop_dict['optimizer'] = optimizer
    model_prop_dict['grad_clip'] = grad_clip
    model_prop_dict['n_features'] = n_features
    model_prop_dict['n_steps_in'] = n_steps_in
    model_prop_dict['n_steps_out'] = n_steps_out
    model_prop_dict['include_lags'] = include_lags
    model_prop_dict['include_avgs'] = include_avgs    
    model_prop_dict['stateful'] = stateful 
    model_prop_dict['include_lags'] = include_lags
    model_prop_dict['architecture'] = architecture
    model_prop_dict['po_size'] = po_size
    model_prop_dict['ker_size'] = ker_size
        
    model_prop_dict['dropout_1'] = random.randint(1,60)/100.0 
    model_prop_dict['dropout_2'] = random.randint(1,60)/100.0 
    model_prop_dict['nodes_rec_1'] = random.choice([8, 16, 32, 64, 128, 256])
    model_prop_dict['nodes_rec_2'] = random.choice([8, 16, 32, 64, 128])
    model_prop_dict['nodes_rec_3'] = random.choice([8, 16, 32, 64, 128])
    model_prop_dict['nodes_rec_4'] = random.choice([8, 16, 32, 64, 128])
    model_prop_dict['nodes_dense_1'] = random.choice([8, 16, 32, 64, 128, 256])
    model_prop_dict['nodes_dense_2'] = random.choice([8, 16, 32, 64, 128])
    model_prop_dict['nodes_dense_3'] = random.choice([8, 16, 32, 64, 128])
    model_prop_dict['nodes_dense_4'] = random.choice([8, 16, 32, 64, 128])
    model_prop_dict['nf_1'] = random.choice([4, 8, 16, 32, 64])
    model_prop_dict['nf_2'] = random.choice([4, 8, 16, 32, 64])
    model_prop_dict['nf_3'] = random.choice([4, 8, 16, 32, 64])
    model_prop_dict['nf_4'] = random.choice([4, 8, 16, 32, 64])      
    model_prop_dict['kernel_reg_1'] = random.choice([None, None, None, 'l1', 'l2'])
    model_prop_dict['kernel_reg_2'] = random.choice([None, None, None, 'l1', 'l2'])
    model_prop_dict['kernel_reg_3'] = random.choice([None, None, None, 'l1', 'l2'])
    model_prop_dict['kernel_reg_4'] = random.choice([None, None, None, 'l1', 'l2'])
    model_prop_dict['kernel_reg_5'] = random.choice([None, None, None, 'l1', 'l2'])   
    
    model_prop_dict['convolve'] = random.randint(0,1) 
    model_prop_dict['stack_layers'] = random.choice([1, 2, 3]) 
    model_prop_dict['stacked'] = random.choice([0,1,2])   
    model_prop_dict['stack_size'] = random.choice([2,3,4])
    model_prop_dict['stack_conv'] = random.choice([2, 3, 4])
    model_prop_dict['dilate'] = random.randint(0,1)    
    model_prop_dict['second_step'] = random.randint(0,1)
    model_prop_dict['first_dense'] = random.randint(0,1)
    model_prop_dict['second_LSTM'] = random.randint(0,1)
    model_prop_dict['first_conv'] = random.randint(0,1)
    model_prop_dict['second_conv'] = random.randint(0,1)        
    
    # check for hyperparameter search progress
    wait_for_files([export_path + dirname + '_real.csv']) 
    glob_res_df = pd.read_csv(export_path + dirname + '_real.csv')       
    trial = check_progress_hyper(glob_res_df[glob_res_df.dataset == dirname], model_prop_dict, trials)
    if trial == 0:
        return     

    # lists for collecting results
    val_losses = []
    epochs_all = []
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
            if architecture != 'Hybrid':
                train_df = insert_avgs(train_df, averages)
                val_df = insert_avgs(val_df, averages)  
        
        # generate datasets
        if architecture == 'Hybrid':
            train_generator = Hgenerator(train_df, averages, n_steps_in, n_steps_out, n_features, batch_size=bat_size,stateful=stateful)
            test_generator = Hgenerator(val_df, averages, n_steps_in, n_steps_out, n_features, batch_size=bat_size,stateful=stateful)
            model = hybrid(model_prop_dict, train_df, val_df)
        elif architecture == 'PLCnet':
            train_generator = PLCgenerator(train_df, n_steps_in, n_steps_out, batch_size=bat_size,stateful=stateful)
            test_generator = PLCgenerator(val_df, n_steps_in, n_steps_out, batch_size=bat_size,stateful=stateful)
            model = plc(model_prop_dict, train_df, val_df)
        else:
            train_generator = NHgenerator(train_df, n_steps_in, n_steps_out, n_features, architecture, batch_size=bat_size,stateful=stateful)
            test_generator = NHgenerator(val_df, n_steps_in, n_steps_out, n_features, architecture, batch_size=bat_size,stateful=stateful)
            model = non_hybrid(model_prop_dict, train_df, val_df)     
            
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
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=n_epoch//8,restore_best_weights=True)
        start = datetime.now()
        if stateful:
            history = model.fit(
                train_generator,
                validation_data= test_generator,
                validation_steps=len(test_generator),
                epochs=n_epoch,
                batch_size=bat_size,
                shuffle=False,
                callbacks=[callback]
                )            
        else:
            history = model.fit(
                train_generator,
                validation_data= test_generator,
                validation_steps=len(test_generator),
                epochs=n_epoch,                    
                shuffle=False,
                callbacks=[callback]
                )           
        end = datetime.now()   
        duration = end-start
        
        # collect relevant results of hyperparameter search trial
        val_hist = history.history['val_loss']   
        train_hist = history.history['loss']        
        val_losses.append(min(val_hist))
        epochs_all.append(val_hist.index(min(val_hist)) + 1)   
        durs_all.append(duration.total_seconds())        
        
        # export model and learning curves
        model_name = uuid.uuid4().hex        
        filename = export_path + 'models/' + str(model_name) + '.sav'
        pickle.dump(model, open(filename, 'wb'))         
        train_hist = pd.DataFrame(train_hist)
        train_hist.to_csv(export_path + 'models/' + str(model_name) + '_train_hist.csv')
        val_hist = pd.DataFrame(val_hist)
        val_hist.to_csv(export_path + 'models/' + str(model_name) + '_val_hist.csv')
        
        
        # reset tensorflow to avoid memory congestion
        reset_tensorflow_keras_backend(os.environ)       
        
    # collect average results 
    model_prop_dict['n_epoch'] = int(np.mean(epochs_all) // 1)
    model_prop_dict['min_val_loss'] = np.mean(val_losses)
    model_prop_dict['dur'] = np.mean(durs_all)          
    model_prop_dict['trial'] = trial                  
    row = pd.DataFrame(columns=glob_res_cols)    
    row.dataset = [dirname]               
    row.model = [model_prop_dict]     
        
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
    
    
    



                
                
    
    
     
        


