import time
import os
import ast


def is_locked(filepath):
    """Checks if a file is locked by opening it in append mode.
    If no exception thrown, then the file is not locked.
    """
    locked = None
    file_object = None
    if os.path.exists(filepath):
        try:
            print("Trying to open " +  filepath)
            buffer_size = 8
            # Opening file in append mode and read the first 8 characters.
            file_object = open(filepath, 'a', buffer_size)
            if file_object:
                print (filepath + " is not locked.")
                locked = False
        except IOError as e:
            print("File is locked (unable to open in append mode.) " + e)
            locked = True
        finally:
            if file_object:
                file_object.close()
                print(filepath + " closed.")
    else:
        print(filepath + " not found.")
    return locked

def wait_for_files(filepaths):
    """Checks if the files are ready.

    For a file to be ready it must exist and can be opened in append
    mode.
    """
    wait_time = 5
    for filepath in filepaths:
        # If the file doesn't exist, wait wait_time seconds and try again
        # until it's found.
        while not os.path.exists(filepath):
            print(filepath + " hasn't arrived. Waiting ")
            time.sleep(wait_time)
        # If the file exists but locked, wait wait_time seconds and check
        # again until it's no longer locked by another process.
        while is_locked(filepath):
            print(filepath + " is currently in use. Waiting ") 
            time.sleep(wait_time)
            

# check how many hyperparameter search trials were already conducted for the given model and forecasting horizon
def check_progress_hyper(df_progress, model_prop_dict, trials):
    trial = 1
    mode = model_prop_dict['mode']
    architecture = model_prop_dict['architecture']
    n_steps_out = model_prop_dict['n_steps_out']    
    for i in range(len(df_progress['model'].values)):              
        dict_progress = ast.literal_eval(df_progress['model'].values[i])        
        if 'architecture' in dict_progress:
            if dict_progress['architecture'] == architecture:
                if dict_progress['n_steps_out'] == n_steps_out:
                    if 'min_val_loss' in dict_progress:                 
                        if 'mode' in dict_progress: 
                            if dict_progress['mode'] == mode:
                                if 'trial' in dict_progress:
                                    trial += 1
    if trial > trials:
        return 0
    return trial

# check whether the forecaster with the given hyperparameter configuration was already evaluated 
def check_progress_forecast(df_progress, model_prop_dict):
    for i in range(len(df_progress['model'].values)):                 
        dict_progress = ast.literal_eval(df_progress['model'].values[i] ) 
        if dict_progress == model_prop_dict:
            return 0
        return 1

# get best performing hyperparameter configuration for a given model and forecasting horizon
def get_best_config(df_progress, model_prop_dict):    
    ind_val = []
    mode = model_prop_dict['mode']
    architecture = model_prop_dict['architecture']  
    n_steps_out = model_prop_dict['n_steps_out'] 
    for i in range(len(df_progress['model'].values)): 
        dict_progress = ast.literal_eval(df_progress['model'].values[i])    
        if 'architecture' in dict_progress:
            if dict_progress['architecture'] == architecture:
                if dict_progress['n_steps_out'] == n_steps_out:
                    if 'min_val_loss' in dict_progress:                 
                        if 'mode' in dict_progress: 
                            if dict_progress['mode'] == mode:    
                                ind_val.append([dict_progress['min_val_loss'],i])
    ind_val = sorted(ind_val, key=lambda x: x[0], reverse = False)    
    if len(ind_val) > 0:        
        return ast.literal_eval(df_progress['model'].values[ind_val[0][1]]) 
    return 0
    

