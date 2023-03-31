import tensorflow as tf
import os
import gc
    

# Set policy to mixed precision if GPU is available  
def set_policy(env):    
    try:
        if 'gpu' in env['CONDA_DEFAULT_ENV']:    
            from tensorflow.compat.v1 import ConfigProto
            from tensorflow.compat.v1 import InteractiveSession
            config = ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.4
            config.gpu_options.allow_growth = True
            session = InteractiveSession(config=config)
            policyConfig = 'mixed_float16'
            policy = tf.keras.mixed_precision.Policy(policyConfig)
            tf.keras.mixed_precision.set_global_policy(policy)
        else:
           env['CUDA_VISIBLE_DEVICES'] = '-1'
    except Exception:
        pass



# Set the environment to 'local' if running on a local machine, otherwise set it to 'cloud'
def set_paths():
    env = 'cloud'
    if (os.path.expanduser("~") == '/home/doktormatte'):
        env = 'local'
    # Set the import and export paths based on the environment (local or cloud)
    import_path = '/home/doktormatte/Dropbox/Dokumente/Studium/MA_SciComp/Data/'
    export_path = '/home/doktormatte/Dropbox/Dokumente/Studium/MA_SciComp/Data/results/'
    if env == 'cloud':
        import_path = '/content/data/Data/'
        export_path = '/drive/MyDrive/'
    return import_path, export_path



    
# Reset the TensorFlow and Keras backend to clear session and reset the default graph
def reset_tensorflow_keras_backend(env):    
    import tensorflow as tf    
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    _ = gc.collect()
    try:
        if 'gpu' in env['CONDA_DEFAULT_ENV']:
            policyConfig = 'mixed_float16'
            policy = tf.keras.mixed_precision.Policy(policyConfig)
            tf.keras.mixed_precision.set_global_policy(policy)
    except Exception:
        pass


# Set seeds for reproducibility in various modules (Python hash, TensorFlow, etc.)
def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)



