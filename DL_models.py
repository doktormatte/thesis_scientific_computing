
import tensorflow as tf
from tensorflow import keras
from keras.layers import LSTM,GRU,ConvLSTM2D
from keras.layers import Dense,Dropout,Flatten,TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D 
from keras.models import Model   
    
	


# non hybrid forecasters
def non_hybrid(model_prop_dict, train_df, val_df):               
      
    architecture = model_prop_dict['architecture']  
    n_features = model_prop_dict['n_features'] 
    n_steps_in = model_prop_dict['n_steps_in']  
    n_steps_out = model_prop_dict['n_steps_out'] 
    po_size = model_prop_dict['po_size'] 
    nf_1 = model_prop_dict['nf_1'] 
    nf_2 = model_prop_dict['nf_2']  
    nf_3 = model_prop_dict['nf_3']  
    nf_4 = model_prop_dict['nf_4']  
    ker_size = model_prop_dict['ker_size'] 
    stacked = model_prop_dict['stacked'] 
    stack_size = model_prop_dict['stack_size'] 
    nodes_dense_1=model_prop_dict['nodes_dense_1']    
    nodes_rec_1=model_prop_dict['nodes_rec_1']  
    nodes_rec_2=model_prop_dict['nodes_rec_2']  
    nodes_rec_3=model_prop_dict['nodes_rec_3']  
    nodes_rec_4=model_prop_dict['nodes_rec_4']
    dropout_1 = model_prop_dict['dropout_1']      
    bat_size = model_prop_dict['bat_size']     
    kernel_reg_1 = model_prop_dict['kernel_reg_1']  
    kernel_reg_2 = model_prop_dict['kernel_reg_2'] 
    kernel_reg_3 = model_prop_dict['kernel_reg_3'] 
    kernel_reg_4=model_prop_dict['kernel_reg_4']     
    stack_conv = model_prop_dict['stack_conv'] 
    dilate = int(model_prop_dict['dilate'])    
    stateful = model_prop_dict['stateful'] 
        
    model = keras.Sequential()
    
    if architecture == 'Conv1D':
        if stateful:
            model.add(Conv1D(
                kernel_regularizer=kernel_reg_1,                                     
                filters=nf_1,
                kernel_size=ker_size,
                activation='relu',
                batch_input_shape=(bat_size,n_steps_in, n_features))) 
        else:                
            model.add(Conv1D(
                kernel_regularizer=kernel_reg_1,                                     
                filters=nf_1,
                kernel_size=ker_size,
                activation='relu',
                input_shape=(n_steps_in, n_features))) 
        if stack_conv == 2:
            model.add(Conv1D(
                kernel_regularizer=kernel_reg_2,                                    
                filters=nf_2,
                kernel_size=ker_size,
                activation='relu',
                dilation_rate=2**dilate)) 
        if stack_conv == 3:
            model.add(Conv1D(
                kernel_regularizer=kernel_reg_2,                    
                filters=nf_2,
                kernel_size=ker_size,
                activation='relu', 
                dilation_rate=2**dilate))
            model.add(Conv1D(
                kernel_regularizer=kernel_reg_3,                    
                filters=nf_3,
                kernel_size=ker_size,
                activation='relu',
                dilation_rate=4**dilate)) 
        if stack_conv == 4:
            model.add(Conv1D(
                kernel_regularizer=kernel_reg_2,
                filters=nf_2,
                kernel_size=ker_size,
                activation='relu',
                dilation_rate=2**dilate))
            model.add(Conv1D(
                kernel_regularizer=kernel_reg_3,                                     
                filters=nf_3,
                kernel_size=ker_size,
                activation='relu',
                dilation_rate=4**dilate)) 
            model.add(Conv1D(
                kernel_regularizer=kernel_reg_4,                                     
                filters=nf_4, 
                kernel_size=ker_size, 
                activation='relu', 
                dilation_rate=8**dilate)) 
        model.add(MaxPooling1D(pool_size=po_size))
        model.add(Flatten())   
        model.add(Dense(
            nodes_dense_1,
            kernel_regularizer=kernel_reg_3,                 
            activation='relu'))
        model.add(Dropout(dropout_1)) 
        model.add(Dense(n_steps_out, activation='sigmoid',dtype=tf.float32))
        
        
    if architecture == 'ConvLSTM':
        if stateful:
            model.add(ConvLSTM2D(
                filters=nf_1,
                kernel_size=(1,ker_size),
                activation='tanh',
                stateful=stateful,
                batch_input_shape=(bat_size,n_steps_in, 1, n_steps_out, n_features)))
        else:
            model.add(ConvLSTM2D(
                filters=nf_1,
                kernel_size=(1,ker_size),
                activation='tanh',
                stateful=stateful,
                input_shape=(n_steps_in, 1, n_steps_out, n_features)))
              
        model.add(Flatten())
        model.add(Dense(
            nodes_dense_1,
            kernel_regularizer=kernel_reg_1,                              
            activation='relu'))
        model.add(Dropout(dropout_1))  
        model.add(Dense(n_steps_out, activation='sigmoid',dtype=tf.float32))
        
    
    if architecture == 'CNN_LSTM':
        if stateful:
            model.add(TimeDistributed(Conv1D(
                kernel_regularizer=kernel_reg_1,                    
                filters=nf_1,
                kernel_size=ker_size,
                padding='same',
                activation='relu'),
                batch_input_shape=(bat_size,n_steps_in, n_steps_out, n_features)))
        else:
            model.add(TimeDistributed(Conv1D(
                kernel_regularizer=kernel_reg_1,                    
                filters=nf_1,
                kernel_size=ker_size,
                padding='same',
                activation='relu'),
                input_shape=(n_steps_in, n_steps_out, n_features)))
        model.add(TimeDistributed(MaxPooling1D(pool_size=po_size)))
        model.add(TimeDistributed(Conv1D(
            kernel_regularizer=kernel_reg_2,
            filters=nf_2,
            kernel_size=ker_size, 
            padding='same',
            activation='relu'))) 
        model.add(TimeDistributed(MaxPooling1D(pool_size=po_size,padding='same')))
        model.add(TimeDistributed(Flatten())) 
        model.add(LSTM(nodes_rec_1, stateful=stateful)) 
        model.add(Dense(
            nodes_dense_1,
            kernel_regularizer=kernel_reg_3,
            activation='relu'))
        model.add(Dropout(dropout_1))   
        model.add(Dense(n_steps_out, activation='sigmoid',dtype=tf.float32))
        

    if architecture == 'CNN_BiLSTM':
        if stateful:
            model.add(TimeDistributed(Conv1D(
                kernel_regularizer=kernel_reg_1,                    
                filters=nf_1, 
                kernel_size=ker_size,
                padding='same',
                activation='relu'),
                batch_input_shape=(bat_size,n_steps_in, n_steps_out, n_features)))
        else:
            model.add(TimeDistributed(Conv1D(
                kernel_regularizer=kernel_reg_1,                    
                filters=nf_1,
                kernel_size=ker_size,
                padding='same',activation='relu'),
                input_shape=(n_steps_in, n_steps_out, n_features)))
        model.add(TimeDistributed(MaxPooling1D(pool_size=po_size)))
        model.add(TimeDistributed(Conv1D(
            kernel_regularizer=kernel_reg_2,
            filters=nf_2,
            kernel_size=ker_size, 
            padding='same',
            activation='relu'))) 
        model.add(TimeDistributed(MaxPooling1D(pool_size=po_size,padding='same')))
        model.add(TimeDistributed(Flatten())) 
        model.add(keras.layers.Bidirectional(LSTM(nodes_rec_1, stateful=stateful))) 
        model.add(Dense(
            nodes_dense_1,
            kernel_regularizer=kernel_reg_3,
            activation='relu'))
        model.add(Dropout(dropout_1))   
        model.add(Dense(n_steps_out, activation='sigmoid',dtype=tf.float32))
        

    if architecture == 'CNN_GRU':
        if stateful:
            model.add(TimeDistributed(Conv1D(
                kernel_regularizer=kernel_reg_1,                    
                filters=nf_1, 
                kernel_size=ker_size,
                padding='same',
                activation='relu'),
                batch_input_shape=(bat_size,n_steps_in, n_steps_out, n_features)))
        else:
            model.add(TimeDistributed(Conv1D(
                kernel_regularizer=kernel_reg_1,                    
                filters=nf_1, 
                kernel_size=ker_size,
                padding='same',
                activation='relu'), 
                input_shape=(n_steps_in, n_steps_out, n_features)))
        model.add(TimeDistributed(MaxPooling1D(pool_size=po_size)))
        model.add(TimeDistributed(Conv1D(
            kernel_regularizer=kernel_reg_2,
            filters=nf_2,
            kernel_size=ker_size, 
            padding='same',
            activation='relu'))) 
        model.add(TimeDistributed(MaxPooling1D(pool_size=po_size,padding='same')))
        model.add(TimeDistributed(Flatten())) 
        model.add(GRU(nodes_rec_1, stateful=stateful))
        model.add(Dense(
            nodes_dense_1,
            kernel_regularizer=kernel_reg_3,
            activation='relu'))
        model.add(Dropout(dropout_1))   
        model.add(Dense(n_steps_out, activation='sigmoid',dtype=tf.float32))
        

    if architecture == 'Stacked':
        if stacked == 1:
            if stateful:
                if stack_size == 2:
                    model.add(keras.layers.LSTM(
                        units = nodes_rec_1,
                        return_sequences=True,
                        batch_input_shape=(bat_size,n_steps_in,train_df.shape[1]-1),
                        activation='tanh',
                        stateful=stateful))
                    model.add(keras.layers.LSTM(nodes_rec_2,activation='tanh', stateful=stateful))
                if stack_size == 3:
                    model.add(keras.layers.LSTM(
                        units = nodes_rec_1,
                        return_sequences=True,
                        batch_input_shape=(bat_size,n_steps_in,train_df.shape[1]-1),
                        activation='tanh', stateful=stateful))
                    model.add(keras.layers.LSTM(
                        nodes_rec_2,
                        return_sequences=True,
                        activation='tanh',
                        stateful=stateful))
                    model.add(keras.layers.LSTM(
                        nodes_rec_3,
                        activation='tanh',
                        stateful=stateful))
                if stack_size == 4:
                    model.add(keras.layers.LSTM(
                        units = nodes_rec_1,
                        return_sequences=True,
                        batch_input_shape=(bat_size,n_steps_in,train_df.shape[1]-1),
                        activation='tanh', 
                        stateful=stateful))
                    model.add(keras.layers.LSTM(
                        nodes_rec_2,
                        return_sequences=True,
                        activation='tanh', 
                        stateful=stateful))
                    model.add(keras.layers.LSTM(
                        nodes_rec_3,
                        return_sequences=True,
                        activation='tanh',
                        stateful=stateful))
                    model.add(keras.layers.LSTM(
                        nodes_rec_4,
                        activation='tanh',
                        stateful=stateful))
            else:
                if stack_size == 2:
                    model.add(keras.layers.LSTM(
                        units = nodes_rec_1,
                        return_sequences=True,
                        input_shape=(n_steps_in,train_df.shape[1]-1),
                        activation='tanh',
                        stateful=stateful))
                    model.add(keras.layers.LSTM(
                        nodes_rec_2,activation='tanh',
                        stateful=stateful))
                if stack_size == 3:
                    model.add(keras.layers.LSTM(
                        units = nodes_rec_1,
                        return_sequences=True,
                        input_shape=(n_steps_in,train_df.shape[1]-1),
                        activation='tanh',
                        stateful=stateful))
                    model.add(keras.layers.LSTM(
                        nodes_rec_2,
                        return_sequences=True,
                        activation='tanh', 
                        stateful=stateful))
                    model.add(keras.layers.LSTM(
                        nodes_rec_3,
                        activation='tanh', 
                        stateful=stateful))
                if stack_size == 4:
                    model.add(keras.layers.LSTM(
                        units = nodes_rec_1,
                        return_sequences=True,
                        input_shape=(n_steps_in,train_df.shape[1]-1),
                        activation='tanh',
                        stateful=stateful))
                    model.add(keras.layers.LSTM(
                        nodes_rec_2,
                        return_sequences=True,
                        activation='tanh',
                        stateful=stateful))
                    model.add(keras.layers.LSTM(
                        nodes_rec_3,
                        return_sequences=True,
                        activation='tanh', 
                        stateful=stateful))
                    model.add(keras.layers.LSTM(
                        nodes_rec_4,
                        activation='tanh',
                        stateful=stateful))
                
        if stacked == 0:
            if stateful:
                if stack_size == 2:                    
                    model.add(keras.layers.GRU(
                        units = nodes_rec_1,
                        return_sequences=True,
                        batch_input_shape=(bat_size,n_steps_in,train_df.shape[1]-1),
                        activation='tanh', 
                        stateful=stateful))
                    model.add(keras.layers.GRU(nodes_rec_2,activation='tanh', stateful=stateful))   
                if stack_size == 3:                    
                    model.add(keras.layers.GRU(
                        units = nodes_rec_1,
                        return_sequences=True,
                        batch_input_shape=(bat_size,n_steps_in,train_df.shape[1]-1),
                        activation='tanh',
                        stateful=stateful))
                    model.add(keras.layers.GRU(
                        nodes_rec_2,
                        return_sequences=True,
                        activation='tanh', 
                        stateful=stateful))   
                    model.add(keras.layers.GRU(
                        nodes_rec_3,
                        activation='tanh', 
                        stateful=stateful))  
                if stack_size == 4:
                    model.add(keras.layers.GRU(
                        units = nodes_rec_1,
                        return_sequences=True,
                        batch_input_shape=(bat_size,n_steps_in,train_df.shape[1]-1),
                        activation='tanh',
                        stateful=stateful))
                    model.add(keras.layers.GRU(
                        nodes_rec_2,
                        return_sequences=True,
                        activation='tanh',
                        stateful=stateful))   
                    model.add(keras.layers.GRU(
                        nodes_rec_3,
                        return_sequences=True,
                        activation='tanh',
                        stateful=stateful))
                    model.add(keras.layers.GRU(
                        nodes_rec_4,
                        activation='tanh',
                        stateful=stateful))
            else:
                if stack_size == 2:                    
                    model.add(keras.layers.GRU(
                        units = nodes_rec_1,
                        return_sequences=True,
                        input_shape=(n_steps_in,train_df.shape[1]-1),
                        activation='tanh',
                        stateful=stateful))
                    model.add(keras.layers.GRU(
                        nodes_rec_2,
                        activation='tanh',
                        stateful=stateful))   
                if stack_size == 3:                    
                    model.add(keras.layers.GRU(
                        units = nodes_rec_1,
                        return_sequences=True,
                        input_shape=(n_steps_in,train_df.shape[1]-1),
                        activation='tanh',
                        stateful=stateful))
                    model.add(keras.layers.GRU(
                        nodes_rec_2,
                        return_sequences=True,
                        activation='tanh',
                        stateful=stateful))   
                    model.add(keras.layers.GRU(
                        nodes_rec_3,
                        activation='tanh',
                        stateful=stateful))  
                if stack_size == 4:
                    model.add(keras.layers.GRU(
                        units = nodes_rec_1,
                        return_sequences=True,
                        input_shape=(n_steps_in,train_df.shape[1]-1),
                        activation='tanh',
                        stateful=stateful))
                    model.add(keras.layers.GRU(
                        nodes_rec_2,
                        return_sequences=True,
                        activation='tanh',
                        stateful=stateful))   
                    model.add(keras.layers.GRU(
                        nodes_rec_3,
                        return_sequences=True,
                        activation='tanh',
                        stateful=stateful))
                    model.add(keras.layers.GRU(
                        nodes_rec_4,
                        activation='tanh',
                        stateful=stateful))                  
                    
        if stacked == 2:
            if stateful:
                if stack_size == 2:
                    model.add(keras.layers.Bidirectional(keras.layers.LSTM(
                        units = nodes_rec_1,
                        return_sequences=True,
                        batch_input_shape=(bat_size,n_steps_in,train_df.shape[1]-1),
                        activation='tanh', 
                        stateful=stateful)))
                    model.add(keras.layers.Bidirectional(keras.layers.LSTM(
                        nodes_rec_2,
                        activation='tanh',
                        stateful=stateful)))
                if stack_size == 3:
                    model.add(keras.layers.Bidirectional(keras.layers.LSTM(
                        units = nodes_rec_1,
                        return_sequences=True,
                        batch_input_shape=(bat_size,n_steps_in,train_df.shape[1]-1),
                        activation='tanh', 
                        stateful=stateful)))
                    model.add(keras.layers.Bidirectional(keras.layers.LSTM(
                        nodes_rec_2,
                        return_sequences=True,
                        activation='tanh',
                        stateful=stateful)))
                    model.add(keras.layers.Bidirectional(keras.layers.LSTM(
                        nodes_rec_3,
                        activation='tanh',
                        stateful=stateful)))
                if stack_size == 4:
                    model.add(keras.layers.Bidirectional(keras.layers.LSTM(
                        units = nodes_rec_1,
                        return_sequences=True,
                        batch_input_shape=(bat_size,n_steps_in,train_df.shape[1]-1),
                        activation='tanh', 
                        stateful=stateful)))
                    model.add(keras.layers.Bidirectional(keras.layers.LSTM(
                        nodes_rec_2,return_sequences=True,
                        activation='tanh',
                        stateful=stateful)))
                    model.add(keras.layers.Bidirectional(keras.layers.LSTM(
                        nodes_rec_3,
                        return_sequences=True,
                        activation='tanh',
                        stateful=stateful)))
                    model.add(keras.layers.Bidirectional(keras.layers.LSTM(
                        nodes_rec_4,
                        activation='tanh',
                        stateful=stateful)))
            else:
                if stack_size == 2:
                    model.add(keras.layers.Bidirectional(keras.layers.LSTM(
                        units = nodes_rec_1,
                        return_sequences=True,
                        input_shape=(n_steps_in,train_df.shape[1]-1),
                        activation='tanh',
                        stateful=stateful)))
                    model.add(keras.layers.Bidirectional(keras.layers.LSTM(
                        nodes_rec_2,
                        activation='tanh',
                        stateful=stateful)))
                if stack_size == 3:
                    model.add(keras.layers.Bidirectional(keras.layers.LSTM(
                        units = nodes_rec_1,
                        return_sequences=True,
                        input_shape=(n_steps_in,train_df.shape[1]-1),
                        activation='tanh',
                        stateful=stateful)))
                    model.add(keras.layers.Bidirectional(keras.layers.LSTM(
                        nodes_rec_2,return_sequences=True,
                        activation='tanh', 
                        stateful=stateful)))
                    model.add(keras.layers.Bidirectional(keras.layers.LSTM(
                        nodes_rec_3,
                        activation='tanh',
                        stateful=stateful)))
                if stack_size == 4:
                    model.add(keras.layers.Bidirectional(keras.layers.LSTM(
                        units = nodes_rec_1,
                        return_sequences=True,
                        input_shape=(n_steps_in,train_df.shape[1]-1),
                        activation='tanh',
                        stateful=stateful)))
                    model.add(keras.layers.Bidirectional(keras.layers.LSTM(
                        nodes_rec_2,
                        return_sequences=True,
                        activation='tanh', 
                        stateful=stateful)))
                    model.add(keras.layers.Bidirectional(keras.layers.LSTM(
                        nodes_rec_3,
                        return_sequences=True,
                        activation='tanh', 
                        stateful=stateful)))
                    model.add(keras.layers.Bidirectional(keras.layers.LSTM(
                        nodes_rec_4,
                        activation='tanh',
                        stateful=stateful)))
            
        model.add(Dense(nodes_dense_1,
                            kernel_regularizer=kernel_reg_1,
                            activation='relu'))
        model.add(Dropout(dropout_1))  
        model.add(Dense(n_steps_out, activation='sigmoid',dtype=tf.float32)) 
        
    if architecture == 'BiLSTM':
        if stateful:
            model.add(keras.layers.Bidirectional(keras.layers.LSTM(
                units=nodes_rec_1,
                stateful=stateful,
                batch_input_shape=(bat_size,n_steps_in,train_df.shape[1]-1))))
        else:
            model.add(keras.layers.Bidirectional(keras.layers.LSTM(
                units=nodes_rec_1,
                stateful=stateful,
                input_shape=(n_steps_in,train_df.shape[1]-1))))
        model.add(Dense(
            nodes_dense_1,
            kernel_regularizer=kernel_reg_1,
            activation='relu'))
        model.add(Dropout(dropout_1))  
        model.add(Dense(n_steps_out, activation='sigmoid', dtype=tf.float32))               
        
    
    if architecture == 'GRU':
        if stateful:
            model.add(keras.layers.GRU(
                units = nodes_rec_1,
                stateful=stateful,
                batch_input_shape=(bat_size,n_steps_in,train_df.shape[1]-1)))
        else:
            model.add(keras.layers.GRU(
                units = nodes_rec_1, 
                stateful=stateful,
                input_shape=(n_steps_in,train_df.shape[1]-1)))
        model.add(Dense(
            nodes_dense_1,
            kernel_regularizer=kernel_reg_1,
            activation='relu'))
        model.add(Dropout(dropout_1))  
        model.add(Dense(n_steps_out, activation='sigmoid',dtype=tf.float32))                
        
        
    if architecture == 'LSTM':
        if stateful:
            model.add(keras.layers.LSTM(
                units = nodes_rec_1,
                stateful=stateful,
                batch_input_shape=(bat_size,n_steps_in,train_df.shape[1]-1)))
        else:
            model.add(keras.layers.LSTM(
                units = nodes_rec_1,
                stateful=stateful,
                input_shape=(n_steps_in,train_df.shape[1]-1)))
        model.add(Dense(
            nodes_dense_1,
            kernel_regularizer=kernel_reg_1,
            activation='relu'))
        model.add(Dropout(dropout_1))  
        model.add(Dense(n_steps_out, activation='sigmoid',dtype=tf.float32))                
        
    return model    
    

# Hybrid LSTM forecasters
def hybrid(model_prop_dict, train_df, val_df):  
    
    n_features=model_prop_dict['n_features']  
    n_steps_in=model_prop_dict['n_steps_in']      
    n_steps_out = model_prop_dict['n_steps_out']    
    dropout_1=model_prop_dict['dropout_1']  
    dropout_2=model_prop_dict['dropout_2']  
    nodes_rec_1=model_prop_dict['nodes_rec_1']  
    nodes_rec_2=model_prop_dict['nodes_rec_2']  
    nodes_rec_3=model_prop_dict['nodes_rec_3']  
    bat_size=model_prop_dict['bat_size']  
    nodes_dense_1=model_prop_dict['nodes_dense_1']  
    nodes_dense_2=model_prop_dict['nodes_dense_2']  
    nodes_dense_3=model_prop_dict['nodes_dense_3']  
    nodes_dense_4=model_prop_dict['nodes_dense_4']  
    nf_1=model_prop_dict['nf_1']  
    nf_2=model_prop_dict['nf_2'] 
    convolve=model_prop_dict['convolve']  
    stack_layers=model_prop_dict['stack_layers']  
    kernel_reg_1=model_prop_dict['kernel_reg_1']  
    kernel_reg_2=model_prop_dict['kernel_reg_2']  
    kernel_reg_3=model_prop_dict['kernel_reg_3']  
    kernel_reg_4=model_prop_dict['kernel_reg_4']  
    kernel_reg_5=model_prop_dict['kernel_reg_5']         
    stateful=model_prop_dict['stateful']         
    
    
    if convolve==1:                                           
        input2 = keras.Input(shape=(106,1))  
        meta_layer=Dense(
            nodes_dense_1,
            kernel_regularizer=kernel_reg_1,
            activation='relu')(input2)
        meta_layer = keras.layers.Conv1D(
            kernel_regularizer=kernel_reg_2,
            filters=nf_1,
            kernel_size=1,
            activation='relu')(input2)
        meta_layer = keras.layers.Conv1D(
            kernel_regularizer=kernel_reg_3,                
            filters=nf_2,
            kernel_size=1,
            activation='relu')(meta_layer)
        meta_layer = keras.layers.MaxPool1D(pool_size=2)(meta_layer)
        meta_layer = keras.layers.Flatten()(meta_layer)
    else:
        input2 = keras.Input(shape=(106,))  
        meta_layer = keras.layers.Dense(
            106,
            kernel_regularizer=kernel_reg_1,                                           
            activation='relu')(input2)                   
    
    if stateful:                
        input1 = keras.Input(shape=(n_steps_in, n_features),batch_size=bat_size)                        
    else:
        input1 = keras.Input(shape=(n_steps_in, n_features))      
        
    if stack_layers==3:
        model_LSTM = LSTM(
            nodes_rec_1,
            return_sequences=True,
            activation='tanh',
            stateful=stateful)(input1)
        model_LSTM = LSTM(
            nodes_rec_2,
            return_sequences=True,
            activation='tanh',
            stateful=stateful)(input1)                        
        model_LSTM=LSTM(
            nodes_rec_3,
            activation='tanh',
            stateful=stateful)(input1)
    if stack_layers==2:
        model_LSTM = LSTM(
            nodes_rec_1,
            return_sequences=True,
            activation='tanh', stateful=stateful)(input1)
        model_LSTM=LSTM(
            nodes_rec_3,
            activation='tanh',
            stateful=stateful)(input1)
    else:                        
        model_LSTM=LSTM(
            nodes_rec_3,
            activation='tanh',
            stateful=stateful)(input1)   
        
    model_LSTM=Dropout(dropout_1)(model_LSTM)
    model_LSTM=Dense(
        nodes_dense_1,
        kernel_regularizer=kernel_reg_4,
        activation='relu')(model_LSTM)                    
    meta_layer = keras.layers.Dense(
        nodes_dense_2,            
        kernel_regularizer=kernel_reg_5, 
        activation='relu')(meta_layer)    
    meta_layer = keras.layers.Dense(
        nodes_dense_3,
        activation='relu')(meta_layer)
    model_merge = keras.layers.concatenate([model_LSTM, meta_layer])
    model_merge = Dense(
        nodes_dense_4,
        activation='relu')(model_merge)
    model_merge = Dropout(dropout_2)(model_merge)     
    
    output = Dense(n_steps_out, activation='sigmoid',dtype=tf.float32)(model_merge)
    model = Model(inputs=[input1, input2], outputs=output) 
    
    return model  
    
    
            
# PLCnet forecasters
def plc(model_prop_dict, train_df, val_df):   
     
    
    n_features=model_prop_dict['n_features']  
    n_steps_in=model_prop_dict['n_steps_in']  
    n_steps_out=model_prop_dict['n_steps_out']   
    n_features=model_prop_dict['n_features']  
    n_steps_in=model_prop_dict['n_steps_in']      
    n_steps_out = model_prop_dict['n_steps_out']  
    dropout_1=model_prop_dict['dropout_1']  
    dropout_2=model_prop_dict['dropout_2']  
    nodes_rec_1=model_prop_dict['nodes_rec_1']  
    nodes_rec_2=model_prop_dict['nodes_rec_2']      
    bat_size=model_prop_dict['bat_size']  
    nodes_dense_1=model_prop_dict['nodes_dense_1']  
    nodes_dense_2=model_prop_dict['nodes_dense_2']  
    nodes_dense_3=model_prop_dict['nodes_dense_3']  
    nodes_dense_4=model_prop_dict['nodes_dense_4']  
    nf_1=model_prop_dict['nf_1']  
    nf_2=model_prop_dict['nf_2'] 
    nf_3=model_prop_dict['nf_3'] 
    kernel_reg_1=model_prop_dict['kernel_reg_1']  
    kernel_reg_2=model_prop_dict['kernel_reg_2']  
    kernel_reg_3=model_prop_dict['kernel_reg_3']  
    kernel_reg_4=model_prop_dict['kernel_reg_4']  
    kernel_reg_5=model_prop_dict['kernel_reg_5']             
    second_step=model_prop_dict['second_step']      
    first_conv=model_prop_dict['first_conv']  
    second_conv=model_prop_dict['second_conv']
    first_dense = model_prop_dict['first_dense']
    stateful=model_prop_dict['stateful']      

    
    input_conv = keras.Input(shape=(n_steps_in,n_features))  
    
    if first_conv == 1:   
        conv_layer = keras.layers.Conv1D(
                                kernel_regularizer=kernel_reg_1,
                                filters=nf_1,
                                kernel_size=1,
                                activation='relu')(input_conv)
        conv_layer = keras.layers.MaxPool1D(pool_size=2)(conv_layer)
        conv_layer = keras.layers.Conv1D(
                                kernel_regularizer=kernel_reg_2,
                                filters=nf_2,
                                kernel_size=1,
                                activation='relu')(conv_layer)
        conv_layer = keras.layers.Flatten()(conv_layer)
        if first_dense == 1:
            conv_layer = Dense(nodes_dense_1,
                               kernel_regularizer=kernel_reg_3,
                               activation='relu')(conv_layer)
    else:
        conv_layer = Dense(nodes_dense_2,
                           kernel_regularizer=kernel_reg_1,
                           activation='relu')(input_conv)
        conv_layer = keras.layers.Flatten()(conv_layer)
        conv_layer = Dense(nodes_dense_3,
                           kernel_regularizer=kernel_reg_2,
                           activation='relu')(conv_layer)
        conv_layer = Dropout(dropout_1)(conv_layer)
        
    if stateful: 
        input_LSTM = keras.Input(shape=(n_steps_in, n_features),batch_size=bat_size) 
    else:
        input_LSTM = keras.Input(shape=(n_steps_in, n_features)) 
    LSTM_layer = LSTM(nodes_rec_1, activation='tanh',stateful=stateful,return_sequences=False)(input_LSTM)  
    model_merge = keras.layers.concatenate([LSTM_layer, conv_layer])
    
    reshape_dim = 0
    if first_conv == 1: 
        if first_dense == 1:
            reshape_dim = nodes_dense_1 + nodes_rec_1   
        else:
            reshape_dim = conv_layer.shape[1] + nodes_rec_1    
    else:
        reshape_dim = nodes_dense_3 + nodes_rec_1    
    
    model_merge = keras.layers.Reshape((1,reshape_dim))(model_merge)
    
    if second_step == 1:
        if second_conv == 1:
            model_merge = keras.layers.Conv1D(
                kernel_regularizer=kernel_reg_4,
                filters=nf_3,
                kernel_size=1,
                activation='relu')(model_merge)
            model_merge = keras.layers.Flatten()(model_merge)    
        else:
            model_merge = LSTM(nodes_rec_2, activation='tanh')(model_merge)  
        
    model_merge = Dense(nodes_dense_4,
                        kernel_regularizer=kernel_reg_5,
                        activation='relu')(model_merge)
    model_merge = Dropout(dropout_2)(model_merge) 
    
    output = Dense(n_steps_out, activation='sigmoid',dtype=tf.float32)(model_merge)
    model = Model(inputs=[input_LSTM, input_conv], outputs=output) 
    
    return model