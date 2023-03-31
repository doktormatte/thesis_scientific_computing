import numpy  as np 
from numpy import array 
from tensorflow.keras.utils import Sequence


# Function for splitting input sequences into subsequences with defined input and output lengths
def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out-1
        if out_end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# Function for reading data for non-hybrid models, and preprocess it based on the selected architecture
def read_data_nh(df, n_steps_in, n_steps_out, n_features, architecture):    
    t_win = n_steps_in*n_steps_out
    Z = df
    Z=Z.to_numpy()   
    
    if architecture in ['CNN_LSTM', 'CNN_BiLSTM', 'CNN_GRU']:
        X, y = split_sequences(Z, t_win, n_steps_out )
        X = X.reshape((X.shape[0], n_steps_in, n_steps_out, n_features))
    elif architecture == 'ConvLSTM':
        X, y = split_sequences(Z, t_win, n_steps_out )
        X = X.reshape((X.shape[0], n_steps_in, 1, n_steps_out, n_features))
    else:
        X, y = split_sequences(Z, n_steps_in, n_steps_out)   
        
    return X, y

# Function for reading data for Hybrid LSTM, and preprocess it
def read_data_h(df1, df2, n_steps_in,n_steps_out,n_features):     
    Z = df1
    Z = Z.to_numpy()     
    X, y = split_sequences(Z[:,10:10+n_features+1], n_steps_in, n_steps_out)     
    Z1 = df2    
    Z1 = Z1.to_numpy()
    Z1 = Z1.transpose()
    Z2 = np.concatenate((Z1,Z1),axis=1) 
    X2 = np.zeros([len(Z),10+96],float)  
  
    for i in range(len(Z)-n_steps_in): 
     if Z[i+n_steps_in-1,-1] == 0:             
          qq = np.array(Z2[0][0:96])
          X2[i] = np.append(Z[i+n_steps_in-1][0:10],qq) 
     else:            
         qq = np.array(Z2[1][0:96])
         X2[i] = np.append(Z[i+n_steps_in-1][0:10],qq) 
    
    X2 = X2[0:len(X),]
    return X, y, X2

# Function for reading data for PLCnet, and preprocess it
def read_data_plc(df, n_steps_in, n_steps_out):       
    Z = df
    Z=Z.to_numpy()      
    X, y = split_sequences(Z, n_steps_in, n_steps_out)           
    return X, y

# Function for inserting lagged variables into given dataframe
def insert_lags(df, n_steps_in):
    for n in range(n_steps_in):
        df.insert(10+n, 'lag_' + str(n+1), [0.0]*len(df), True)
    x = df['value']
    for n in range(n_steps_in):
        df['lag_' + str(n+1)] = np.roll(x,n+1)
    df.insert(10, 'lag_1w', np.roll(x,96*7), True)
    df.insert(10, 'lag_1d', np.roll(x,96), True)
    df = df[96*7:]      
    return df

# Function to insert observation averages into given dataframe
def insert_avgs(df, averages):
    for m in range(96):
        df.insert(10+m, str(10+m), [0.0]*len(df), True)    
    df.loc[df['weekend'] == 0, [str(x) for x in list(range(10,106))]] = np.array(averages.weekday.T)
    df.loc[df['weekend'] == 1, [str(x) for x in list(range(10,106))]] = np.array(averages.weekend.T)  
    return df    


# Generator class for non-hybrid models, inherits from Sequence class
class NHgenerator(Sequence):
    def __init__(self, df, n_steps_in, n_steps_out, n_features, architecture, batch_size,stateful=False):
        self.df = df        
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out
        self.n_features = n_features       
        self.architecture = architecture
        self.batch_size = batch_size        
        
        # Call your data read function to create the input data and target values
        X, y = read_data_nh(self.df, self.n_steps_in, self.n_steps_out, self.n_features, self.architecture)
        
        if stateful: 
            train_size = X.shape[0]
            train_size = train_size-(train_size % 64)            
            X = X[:train_size]            
            y = y[:train_size]           
            self.batch_size = 64
        
        self.data = X
        self.targets = y         


    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_data = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_targets = self.targets[idx * self.batch_size:(idx + 1) * self.batch_size]        

        return batch_data, batch_targets
    
    def get_data(self):
        return self.targets


# Generator class for Hybrid LSTM, inherits from Sequence class
class Hgenerator(Sequence):
    def __init__(self, df, df2, n_steps_in, n_steps_out, n_features, batch_size,stateful=False):
        self.df = df
        self.df2 = df2
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out
        self.n_features = n_features        
        self.batch_size = batch_size        
        
        # Call your data read function to create the input data and target values
        X, y, X2 = read_data_h(self.df, self.df2, self.n_steps_in, self.n_steps_out, self.n_features)
        
        if stateful: 
            train_size = X.shape[0]
            train_size = train_size-(train_size % 64)            
            X = X[:train_size]
            X2 = X2[:train_size]
            y = y[:train_size]           
            self.batch_size = 64
        
        self.data = X
        self.targets = y  
        self.data2 = X2


    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_data = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_targets = self.targets[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_data2 = self.data2[idx * self.batch_size:(idx + 1) * self.batch_size]

        return [batch_data,batch_data2], batch_targets
    
    def get_data(self):
        return self.targets


# Generator class for PLCnet, inherits from Sequence class
class PLCgenerator(Sequence):
    def __init__(self, df, n_steps_in, n_steps_out, batch_size,stateful=False):
        self.df = df
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out
        # self.n_features = n_features        
        self.batch_size = batch_size        
        
        # Call your data read function to create the input data and target values
        X, y = read_data_plc(self.df, self.n_steps_in, self.n_steps_out)
        
        if stateful: 
            train_size = X.shape[0]
            train_size = train_size-(train_size % 64)            
            X = X[:train_size]
            y = y[:train_size]           
            self.batch_size = 64
        
        self.data = X
        self.targets = y  


    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_data = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_targets = self.targets[idx * self.batch_size:(idx + 1) * self.batch_size]

        return [batch_data,batch_data], batch_targets
    
    def get_data(self):
        return self.targets
    
    
