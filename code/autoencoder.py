import scipy
import numpy as np
from keras.layers import Dense, Flatten, Input, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model
import pandas as pd
import random

from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, LSTM, RepeatVector
from keras.models import Model, Sequential
from keras.models import model_from_json
from keras import regularizers
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
import itertools

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from sklearn.cluster import DBSCAN
from collections import Counter

def load_data(path):
    #path to data
    #data is a matrix num_datapoints x num_dimension
    df = pd.read_csv(path, header=None)
    data = df.values
    if(data.shape[0] < data.shape[1]):
        data = np.transpose(data)
    return data

def random_signal(num_dim):
    #generate a smooth signal based on random-walk alg
    random_walk = list()
    random_walk.append(-1 if random.random() < 0.5 else 1)
    for i in range(1, num_dim):
        movement = -1 if random.random() < 0.5 else 1
        value = random_walk[i-1] + movement
        random_walk.append(value)  
    
    x = np.array(list(range(len(random_walk))))
    y = np.array(random_walk)
    p30 = np.poly1d(np.polyfit(x, y, 10))
    return p30(x)/max(p30(x))   

def generate_signal(data):
    #this is for training autoencoder
    #generate 3 signals
    num_dim = data.shape[1]
    signal1 = random_signal(num_dim)
    signal2 = random_signal(num_dim)
    signal3 = random_signal(num_dim)
    
    #generate a random abundance fraction for the 3 signals
    #this allows more variation in signal shapes
    prob_list = []
    for i in range(0, 3):
        prob_list.append(random.randint(1, 10))
    prob_list1 = [x / sum(prob_list) for x in prob_list]    
    combined = prob_list1[0]*signal1 + prob_list1[1]*signal2 + prob_list1[2]*signal3
    
    #gaussian noise
    noise_r = np.random.normal(0,0.2,combined.shape[0])
    noise = np.random.poisson(5, combined.shape[0])
    
    combined_noisy = combined + noise_r + noise
    return combined, combined_noisy, prob_list1

def autoencoder(data_train, data_test, visualization = False):
    #model
    x_train_enc = []
    x_train_dec = []
    x_test_enc = []
    x_test_dec = []
    for val in data_train:
        x_train_enc.append(val[0])
        x_train_dec.append(val[1])
    for val in data_test:
        x_test_enc.append(val[0])
        x_test_dec.append(val[1])
        
    x_train_enc = np.array(x_train_enc)
    x_train_dec = np.array(x_train_dec)
    x_test_enc = np.array(x_test_enc)
    x_test_dec = np.array(x_test_dec)
    
    #x_train_enc = x_train_enc.reshape((len(x_train_enc), np.prod(x_train_enc.shape[1:]), 1))
    #x_train_dec = x_train_dec.reshape((len(x_train_dec), np.prod(x_train_dec.shape[1:]), 1))
    #x_test_enc = x_test_enc.reshape((len(x_test_enc), np.prod(x_test_enc.shape[1:]), 1))
    #x_test_dec = x_test_dec.reshape((len(x_test_dec), np.prod(x_test_dec.shape[1:]), 1))
    
    window_length = x_train_enc.shape[1]
    
    #===========================================================================
    #structure of the deep autoencoder
    
    input_window = Input(shape=(window_length,))
    #input_window = Input(shape=(window_length,1))
    encoding_dim = 15
    x = Dense(80, activation='relu')(input_window)
    x = BatchNormalization()(x)
    x = Dense(60, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(40, activation='relu')(x)
    x = BatchNormalization()(x)     
    x = Dense(20, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(15, activation='relu')(x)
    x = BatchNormalization()(x)    
    #embedding dimension is 15
    encoded = Dense(encoding_dim, activation='relu')(x)
    x = Dense(15, activation='relu')(encoded)
    x = BatchNormalization()(x)   
    x = Dense(20, activation='relu')(x)
    x = BatchNormalization()(x)  
    x = Dense(40, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(60, activation='relu')(x)
    x = BatchNormalization()(x)    
    x = Dense(80, activation='relu')(x)
    x = BatchNormalization()(x)
    decoded = Dense(window_length, activation='sigmoid')(x)
    autoencoder = Model(input_window, decoded)
    
    autoencoder.summary()    
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    history = autoencoder.fit(x_train_enc, x_train_dec,
                              epochs=5,
                              batch_size=1024,
                              shuffle=True,
                              validation_data=(x_test_enc, x_test_dec))   
    #=======================================================================
    
    denoised_signals = autoencoder.predict(x_test_enc)  
    noises = []
    
    #this is for visualization of predicited noise vs. real noise (on 10 data points)
    #set visualization parameter to be True to see the plots
    if(visualization):
        for i in range(0, 10):
            current_denoised_signal = denoised_signals[i].reshape(data_train[0][0].shape[0],)
            current_noisy_signal = x_test_enc[i].reshape(data_train[0][0].shape[0],)
            predicted_noise = current_noisy_signal - current_denoised_signal
            noises.append(predicted_noise)
            ground_truth_noise = x_test_enc[i] - x_test_dec[i]
            plt.plot(ground_truth_noise)
            plt.plot(predicted_noise)
            plt.gca().legend(('ground truth', 'predicted'))
            plt.show()
    return autoencoder

def generate_data(data, size):
    output = []
    for i in range(size):
        combined, combined_noisy, prob_list1 = generate_signal(data)
        output.append([combined_noisy, combined, prob_list1])
    return output

def denoise_data(data, trained_autoencoder, output_file_path):
    output_data = []
    num_dim = data[0].shape[0]
    for i in range(0, data.shape[0]):
        current_data = data[i]
        combined_acinput = np.array(current_data).reshape(1, num_dim)
        denoised_signal = trained_autoencoder.predict(combined_acinput)
        denoised_signal = denoised_signal.reshape(num_dim,)
        output_data.append(denoised_signal)
        
    df_out = pd.DataFrame(output_data)
    df_out.to_csv(output_file_path, index=False, header=None)    
    

def do_all():
    path = '../data/new651_data.csv'
    data = load_data(path)
    #currently set training data size to be 300, validation to be 100 for code testing purpose
    #increase these 2 numbers for better model - training size should be more than 100,000
    data_train = generate_data(data, 300)
    data_test = generate_data(data, 100)
    au_enc = autoencoder(data_train, data_test, visualization=False)
    denoise_data(data, au_enc, "../data/new651_data_denoised.csv")
    
#modify the data you want to denoise, output denoised data path and training data size in do_all()   
#i did not implement code to save the model and reuse the trained model for future predictions, this
#could be something good to do for saving training time
#data of different dimensions require different models (current trained dimension is 351)
do_all()
