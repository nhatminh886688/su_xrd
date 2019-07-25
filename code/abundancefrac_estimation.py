import pandas as pd
import numpy as np
from random import random
import random
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import random
from numpy import diff
import itertools
from scipy import signal
from scipy.signal import resample
import nimfa

#this script implement the proposed matrix factorization method
#and also compare the performance with several other non-negative matrix factorization methods

def signal_change(input_signal, variation_ratio):
    output_signal = []
    for i in range(0, input_signal.shape[0]):
        lower_bound = input_signal[i] - variation_ratio*input_signal[i]
        upper_bound = input_signal[i] + variation_ratio*input_signal[i]
        output_signal.append(np.random.uniform(lower_bound, upper_bound))
    output_signal = np.array(output_signal)
    x = np.array(list(range(len(output_signal))))
    y = np.array(output_signal).reshape(x.shape[0],)
    p30 = np.poly1d(np.polyfit(x, y, 10))
    return p30(x).reshape(-1,1)

def generate_data(variation_ratio, num_dim):
    df1 = pd.read_csv('../data/groundtruth_tissues.csv', header=None)
    data1 = df1.values
    data1 = np.transpose(data1)
    signal1 = np.array(data1[1]).reshape(-1, 1)
    signal2 = np.array(data1[2]).reshape(-1, 1)
    signal3 = np.array(data1[4]).reshape(-1, 1)
    signal1 = signal.resample(signal1, num_dim)
    signal2 = signal.resample(signal2, num_dim)
    signal3 = signal.resample(signal3, num_dim)    
    
    Vs = []
    Hs = []
    Ws = []
    for i in range(0, 100):
        signal1_prime = signal_change(signal1, variation_ratio)
        signal2_prime = signal_change(signal2, variation_ratio)
        signal3_prime = signal_change(signal3, variation_ratio)
        
        W = np.concatenate((signal1_prime, signal2_prime, signal3_prime), axis = 1)
        H = np.random.rand(3 , 1)
        H = H / sum(H)
        V = np.matmul(W, H)
        Vs.append(V)
        Hs.append(H)
        Ws.append(W)
    return Vs, Ws, Hs

def normalized(input_H):
    return [x/sum(input_H) for x in input_H]

class nmf_mv2():
    #this class implements exactly Alg 2 in the write-up
    #for the variation ratio test, var_ratio needs to be specified
    #for experimental data, var_ratio can be ignored
    def __init__(self, V, W_init,iterations, n_split, data_path, var_ratio = None):
        self.V = V
        self.iterations = iterations    
        self.W = W_init
        num_row = W_init.shape[0]
        num_col = W_init.shape[1]
        if(var_ratio is None):
            self.W_low = np.zeros((num_row, num_col))
            self.W_up = np.ones((num_row, num_col))
        else:
            self.W_low = np.ones((num_row, num_col))*(1-var_ratio)
            self.W_up = np.ones((num_row, num_col)) *(1+var_ratio)           
        self.iternum = 1
        self.reconstructed_V = None
        self.H = None   
        self.n_split = n_split
        self.H_low = np.zeros(num_col)
        self.H_up = np.ones(num_col)
        self.n_split_H = 101
        self.prev_W = None
        self.data_path = data_path
                  
    def get_var(self, W_bool, row_num=None):
        #this fucntion generates all the combinations of H (for 3 endmembers, 5050 possible H are generated)
        #this function generates all combinations for W if W is chosen to be updated
        num_split = self.n_split
        if(W_bool):
            output_W = []
            pos_vals = []
            row_low = self.W_low[row_num]
            row_high = self.W_up[row_num]
            for i in range(len(row_low)):
                current_low = row_low[i]
                current_high = row_high[i]
                pos_vals.append(list(np.linspace(current_low, current_high, num_split + 1))) #inclusive
            
            output_W = list(itertools.product(*pos_vals))
            return output_W
        else:
            #generating W set, adding sum to 1 constraint
            num_split = self.n_split_H
            output_H = []
            pos_vals = []
            H_low = self.H_low
            H_high = self.H_up
            for i in range(len(H_low)):
                current_low = H_low[i]
                current_high = H_high[i]
                pos_vals.append(list(np.linspace(current_low, current_high, num_split + 1))) #inclusive
            temp_output_H = list(itertools.product(*pos_vals))
            for i in range(len(temp_output_H)):
                if(sum(temp_output_H[i]) == 1):
                    output_H.append(temp_output_H[i])
            return output_H
        
        
    def updateW(self):
        num_split = self.n_split
        num_row = self.W_up.shape[0]
        num_low = self.W_low.shape[0]
        H = np.array(self.H).reshape(-1,1)
        V = np.array(self.V).reshape(-1,)
        W_candidates = []
        for i in range(0, num_row):
            row_up = self.W_up[i]
            row_low = self.W_low[i]
            W_var = self.get_var(W_bool=True, row_num=i)
            error_rate = []
            for j in range(len(W_var)):
                current_candidate_W = np.array(W_var[j]).reshape(1,-1)
                pred_V = np.matmul(current_candidate_W, H)
                error_rate.append(abs(pred_V - V[i]))
            #best_candidate = list(W_var[error_rate.index(min(error_rate))])
            best_candidates = []
            
            #*** the 1.0 here can be change to a small number [1.0, 1.3] to
            #allow for more choice for updating W to make it smooth
            error_threshold = min(error_rate) * 1.0
            for j in range(len(W_var)):
                if(error_rate[j] <= error_threshold):
                    best_candidates.append(list(W_var[j]))
            #update all of W at once to create smooth curve
            #self.W[i] = best_candidate
            W_candidates.append(best_candidates)
            
            
            #update W_up and W_low here for the boundaries of the next iteration
            for j in range(0, len(self.W[i])):
                current_val = self.W[i][j]
                current_W_low = self.W_low[i][j]
                current_W_up = self.W_up[i][j]
                current_spacing, stepsize = np.linspace(current_W_low, current_W_up, num_split + 1, retstep = True)
                current_spacing = list(current_spacing)
                
                self.W_low[i][j] = current_val - stepsize
                if(self.W_low[i][j] < 0):
                    self.W_low[i][j] = 0
                self.W_up[i][j] = current_val + stepsize
                if(self.W_up[i][j] > 1):
                    self.W_up[i][j] = 1
                    
        self.smooth_sig_genv2(W_candidates)
                    
    def updateH(self):
        #obtain the best H out of all possible choices based on reconstruction error
        H_var = self.get_var(W_bool=False)
        error_rate = []
        W = self.W
        V = self.V
        num_split = self.n_split
        
        for i in range(len(H_var)):
            current_H = np.array(H_var[i]).reshape(-1,1)
            pred_V = np.matmul(W, current_H)
            error_rate.append(mean_absolute_error(V, pred_V))
        self.H = list(H_var[error_rate.index(min(error_rate))])
        print(self.H)
        #update upper and lower bound for H
        for i in range(len(self.H)):
            current_Hlow = self.H_low[i]
            current_Hup = self.H_up[i]
            current_val = self.H[i]
            current_spacing, stepsize = np.linspace(current_Hlow, current_Hup, num_split + 1, retstep = True)
            current_spacing = list(current_spacing)
            #for x in range(0, len(current_spacing) - 1):
            self.H_low[i] = current_val - stepsize
            if(self.H_low[i] < 0):
                self.H_low[i] = 0                
            self.H_up[i] = current_val + stepsize
            if(self.H_up[i] > 1):
                self.H_up[i] = 1                  
                
    def converge(self, plot=False):
        #this function performs convergence for both W and H
        num_iterations = self.iterations
        mae = [np.infty]
        maeh = []
        for i in range(num_iterations):
            try:
                self.updateH()
                #self.reconstructed_V = np.matmul(self.W, self.H)
                #return
            except:
                pass
            self.updateW()
            #print('----')
            #print('Reconstruction error: ', mean_absolute_error(self.V, np.matmul(self.W, self.H)))
            #print('H error: ', mean_absolute_error(Hs[0], self.H))
            if(mean_absolute_error(self.V, np.matmul(self.W, self.H)) > mae[-1]):
                self.W = self.prev_W
                break
            mae.append(mean_absolute_error(self.V, np.matmul(self.W, self.H)))
            #maeh.append(mean_absolute_error(Hs[0], self.H))
        if(plot):
            plt.plot(mae)
            #plt.plot(maeh)
            plt.show()
        self.reconstructed_V = np.matmul(self.W, self.H)
        
        #return mean_absolute_error(self.true_H, self.H)
        
    def converge_noW(self, plot=False):
        #this fucntion only estimate H
        num_iterations = self.iterations
        mae = [np.infty]
        maeh = []
        for i in range(num_iterations):
            try:
                self.updateH()
                self.reconstructed_V = np.matmul(self.W, self.H)
                return
            except:
                pass
            self.updateW()
            #print('----')
            #print('Reconstruction error: ', mean_absolute_error(self.V, np.matmul(self.W, self.H)))
            #print('H error: ', mean_absolute_error(Hs[0], self.H))
            if(mean_absolute_error(self.V, np.matmul(self.W, self.H)) > mae[-1]):
                self.W = self.prev_W
                break
            mae.append(mean_absolute_error(self.V, np.matmul(self.W, self.H)))
            #maeh.append(mean_absolute_error(Hs[0], self.H))
        if(plot):
            plt.plot(mae)
            #plt.plot(maeh)
            plt.show()
        self.reconstructed_V = np.matmul(self.W, self.H)    

    def smooth_sig_genv2(self, W_candidates):
        #choose the best candidates based on how close 
        #they are to the estimated abundance fractions
        num_sig = len(W_candidates[0][0])
        Y_val_collectors = []
        for i in range(num_sig):
            Y_val_collectors.append([])
        for i in range(len(W_candidates)):
            current_q = W_candidates[i]
            for j in range(num_sig):            
                current_output = []
                for val in current_q:
                    current_output.append(val[j])
                Y_val_collectors[j].append(current_output)
                
                
        #be careful here and set the correct value for signal1,2 and 3
        #for the ground truth file, they are 1,2 and 4
        #for the extracted endmembers, they are 0,1 and 2
        df1 = pd.read_csv(self.data_path, header=None)
        data1 = df1.values
        data1 = np.transpose(data1)
        signal1 = np.array(data1[0]).reshape(-1, 1)
        signal2 = np.array(data1[2]).reshape(-1, 1)
        signal3 = np.array(data1[4]).reshape(-1, 1)  
        #for ground truth data only (i.e. synthesized experiment in the paper)
        signal1 = signal.resample(signal1, 351)
        signal2 = signal.resample(signal2, 351)
        signal3 = signal.resample(signal3, 351)          
        std_sig = [signal1, signal2, signal3]
                
        Y_val_mean = []
        for i in range(num_sig):
            Y_val_mean.append([])    
        for i in range(num_sig):
            for j in range(len(Y_val_collectors[i])):
                current_q = Y_val_collectors[i][j]
                orig_val = std_sig[i][j]
                Y_val_mean[i].append(min(current_q, key=lambda x:abs(x-orig_val)))
        update_W = []
        for i in range(num_sig):  
            x = np.array(list(range(len(Y_val_mean[i]))))
            y = np.array(Y_val_mean[i])
            p30 = np.poly1d(np.polyfit(x, y, 10))
            update_W.append(p30(x))
        
        self.prev_W = (self.W).copy()
        self.W = np.array(update_W).transpose()
        
def nmf_library(V, W_init, correct_H):
    #comparisons with non-negative matrix factorization
    lsnmf = nimfa.Lsnmf(V, seed=None, rank=3, max_iter=100, H = np.array([0.,0.,0.]).reshape(-1,1), W = W_init)
    nmf = nimfa.Nmf(V, seed=None, rank=3, max_iter=100, H = np.array([0.,0.,0.]).reshape(-1,1), W = W_init)
    icm = nimfa.Icm(V, seed=None, rank=3, max_iter=100, H = np.array([0.,0.,0.]).reshape(-1,1), W = W_init)
    bd = nimfa.Bd(V, seed=None, rank=3, max_iter=100, H = np.array([0.,0.,0.]).reshape(-1,1), W = W_init)
    pmf = nimfa.Pmf(V, seed=None, rank=3, max_iter=100, H = np.array([0.,0.,0.]).reshape(-1,1), W = W_init)
    #lfnmf = nimfa.Lfnmf(V, seed=None, rank=3, max_iter=100, H = np.array([0.,0.,0.]).reshape(-1,1), W = W_init)
    
    lsnmf_fit = lsnmf()
    nmf_fit = nmf()
    icm_fit = icm()
    bd_fit = bd()
    pmf_fit = pmf()
    
    
    lsnmf_error = mean_absolute_error(correct_H, normalized(np.array(lsnmf.H).reshape(-1,)))
    nmf_error = mean_absolute_error(correct_H, normalized(np.array(nmf.H).reshape(-1,)))
    icm_error = mean_absolute_error(correct_H, normalized(np.array(icm.H).reshape(-1,)))
    bd_error = mean_absolute_error(correct_H, normalized(np.array(bd.H).reshape(-1,)))
    pmf_error = mean_absolute_error(correct_H, normalized(np.array(pmf.H).reshape(-1,)))
    
    return [lsnmf_error, nmf_error, icm_error, bd_error, pmf_error]

def do_synthesize():    
    #===========================================================================
    #synthesized data
    path1 = '../data/groundtruth_tissues.csv'
    df1 = pd.read_csv(path1, header=None)
    data1 = df1.values
    data1 = np.transpose(data1)
    signal1 = np.array(data1[1]).reshape(-1, 1)
    signal2 = np.array(data1[2]).reshape(-1, 1)
    signal3 = np.array(data1[4]).reshape(-1, 1)
    signal1 = signal.resample(signal1, 351)
    signal2 = signal.resample(signal2, 351)
    signal3 = signal.resample(signal3, 351)    
    W = np.concatenate((signal1, signal2, signal3), axis = 1)
    #=====================================================================
    
    variation_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    num_dim = 351
    avg_H_error = []
    for variation_ratio in variation_ratios:
        Vs, Ws, Hs = generate_data(variation_ratio, num_dim)
        errors = []
        for i in range(0, 3):
            print('iter ', i)
            V = Vs[i]
            correct_H = Hs[i]
            #W_init = np.random.random((V.shape[0],3))
            W_init = W
            iterations = 5
            n_split = 10
            
            #proposed method
            nmf = nmf_mv2(V, W_init, iterations, n_split, path1, variation_ratio)
            nmf.converge(False)
            
            nmf_noW = nmf_mv2(V, W_init, iterations, n_split, path1, variation_ratio)
            nmf_noW.converge_noW(False)  
            
            #====================================================================
            V = np.append(V, np.array([1]).reshape(-1,1), axis = 0)
            W_init = np.append(W_init, np.array([1,1,1]).reshape(1,3), axis = 0)   
            #====================================================================
            #combined = list(V.reshape(-1,))
            #signal1 = list(W_init[:,0])
            #signal2 = list(W_init[:,1])
            #signal3 = list(W_init[:,2])
            A = nmf_library(V, W_init, correct_H)
            
            #===================================================================
            #pseudo_inverse matrix erorr
            A1 = np.linalg.pinv(W_init)
            pred_H = np.matmul(A1, V)
            #errors.append([mean_absolute_error(correct_H, normalized(pred_H))])
            #===================================================================
            errors.append(A + [mean_absolute_error(correct_H, pred_H), mean_absolute_error(correct_H, nmf.H), mean_absolute_error(correct_H, nmf_noW.H)])
        errors = np.array(errors)
        avg_H_error.append(np.mean(errors, axis = 0))
        print('Done with: ', variation_ratio, 'error found: ', np.mean(errors, axis = 0))
        
    print(avg_H_error)
    avg_df = pd.DataFrame(avg_H_error)
    avg_df.to_csv('../data/error_exp.csv', header=None)    

def do_experimental():
    #==========================================================================
    #experimental data
    path_experimental = '../data/estimated_signals.csv'
    df1 = pd.read_csv(path_experimental, header=None)
    data1 = df1.values
    signal1 = np.array(data1[0]).reshape(-1, 1)
    signal2 = np.array(data1[1]).reshape(-1, 1)
    signal3 = np.array(data1[2]).reshape(-1, 1)
    #W intialization for experimental data
    W = np.concatenate((signal1, signal2, signal3), axis = 1)
    iterations = 3
    n_split = 10
    
    #load data
    path_data = '../data/new651_data_smooth.csv'
    df = pd.read_csv(path_data, header=None)
    data = df.values
    
    #load ground-truth results
    df = pd.read_csv('../data/pathology_result.csv', header=None)
    data1 = df.values
    assert(data.shape[0] == data1.shape[0])
    
    H_features = []
    for i in range(0, data.shape[0]):
        print('starting iter ', i)
        if(i == 244):
            pred_H = [0.0,0.0,0.0]
            label = data1[i][0]
            H_features.append(pred_H + [int(label)])        
            continue
        V = data[i]
        nmf = nmf_mv2(V, W, iterations, n_split, path_data)
        
        #converge with static W is shown to perform the best
        nmf.converge_noW()
        pred_H = nmf.H
        print('mean absolute error: ', mean_absolute_error(V, nmf.reconstructed_V))
        label = data1[i][0]
        H_features.append(pred_H + [int(label)])
    df_out = pd.DataFrame(H_features)
    df_out.to_csv('../data/pred_features_data651denoise.csv', header=None, index=False)
    
#modify the csv file name to match your previous steps (abundance extraction) / new input data
do_synthesize()
do_experimental()
