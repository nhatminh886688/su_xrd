import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate
from scipy import signal

from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import silhouette_score
from collections import Counter
from sklearn.cluster import DBSCAN

#pip install pysptools
import pysptools.eea as eea
#the VCA.py file is obtained from https://github.com/Laadr/VCA
import VCA


def load_data(path):
    df = pd.read_csv(path, header=None)
    data = df.values
    return data


class dynamic_DBSCAN():
    #this class helps to find the appropriate parameters for DBSCAN
    #return the corresponding cluster labels for each data point with the chosen parameters
    def __init__(self, input_data, max_eps, min_eps, N):
        #[min_eps, max_eps] is the range of low and high values of eps (in this case, i set them to be 0.1 and 5)
        #N is the # of clusters you want to find
        self.data = input_data
        self.max_eps = max_eps
        self.min_eps = min_eps
        self.N = N
        
    def find_eps(self):
        #split eps range into 100 values and find the first value return at least N clusters
        testing_list = list(np.linspace(self.min_eps, self.max_eps, 100))
        min_pts =  np.log(self.data.shape[0])
        
        for eps in reversed(testing_list):
            temp_labels = DBSCAN(eps=eps, min_samples=min_pts).fit(self.data).labels_
            if(len(list(set(temp_labels))) >= self.N):
                return eps
        #in case no such eps is found, return the lowest eps possible (so most cluster would appear)
        return np.min(testing_list)
    
    def dynamic_clustering(self):
        #return the labels for the chosen parameters
        best_eps = self.find_eps()
        min_pts =  np.log(self.data.shape[0])
        return DBSCAN(eps=best_eps, min_samples=min_pts).fit(self.data).labels_
    
class mclustering():
    def __init__(self, data, N):
        self.data = data
        self.N = N
    
    def do_clustering(self, data):
        #====================================================
        dynamic_clustering = dynamic_DBSCAN(data, 5, 0.1, self.N)
        labels = dynamic_clustering.dynamic_clustering()
        #====================================================
        #calculate the centroid for each returning clusters
        num_cluster = len(list(set(labels)))
        data_grouped = []
        for i in range(num_cluster):
            data_grouped.append([])
        for i in range(len(labels)):
            if(labels[i] != -1):
                data_grouped[labels[i]].append(data[i])
        centroids = []
        for i in range(num_cluster-1):
            centroids.append(np.mean(data_grouped[i], axis = 0))
        
        #smooth the centroid with polynomial fit
        centroid_smooth = []
        for centroid in centroids:
            x = np.array(list(range(len(centroid))))
            y = np.array(centroid)
            p30 = np.poly1d(np.polyfit(x, y, 10))
            centroid_smooth.append(p30(x)/max(p30(x))) 
            
        #uncomment these if you want to see the figures of the found endmembers    
        #cnt = 0
        #for val in centroid_smooth:
            #plt.plot(val)
            #plt.savefig('../figs/dbscan_v1' + str(cnt) +'.png')
            #cnt += 1
        return centroid_smooth
    
    def mapping_to_class(self, centroid_smooth):
        #this function map the found centroids to the 3 known tissue classes ground truth
        #so we can know which found endmember is cancer, which one is adipose etc.
        df1 = pd.read_csv('../data/groundtruth_tissues.csv', header=None)
        data1 = df1.values
        data1 = np.transpose(data1)
        signal1 = np.array(data1[1]).reshape(-1, 1)   #adipose
        signal2 = np.array(data1[2]).reshape(-1, 1)   #fibroglandular
        signal3 = np.array(data1[4]).reshape(-1, 1)   #cancer  
        
        output_signals = []
        signal1_DISTANCES = []
        for i in range(len(centroid_smooth)):
            signal1_DISTANCES.append(distance.euclidean(signal.resample(centroid_smooth[i], 235), signal1))
        output_signals.append(centroid_smooth[signal1_DISTANCES.index(min(signal1_DISTANCES))])
        del centroid_smooth[signal1_DISTANCES.index(min(signal1_DISTANCES))]
        
        signal2_DISTANCES = []
        for i in range(len(centroid_smooth)):
            signal2_DISTANCES.append(distance.euclidean(signal.resample(centroid_smooth[i], 235), signal2))
        output_signals.append(centroid_smooth[signal2_DISTANCES.index(min(signal2_DISTANCES))])
        del centroid_smooth[signal2_DISTANCES.index(min(signal2_DISTANCES))]
        
        signal3_DISTANCES = []
        for i in range(len(centroid_smooth)):
            signal3_DISTANCES.append(distance.euclidean(signal.resample(centroid_smooth[i], 235), signal3))
            
        #the 3 most similar signals are written to the specified output path
        output_signals.append(centroid_smooth[signal3_DISTANCES.index(min(signal3_DISTANCES))])        
        output_df = pd.DataFrame(output_signals)
        output_df.to_csv('../data/estimated_signals.csv', index=False, header=None)
        
def nfindr(data, N):
    #N is the # of endmembers you want to find
    
    data = data.reshape(1, data.shape[0], data.shape[1])
    #data = data.reshape(data.shape[1], data.shape[0])
    #====================================================
    #Ae,indice,Yp = VCA.vca(data.transpose(), 3)
    #endmembers = Ae.transpose()
    #====================================================
    nfindr = eea.NFINDR()
    endmembers = nfindr.extract(M=data, q=N)
    centroid_smooth = []
    for centroid in endmembers:
        x = np.array(list(range(len(centroid))))
        y = np.array(centroid)
        p30 = np.poly1d(np.polyfit(x, y, 10))
        centroid_smooth.append(p30(x)/max(p30(x))) 
    centroid_smooth_df = pd.DataFrame(centroid_smooth)
    #the outputfile will contain exactly N endmembers, you can choose closest signals to ground truth for comparison of methods
    centroid_smooth_df.to_csv('../data/NFINDR_estimations.csv', header=None, index=False)
    
def ppi(data, N):
    #N is the # of endmembers you want to find
    
    data = data.reshape(1, data.shape[0], data.shape[1])
    #data = data.reshape(data.shape[1], data.shape[0])
    #====================================================
    #Ae,indice,Yp = VCA.vca(data.transpose(), 3)
    #endmembers = Ae.transpose()
    #====================================================
    ppi = eea.PPI()
    endmembers = ppi.extract(M=data, q=N)
    centroid_smooth = []
    for centroid in endmembers:
        x = np.array(list(range(len(centroid))))
        y = np.array(centroid)
        p30 = np.poly1d(np.polyfit(x, y, 10))
        centroid_smooth.append(p30(x)/max(p30(x))) 
    centroid_smooth_df = pd.DataFrame(centroid_smooth)
    centroid_smooth_df.to_csv('../data/PPI_estimations.csv', header=None, index=False)    
    
def vca(data, N):
    #N is the # of endmembers you want to find
    
    #data = data.reshape(1, data.shape[0], data.shape[1])
    #data = data.reshape(data.shape[1], data.shape[0])
    #====================================================
    Ae,indice,Yp = VCA.vca(data.transpose(), 3)
    endmembers = Ae.transpose()
    #====================================================
    centroid_smooth = []
    for centroid in endmembers:
        x = np.array(list(range(len(centroid))))
        y = np.array(centroid)
        p30 = np.poly1d(np.polyfit(x, y, 10))
        centroid_smooth.append(p30(x)/max(p30(x))) 
    centroid_smooth_df = pd.DataFrame(centroid_smooth)
    centroid_smooth_df.to_csv('../data/VCA_estimations.csv', header=None, index=False)  
    
def clustering(data):
    mcluster = mclustering(data, 5)
    N = 5
    smooth_centroids = mcluster.do_clustering(data)
    mcluster.mapping_to_class(smooth_centroids)
    
    
#=======================================================================
#load data
path = '../data/new651_data_denoised.csv'
data = load_data(path)
#=======================================================================
#this part run the proposed endmember extraction algorithm
#for this data, need to delete the 244 row which contains all nan
#for other data, uncomment this following line (or drop any row containing nan)
data = np.delete(data, (244), axis=0)
#clustering(data)


#=======================================================================
#this part run the state-of-the-art endmember extraction algorithms from literature
N = 3
nfindr(data, N)
ppi(data, N)
vca(data, N)
#=======================================================================
