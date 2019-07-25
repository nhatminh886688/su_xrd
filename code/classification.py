import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def load_data(path):
    df = pd.read_csv(path, header=None)
    data = df.values
    output = data[:,0:3]
    target = data[:,-1]
    target = target.astype(int)
    return output, target

def load_data_normal(path):
    df = pd.read_csv(path, header=None)
    return df.values

def manual_classification(percent_features, target_labels):
    #endmember with highest proportion gives the label, totally unsupervised
    predicted_labels = []
    for i in range(len(percent_features)):
        current_feature = list(percent_features[i])
        if(current_feature.index(max(current_feature)) == 1):
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)
    predicted_labels[244] = 1
    print('ACC: ', accuracy_score(target_labels, predicted_labels))
    return predicted_labels

def plot_tissue(predicted_labels):
    data  = load_data_normal('../data/spot_position.csv')
    data2 = load_data_normal('../data/pathology_result.csv')
    colors = []
    
    for i in range(0, len(predicted_labels)):
        if(predicted_labels[i] == 1):
            colors.append('red')
        else:
            colors.append('green')
            
    color_gt = []
    for i in range(0, len(data2)):
        if(data2[i][0] == 1):
            color_gt.append('red')
        else:
            color_gt.append('green')    
    plt.scatter(data[:,0], data[:,1]*(-1), c = colors)
    plt.title('Prediction')
    plt.show()
    plt.close()
    plt.scatter(data[:,0], data[:,1]*(-1), c = color_gt)
    plt.title('Ground truth')
    plt.show()
    plt.close()
    
    

features, target = load_data('../data/pred_features_data651.csv')
X = np.array(features)
y = np.array(target)
predicted_labels = manual_classification(features, target)
plot_tissue(predicted_labels)