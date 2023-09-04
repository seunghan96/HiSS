import os
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import f1_score, roc_auc_score

warnings.filterwarnings(action='ignore')

DATA_DIR = '/Users/LSH/Desktop/hyodoll/data'

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default = 200, help = '# of epochs')
parser.add_argument('--alpha', type=float, default = 0.2, help = '# of epochs')

args = parser.parse_args()

epochs = args.epochs
alpha = args.alpha

class Net(nn.Module):
    def __init__(self, input_size, num_clusters):
        super(Net, self).__init__()
        self.shared_layer = nn.Linear(input_size, 64)
        self.aux_layer = nn.Linear(64, num_clusters)
        self.month_shared_layer = nn.Linear(64, 32)
        self.time_shared_layer = nn.Linear(64, 32)
        self.time_layer1 = nn.ModuleList([nn.Linear(32, 32) for _ in range(6)])
        self.month_layer1 = nn.ModuleList([nn.Linear(32, 32) for _ in range(12)])
        self.time_layer2 = nn.ModuleList([nn.Linear(32, 1) for _ in range(6)])
        self.month_layer2 = nn.ModuleList([nn.Linear(32, 1) for _ in range(12)])
        
        self.activation = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        x = self.activation(self.shared_layer(x))
        cluster_pred = self.aux_layer(x)
        
        x_month = self.activation(self.month_shared_layer(x))
        x_time = self.activation(self.time_shared_layer(x))
        
        time_out_list = []
        month_out_list = []
        
        for layer in self.time_layer1:
            out = layer(x_time)
            time_out_list.append(self.activation(out))
        
        for layer in self.month_layer1:
            out = layer(x_month)
            month_out_list.append(self.activation(out))
        
        time_final_pred = []
        month_final_pred = []
        for layer, out in zip(self.time_layer2, time_out_list):
            #time_final_pred.append(self.sigmoid(layer(out)))
            time_final_pred.append(layer(out))
            
        for layer, out in zip(self.month_layer2, month_out_list):
            #month_final_pred.append(self.sigmoid(layer(out)))
            month_final_pred.append(layer(out))
            
        return month_final_pred + time_final_pred, cluster_pred

if __name__ == '__main__':
    data = pd.read_csv(os.path.join(DATA_DIR,'cluster_df_merged.csv'))
    dummy_col = pd.get_dummies(data['cluster'])
    dummy_col.columns = ['cluster_'+str(i) for i in dummy_col.columns]
    data = pd.concat([data,dummy_col],axis=1)
    
    N = data.shape[0]
    train_ratio = 0.8
    np.random.seed(19960729)
    train_index = list(np.random.choice(N, int(N*train_ratio), replace=False))
    test_index = list(set(range(N)) - set(train_index))
    
    #X_cols = [x for x in data.columns if 'FA_' in x] + [x for x in data.columns if 'cluster_' in x]
    X_cols = [x for x in data.columns if 'FA_' in x] 
    y_aux_cols = [x for x in data.columns if 'cluster_(' in x]
    num_clusters = len(y_aux_cols)
    y_cols = [x for x in data.columns if 'emergency_month' in x] + [x for x in data.columns if 'emergency_time' in x]
    X = data[X_cols].values 
    
    thres_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9,0.95]
    
    X = torch.tensor(data[X_cols].values).float()#.cuda()
    y = torch.tensor(data[y_cols].values).float()#.cuda()
    y_aux = torch.tensor(data[y_aux_cols].values)
    y_aux = torch.argmax(y_aux, dim=1)
    y_aux = y_aux.squeeze().long()
    #y_aux = y_aux.long()
    y[y > 0] = 1
    
    X_train = X[train_index]
    X_test = X[test_index]
    y_train = y[train_index]
    y_test = y[test_index]
    y_aux_train = y_aux[train_index]
    y_aux_test = y_aux[test_index]
    
    model = Net(X.shape[1], num_clusters)#.cuda()
    criterion = nn.BCEWithLogitsLoss()
    criterion_aux = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    predicted_df = pd.DataFrame(index=range(len(y_test)), columns=['thres'+str(thres)+'_'+y_col for thres in thres_list for y_col in y_cols])
    performance_index = ['f1', 'acc', 'auroc']
    performance_df = pd.DataFrame(index=performance_index, columns=['thres'+str(thres)+'_'+y_col for thres in thres_list for y_col in y_cols])
    
    for epoch in range(epochs):
        # (1) Train
        optimizer.zero_grad()
        y_hats,y_aux_hats = model(X_train)
        loss = 0
        for i in range(y.shape[1]):
            loss += criterion(y_hats[i], y_train[:, i].unsqueeze(1))
        
        loss += alpha*criterion_aux(y_aux_hats, y_aux_train)
        loss.backward()
        optimizer.step()
        
        # (2) Test
        y_hats, _ = model(X_test)
        y_hats = [i.detach().cpu().numpy() for i in y_hats]        
    
        for i in range(len(y_hats)):
            predicted_df.iloc[:,i]=y_hats[i]
            
        for thres in thres_list:
            for idx in range(len(y_cols)):
                y_pred_class=(y_hats[idx].flatten()>thres).astype('int')
                #print(y_pred_class, y_test.detach().cpu().numpy()[:,idx])
                score_f1 = f1_score(y_test.detach().cpu().numpy()[:,idx], y_pred_class)
                score_acc = np.mean((y_test.detach().cpu().numpy()[:,idx]==y_pred_class).astype('int'))
                df_col='thres'+str(thres)+'_'+str(y_cols[idx])
                performance_df[df_col]=[score_f1,score_acc,0]

        for idx in range(len(y_cols)):
            performance_df.loc['auroc'][performance_df.loc['auroc'].index.str.endswith(y_cols[idx])]=roc_auc_score(y_test[:,idx].flatten(), y_hats[idx].flatten())
            
        best_cols=[]
        for y_col in y_cols:
            tmp = performance_df[performance_df.columns[performance_df.columns.str.endswith(y_col)]].idxmax(axis=1)['f1']
            best_cols.append(tmp)

        if epoch%10==0:
            print(epoch)
            print(performance_df[best_cols].mean(axis=1))    
            print('='*50)