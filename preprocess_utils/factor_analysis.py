import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA,FactorAnalysis

import warnings

warnings.filterwarnings("ignore")

def pca(log_df,pca_thres):
    # 1) scaling
    scaler = StandardScaler()
    scaler.fit(log_df)
    log_df_vals = scaler.transform(log_df)
    
    # 2) pca
    pca = PCA()
    pca.fit(log_df_vals)
    log_df_pcs= pd.DataFrame(pca.fit_transform(log_df_vals),
                             index=log_df.index, columns=['pc'+str(i) for i in range(1,1+log_df.shape[1])])
    
    # 3) plotting
    pca_evr = pca.explained_variance_ratio_.cumsum()
    
    # 4) appropriate number of PCs
    pc_nums = sum(pca_evr<pca_thres)+1
    log_df_pcs = log_df_pcs[['pc'+str(i) for i in range(1,pc_nums+1)]]
    
    return log_df_pcs


def fa(log_df,n_components,type='doll'):
    # 1) scaling
    scaler = StandardScaler()
    scaler.fit(log_df)
    log_df_vals = scaler.transform(log_df)
    
    # 2) fa
    fa = FactorAnalysis(n_components=n_components)
    fa.fit(log_df_vals)
    log_df_factors= pd.DataFrame(fa.fit_transform(log_df_vals),
                             index=log_df.index, columns=['factor'+str(i) for i in range(1,1+n_components)])
        
    # 5) Interpreatation
    FA_components=pd.DataFrame(fa.components_,
                 columns=log_df.columns,index=['factor'+str(i) for i in range(1,n_components+1)])
    
    log_df_factors.columns=['FA_{}_log{}'.format(type,i) for i in range(1,1+n_components)]
    return log_df_factors,FA_components


def z_scale_wo_outliers(df,cols_list):
    for col in cols_list:
        scaler_for_OD = StandardScaler()
        dat=df[[col]]
        scaler_for_OD.fit(dat)
        col_prescaled = scaler_for_OD.transform(dat)

        scaler = StandardScaler()
        scaler.fit(dat[(np.abs(col_prescaled)>3)==False])
        df[col] = scaler.transform(dat) 