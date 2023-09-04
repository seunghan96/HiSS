
import pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler

from cluster_utils.som import Som

import warnings

warnings.filterwarnings("ignore")

def merge_and_scale(df1,df2,df3):
    df = pd.merge(pd.merge(df1,df2,left_index = True,right_index = True),
                df3,left_index = True,right_index = True)
    df_scaled = df.copy()
    sc = StandardScaler()
    sc.fit(df)
    df_scaled.iloc[:,:]=sc.transform(df)
    return df_scaled
    
def som_clustering(df_scaled, num_x=2, num_y=3, iter=20000):
    som_X = df_scaled.values
    som_ = Som(num_x, num_y, df_scaled.shape[1],sigma=1.0, learning_rate=0.5)
    som_.random_weights_init(som_X)
    som_.train_random(som_X, iter)
    grid_total = [som_.winner(som_X[i]) for i in range(len(som_X))]
    grid_count = Counter(grid_total)
    print(grid_count)
    df_scaled['cluster'] = grid_total
    df_scaled['cluster_x'] = df_scaled['cluster'].apply(lambda x:x[0])
    df_scaled['cluster_y'] = df_scaled['cluster'].apply(lambda x:x[1])
    return df_scaled
