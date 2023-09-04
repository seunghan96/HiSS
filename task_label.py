import pandas as pd
import numpy as np

DATA_DIR = '/Users/seunghan96/Desktop/hyodoll_yonsei/data/'


if __name__ == '__main__':
    cluster_df = pd.read_csv(DATA_DIR + 'FA_with_cluster.csv')
    cluster_df=cluster_df.set_index('Unnamed: 0')
    cluster_df.index.names = ['doll_id']

    scc_emergency_call=pd.read_csv(DATA_DIR+'scc_emergency_call.csv',sep=';')

    scc_emergency_call['temp_val'] = 1
    scc_emergency_call1=pd.pivot_table(scc_emergency_call, values='temp_val', index=['doll_id'],
                        columns=['call_month'], aggfunc=np.sum,fill_value=0)

    scc_emergency_call2=pd.pivot_table(scc_emergency_call, values='temp_val', index=['doll_id'],
                        columns=['call_hour'], aggfunc=np.sum,fill_value=0)

    scc_emergency_call1.columns=['emergency_month'+str(col) for col in scc_emergency_call1.columns]
    scc_emergency_call2.columns=['emergency_hour'+str(col) for col in scc_emergency_call2.columns]

    cluster_df_merged = cluster_df.merge(scc_emergency_call1,left_index=True,right_index=True,how='left')
    cluster_df_merged = cluster_df_merged.merge(scc_emergency_call2,left_index=True,right_index=True,how='left')
    cluster_df_merged=cluster_df_merged.fillna(0)


    cluster_df_merged['emergency_time1'] = cluster_df_merged[['emergency_hour'+str(i) 
                                                            for i in [0,1,2,3,4,5]]].sum(axis=1)
    cluster_df_merged['emergency_time2'] = cluster_df_merged[['emergency_hour'+str(i) 
                                                            for i in [6,7,8,9,10,11]]].sum(axis=1)
    cluster_df_merged['emergency_time3'] = cluster_df_merged[['emergency_hour'+str(i) 
                                                            for i in [12,13,14,15,16,17]]].sum(axis=1)
    cluster_df_merged['emergency_time4'] = cluster_df_merged[['emergency_hour'+str(i) 
                                                            for i in [18,19,20,21,22,23]]].sum(axis=1)

    cluster_df_merged.to_csv(DATA_DIR+'cluster_df_merged.csv')
    print('Finished Saving!')
