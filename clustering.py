import pandas as pd
from cluster_utils.clustering_utils import *
import argparse

DATA_DIR = '/Users/seunghan96/Desktop/hyodoll_yonsei/data/'

parser = argparse.ArgumentParser()
parser.add_argument('--grid_x', type=int, default = 2, help = '# of X grid in SOM')
parser.add_argument('--grid_y', type=int, default = 3, help = '# of Y grid in SOM')
parser.add_argument('--iter', type=int, default = 20000, help = '# of iterations in SOM')

args = parser.parse_args()

num_x = args.grid_x
num_y = args.grid_y
iter = args.iter

if __name__ == '__main__':
    FA_doll_merged = pd.read_csv(DATA_DIR+'FA_doll_merged.csv').set_index('doll_id')
    FA_log_action = pd.read_csv(DATA_DIR+'FA_log_action.csv').set_index('Unnamed: 0')
    FA_log_program = pd.read_csv(DATA_DIR+'FA_log_program.csv').set_index('Unnamed: 0')

    df_scaled = merge_and_scale(FA_doll_merged,FA_log_action,FA_log_program)
    #--------------------------------------------------------------------------#
    ############################################################################
    '''
    print('여기코드수정해야함')
    
    # DROP OUTLIERS HERE 
    # (1) outlier로써, 버려야 할 애들 id를 따로 저장한다음 여기서 불러오기
    DROP_IDs = pd.read_csv('xxxxxxx')
    DROP_IDs = list(DROP_IDs)
    
    # (2) 제거하기
    df_scaled = df_scaled.index
    df_scaled = df_scaled.loc[~df_scaled.index.isin(DROP_IDs)]
    '''
    #--------------------------------------------------------------------------#
    df_with_cluster = som_clustering(df_scaled, num_x, num_y, iter )
    print('Clustering with SOM({},{})'.format(num_x,num_y))
    #df_scaled.to_csv(DATA_DIR + 'FA_total_scaled.csv')
    df_with_cluster.to_csv(DATA_DIR + 'FA_with_cluster.csv')
    print('Finished Saving!')
    