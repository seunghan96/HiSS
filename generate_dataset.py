from preprocess_utils.preprocess_table import *
from preprocess_utils.preprocess_log import *
from preprocess_utils.preprocess_doll import *
from preprocess_utils.factor_analysis import *


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--doll', type=int, default = 6, help = '# of components (in Factor Analysis) of "DOLL MERGED" data')
parser.add_argument('--action', type=int, default = 3, help = '# of components (in Factor Analysis) of "ACTION" data')
parser.add_argument('--program', type=int, default = 4, help = '# of components (in Factor Analysis) of "PROGRAM" data')

args = parser.parse_args()

DATA_DIR = '/Users/seunghan96/Desktop/hyodoll_yonsei/data/'
NUM_COMP_FA_DOLL = args.doll
NUM_COMP_FA_ACTION = args.action
NUM_COMP_FA_PROGRAM = args.program

if __name__ == '__main__':
    # (1) Tabular (Meta) Dataset
    agency = pd.read_csv(DATA_DIR+'scc_agency.csv')
    doll = pd.read_csv(DATA_DIR+'scc_doll.csv', sep = ',')
    doll_option = pd.read_csv(DATA_DIR+'scc_doll_option.csv',sep = ',')
    user = pd.read_csv(DATA_DIR+'scc_user.csv',sep = ';')
    user_doll_group = pd.read_csv(DATA_DIR+'scc_user_doll_group.csv',sep = ';')

    user, user_doll_group, merged = preprocess_meta(user, agency, doll, user_doll_group)
    doll_merged = merge_doll_user(doll,merged, doll_option)
    #---------------------------------------------------------------------------------#
    #---------------------------------------------------------------------------------#
    # (2-1) [FA] Tabular (Meta) Dataset
    doll_merged = preprocess_doll(doll_merged)
    z_scale_wo_outliers(doll_merged,['age'])
    FA_doll_merged,_ = fa(doll_merged,n_components = NUM_COMP_FA_DOLL,type='doll')
    #---------------------------------------------------------------------------------#
    # (2-2) [FA] Log (Action) Dataset
    log_action = pd.read_csv(DATA_DIR+'log_doll_summary_YMD.csv',sep=';')

    log_action = preprocess_log_action(log_action)
    log_action_filtered = filter_df_action(log_action,activities_binary=True,month_binary=True)
    cols_act=list(log_action_filtered.columns[~log_action_filtered.columns.str.contains('M')])
    z_scale_wo_outliers(log_action_filtered,cols_act)
    FA_log_action, _ = fa(log_action_filtered,n_components = NUM_COMP_FA_ACTION, type='action')
    #---------------------------------------------------------------------------------#
    # (2-3) [FA] Log (Program) Dataset
    log_program = pd.read_csv(DATA_DIR+'ear_function_log.csv',sep=';')

    log_program = preprocess_log_program(log_program)
    log_program_filtered = filter_df_program(log_program,programs_binary=True,month_binary=True)
    cols_act=list(log_program_filtered.columns[~log_program_filtered.columns.str.contains('M')])
    z_scale_wo_outliers(log_program_filtered,cols_act)
    FA_log_program, _ = fa(log_program_filtered,n_components=NUM_COMP_FA_PROGRAM, type='program')
    #---------------------------------------------------------------------------------#
    # (4) Save Dataset
    doll_merged.to_csv(DATA_DIR + 'doll_merged.csv',index=False)
    FA_doll_merged.to_csv(DATA_DIR + 'FA_doll_merged.csv')
    FA_log_action.to_csv(DATA_DIR + 'FA_log_action.csv')
    FA_log_program.to_csv(DATA_DIR + 'FA_log_program.csv')
    #---------------------------------------------------------------------------------#
    print('Finished Saving')
    print('='*30)
    print('shape of "doll_merged" :',doll_merged.shape)
    print('shape of "FA_doll_merged" :',FA_doll_merged.shape)
    print('shape of "FA_log_action" :',FA_log_action.shape)
    print('shape of "FA_log_program" :',FA_log_program.shape)
    