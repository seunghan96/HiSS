import pandas as pd
import warnings

warnings.filterwarnings("ignore")

def preprocess_meta(user_df, agency_df, doll_df, user_doll_group_df):
    # 1. Drop users w.o "agency code"
    user_df = user_df[~user_df['agency_code'].isnull().values]
    user_df['agency_code'] = user_df['agency_code'].apply(int)
    
    # 2. Get "agency name" with "agency code"
    user_df = user_df[ user_df['agency_code'].apply(lambda x: x not in [3,4,1111, 7, 8, 9, 324, 104211001000, 
                                                             991234123001, 999999901001])]
    agency_name_dict = dict(zip(agency_df['agency_id'], agency_df['agency_name']))
    
    user_df['agency_name'] = user_df['agency_code'].apply(lambda x: agency_name_dict[x] )
    
    # 3. Drop users w.o "doll"
    doll_user_dict = dict(zip( doll_df['doll_id'],doll_df['user_id']))
    user_wo_doll = user_doll_group_df['doll_id'].apply(lambda x: x not in doll_user_dict.keys() )
    user_doll_group_df = user_doll_group_df[~user_wo_doll]
    user_doll_group_df['user_id'] = user_doll_group_df['doll_id'].apply(lambda x: doll_user_dict[x])
    

    # 4. Rename "agency_code" -> "gp_agency_code" (for merge)
    user_doll_group_df.rename(columns= {'agency_code': 'gp_agency_code'}, inplace = True)
    
    # 5. Merge
    merged_df = pd.merge(user_df, user_doll_group_df, on = 'user_id')
    merged_df['gp_agency_name']= merged_df.loc[:, 'gp_agency_code'].apply(lambda x: 
        agency_name_dict[x] if x in agency_name_dict.keys() else None )
    
    # 6. Drop rows with any null values
    merged_df = merged_df[ ~merged_df.isnull().any(axis=1)]
    merged_df['gp_agency_code'] = merged_df['gp_agency_code'].apply(int)
    
    # 7. Drop columns & Rename column ( remove "gp_" )
    merged_df['match_agc'] = 1 * (merged_df['agency_code'] == merged_df['gp_agency_code'] )
    merged_df.drop(columns = ['agency_code', 'agency_name'], inplace = True)
    merged_df.rename(columns = {'gp_agency_code': 'agency_code', 'gp_agency_name':'agency_name'}, inplace = True)
    
    return user_df, user_doll_group_df, merged_df

def merge_doll_user(doll_df,merged_df, doll_option_df):
    # 1. Merge "doll", "merged", "doll_option"
    doll_option_df.drop(columns = 'is_edited', inplace = True)
    doll_merged_df = pd.merge(doll_df, merged_df, on = ['user_id', 'doll_id'])
    doll_merged_df = pd.merge(doll_merged_df, doll_option_df, on = 'doll_id' )
    
    return doll_merged_df

