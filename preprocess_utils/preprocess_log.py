import pandas as pd

import warnings

warnings.filterwarnings("ignore")


def preprocess_log_program(log_df):
    programs=[ 'story', 'religion', 'music', 'english', 'remembrance', 'quiz', 'gymnastics']
    log_df['month']=pd.to_datetime(log_df['reg_date']).dt.month
    log_df['year']=pd.to_datetime(log_df['reg_date']).dt.year
    log_df['day']=pd.to_datetime(log_df['reg_date']).dt.day
    log_df.drop(['classic_music','religion_music'],axis=1,inplace=True)
    
    log_df['use_total']=log_df[programs].sum(axis=1)
    log_df=log_df[log_df['doll_id']>99]
    log_df_groupby = log_df.groupby('doll_id')
    log_program = pd.DataFrame(index=sorted(list(set(log_df['doll_id']))))
    log_df_sum = log_df_groupby.sum()[programs]
    log_df_mean = log_df_groupby.mean()[programs]
    log_df_sum_binary = log_df_sum.copy()
    log_df_sum_binary[log_df_sum_binary>0]=1
    log_df_sum.columns=[act+'_sum' for act in programs]
    log_df_mean.columns=[act+'_mean' for act in programs]
    log_df_sum_binary.columns=[act+'_binary' for act in programs]
    log_program=pd.concat([log_program,log_df_sum,log_df_mean,log_df_sum_binary],axis=1)
    monthly_program_origin = pd.pivot_table(log_df, index = ['doll_id'], values = 'use_total', columns = 'month', aggfunc = ['sum']).fillna(0)
    monthly_program_binary = monthly_program_origin.copy()
    monthly_program_binary[monthly_program_binary>0]=1

    monthly_program_origin.columns=['M'+str(i) for i in range(1,13)]
    monthly_program_binary.columns=['M'+str(i)+'_binary' for i in range(1,13)]
    log_program=pd.concat([log_program,monthly_program_origin,monthly_program_binary],axis=1)
     
    ear_use_days = pd.DataFrame(log_df[['doll_id','year','month','day']].drop_duplicates()['doll_id'].value_counts())
    ear_use_days.columns=['ear_use_days']
    log_program_merged=pd.merge(log_program,ear_use_days,left_index=True, right_index=True)
    return log_program_merged

def preprocess_log_action(log_df):
    activities=['sum_stroke','sum_hand_hold','sum_knock','sum_human_detection','sum_gymnastics','sum_brain_tier']
    log_df=log_df[log_df['doll_id']>99]
    log_df['M']=pd.to_datetime(log_df['YM']).dt.month
    log_df_binary=log_df.copy()
    log_df_binary[activities]=(log_df_binary[activities]>0).astype('int')
    
    log_action = pd.DataFrame(index=sorted(list(set(log_df['doll_id']))))
    log_action['date_min']=pd.to_datetime(log_df.groupby('doll_id').min()['YM'].values)
    log_action['date_max']=pd.to_datetime(log_df.groupby('doll_id').max()['YM'].values)
    log_action['date_period']=(log_action['date_max']-log_action['date_min']).dt.days
    log_action['use_days']=log_df.groupby('doll_id').count()['YM']
    log_action['date_over_20']=(log_action['date_period']>19).astype('int')
    log_action[[str(act) +'_binary' for act in activities]]=log_df_binary.groupby('doll_id').sum()[activities]
    log_action[activities]=log_df.groupby('doll_id').sum()[activities]
    
    M_crosstab=pd.crosstab(log_df['doll_id'], log_df['M'].fillna('n/a'))
    M_crosstab2=M_crosstab.copy()
    M_crosstab2[M_crosstab2>0]=1

    M_crosstab.columns=['M'+str(i) for i in range(1,13)]
    M_crosstab2.columns=['M'+str(i)+'_binary' for i in range(1,13)]

    log_action_merged = pd.concat([log_action,M_crosstab,M_crosstab2],axis=1)
    
    return log_action_merged
        
        

def filter_df_program(df_origin,programs_binary,month_binary):
    df = df_origin.copy()
    programs=['story', 'religion', 'music', 'english', 'remembrance', 'quiz', 'gymnastics']
    months = ['M'+str(i) for i in range(1,13)]
    
    
    if programs_binary:
        df.drop([act+'_sum' for act in  programs],axis=1,inplace=True)
    else:
        df.drop([act+'_binary' for act in  programs],axis=1,inplace=True)
        
    if month_binary:
        df.drop(months,axis=1,inplace=True)
    else:
        df.drop([m+'_binary' for m in  months],axis=1,inplace=True)
    return df


def filter_df_action(df_origin,activities_binary,month_binary):
    df = df_origin.copy()
    activities = ['sum_stroke', 'sum_hand_hold', 'sum_knock', 'sum_human_detection', 'sum_gymnastics', 'sum_brain_tier']
    months = ['M'+str(i) for i in range(1,13)]
    
    df=df.drop(['date_min','date_max'],axis=1)
    
    if activities_binary:
        df.drop(activities,axis=1,inplace=True)
    else:
        df.drop([act+'_binary' for act in  activities],axis=1,inplace=True)
        
    if month_binary:
        df.drop(months,axis=1,inplace=True)
    else:
        df.drop([m+'_binary' for m in  months],axis=1,inplace=True)
    return df
