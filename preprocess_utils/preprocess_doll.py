from preprocess_utils.factor_analysis import *
#import pandas as pd

def preprocess_doll(doll_merged):
    doll_merged.drop(['group_id','user_group_member_id'],axis=1,inplace=True)
    doll_merged = doll_merged.drop_duplicates(keep='first')
    doll_merged = doll_merged[doll_merged.agency_name != '전체']
    doll_merged.doll_id.value_counts()
    doll_merged = doll_merged.set_index('doll_id')

    doll_merged = doll_merged[doll_merged.columns[~doll_merged.columns.str.contains('FM')]]
    
    drop_cols = ['user_id','battery','active_monitor','is_edited','regsted_year','regsted_month',
            'relig1','relig2','relig3','relig4','relig5',
            'mac_id','is_host','agency_name','serial_number_x','agency_code',
            'match_agc','religion_alarm','calender_type','right_ear_function','left_ear_function','is_active_detect',
            'serial_number_y','alarm_low_battery','alarm_disconnect']

    doll_merged.drop(drop_cols,axis=1,inplace=True)

    # [Information] Age & Sex 
    doll_merged = doll_merged[doll_merged['sex']!=0]
    doll_merged['sex'] = doll_merged['sex']-1
    doll_merged['age'][doll_merged['age']>100] =100
    doll_merged['age'][doll_merged['age']<40] =40

    # [Information] Time ( Sleep & Food )
    doll_merged['wake_dawn'] = doll_merged['HOUR(wakeup)'].isin([4,5,6]).astype('int')
    doll_merged['wake_morning'] = doll_merged['HOUR(wakeup)'].isin([7,8,9,10]).astype('int')

    doll_merged['breakfast_dawn'] = doll_merged['HOUR(breakfast)'].isin([4,5,6]).astype('int')
    doll_merged['breakfast_morning'] = doll_merged['HOUR(breakfast)'].isin([7,8,9,10]).astype('int')
    doll_merged['breakfast_afternoon'] = doll_merged['HOUR(breakfast)'].isin([11,12,13]).astype('int')

    doll_merged['lunch_afternoon'] = doll_merged['HOUR(lunch)'].isin([11,12,13,14,1,2]).astype('int')

    doll_merged['dinner_dinner'] = doll_merged['HOUR(dinner)'].isin([17,18,19,5,6,7]).astype('int')
    doll_merged['dinner_late'] = doll_merged['HOUR(dinner)'].isin([20,21,22,23,8,9,10,11]).astype('int')

    doll_merged['sleep_early'] = doll_merged['HOUR(sleep)'].isin([20,21,22,23]).astype('int')
    doll_merged['sleep_late'] = doll_merged['HOUR(sleep)'].isin([0,1,2,3]).astype('int')

    doll_merged.drop('HOUR(wakeup)',axis=1,inplace=True)
    doll_merged.drop('HOUR(breakfast)',axis=1,inplace=True)
    doll_merged.drop('HOUR(lunch)',axis=1,inplace=True)
    doll_merged.drop('HOUR(dinner)',axis=1,inplace=True)
    doll_merged.drop('HOUR(sleep)',axis=1,inplace=True)

    doll_merged= doll_merged[doll_merged[['disease1','disease2','disease3']].isna().sum(axis=1)==0]

    return doll_merged
    