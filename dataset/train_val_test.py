import pandas as pd
import numpy as np


compute_train_test_val = False

np.random.seed(42)


def split_events_by_mag(mag, data):
    min_mag = mag[0]
    max_mag = mag[1]
    temp = data[(data['magnitude'] >= min_mag) & (data['magnitude'] < max_mag)].copy()
    events = temp['source_id'].tolist()
    np.random.shuffle(events)
    total_size = len(events)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size  
    train_ev = events[:train_size]
    val_ev = events[train_size:train_size + val_size]
    test_ev = events[train_size + val_size:]

    return train_ev, val_ev, test_ev


def count_per_magnitude(df, col):

    mag_ranges = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7)]
    
    for mag_range in mag_ranges:
        data_range = df[(df['magnitude'] >= mag_range[0]) & (df['magnitude'] < mag_range[1])]
        print(f'{col} with {mag_range[0]}<= magnitude <{mag_range[1]}: ', len(set(data_range[col])))
    
    return



if compute_train_test_val:
    data_p = pd.read_csv('./dataset/input_output_P.csv')
    data_s = pd.read_csv('./dataset/input_output_S.csv')
    
    # Define proportions of train-val-test
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    # Initialize train-val-test dataframe for both dataframe
    train_p = pd.DataFrame()
    val_p = pd.DataFrame()
    test_p = pd.DataFrame()
    train_s = pd.DataFrame()
    val_s = pd.DataFrame()
    test_s = pd.DataFrame()
    
    # Define magnitude ranges
    ranges_mag = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7)]

    for mag in ranges_mag:
        train_events, val_events, test_events = split_events_by_mag(mag, data_p)
        train_p = pd.concat([train_p, data_p[data_p['source_id'].isin(train_events)]], ignore_index = True)
        val_p = pd.concat([val_p, data_p[data_p['source_id'].isin(val_events)]], ignore_index = True)
        test_p = pd.concat([test_p, data_p[data_p['source_id'].isin(test_events)]], ignore_index = True)

        train_s = pd.concat([train_s, data_s[data_s['source_id'].isin(train_events)]], ignore_index = True)
        val_s = pd.concat([val_s, data_s[data_s['source_id'].isin(val_events)]], ignore_index = True)
        test_s = pd.concat([test_s, data_s[data_s['source_id'].isin(test_events)]], ignore_index = True)
    
    train_p.to_csv('./dataset/train_p.csv', index = False)
    val_p.to_csv('./dataset/val_p.csv', index = False)
    test_p.to_csv('./dataset/test_p.csv', index = False)
    train_s.to_csv('./dataset/train_s.csv', index = False)
    val_s.to_csv('./dataset/val_s.csv', index = False)
    test_s.to_csv('./dataset/test_s.csv', index = False)

else:
    train_p = pd.read_csv('./dataset/train_p.csv')
    val_p = pd.read_csv('./dataset/val_p.csv')
    test_p = pd.read_csv('./dataset/test_p.csv')
    train_s = pd.read_csv('./dataset/train_s.csv')
    val_s = pd.read_csv('./dataset/val_s.csv')
    test_s = pd.read_csv('./dataset/test_s.csv')

count_per_magnitude(train_p, 'source_id')
count_per_magnitude(train_s, 'source_id')
count_per_magnitude(val_p, 'source_id')
count_per_magnitude(val_s, 'source_id')
count_per_magnitude(test_p, 'source_id')
count_per_magnitude(test_s, 'source_id')


def compute_len(row):
    return len(row['Input']) + len(row['Output'])


train_p['len'] = train_p.apply(compute_len, axis = 1)
print(max(train_p['len'])) #1594
print('train: ', train_p.shape[0]) #28295
val_p['len'] = val_p.apply(compute_len, axis = 1)
print(max(val_p['len'])) #932
print('val: ', val_p.shape[0]) #3535
test_p['len'] = test_p.apply(compute_len, axis = 1)
print(max(test_p['len'])) # 852
print('test: ', test_p.shape[0]) #3541

train_s['len'] = train_s.apply(compute_len, axis = 1)
print(max(train_s['len'])) #1594
print('train: ', train_s.shape[0]) #28295
val_s['len'] = val_s.apply(compute_len, axis = 1)
print(max(val_s['len'])) #932
print('val: ', val_s.shape[0]) #3535
test_s['len'] = test_s.apply(compute_len, axis = 1)
print(max(test_s['len'])) # 852
print('test: ', test_s.shape[0]) #3541

