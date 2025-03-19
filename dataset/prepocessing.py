# libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
import h5py

compute_dist = False
compute_PGV = False
compute_input_output = True

def take_PGV(row):
    # get the name of the trace 
    data = data_gm.get('data/'+row['trace_name'])

    # Compute vertical PGV around P-wave arrival time in a 0.2 s time window 
    PGV_p = max(abs(data[2, int(row['trace_P_arrival_sample']) - 10: int(row['trace_P_arrival_sample']) + 10]))


    # Compute horizontal PGV around S-wave arrival time in a 2 s time window
    if pd.isna(row['trace_S_arrival_sample']):
        PGV_s = np.nan
    else:
        first_component = np.abs(data[0, int(row['trace_S_arrival_sample']) - 50: int(row['trace_S_arrival_sample']) + 150])
        second_component = np.abs(data[1, int(row['trace_S_arrival_sample']) - 50: int(row['trace_S_arrival_sample']) + 150])

        PGV_s = (np.max(first_component)+np.max(second_component))/2
    return PGV_p, PGV_s

def print_shapes(df):
    print('Number traces: ', df.shape[0])
    print('Number events: ', len(set(df['source_id'])))






# Read instance metadata with usefull columns 
path_md = './INSTANCE/metadata_Instance_events_v2.csv'

data_md = pd.read_csv(path_md, usecols = ['source_id', 'trace_name', 
 'path_travel_time_P_s', 'trace_P_arrival_time', 'trace_P_arrival_sample','trace_S_arrival_sample',
  'source_magnitude', 'source_latitude_deg', 'source_longitude_deg', 'source_depth_km',
  'trace_deconvolved_units','trace_dt_s',
  'station_latitude_deg', 'station_longitude_deg'])

print('All metadata:\n')
# Print shapes 
print_shapes(data_md)
# Show percentage nan in each column
print('Nan percentage in each column: \n', data_md.isna().sum()/data_md.shape[0])  


# Drop traces in mps2 -> take only data with gm in velocities
data_md = data_md[data_md['trace_deconvolved_units'] == 'mps'].copy()

print('Only traces with velocities ground motion data:\n ')
# Print shapes 
print_shapes(data_md)
# Show percentage nan in each column
print('Nan percentage in each column: \n', data_md.isna().sum()/data_md.shape[0])  

if compute_dist:
    # For each event find first station that recorded it with velocities ground motion traces and compute distance in seconds 
    # with all other stations
    # Change in datetime type the column traca_P_arrival_time
    data_md['timestamp'] = pd.to_datetime(data_md['trace_P_arrival_time'])
    data_md['distance_from_first'] = np.nan

    for event in tqdm(set(data_md['source_id'])):
        first_station = data_md[data_md['source_id'] == event]['timestamp'].min()
        data_md.loc[data_md['source_id'] == event, 'distance_from_first'] = data_md[data_md['source_id'] == event]['timestamp'].apply(lambda x: (x - first_station).total_seconds())
    data_md.to_csv('./dataset/metadata_with_dist_sec.csv', index = False)
else:
    data_md = pd.read_csv('./dataset/metadata_with_dist_sec.csv')






if compute_PGV:
    # ground motion data path
    path_gm = './INSTANCE/Instance_events_gm.hdf5'
    # open gm data 
    data_gm = h5py.File(path_gm, 'r+')

    tqdm.pandas()
    data_md[['PGV_p', 'PGV_s']] = data_md.progress_apply(take_PGV, axis = 1, result_type = 'expand')
    data_md.to_csv('./dataset/metadata_with_PGV.csv', index = False)

else:
    data = pd.read_csv('./dataset/metadata_with_PGV.csv')


print('After computing PGVs:\n ')
print('Nan percentage in each column: \n', data.isna().sum()/data_md.shape[0]) 


data.dropna(inplace = True)

print('After deleting all nans:\n')
# Print shapes 
print_shapes(data)
# Show percentage nan in each column
print('Nan percentage in each column: \n', data.isna().sum()/data.shape[0]) 


# Select, for each event, only traces with a distance from the first one smaller than 5 sec
data = data[data['distance_from_first'] <= 5].copy()
# Drop events with less than 5 traces 
count_traces = data['source_id'].value_counts()
selected_events = count_traces[count_traces > 4].index
data = data[data['source_id'].isin(selected_events)].copy()

print('After taking only events with at least 5 traces with a distance from the first one smaller than 5 seconds:\n')
# Print shapes 
print_shapes(data)



if compute_input_output:
    # Convert PGVs in micrometers/second
    data['PGV_p'] = data['PGV_p']*100000
    data['PGV_s'] = data['PGV_s']*100000



    # form groups by source_id
    groups = data.groupby(['source_id'])
    list_of_group_dfs = [group[['source_id', 'station_latitude_deg', 'station_longitude_deg', 'source_magnitude', 'source_latitude_deg', 'source_longitude_deg', 'PGV_s', 'PGV_p', 'path_travel_time_P_s', 'source_depth_km', 'distance_from_first']] for _, group in groups]



    # empty list for formatted strings for each dataframes
    formatted_strings_per_df_p = []
    formatted_strings_per_df_s = []

    output = []

    source_id = []

    number_traces = []

    magnitude = []


    # iterate over groups of dataframes
    for df in tqdm(list_of_group_dfs):
        # reorder rows based on distance_from_first
        df.sort_values(by = 'distance_from_first', inplace = True)

        # empty list for formatted strings of each given dataframe
        rows_strings_p = []
        rows_strings_s = []

        # select the first row in order to get the magnitude that is equal for each row in the given dataframe
        first_row = df.iloc[0]
        out = f"{int(first_row['source_latitude_deg']*10000)},{int(first_row['source_longitude_deg']*10000)},{int(first_row['source_depth_km']*100)},{int(first_row['source_magnitude']*100)},{int(first_row['path_travel_time_P_s']*100)}"
        output.append(out)

        source_id.append(first_row['source_id'])

        number_traces.append(df.shape[0])

        magnitude.append(first_row['source_magnitude'])

        # iteret over rows in the dataframe
        for index, row in df.iterrows():
            # formatted string for each row in the group
            row_string_p = f"({int(row['station_latitude_deg']*10000)},{int(row['station_longitude_deg']*10000)},{int(row['distance_from_first']*100)},{int(row['PGV_p']*10000)})"
            rows_strings_p.append(row_string_p)
            row_string_s = f"({int(row['station_latitude_deg']*10000)},{int(row['station_longitude_deg']*10000)},{int(row['distance_from_first']*100)},{int(row['PGV_s']*10000)})"
            rows_strings_s.append(row_string_s)
        
        # join formatted strings of the same dataframe
        df_string_p = "\n".join(rows_strings_p)
        formatted_strings_per_df_p.append(df_string_p)

        df_string_s = "\n".join(rows_strings_s)
        formatted_strings_per_df_s.append(df_string_s)
        
        


    # Create final dataframe
    final_df_p = pd.DataFrame({
        'Input': formatted_strings_per_df_p,
        'Output': output,
        'source_id': source_id,
        'number_traces': number_traces,
        'magnitude': magnitude
    })

    final_df_p.to_csv('./dataset/input_output_P.csv', index= False)

    final_df_s = pd.DataFrame({
        'Input': formatted_strings_per_df_s,
        'Output': output,
        'source_id': source_id,
        'number_traces': number_traces,
        'magnitude': magnitude
    })

    final_df_s.to_csv('./dataset/input_output_S.csv', index= False)

else:
    data_p = pd.read_csv('./dataset/input_output_P.csv')
    data_s = pd.read_csv('./dataset/input_output_S.csv')