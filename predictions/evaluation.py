import pandas as pd
import numpy as np
import re
from obspy.geodetics import gps2dist_azimuth

preds_P = pd.read_csv('./predictions/predictions_P.csv', usecols= ['Output', 'prediction', 'Input', 'number_traces','source_id'])
preds_S = pd.read_csv('./predictions/predictions_S.csv', usecols= ['Output', 'prediction', 'Input', 'number_traces','source_id'])
print(preds_P['number_traces'].max())

def extract_preds(row):
    pred = re.sub(r'[\n\)]', ',', row['prediction'])
    predictions = pred.split(',')
    targets = row['Output'].split(',')

    lat_target = float(targets[0])/10000
    lat_pred = float(predictions[0])/10000
    lon_target = float(targets[1])/10000
    lon_pred = float(predictions[1])/10000
    depth_target = float(targets[2])/100
    depth_pred = float(predictions[2])/100
    mag_target = float(targets[3])/100
    mag_pred = float(predictions[3])/100
    first_p_target = float(targets[4])/100
    first_p_pred = float(predictions[4])/100
    return lat_target, lon_target, depth_target, mag_target,first_p_target, lat_pred, lon_pred, depth_pred, mag_pred, first_p_pred

preds_P[['lat_target', 'lon_target', 'depth_target', 'mag_target', 'first_p_target', 'lat_pred', 'lon_pred', 'depth_pred', 'mag_pred', 'first_p_pred']] = preds_P.apply(extract_preds, axis = 1, result_type= 'expand')
preds_S[['lat_target', 'lon_target', 'depth_target', 'mag_target', 'first_p_target', 'lat_pred', 'lon_pred', 'depth_pred', 'mag_pred', 'first_p_pred']] = preds_S.apply(extract_preds, axis = 1, result_type= 'expand')



def compute_distance(row):
    distance, _, _ = gps2dist_azimuth(row['lat_pred'], row['lon_pred'], row['lat_target'], row['lon_target'])
    return distance/1000


preds_P['epicentral_distance'] = preds_P.apply(compute_distance,axis = 1,)
preds_P['z_dist'] = preds_P['depth_pred'] - preds_P['depth_target']
preds_P['hypocentral_dist'] = (preds_P['z_dist']**2 + preds_P['epicentral_distance']**2)**(0.5)

preds_S['epicentral_distance'] = preds_S.apply(compute_distance,axis = 1)
preds_S['z_dist'] = preds_S['depth_pred'] - preds_S['depth_target']
preds_S['hypocentral_dist'] = (preds_S['z_dist']**2 + preds_S['epicentral_distance']**2)**(0.5)



preds_P['mag_diff'] = preds_P['mag_pred'] - preds_P['mag_target']
preds_S['mag_diff'] = preds_S['mag_pred'] - preds_S['mag_target']

preds_P.to_csv('./evaluation_P.csv')
preds_S.to_csv('./evaluation_S.csv')

def metrics_for_mag_range(mag_range, preds, data_type):
    print(f'\nResults with {data_type}')
    preds_range = preds[(preds['mag_target']>=mag_range[0]) & (preds['mag_target']<mag_range[1])]
    print(f'Number of test samples with {mag_range[0]}<= magnitude <{mag_range[1]}: {preds_range.shape[0]}')

    print('Proportion od test samples with respect the whole test: ', round((preds_range.shape[0]/preds.shape[0])*100,2))
    print('Average geodesic distances: ', round(np.mean(preds_range['epicentral_distance']),2))
    print('Average hypocentral distances: ', round(np.mean(preds_range['hypocentral_dist']),2))
    print('Average absolute value differences between predicted and true depth: ', round(np.mean(np.abs(preds_range['z_dist'])),2))

    mae_mag = np.mean(np.abs(preds_range['mag_diff']))
    mse_mag = np.sqrt(np.mean((preds_range['mag_diff'])**2))
    print('RMSE magnitude: ', round(mse_mag, 2), '\n')
    print('MAE magnitude: ', round(mae_mag, 2), '\n')

    mae_first = np.mean(np.abs(preds_range['first_p_pred']-preds_range['first_p_target']))
    print('MAE first P time: ', round(mae_first, 2), '\n')

metrics_for_mag_range((0,7), preds_P, 'P data')
metrics_for_mag_range((0,7), preds_S, 'S data')


