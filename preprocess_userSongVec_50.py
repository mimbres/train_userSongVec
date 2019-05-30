#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 14:34:50 2019

@author: sungkyun

Total num_songs =  132439
num_duplicated = 2529 (1.9%) , num_unmatched = 19207 (14.5%)
top duplicated songs:
39103     13
29028     12
21678     10
158131     9
139942     9

We use 110703 songs out of 132439 (reducing 16.4%)
"""

import pandas as pd
import numpy as np
from tqdm import trange, tqdm

AUDIO_FEAT_WITH_SPOTIFY_ID_FILEPATH = './data/final_mapping.json'
CF_ID_FILEPATH = './data/user_song_feat_201905/features_spotify_id_50.json'
#CF_FEAT_FILEPATH = './data/cf_features_spotify_id.npy' # No modifications is required for this file!!
OUTPUT_AUDIO_FEAT_FILEPATH = './data/audio_featmtx_50.npy'
SAVED_SCALER_FILEPATH = './data/std_scaler.sav'

_dict = {'major': 1, 'minor': 0}



# Load .json file...

cf_id = pd.read_json(CF_ID_FILEPATH)
#cf_feat = np.load(CF_FEAT_FILEPATH)

df_fm = pd.read_json(AUDIO_FEAT_WITH_SPOTIFY_ID_FILEPATH)[1]
fm_id = list()
for i in trange(len(df_fm)):
    fm_id.append(df_fm[i][0])
    


# Create an empty result matrix
num_cf_items = len(cf_id)
cf_mtx = np.zeros((num_cf_items,200))
audiofeat_mtx = np.zeros((num_cf_items,29))

num_duplicated = 0
num_unmatched = 0
_ids = list()
i=0
for ci in tqdm(cf_id[0]):
    try:
        sel_fm_id = fm_id.index(ci)
        if sel_fm_id in _ids:
            num_duplicated += 1
            pass
        else:
            _ids.append(sel_fm_id)
            _feat = np.asarray(df_fm[sel_fm_id])
            _feat[20] = _dict.get(_feat[20])
            _feat[5] = _feat[5][:4] # '2005-01-01' --> '2005'
            audiofeat_mtx[i,:] = _feat[[4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]]
    except ValueError:
        num_unmatched += 1 
    


print('Total num_songs = ', len(cf_id))
print('num_duplicated = {} , num_unmatched = {}'.format(num_duplicated, num_unmatched))
print('We use {} songs out of {}'.format(len(_ids), len(cf_id)))
#print('top duplicated songs:\n', pd.DataFrame(_ids)[0].value_counts().iloc[:10])

# Feature normalization
import pickle
#from sklearn.preprocessing import StandardScaler
scaler = pickle.load(open(SAVED_SCALER_FILEPATH, 'rb'))
audiofeat_mtx_new = scaler.fit_transform(audiofeat_mtx)
audiofeat_mtx_new[:,15] = audiofeat_mtx[:,15]

# Save results as .npy
np.save(OUTPUT_AUDIO_FEAT_FILEPATH, audiofeat_mtx_new.astype(np.float32))
