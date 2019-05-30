#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 02:19:59 2019

@author: mimbres
"""

import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
#from utils.matrix_math import indices_to_one_hot

#TAG_PATH = './data/cf_features_spotify_id.npy'
#FEAT_PATH = './data/audio_featmtx.npy'
NUM_TRSET = 100000
# TRSET: 100,000
# TSSET: 10,703


def CFDataloader( mtrain_mode=True,
                      data_sel=None,
                      normalization_factor=100,
                      batch_size=1,
                      shuffle=False,
                      num_workers=8,
                      pin_memory=True,
                      tag_path='./data/user_song_Feat_201904/cf_features_spotify_id.npy',
                      feat_path='./data/audio_featmtx.npy'):
    
    dset = CFDataset(mtrain_mode=mtrain_mode,
                         normalization_factor=normalization_factor,
                         data_sel=data_sel,
                         tag_path=tag_path,
                         feat_path=feat_path)
    dloader = DataLoader(dset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         num_workers=num_workers,
                         pin_memory=pin_memory)
    return dloader



class CFDataset(Dataset):
    def __init__(self,
                 mtrain_mode=True,
                 normalization_factor=100,
                 data_sel=None,
                 tag_path=str(),
                 feat_path=str()):
        
        self.mtrain_mode = mtrain_mode
        self.data_sel    = data_sel # NOT IMPLEMENTED YET
        self.tag_all     = []
        self.feat_all    = []
        
        # Import data
        tag_all = np.load(tag_path)
        feat_all = np.load(feat_path)
        
        # Scaling 20190531
        from sklearn.preprocessing import StandardScaler 
        scaler = StandardScaler()
        tag_all = scaler.fit_transform(tag_all)
        
        # Train/test split (8:2 by default)
        if self.mtrain_mode:
            self.tag_all  = tag_all[:NUM_TRSET,:]
            self.feat_all = feat_all[:NUM_TRSET,:]
        else:
            self.tag_all  = tag_all[NUM_TRSET:,:]
            self.feat_all = feat_all[NUM_TRSET:,:]
            
        # Normalize tag probability
        self.tag_all = self.tag_all.astype(np.float32) / normalization_factor
        return None
    
    
    
    def __getitem__(self, index):
        tag = self.tag_all[index,:]
        feat = self.feat_all[index, :]
        return index, tag, feat 
    
    
    
    def __len__(self):
        return len(self.tag_all) # return the total number of items
    
    
    