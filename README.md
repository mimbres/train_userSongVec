# trainUserSongVec
## Dataset Info
* CF vectors representing User-Song relations + correcsponding Spotify acoustic features. 
* Song duplication issue:
```
Total num_songs =  132439
num_duplicated = 2529 (1.9%) , num_unmatched = 19207 (14.5%)
top duplicated songs:
39103     13
29028     12
21678     10
158131     9
139942     9

We use 110703 songs out of 132439 (reducing 16.4%)
```


# /data
* 1. audio_featmtx.npy: Equivalent with spotify audio features. This was created by running preprocess_userSongVec.py that connect final_mapping.json and cf_feature_spotify_id.json.
* 2. cf_features_spotify_id.npy: (Andres) CF features
* 3. cf_features_spotify_id.json: (Andres) Spotify id for CF features. After preprocessing, we can ignore this.
* 4. final_mapping.json: (Andres) spotify audio features with spotify ID(but sorted differently from 3.)

# For training a model:
* After running preprocess+userSongVec.py, we will need only two *.npy files (1 and 2 in the list above) for training!
