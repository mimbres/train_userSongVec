# trainUserSongVec
![85d04e1f-130d-466b-8880-43ee42c9993c_200x200](https://user-images.githubusercontent.com/26891722/58658328-41c1e700-835b-11e9-96f7-6865dd885fe5.png)

[![PyPI](https://img.shields.io/badge/python-3.6%2C%203.7%20-blue.svg)]() 

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


# Install:
* Install git-lfs by following guide: https://git-lfs.github.com/
```
git lfs install
git config credential.helper store # Without this, we will have to type ID/password 3times... 
git clone <THIS REPO>
```

# /data
* audio_featmtx.npy: Equivalent with spotify audio features. This was created by running preprocess_userSongVec.py that connect final_mapping.json and cf_feature_spotify_id.json.
* cf_features_spotify_id.npy: (Andres) CF features
* cf_features_spotify_id.json: (Andres) Spotify id for CF features. After preprocessing, we can ignore this.
* final_mapping.json: (Andres) spotify audio features with spotify ID(but sorted differently from 3.)

# For training a model:
* After running preprocess+userSongVec.py, we will need only two *.npy files (1 and 2 in the list above) for training!

