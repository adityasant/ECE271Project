'''
This code is used to read the metadata from the csv files
It starts with the genre information

Following this the track information is read

This is used to create a dictionary matching the tracks to the metadata

This code is used as a precursor to extract the features for training; it is used to create
an index of the tracks and genres
'''

import numpy as np
import pandas as pd

metadata_path = '..\\Dataset\\fma_metadata'

## Read genre information
genres = pd.read_csv(metadata_path+'\\genres.csv')
genre_grps = genres.groupby('parent')
genre_id_root = genre_grps.get_group(0)['genre_id'] # Extract the parent groups


## Read track information
fields = ['track']
tracks = pd.read_csv(metadata_path+'\\tracks_mod.csv',nrows=85000)
tracks = tracks.dropna()
tgen = list(tracks['genres'])
tgen = [t.replace('[','') for t in tgen]
tgen = [t.replace(']','') for t in tgen]
tgen = [t.replace(' ','') for t in tgen]
for k in  range(len(tgen)):
    if(len(tgen[k])!=0):
        tgen[k] = [i for i in map(int, tgen[k].split(','))]
    else:
        tgen[k]=[]
tracks['genres']=tgen

## Match the tracks to the genre data: Create dictionary with genre label as key
genre_dict = {}
for i in range(len(tgen)):
    if i in tracks.index:
        ti = tracks['track_id'][i]
        for k in tgen[i]:
            if k is not None:
                if (k in genre_dict.keys()):
                    genre_dict[k]+=[ti]
                else:
                    genre_dict.update({k:[ti]})


## Total number of samples for the different root genres
for i in genre_id_root:
    print(i,len(genre_dict[i]))
    