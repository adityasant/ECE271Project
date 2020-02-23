'''
This code reads the metadata from the features.csv file. 
The aim is to visualize and extract information from this metadata and create the 
training set based on the extracted features. 

The earlier script can calculate the genre for the specific track. 
'''

import numpy as np
import pandas as pd

metadata_path = '..\\Dataset\\fma_metadata'


'''
Based on the information in the Bahuleya paper, the following fields are selected:
mfcc_4_mean, mfcc_1_std, spectral_contrast_2_mean, spectral_contrast_3_mean, mfcc_5_std,
spectral_contrast_1_mean, mfcc_9_mean, mfcc_3_mean, mfcc_17_mean, mfcc_1_mean,
spectral_contrast_2_std, tempo, mfcc_11_mean, mfcc_6_mean, mfcc_5_mean, spectral_contrast_3_std,
mfcc_2_std, mfcc_14_mean, mfcc_3_std, mfcc_10_mean
'''
## Column information
fields = ['mfcc.40','mfcc.42','mfcc.43','mfcc.44','mfcc.45','mfcc.48','mfcc.49','mfcc.50',\
    'mfcc.53','mfcc.56','mfcc.120','mfcc.121','mfcc.122','mfcc.124','spectral_contrast.14',\
        'spectral_contrast.15','spectral_contrast.16','spectral_contrast.43',\
            'spectral_contrast.44','feature']


## Read feature information 
features = pd.read_csv(metadata_path+'\\features.csv',usecols=fields)
feat_temp = features['feature']
for i in range(3,len(feat_temp)):
    feat_temp[i]=int(feat_temp[i])
features['feature'] = feat_temp


## Include code to filter the rows based on the training id
temp_id_set =[2,3,5,10,20,26]
feat1 = features.loc[features['feature'].isin(temp_id_set)]