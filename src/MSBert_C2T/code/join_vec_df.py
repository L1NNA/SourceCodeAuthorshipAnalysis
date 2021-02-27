import pandas as pd
import numpy as np
import sys

language = sys.argv[1]

if os.path.exist('/home/jovyan/Source_Code_Veri/data/GCJ/gcj2017_'+language+'.csv'):
    df = pd.read_csv('/home/jovyan/Source_Code_Veri/data/GCJ/gcj2017_'+language+'.csv')
else:
    df = pd.read_csv('/home/jovyan/Source_Code_Veri/data/GCJ/gcj2017.csv')
    df['language'] = df.apply(lambda row: os.path.splitext(row['file'])[1][1:] , axis=1)
    df = df.loc[df['language']==language]
    
df['flines_vec_MSC2T'] = ''


corpus_vec = np.load('/home/jovyan/Source_Code_Veri/data/GCJ/MS_C2T/gcj2017'+language+'.npy')
corpus_vec_swap = np.swapaxes(corpus_vec,0,1)
corpus_vec_swap_mean = np.mean(corpus_vec_swap, axis=1)
corpus_vec_swap_mean.shape


for i in range(df.shape[0]):
    df['flines_vec_MSC2T'].iloc[i] = list(corpus_vec_swap_mean[i])

    df.to_csv('/home/jovyan/Source_Code_Veri/data/GCJ/gcj2017_'+language+'.csv')
df.to_csv('/home/jovyan/Source_Code_Veri/data/GCJ/MS_C2T/gcj2017_'+language+'.csv')