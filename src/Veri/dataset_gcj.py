import sys
import os
import re
import json
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import pickle
import gzip
import pathlib
import code_bert_format as cbf


def gen_positive(df_in, authors, num):
    for a in tqdm(authors):
        af = df_in[df_in['author'] == a]
        for _ in range(num):
            success = False
            try_num = 0
            while success == False:
                try:
                    e = af.sample(2)
                    f0 = e.iloc[0]['full_path']
                    f1 = e.iloc[1]['full_path']
                    a0 = e.iloc[0]['author']
                    a1 = e.iloc[1]['author']
                    t0 = e.iloc[0]['flines']
                    t1 = e.iloc[1]['flines']
                    p0 = e.iloc[0]['problem']
                    p1 = e.iloc[1]['problem']
                    label_p = 1 if e.iloc[0]['problem'] == e.iloc[1]['problem'] else 0
                    assert a1 == a0
                    assert f0 != f1
                    success = True
                    yield f0, f1, 1, a0, a1, t0, t1, p0, p1, label_p
                except:
                    if try_num < 10:
                        try_num += 1
                    else:
                        success = True
                        print(a)


def gen_negative(df_in, authors, num):
    df_ds = df_in[df_in['author'].isin(authors)]
    for a in tqdm(authors):
        af = df_ds[df_ds['author'] == a]
        nf = df_ds[df_ds['author'] != a]
        for _ in range(num):
            success = False
            try_num = 0
            while success == False:
                try:
                    e0 = af.sample(1)
                    e1 = nf.sample(1)
                    f0 = e0.iloc[0]['full_path']
                    f1 = e1.iloc[0]['full_path']
                    a0 = e0.iloc[0]['author']
                    a1 = e1.iloc[0]['author']
                    t0 = e0.iloc[0]['flines']
                    t1 = e1.iloc[0]['flines']
                    p0 = e0.iloc[0]['problem']
                    p1 = e1.iloc[0]['problem']
                    label_p = 1 if e0.iloc[0]['problem'] == e1.iloc[0]['problem'] else 0
                    assert a0 != a1
                    assert f0 != f1
                    success = True
                    yield f0, f1, 0, a0, a1, t0, t1, p0, p1, label_p
                except:
                    if try_num < 10:
                        try_num += 1
                    else:
                        success = True
                        print(a)


def gen_all(df_in, name, authors, pos, neg):
    ''' 0. Generate pos+neg samples for giving dataset '''
    # gen_positive/negative(df_in, authors, num):
    entries = [x for x in gen_positive(df_in, authors, pos)] \
        + [x for x in gen_negative(df_in, authors, neg)]
    entries_df = pd.DataFrame(
        entries, columns=[
            'file0', 'file1', 'label', 'author0', 'author1', 'flines0', 'flines1', 'problem0', 'problem1', 'label_p'])

    ''' 1. For dataset used in training, split traing/test0 '''
    if name == 'train':
        msk = np.random.rand(len(entries_df)) < 0.8
        entries_train_df = entries_df.iloc[msk]
        entries_test0_df = entries_df.iloc[~msk]
        return entries_train_df, entries_test0_df

    return entries_df


def gen_language(
        df_in: pd.DataFrame, language,
        pos_per_author_tr, neg_per_author_tr,
        pos_per_author_te, neg_per_author_te,):
#     print(df.loc[df.author=='musouka'])

    ''' 1. Sample train and test authors 0.7:0.3 '''
    np.random.seed(0)
    authors = sorted(np.unique(df_in['author']))
    authors_train = np.random.choice(
        authors, size=round(0.7*len(authors)), replace=False)
    authors_test = [a for a in authors if a not in authors_train]

    ''' 2. Sample train and test 1 files 0.7:0.3'''
    train_author_df = df_in[df_in['author'].isin(authors_train)]
#     print(train_author_df.loc[train_author_df.author=='musouka'])
    files_train_df = train_author_df.groupby('author').apply(lambda x: x.head(round(len(x)*0.7)))
    files_test1_df = train_author_df.loc[~train_author_df.full_path.isin(files_train_df.full_path)].dropna()
#     print(files_train_df.loc[files_train_df.author=='musouka'])

    ''' 3. Gen train(test0), test1, test2 from file and author sets '''
    print('sample train: use train authors + train files')
    train_df, test0_df = gen_all(
        files_train_df, 'train', authors_train,
        pos_per_author_tr, neg_per_author_tr)
    print('sample test 1: use train authors + test files')
    test1_df = gen_all(
        files_test1_df, 'test1', authors_train,
        pos_per_author_te, neg_per_author_te)

    print('sample test 2: use test authors + corresponding files')
    test2_df = gen_all(
        df_in, 'test2', authors_test,
        pos_per_author_te, neg_per_author_te
    )

    return {
        language.lower() + '_training': train_df,
        language.lower() + '_t0_oos_pairs': test0_df,
        language.lower() + '_t1_oos_files': test1_df,
        language.lower() + '_t2_oos_authors': test2_df
    }


def save_file(lan, in_, path):
    fname = '_'.join(in_['full_path'].split('/'))
    flines = in_['flines']
    with open(os.path.join(path, lan, fname), 'w') as f:
        f.write(flines)
        
        
def get_gcj_dataset(
    df_home, year, languages,
    pos_per_author_tr=10,
    neg_per_author_tr=10,
    pos_per_author_te=10,
    neg_per_author_te=10,
):
    code_file_path = os.path.join(df_home, 'files')
    c2v_path = os.path.join(df_home, 'c2v')
    split_path = os.path.join(df_home, 'splits')
    
    df = pd.read_csv(df_home+'gcj'+year+'.csv')
    df['language'] = df.apply(lambda row: os.path.splitext(row['file'])[1][1:] , axis=1)
    df.columns = ['_', 'year', 'round', 'author', 'problem', 'solution', 'file', 'full_path', 'flines', 'language']
    
    # 1. generate data for C2V
    if not os.path.exists(c2v_path):
        os.makedirs(c2v_path)
        for lan in languages:
            df_lan = df.loc[df.language==lan]
            df_lan[['full_path', 'flines']].progress_apply(lambda row: save_file(lan, row, code_file_path), axis=1)
        
    # 2. generate data for CodeBERT

    
    # 3. split data
    if not os.path.exists(split_path):
        os.makedirs(split_path)

        ''' 0. sample data for java and c '''
        datasets = {}
        for language in languages:
            df_language = df.loc[df['language']==language].groupby('author').filter(lambda x: len(x) > 5)
            print('#@### Language #@###: '+language)
            print('#@### DF length #@###: '+str(df_language.shape[0]))
            datasets.update(
                gen_language(
                    df_language, language,
                    pos_per_author_tr=pos_per_author_tr,
                    neg_per_author_tr=neg_per_author_tr,
                    pos_per_author_te=pos_per_author_te,
                    neg_per_author_te=neg_per_author_te)
            )
        for k, v in datasets.items():
            csv_path = os.path.join(split_path, k+'.csv')
            v.to_csv(csv_path, index=False)
    
    datasets = {}
    for csv_file in os.listdir(split_path):
        if '.csv' not in csv_file:
            continue
        v = pd.read_csv(os.path.join(split_path, csv_file))
        datasets[csv_file.replace('.csv', '')] = v

    return datasets


def formatting_codebert(df_home, ds, name):
    code_bert_path = os.path.join(df_home, 'MS_bert', 'GCJ')
    tqdm.pandas()
    ds['flines_clean0'] = ds.flines0.progress_apply(lambda x: cbf.clean(x))
    ds['flines_clean1'] = ds.flines1.progress_apply(lambda x: cbf.clean(x))

    cbf.gen_dscb(ds, os.path.join(code_bert_path, name))
        
    

if __name__ == '__main__':
    # to run : on the package level directory issue `python -m VeriBin.dataset_gcj`
    

    
        
    formatting_codebert('/home/jovyan/Source_Code_Veri/data/',
                        dataset, dataset_name)
    
    
    print('generating gcj_dataset samples...')
    datasets = get_gcj_dataset(
        df_home = '/home/jovyan/Source_Code_Veri/data/',
        year = '2017', languages = ['java', 'py']
    )

        

#     print(testing)

#     for i in training.to_dict('records'):
#         print(i)
#         break