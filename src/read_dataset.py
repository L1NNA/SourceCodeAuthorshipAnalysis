import tensorflow as tf
import sys
from collections import defaultdict
import os
import re
import json
from typing import NamedTuple, List, Tuple
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
import pickle
import gzip
import zipfile
from pygments.lexers import guess_lexer
from numpy import dot
from numpy.linalg import norm
import ast

from dataset_gcj import get_gcj_dataset
from utils import libc
from tfr import write_objs_as_tfr, read_objs_trf_ds, _default_fs_mapper
from tokenizer_sp import get_tokenizer_lg
from utils import AttrDict


_keywords = ('loc_', 'loc', 'sub_', 'var_', 'var', 'arg_', 'word_',
             'off_', 'locret_', 'flt_', 'dbl_', 'qword',
             'dword', 'offset',)


max_seq_len = 2400

def _binary_to_sequence(all_df, file_name, tokenizer, max_blk=300):
    ###
    ### for source code, we read files directly from the all_df
    ###
    code_sample = all_df.loc[all_df['full_path']==file_name]
    if code_sample.shape[0] >= 1:
        # extract block texts for ordered blocks
        lexer = guess_lexer(code_sample.iloc[0]['flines'])
        tokens = ' '.join([t[1] for t in lexer.get_tokens(code_sample.iloc[0]['flines']) if len(t[1].strip()) > 0 and 'comment' not in str(t[0]).lower()])
        tokens = tokenizer.EncodeAsIds(tokens)
        tokens = [t for t in tokens if t != 5]
        if len(tokens) < max_seq_len:
             tokens = tokens + [5] * (max_seq_len - len(tokens))
        else:
            tokens = tokens[:max_seq_len] 

        return tokens
    else:
        return []


def _mapper(all_df, entry, lang, *args):
    tokenizer_folder = '../../models/tokenizers/'+lang+'/'
    tokenizer = get_tokenizer_lg(all_df, entry, tokenizer_folder)
    seq0 = _binary_to_sequence(all_df, entry['file0'], tokenizer)
    seq1 = _binary_to_sequence(all_df, entry['file1'], tokenizer)
    vMS0 = ast.literal_eval(entry['vMS0'])
    if lang == 'py':
        vMS1 = ast.literal_eval(entry['vMS01'])
    else:
        vMS1 = ast.literal_eval(entry['vMS1'])
    ber_sim = (dot(vMS0, vMS1)/(norm(vMS0)*norm(vMS1))).item()

    return {
        'seq0': seq0,
        'ber0': vMS0,
        'seq1': seq1,
        'ber1': vMS1,
        'authorship': entry['label'],
        'problem': entry['label_p'],
        'ber_sim': ber_sim,
    }


def _mapper_c(all_df, entry, lang, *args):
    tokenizer_folder = '../../models/tokenizers/'+lang+'/'
    tokenizer = get_tokenizer_lg(all_df, entry, tokenizer_folder)
    seq0 = _binary_to_sequence(all_df, entry['file0'], tokenizer)
    seq1 = _binary_to_sequence(all_df, entry['file1'], tokenizer)

    return {
        'seq0': seq0,
        'seq1': seq1,
        'authorship': entry['label'],
        'problem': entry['label_p'],
    }


def _gen_ds_tfrs(all_df, df, lang, tfr_path):
    print(lang)
    entries = df.to_dict('records')
    if 'c' in lang:
        write_objs_as_tfr(all_df, entries, lang, tfr_path, obj_mapper=_mapper_c)
    else:
        write_objs_as_tfr(all_df, entries, lang, tfr_path, obj_mapper=_mapper)


def get_tfrs(
        data_path, 
        tfr_name,
        year, 
        lang):
    
    all_df = pd.read_csv(os.path.join(data_path, 'gcj'+year+'.csv'))
    if len(tfr_name) > 0:
        gcj_path = os.path.join(data_path, 'splits', lang, 'tfrs'+'_'+tfr_name)
    else:
        gcj_path = os.path.join(data_path, 'splits', lang, 'tfrs')
    print('== gcj_path ==: ' + gcj_path)
    if not os.path.exists(gcj_path):
        os.makedirs(gcj_path)

        training, validation, testing = get_gcj_dataset(data_path, year, lang)
        _gen_ds_tfrs(all_df, training, lang, os.path.join(gcj_path, 'training'))
        _gen_ds_tfrs(all_df, validation, lang, os.path.join(gcj_path, 'validation'))
        for k, v in testing.items():
            _gen_ds_tfrs(all_df, v, lang, os.path.join(gcj_path, k))
            
    training = read_objs_trf_ds(os.path.join(gcj_path, 'training'))
    validation = read_objs_trf_ds(os.path.join(gcj_path, 'validation'))
    testing = {}
    for t in os.listdir(gcj_path):
        if t != 'training' and t != 'validation' and 'ipynb_checkpoints' not in t:
            testing[t] = read_objs_trf_ds(os.path.join(gcj_path, t))
    return training, validation, testing


if __name__ == '__main__':
    year = sys.argv[1]
    lang = sys.argv[2]
    training, validation, testing = get_tfrs(
        '../../data/GCJ', year, lang)

    count = 2
    for i, t in enumerate(training):
        for k, v in t.items():
            print(k)
            print(v if not isinstance(v, tf.SparseTensor)
                  else tf.sparse.to_dense(v))
        if i > count:
            break

    # print(testing)
