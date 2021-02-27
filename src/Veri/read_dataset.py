import sys
from collections import defaultdict
import os
import re
import json
from typing import NamedTuple, List, Tuple
import pandas as pd
from PIL import Image
import numpy as np
import pickle
from tqdm import tqdm
import pickle
import gzip
import zipfile
from dataset_gcj import get_gcj_dataset
from utils import libc
from tfr import write_objs_as_tfr, read_objs_trf_ds, _default_fs_mapper
from tokenizer_sp import get_tokenizer_lg
import tensorflow as tf
from utils import AttrDict


_keywords = ('loc_', 'loc', 'sub_', 'var_', 'var', 'arg_', 'word_',
             'off_', 'locret_', 'flt_', 'dbl_', 'qword',
             'dword', 'offset',)

default_bin_home = '/home/shared-data/GCJ_bins/bins/'
default_asm_bin_home = '/home/shared-data/GCJ_bins/bins/'


def _binary_to_sequence(js_file, tokenizer, max_blk=300):

    js_file = os.path.join(default_asm_bin_home, js_file)
    if os.path.exists(js_file):

        obj = _default_fs_mapper(os.path.join(default_bin_home, js_file))

        # access dict keys through attribute.
        obj = AttrDict.from_nested_dict(obj)

        # sort blocks
        start_funcs = set([
            f._id for f in obj.functions
            if 'main' in f.name.lower()
            or 'start' in f.name.lower()])
        if len(start_funcs) < 1:
            start_funcs = set([obj.functions[0]._id])

        blk_map = {b._id: b for b in obj.blocks}
        block_ids = [b._id for b in obj.blocks if b.func_id in start_funcs]
        covered = set(block_ids)
        stack = list(block_ids)

        while len(block_ids) < max_blk and len(stack) > 0:
            _next = stack.pop(0)
            for call in blk_map[_next].calls:
                if call in blk_map:
                    if not call in covered:
                        covered.add(call)
                        stack.append(call)
                        block_ids.append(call)

        new_blks = [blk_map[b] for b in block_ids]
        new_blks = sorted(new_blks, key=lambda x: -1*len(x['ins']))
        # extract block texts for ordered blocks

        tokens = [t for b in new_blks for i in b.ins
                  for t in [i.mne] + i.oprs]

        tokens = [p for t in tokens for p in re.split(
            r'[+\-*\\\[\]:()_\s]', t.lower())
            if len(p) > 0 and not p.startswith(_keywords)]

        # strs = tokenizer.encode(' '.join(tokens), out_type=str)
        tokens = tokenizer.EncodeAsIds(' '.join(tokens))
        tokens = [t for t in tokens if t != 5]

        # for s,t in zip(strs, tokens):
        #    print('#', s, t)

        return tokens
    else:
        js_file = os.path.basename(js_file).replace('.o.asm.json', '')
        js_file = [f for f in os.listdir(default_bin_home) if js_file in f]
        js_file = js_file[0]
        js_file = os.path.join(default_bin_home, js_file)

        with open(js_file) as f:
            data = json.loads(f.read())

        # extract all block texts
        bs = [[b['asm'], b['id']] for b in data[0]]

        # extract ordered ids in graph (stored in data[1][0])
        order = [o for li in data[1][0] for o in li]
        # get all unique ids
        order2 = list(set(order))
        # order the unique ids based on ordered id
        order2.sort(key=order.index)

        # order blocks based on ordered unique ids
        b_sorted = [0] * len(order2)
        for i in range(len(b_sorted)):
            b_sorted[i] = bs[order2[i]]

        # extract block texts for ordered blocks
        ins = [0] * len(b_sorted)
        for i in range(len(b_sorted)):
            # x[1:] removing machine location
            ins[i] = [s for x in b_sorted[i][0] for s in x[1:]]
        ins_str = ""

        for l in ins:
            ins_str += " ".join(l)+" "

        tokens = [p for p in re.split(
            r'[+\-*\\\[\]:()_\s]', ins_str.lower())
            if len(p) > 0 and not p.startswith(_keywords)]

        # strs = tokenizer.encode(' '.join(tokens), out_type=str)
        tokens = tokenizer.EncodeAsIds(' '.join(tokens))
        tokens = [t for t in tokens if t != 5]

        # for s,t in zip(strs, tokens):
        #    print('#', s, t)

        return tokens


def _mapper(entry, *args):
    tokenizer = get_tokenizer_lg()
    seq0 = _binary_to_sequence(entry['file0'], tokenizer)
    seq1 = _binary_to_sequence(entry['file1'], tokenizer)
    return {
        'seq0': seq0,
        'seq1': seq1,
        'authorship': entry['label'],
        'problem': entry['label_p']
    }


def _gen_ds_tfrs(df, tfr_path):
    entries = df.to_dict('records')
    write_objs_as_tfr(entries, tfr_path, obj_mapper=_mapper)


def get_veribin_tfrs(
        bin_home,
        data_path):
    global default_bin_home
    default_bin_home = bin_home

    gcj_path = os.path.join(data_path, 'veri', 'tfrs')
    if not os.path.exists(gcj_path):
        os.makedirs(gcj_path)

        training, validation, testing = get_gcj_dataset(
            bin_home, data_path
        )
        _gen_ds_tfrs(training, os.path.join(gcj_path, 'training'))
        _gen_ds_tfrs(validation, os.path.join(gcj_path, 'validation'))
        for k, v in testing.items():
            _gen_ds_tfrs(v, os.path.join(gcj_path, k))
    training = read_objs_trf_ds(os.path.join(gcj_path, 'training'))
    validation = read_objs_trf_ds(os.path.join(gcj_path, 'validation'))
    testing = {}
    for t in os.listdir(gcj_path):
        if t != 'training' and t != 'validation' and 'ipynb_checkpoints' not in t:
            testing[t] = read_objs_trf_ds(os.path.join(gcj_path, t))
    return training, validation, testing


if __name__ == '__main__':
    training, validation, testing = get_veribin_tfrs(
        default_bin_home,
        data_path='data')

    count = 2
    for i, t in enumerate(training):
        for k, v in t.items():
            print(k)
            print(v if not isinstance(v, tf.SparseTensor)
                  else tf.sparse.to_dense(v))
        if i > count:
            break

    # print(testing)
