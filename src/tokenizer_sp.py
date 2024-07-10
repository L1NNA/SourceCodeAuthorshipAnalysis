import gzip
import json
import pathlib
from argparse import ArgumentParser
from random import seed, shuffle
import os
import re
from pygments.lexers import guess_lexer
import sentencepiece as spm
import pandas as pd
# seed(0)
import sys
from tqdm import tqdm
oprs_filter = (
    'loc_', 'sub_', 'arg_', 'var_', 'unk_',
    'word_', 'off_', 'locret_', 'flt_', 'dbl_', 'param_', 'local_')

TOKEN_PAD = '[PAD]'  # Token for padding
TOKEN_UNK = '[UNK]'  # Token for unknown words
TOKEN_CLS = '[CLS]'  # Token for classification
TOKEN_SEP = '[SEP]'  # Token for separation
TOKEN_MASK = '[MASK]'  # Token for masking

ID_PAD = 0
ID_UNK = 1
ID_CLS = 2
ID_SEP = 3
ID_MASK = 4

base_dict = {
    TOKEN_PAD: ID_PAD,
    TOKEN_UNK: ID_UNK,
    TOKEN_CLS: ID_CLS,
    TOKEN_SEP: ID_SEP,
    TOKEN_MASK: ID_MASK,
}


base_dict_rev = {
    ID_PAD: TOKEN_PAD,
    ID_UNK: TOKEN_UNK,
    ID_CLS: TOKEN_CLS,
    ID_SEP: TOKEN_SEP,
    ID_MASK: TOKEN_MASK,
}
supported_types = ['metapc', 'ppc']

# tokenizer_folder = '/home/jovyan/Source_Code_Veri/models/tokenizers/'
default_bin_home = ''


def ins2seq(i):
    tkns = []
    for tkn in [i['mne']] + i['oprs']:
        for p in re.split(r'([+\-*\\\[\]:()\s@?_$])', tkn.lower()):
            if not p.startswith(oprs_filter) and len(p) > 0:
                tkns.append(p)
    return tkns


def blk2seq(b, sp: spm.SentencePieceProcessor = None, flatten=False):
    # vals = [' '.join(ins2seq(i)) for i in b['ins']]
    vals = b
    if sp:
        vals = [sp.EncodeAsIds(v) for v in vals]
    if flatten:
        vals = [x for v in vals for x in v]
    return vals


def get_files(bins_path):
    return list(
        pathlib.Path(bins_path).glob('*.asm.json.gz'))


def get_tokenizer_lg(all_df, df, tokenizer_folder, bins_path=default_bin_home):
    return get_tokenizer_xl(all_df, df, tokenizer_folder, bins_path)


def get_tokenizer_xl(all_df, df, tokenizer_folder, bins_path=default_bin_home):
    return get_tokenizer(all_df, df, tokenizer_folder, bins_path, num_ins=10000000, vocab_size=30000, re_train=False)


def get_tokenizer_md(bins_path=default_bin_home):
    return get_tokenizer(bins_path, num_ins=1000000, vocab_size=20000, re_train=False)


def get_tokenizer_sm(bins_path=default_bin_home):
    return get_tokenizer(bins_path, num_ins=800000, vocab_size=10000, re_train=False)


def get_tokenizer(all_df, df, tokenizer_folder, bins_path, num_ins=10000000, vocab_size=30000, re_train=False):
    folder = tokenizer_folder
    if not os.path.exists(folder):
        os.makedirs(folder)
        gen_tokenizer(all_df, df, tokenizer_folder, num_ins=10000000, vocab_size=30000, re_train=False)
    saved = 'tokenizer_{}_{}.sp'.format(
        num_ins, vocab_size)
    saved = os.path.join(folder, saved)
    model = saved + '.model'
    trained = False

    sp = spm.SentencePieceProcessor()
    sp.Load(model)

    setattr(sp, 'tokenize_blk', lambda s, b: [i['mne'] + ' ' + ' '.join(
        i['oprs']) for i in b['ins']])

    return sp


def gen_tokenizer(all_df, df, tokenizer_folder, num_ins=10000000, vocab_size=30000, re_train=False):
    if not os.path.exists(tokenizer_folder):
        os.mkdir(tokenizer_folder)
    saved = 'tokenizer_{}_{}.sp'.format(
            num_ins, vocab_size)
    saved = os.path.join(tokenizer_folder, saved)
    model = saved + '.model'
    
    if not os.path.exists(model) or re_train:
       
        files = []
        pass_num = 0
        print(df.columns.values)
        file_name0 = list(df.loc[df['file0'].isin(all_df.full_path.unique())].file0.unique())
        file_name1 = list(df.loc[df['file1'].isin(all_df.full_path.unique())].file1.unique())
        file_names = list(set(file_name0 + file_name1))
        print('len(file_names)', len(file_names))
        files = list(all_df.loc[all_df['full_path'].isin(file_names)].flines)
        print('len(files)', len(files))
        shuffle(files)

        print('total', len(files), 'files in df')
        x = set()
        def generator():
            count = 0
            for file in files:
                lexer = guess_lexer(file)
                file_text = ' '.join([t[1] for t in lexer.get_tokens(file) if len(t[1].strip()) > 0 and 'comment' not in str(t[0]).lower()])
                yield file_text.lower()

        spm.SentencePieceTrainer.Train(
            sentence_iterator=generator(),
            model_prefix=saved,
            vocab_size=vocab_size,
            hard_vocab_limit=False,
            pad_piece=TOKEN_PAD,
            pad_id=ID_PAD,
            unk_piece=TOKEN_UNK,
            unk_id=ID_UNK,
            unk_surface=TOKEN_UNK,
            eos_id=-1,
            bos_id=-1,
            user_defined_symbols=[TOKEN_CLS, TOKEN_SEP, TOKEN_MASK])
        sp = spm.SentencePieceProcessor()
        sp.Load(model)
        print('actual vocab size', sp.vocab_size())
        print('len(files)', len(files))
        print('pass_num', pass_num)

        
if __name__ == '__main__':
#     parser = ArgumentParser()
#     parser.add_argument(
#         '-p', '--json_path', type=str,
#         default=default_bin_home
#     )
#     parser.add_argument(
#         '-n', '--num_ins', type=int, default=10000)
#     parser.add_argument(
#         '-v', '--vocab_size', type=int, default=30000)
#     flags = parser.parse_args()
#     get_tokenizer(
#         flags.json_path, flags.num_ins, flags.vocab_size,
#         re_train=True)

    data_path = '/home/jovyan/Source_Code_Veri/data/GCJ'
    year = '2017'
    lang = sys.argv[1]
    all_df = pd.read_csv(os.path.join(data_path, 'gcj'+year+'.csv'))
    df = pd.read_csv(os.path.join(data_path, 'splits', lang, 'training.csv'))
    gen_tokenizer(all_df, df, '/home/jovyan/Source_Code_Veri/models/tokenizers/'+lang)