from nltk.tokenize import word_tokenize
import re
import itertools
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors as word2vec
from tqdm import tqdm
import os
import sys


def code_file_vector(model, text):
    # remove out-of-vocabulary words
    text_f = [word if word in model.vocab else 'OWH_NONE' for word in text]
    if 'OWH_NONE' in text_f:
        print(text)
        print(text_f)
    return np.mean(model[text_f], axis=0)
  
    
def preprocess(model, code_text):
    return_words = ['']
    try:
        # Tokenize the code text into words
        tokens = word_tokenize(code_text)
        # Lower the tokens, keep punctuations
        words = [word.lower() for word in tokens]
        # split words if it has both dots and words
        words = list(itertools.chain.from_iterable([word.split(".") if not bool(re.match('^[0-9.]*$', word)) else [word] for word in words]))
        # create <PAD_OR_OOV> for unseen words
        words = [word if word in model.vocab else 'oov' for word in words]
        if len(words) >= 200:
            words = words[:200]
        else:
            words = words + ['pad']*(200-len(words))
        return_words = words
    except:
        print(code_text)
        return_words = ['']
    return return_words


if __name__ == '__main__':
    language = sys.argv[1]
    c2v_model = word2vec.load_word2vec_format('/home/jovyan/Source_Code_Veri/data/GCJ/C2V/git_token_vecs.txt', binary=False)
    test_file = pd.read_csv('/home/jovyan/Source_Code_Veri/data/GCJ/gcj2017.csv')
    test_file['language'] = test_file.apply(lambda row: os.path.splitext(row['file'])[1][1:] , axis=1)
    test_file_lan = test_file.loc[test_file.language==language]


    tqdm.pandas()
    # preprocess
    corpus = pd.DataFrame(test_file_lan.progress_apply(lambda row: preprocess(c2v_model, row['flines']), axis=1), columns = ['flines_vec'])
    # filter vector list, include only those c2v has a vector for
    corpus_vec = pd.DataFrame(corpus.progress_apply(lambda row: code_file_vector(c2v_model, row['flines_vec']), axis=1), columns = ['flines_vec_c2v'])
    # concat test df and corresponding vectors
    test_df = pd.concat([test_file_lan, corpus_vec], axis=1)
    
    test_df.to_csv('/home/jovyan/Source_Code_Veri/data/GCJ/gcj2017_java.csv')
    test_df.to_csv('/home/jovyan/Source_Code_Veri/data/GCJ/C2V/gcj2017_java.csv')