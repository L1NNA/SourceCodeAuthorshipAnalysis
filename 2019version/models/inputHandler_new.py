from keras.preprocessing.sequence import pad_sequences
#from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
import numpy as np
import gc
import pickle
import ast
import pandas as pd

def train_word2vec(documents, embedding_dim):
    """
    train word2vector over traning documents
    Args:
        documents (list): list of document
        embedding_dim (int): outpu wordvector size
    Returns:
        word_vectors(dict): dict containing words and their respective vectors
    """
    model = Word2Vec(documents, min_count=10, size=embedding_dim)
    print('min_count: '+str(10))
    word_vectors = model.wv
    del model
    return word_vectors


def setup_tokenizer(docs):
    #unq_words = set()
    #for d in range(len(docs)):
    #    unq_words = unq_words | set(docs[d])
    #np.save('/content/drive/My Drive/Projects/1_Verification_Sourcecode_Siamese/data_gen/siamese/lstm-siamese-text-similarity/unq_words.npy', unq_words)
    unq_words = np.load('/content/drive/My Drive/Projects/1_Verification_Sourcecode_Siamese/data_gen/siamese/lstm-siamese-text-similarity/unq_words.npy', allow_pickle = True)
    print(len(unq_words))
    indexs = np.arange(len(unq_words))
    print(len(indexs))
    tokenizer = {}
    tokenizer['word_index'] = dict(zip(unq_words, indexs))
    tokenizer['nb_words'] = len(unq_words)
    return tokenizer


def create_embedding_matrix(tokenizer, word_vectors, embedding_dim):
    """
    Create embedding matrix containing word indexes and respective vectors from word vectors
    Args:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object containing word indexes
        word_vectors (dict): dict containing word and their respective vectors
        embedding_dim (int): dimention of word vector

    Returns:

    """
    nb_words = tokenizer['nb_words']
    print('nb_words:' + str(nb_words))
    i_n = 0
    word_index = tokenizer['word_index']
    embedding_matrix = np.zeros((nb_words, embedding_dim))
    print("Embedding matrix shape: %s" % str(embedding_matrix.shape))
    for word, i in word_index.items():
        try:
            embedding_vector = word_vectors[word]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        except KeyError:
            #print("vector not found for word - %s" % word)
            i_n += 1
            print(i)
    
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    print("number of vector not found for word: "+str(i_n))
    return embedding_matrix


def word_embed_meta_data(documents, embedding_dim, language):

    """
    Load tokenizer object for given vocabs list
    Args:
        documents (list): list of document
        embedding_dim (int): embedding dimension
    Returns:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object
        embedding_matrix (dict): dict with word_index and vector mapping
    """
    documents = [ast.literal_eval(fline) for fline in documents]

    #tokenizer = Tokenizer()
    #tokenizer.fit_on_texts(documents)
    print('before setup tokenizer')
    tokenizer = setup_tokenizer(documents)
    print('after setup tokenizer')
    # save tokenizer for testing
    with open('checkpoints/tokenizer/'+language+'_tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    word_vector = train_word2vec(documents, embedding_dim)
    embedding_matrix = create_embedding_matrix(tokenizer, word_vector, embedding_dim)
    print('after setup embedding_matrix')
    del word_vector
    gc.collect()
    return tokenizer, embedding_matrix

def create_train_dev_set(tokenizer, sentences_pair, is_similar, max_sequence_length, validation_split_ratio):
    """
    Create training and validation dataset
    Args:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object
        sentences_pair (list): list of tuple of sentences pairs
        is_similar (list): list containing labels if respective sentences in sentence1 and sentence2
                           are same or not (1 if same else 0)
        max_sequence_length (int): max sequence length of sentences to apply padding
        validation_split_ratio (float): contain ratio to split training data into validation data

    Returns:
        train_data_1 (list): list of input features for training set from sentences1
        train_data_2 (list): list of input features for training set from sentences2
        labels_train (np.array): array containing similarity score for training data
        leaks_train(np.array): array of training leaks features

        val_data_1 (list): list of input features for validation set from sentences1
        val_data_2 (list): list of input features for validation set from sentences1
        labels_val (np.array): array containing similarity score for validation data
        leaks_val (np.array): array of validation leaks features
    """
    #sentences1 = [x[0].lower() for x in sentences_pair]
    #sentences2 = [x[1].lower() for x in sentences_pair]
    print('before setup sentences')
    sentences1 = [ast.literal_eval(x[0]) for x in sentences_pair]
    sentences2 = [ast.literal_eval(x[1]) for x in sentences_pair]
    #train_sequences_1 = tokenizer.texts_to_sequences(sentences1)
    #train_sequences_2 = tokenizer.texts_to_sequences(sentences2)
    print('before setup mapping')
    train_sequences_1 = [list(map(tokenizer['word_index'].get, x)) for x in sentences1]
    train_sequences_2 = [list(map(tokenizer['word_index'].get, x)) for x in sentences2]
    print('after mapping')
    #leaks = [[len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
    #         for x1, x2 in zip(train_sequences_1, train_sequences_2)]
    leaks = [[len(set(x1)), len(set(x2))]
             for x1, x2 in zip(train_sequences_1, train_sequences_2)]
             
    train_padded_data_1 = pad_sequences(train_sequences_1, maxlen=max_sequence_length)
    train_padded_data_2 = pad_sequences(train_sequences_2, maxlen=max_sequence_length)
    train_labels = np.array(is_similar)
    leaks = np.array(leaks)

    shuffle_indices = np.random.permutation(np.arange(len(train_labels)))
    train_data_1_shuffled = train_padded_data_1[shuffle_indices]
    train_data_2_shuffled = train_padded_data_2[shuffle_indices]
    train_labels_shuffled = train_labels[shuffle_indices]
    leaks_shuffled = leaks[shuffle_indices]

    dev_idx = max(1, int(len(train_labels_shuffled) * validation_split_ratio))

    del train_padded_data_1
    del train_padded_data_2
    gc.collect()

    train_data_1, val_data_1 = train_data_1_shuffled[:-dev_idx], train_data_1_shuffled[-dev_idx:]
    train_data_2, val_data_2 = train_data_2_shuffled[:-dev_idx], train_data_2_shuffled[-dev_idx:]
    labels_train, labels_val = train_labels_shuffled[:-dev_idx], train_labels_shuffled[-dev_idx:]
    leaks_train, leaks_val = leaks_shuffled[:-dev_idx], leaks_shuffled[-dev_idx:]
    return train_data_1, train_data_2, labels_train, leaks_train, val_data_1, val_data_2, labels_val, leaks_val


def create_test_data(tokenizer, test_sentences_pair, max_sequence_length):

    """
    Create training and validation dataset
    Args:
        tokenizer (keras.preprocessing.text.Tokenizer): keras tokenizer object
        test_sentences_pair (list): list of tuple of sentences pairs
        max_sequence_length (int): max sequence length of sentences to apply padding

    Returns:
        test_data_1 (list): list of input features for training set from sentences1
        test_data_2 (list): list of input features for training set from sentences2
    """
    #test_sentences1 = [x[0].lower() for x in test_sentences_pair]
    #test_sentences2 = [x[1].lower() for x in test_sentences_pair]
    test_sentences1 = [ast.literal_eval(x[0]) for x in test_sentences_pair]
    test_sentences2 = [ast.literal_eval(x[1]) for x in test_sentences_pair]

    #test_sequences_1 = tokenizer.texts_to_sequences(test_sentences1)
    #test_sequences_2 = tokenizer.texts_to_sequences(test_sentences2)
    nb_w = len(tokenizer['word_index'])
    print(nb_w)
    
    def dict_get(dict_, in_):
        return dict_.get(in_, 0)
        
    def func_map(in_seq):
        tkn_num = np.asarray(list(map(lambda x: dict_get(tokenizer['word_index'], x), in_seq)))
        #return np.where(tkn_num is None, nb_w, tkn_num)
        return tkn_num

    test_sequences_1 = [func_map(x) for x in test_sentences1]
    test_sequences_2 = [func_map(x) for x in test_sentences2]

    leaks_test = [[len(set(x1)), len(set(x2))]
                  for x1, x2 in zip(test_sequences_1, test_sequences_2)]
    leaks_test = np.array(leaks_test)
    print(max_sequence_length)
    try:
        test_data_1 = pad_sequences(test_sequences_1, maxlen=max_sequence_length)
        test_data_2 = pad_sequences(test_sequences_2, maxlen=max_sequence_length)
    except:
        print(test_sequences_1)
        print(test_sentences1)
        tt()
    print('===')
    return test_data_1, test_data_2, leaks_test
