# keras imports
import keras.backend as K
from keras.layers import Dense, Input, LSTM, Dropout, Bidirectional, Multiply, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.models import Model
import tensorflow as tf
# std imports
import time
import gc
import os

from inputHandler import create_train_dev_set

class SiameseBiLSTM:
    def __init__(self, embedding_dim, max_sequence_length, number_lstm, number_dense, rate_drop_lstm, 
                 rate_drop_dense, hidden_activation, validation_split_ratio):
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.number_lstm_units = number_lstm
        self.rate_drop_lstm = rate_drop_lstm
        self.number_dense_units = number_dense
        self.activation_function = hidden_activation
        self.rate_drop_dense = rate_drop_dense
        self.validation_split_ratio = validation_split_ratio
    

    def train_model(self, sentences_pair, is_similar, embedding_meta_data, language, model_save_directory='./'):
        """
        Train Siamese network to find similarity between sentences in `sentences_pair`
            Steps Involved:
                1. Pass the each from sentences_pairs  to bidirectional LSTM encoder.
                2. Merge the vectors from LSTM encodes and passed to dense layer.
                3. Pass the  dense layer vectors to sigmoid output layer.
                4. Use cross entropy loss to train weights
        Args:
            sentences_pair (list): list of tuple of sentence pairs
            is_similar (list): target value 1 if same sentences pair are similar otherwise 0
            embedding_meta_data (dict): dict containing tokenizer and word embedding matrix
            model_save_directory (str): working directory for where to save models

        Returns:
            return (best_model_path):  path of best model
        """
        def real_cos_loss(y_true, output1, output2):
            print('1')
            output1 = K.l2_normalize(output1)
            output2 = K.l2_normalize(output2)
            print('2')
            cosine_sim = tf.reduce_sum(output1 * output2, axis=-1)
            print('3')
            loss = abs(y_true - cosine_sim) 
            print('4')
            loss = K.mean(loss)
            return loss
            
        def my_cos_loss_func(output1, output2):
            def loss_stand(y_true, y_pred):
                return real_cos_loss(y_true, output1, output2)
            return loss_stand

           
        tokenizer, embedding_matrix = embedding_meta_data['tokenizer'], embedding_meta_data['embedding_matrix']

        train_data_x1, train_data_x2, train_labels, leaks_train, \
        val_data_x1, val_data_x2, val_labels, leaks_val = create_train_dev_set(tokenizer, sentences_pair,
                                                                               is_similar, self.max_sequence_length,
                                                                               self.validation_split_ratio)

        if train_data_x1 is None:
            print("++++ !! Failure: Unable to train model ++++")
            return None

        nb_words = len(tokenizer.word_index) + 1

        # Creating word embedding layer
        embedding_layer = Embedding(nb_words, self.embedding_dim, weights=[embedding_matrix],
                                    input_length=self.max_sequence_length, trainable=False)

        # Creating LSTM Encoder + a dense layer
        bilstm_layer = Bidirectional(LSTM(self.number_lstm_units, dropout=self.rate_drop_lstm, 
                                        recurrent_dropout=self.rate_drop_lstm, return_sequences=True), 
                                    merge_mode='sum', name='bilstm')
        lstm_layer = LSTM(self.number_lstm_units, dropout=self.rate_drop_lstm, recurrent_dropout=self.rate_drop_lstm, name='lstm')
        dense_layer = Dense(self.number_dense_units, activation=self.activation_function, name='dense_layer')
        
        # Creating LSTM Encoder layer for First Sentence
        sequence_1_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_sequences_1 = embedding_layer(sequence_1_input)
        x1_inter = bilstm_layer(embedded_sequences_1)
        atten_probs_x1 = Dense(self.number_lstm_units, activation='softmax', name='atten_probs_x1')(x1_inter)
        atten_out_x1 = Multiply(name='atten_out_x1')([x1_inter, atten_probs_x1])
        x1 = lstm_layer(atten_out_x1)
        
        # Creating LSTM Encoder layer for Second Sentence
        sequence_2_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_sequences_2 = embedding_layer(sequence_2_input)
        x2_inter = bilstm_layer(embedded_sequences_2)
        atten_probs_x2 = Dense(self.number_lstm_units, activation='softmax', name='atten_probs_x2')(x2_inter)
        atten_out_x2 = Multiply(name='atten_out_x2')([x2_inter, atten_probs_x2])
        x2 = lstm_layer(atten_out_x2)
        
        # Creating leaks input
        leaks_input = Input(shape=(leaks_train.shape[1],))
        leaks_dense = Dense(int(self.number_dense_units/2), activation=self.activation_function)(leaks_input)
        # Add leaks to first encoder
        merged_x1 = concatenate([x1, leaks_dense])
        merged_x1 = BatchNormalization()(merged_x1)
        merged_x1 = Dropout(self.rate_drop_dense)(merged_x1)
        merged_x1 = dense_layer(merged_x1)
        # Add leaks to second encoder
        merged_x2 = concatenate([x2, leaks_dense])
        merged_x2 = BatchNormalization()(merged_x2)
        merged_x2 = Dropout(self.rate_drop_dense)(merged_x2)
        merged_x2 = dense_layer(merged_x2)
        
        merged_x1 = Activation('linear', name='output1')(merged_x1)
        merged_x2 = Activation('linear', name='output2')(merged_x2)
        #loss = K.mean(mse(output_1, label_layer_2) * mse(output_2, label_layer_2))
        
        #loss = abs(y_true - tf.reduce_sum(K.l2_normalize(merged_x1) * K.l2_normalize(merged_x2), axis=-1))
        
        model = Model(inputs=[sequence_1_input, sequence_2_input, leaks_input], outputs=[merged_x1, merged_x2])
        #model.add_loss(loss)
        model.compile(loss=my_cos_loss_func, optimizer='nadam', metrics=['acc'])
        #model.compile(loss={'merged_x1': Myloss, 'merged_x2': Myloss}, optimizer='adam')

        early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        STAMP = 'lstm_%d_%d_%.2f_%.2f' % (self.number_lstm_units, self.number_dense_units, self.rate_drop_lstm, self.rate_drop_dense)

        checkpoint_dir = model_save_directory + 'checkpoints/model/' + language + '_' + str(int(time.time())) + '/'

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        bst_model_path = checkpoint_dir + STAMP + '.h5'

        model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)

        tensorboard = TensorBoard(log_dir=checkpoint_dir + "logs/{}".format(time.time()))
        model.fit([train_data_x1, train_data_x2, leaks_train], train_labels,
                  validation_data=([val_data_x1, val_data_x2, leaks_val], val_labels),
                  epochs=200, batch_size=64, shuffle=True,
                  callbacks=[early_stopping, model_checkpoint, tensorboard])
        #model.fit([train_data_x1, train_data_x2], train_labels,
        #          validation_data=([val_data_x1, val_data_x2], val_labels),
        #          epochs=200, batch_size=64, shuffle=True,
        #          callbacks=[early_stopping, model_checkpoint, tensorboard])

        return bst_model_path


    def update_model(self, saved_model_path, new_sentences_pair, is_similar, embedding_meta_data):
        """
        Update trained siamese model for given new sentences pairs 
            Steps Involved:
                1. Pass the each from sentences from new_sentences_pair to bidirectional LSTM encoder.
                2. Merge the vectors from LSTM encodes and passed to dense layer.
                3. Pass the  dense layer vectors to sigmoid output layer.
                4. Use cross entropy loss to train weights
        Args:
            model_path (str): model path of already trained siamese model
            new_sentences_pair (list): list of tuple of new sentences pairs
            is_similar (list): target value 1 if same sentences pair are similar otherwise 0
            embedding_meta_data (dict): dict containing tokenizer and word embedding matrix

        Returns:
            return (best_model_path):  path of best model
        """
        tokenizer = embedding_meta_data['tokenizer']
        train_data_x1, train_data_x2, train_labels, leaks_train, \
        val_data_x1, val_data_x2, val_labels, leaks_val = create_train_dev_set(tokenizer, new_sentences_pair,
                                                                               is_similar, self.max_sequence_length,
                                                                               self.validation_split_ratio)
        model = load_model(saved_model_path)
        model_file_name = saved_model_path.split('/')[-1]
        new_model_checkpoint_path  = saved_model_path.split('/')[:-2] + str(int(time.time())) + '/' 

        new_model_path = new_model_checkpoint_path + model_file_name
        model_checkpoint = ModelCheckpoint(new_model_checkpoint_path + model_file_name,
                                           save_best_only=True, save_weights_only=False)

        early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        tensorboard = TensorBoard(log_dir=new_model_checkpoint_path + "logs/{}".format(time.time()))

        model.fit([train_data_x1, train_data_x2, leaks_train], train_labels,
                  validation_data=([val_data_x1, val_data_x2, leaks_val], val_labels),
                  epochs=50, batch_size=3, shuffle=True,
                  callbacks=[early_stopping, model_checkpoint, tensorboard])
        #model.fit([train_data_x1, train_data_x2], train_labels,
        #          validation_data=([val_data_x1, val_data_x2], val_labels),
        #          epochs=50, batch_size=3, shuffle=True,
        #          callbacks=[early_stopping, model_checkpoint, tensorboard])

        return new_model_path
