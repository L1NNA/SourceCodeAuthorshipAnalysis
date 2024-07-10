from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.regularizers import l1, l2
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    LSTM, Bidirectional, Dense, Dot, Embedding, Input, Dropout, Lambda)
from tensorflow.keras.callbacks import (
    EarlyStopping, LambdaCallback, ModelCheckpoint)
from tensorflow.keras import Model, regularizers
from gensim.models import Word2Vec
from tqdm import tqdm
import tensorflow as tf
import pandas as pd
import numpy as np
import re
import time
import pickle
import os
import sys
sys.path.insert(0, '/home/jovyan/Source_Code_Veri/src/data_split')
from tokenizer_sp import get_tokenizer_lg
from read_dataset import get_veribin_tfrs
from tfr import make_batch


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def build_model(
    vocab_size,
    embedding_dim=128,
    max_sequence_length=1000,
    number_lstm_units=128,
    num_head=16,
    batch_size=128,

):
    def get_discriminator(
        in_dim=number_lstm_units*2,
        hidden_dim=64
    ):

        input = Input(shape=(in_dim,))
        l1 = Dense(hidden_dim)
        l2 = Dense(1, activation='sigmoid')
        return Model(
            inputs=input, outputs=l2(l1(input))
        )

    def get_encoder():
        seq_input = Input(shape=(max_sequence_length,), dtype='int32')
        embedding_layer = Embedding(
            vocab_size,
            embedding_dim,
            input_length=max_sequence_length,
            trainable=True)

        bilstm_layer = Bidirectional(
            CuDNNLSTM(number_lstm_units, return_sequences=True),
            merge_mode='sum', name='lstm')

        get_key = Dense(num_head, name='attn-key', activation='softmax')
        get_val = Dense(number_lstm_units, name='attn-val')
        head_dim = int(number_lstm_units / num_head)

        embedding_seq = embedding_layer(seq_input)
        inter = bilstm_layer(embedding_seq)
        # [batch, seq, num_head]
        # if any query, concate to inter axis -1
        keys = get_key(inter)
        # [batch, seq, num_head, head_dim]
        vals = tf.reshape(
            get_val(inter),
            [tf.shape(inter)[0],
             tf.shape(inter)[1],
             num_head, head_dim]
        )
        print('keys.shape:')
        print(keys.shape)
        # print((tf.reduce_max(keys, axis=1)).shape)
        attn_outputs = tf.expand_dims(keys, -1) * vals
        # average across time step (weighted avg for each head)
        # [batch, num_head, head_dim]
        out_ = tf.reduce_mean(attn_outputs, axis=1)
        # concat all the heads:
        out_ = tf.reshape(out_, [tf.shape(inter)[0], -1])

        _encoder = Model(inputs=seq_input, outputs=[out_, keys])
        return _encoder

    encoder = get_encoder()
    training_inputs = {
        'seq0': Input(shape=(max_sequence_length), dtype=tf.int32),
        'seq1': Input(shape=(max_sequence_length), dtype=tf.int32),
    }
    out_x1, _ = encoder(training_inputs['seq0'])
    out_x2, _ = encoder(training_inputs['seq1'])

    cos_sim = Dot(axes=1, normalize=True, name='cos_sim')(
        [out_x1, out_x2])
    cos_sim = Lambda(lambda x: tf.clip_by_value(
        x, 0, 1), name='a')(cos_sim)
    
    concated = tf.concat([out_x1, out_x2], axis=-1)
    concated_stop = tf.stop_gradient(concated)

    discriminator = get_discriminator()

    dis_pred = discriminator(concated_stop)
    gen_pred = Lambda(lambda x: discriminator(x), name='pg')(concated)


    adam = tf.keras.optimizers.Nadam(lr=0.04, clipnorm=.1)
    
    discriminator.trainable = True
    discriminator_trainer = Model(
        inputs=training_inputs, outputs={
            'problem':  dis_pred
        }
    )
    discriminator_trainer.compile(
        optimizer=adam,
        loss={
            'problem': 'binary_crossentropy'
        },
        metrics={
            'problem': ['acc', 'AUC'],
        },
    )

    discriminator.trainable = False
    generator_trainer = Model(
        inputs=training_inputs, outputs={
            'authorship': cos_sim,
            'problem': gen_pred,
        })

    generator_trainer.compile(
        optimizer=adam,
        loss={
            'authorship': 'binary_crossentropy',
            'problem': 'binary_crossentropy'
        },
        loss_weights={
            'authorship': 0.9,
            'problem': .1
        },
        metrics={
            'authorship': ['acc', 'AUC', 'Precision', 'Recall'],
            'problem': ['acc', 'AUC'],
        },
    )
    return generator_trainer, discriminator_trainer, encoder


def preprocess(ds_name, ds, batch_size, max_seq_len):
    if isinstance(ds, dict):
        for k, v in ds.items():
            ds[k] = preprocess(k, v, batch_size, max_seq_len)
        return ds
    if isinstance(ds, list):
        return [preprocess('basic', i, batch_size, max_seq_len) for i in ds]
    else:
        # seperate input and output
        return make_batch(ds_name, ds, batch_size, padded=True).map(
            lambda x: ({
                'seq0': x['seq0'][:, :max_seq_len],
                'seq1': x['seq1'][:, :max_seq_len],
            }, {
                'authorship': x['authorship'],
                'problem': x['ber_sim'],
            })
        )


def preprocess_c(ds_name, ds, batch_size, max_seq_len):
    if isinstance(ds, dict):
        for k, v in ds.items():
            ds[k] = preprocess_c(k, v, batch_size, max_seq_len)
        return ds
    if isinstance(ds, list):
        return [preprocess_c('basic', i, batch_size, max_seq_len) for i in ds]
    else:
        # seperate input and output
        return make_batch(ds_name, ds, batch_size, padded=True).map(
            lambda x: ({
                'seq0': x['seq0'][:, :max_seq_len],
                'seq1': x['seq1'][:, :max_seq_len],
            }, {
                'authorship': x['authorship'],
                'problem': x['problem'],
            })
        )
    
    
class EvalCallback(tf.keras.callbacks.Callback):
    def __init__(self, testing_set, model):
        # correct problem label for generator
        # (it should always fool discriminator to output zero (not same problem))
        self.testing_set = {
            k: v.map(
                lambda x, y: (x, {
                    'authorship': y['authorship'],
                    'problem': tf.zeros_like(y['problem'])
                }
                ))
            for k, v in testing_set.items()
        }
        self.model = model

    def on_epoch_begin(self, *args, **kwargs):
        for k, tv in self.testing_set.items():
            result = self.model.evaluate(tv, verbose=-1)
            result = dict(zip(self.model.metrics_names, result))
            result = {k: "{:.4f}".format(v) for k, v in result.items()}
            print(k, '\t', result)
        


def run_training(lang,
        bin_home, proj_path,
        max_seq_len=1200, batch_size=64,
        train_op=True, load_op=True):

    model_path = os.path.join(proj_path, 'models', 'Siamese', lang)
    data_path = os.path.join(proj_path, 'data', 'GCJ')

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    training, validation, testing = get_veribin_tfrs(
        data_path, 'task_cluster', '2017', lang
    )
    if lang == 'c':
        training, validation, testing = preprocess_c(
            '', [training, validation, testing], batch_size, max_seq_len)
    else:
        training, validation, testing = preprocess(
            '', [training, validation, testing], batch_size, max_seq_len)
        
    testing['val'] = validation
    
    
    ###
    
    training = training.shuffle(10000, seed=10).take(10000)
    for k, kv in testing.items(): 
        print(kv)
        testing[k] = testing[k].shuffle(3000, seed=10).take(3000)
    ###
    # remove dis remove problem
    ###
    
    
    tokenizer = get_tokenizer_lg(pd.DataFrame(), pd.DataFrame(), proj_path+'/models/tokenizers/'+lang+'/')
    
    generator_trainer, discriminator_trainer, encoder = build_model(
        vocab_size=tokenizer.vocab_size(), max_sequence_length=max_seq_len)

    # early_stopping = EarlyStopping(monitor='val_acc', patience=100)
    # bst_model_path = os.path.join(
    #     model_path, 'checkpoints.ckpt'
    # )
    # model_checkpoint = ModelCheckpoint(
    #     bst_model_path, save_best_only=False, save_weights_only=True)

    epoches = 600
    if train_op:
        # model_train.fit(
        #     training,
        #     validation_data=None,
        #     epochs=100, batch_size=batch_size, shuffle=True,
        #     callbacks=[
        #         EvalCallback(testing, model_train),
        #         model_checkpoint,
        #     ])
        eval = EvalCallback(testing, generator_trainer)
        steps_per_epoch = 0
#         generator_trainer.load_weights(os.path.join(
#                         model_path, 'gan16', '11.09', 'generator', 'sgd_default_ep69.ckpt'
#                     ))
#         discriminator_trainer.load_weights(os.path.join(
#                         model_path, 'gan16', '11.09', 'discriminator', 'sgd_default_ep69.ckpt'
#                     ))
        for i in range(epoches):
            print()
            print('ep', i)
            if i > 0:
                eval.on_epoch_begin()
            progress = tqdm(total=steps_per_epoch)
            train_out_dict = {'loss':0, 'a_loss':0, 'pg_loss':0,'a_acc':0, 'a_auc':0, 
                'a_precision':0, 'a_recall':0, 'pg_acc':0, 'pg_auc_1':0}
            for x, y in training:
                ret = generator_trainer.train_on_batch(
                        x, {
                            'authorship': y['authorship'],
                            'problem': tf.zeros_like(y['problem'])
                        },reset_metrics=True if i == 0 else False, return_dict=True)
#                 for o in range(len(ret)):
#                     train_out_dict[list(train_out_dict.keys())[o]] += ret[o]
                    
#                 discriminator_trainer.train_on_batch(
#                     x, {
#                         'problem': y['problem']
#                     })
                progress.update()
                if i < 1:
                    steps_per_epoch += 1
                
#                 generator_trainer.save_weights(
#                     os.path.join(
#                         model_path, 'generator_16no_gan', 'select_adam005_1-9','sgd_0.01_ep'+str(i)+'.ckpt'
#                     )
#                 )
#                 discriminator_trainer.save_weights(
#                     os.path.join(
#                         model_path, 'discriminator', 'sgd_0.01_ep'+str(i)+'.ckpt'
#                     )
#                 )
            print('training set \t' + str(ret))
        
    print('done')
    return encoder


if __name__ == '__main__':
    run_training(
        lang=sys.argv[1],
        bin_home='',
        proj_path='/home/jovyan/Source_Code_Veri',
        batch_size=192,
        max_seq_len=2400,
        train_op=True, load_op=False)
