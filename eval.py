# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
from typing import Sequence, Any, ContextManager

from absl import app
from absl import flags
from absl import logging

import numpy as np
import pandas as pd
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    'platform',
    'gpu',
    ['cpu', 'gpu', 'tpu'],
    'What platform to train on')
flags.DEFINE_integer('embed_dim', 64, 'Embedding Dimension')
flags.DEFINE_integer('ff_dim', 64, 'FF Dimension')
flags.DEFINE_integer('num_heads', 1, 'Num heads')
flags.DEFINE_integer('model_depth_enc', 6, 'Num Encoder Layers')
flags.DEFINE_integer('model_depth_dec', 1, 'Num Decoder Layers')
flags.DEFINE_integer('mask_pct', 70, 'Mask Pct')
flags.DEFINE_float('lr',1e-5,'Learning rate')
flags.DEFINE_integer('num_classes', 10, 'Number of Classes')
flags.DEFINE_string('model_path','', 'Path for saved model')
flags.DEFINE_string('model_path_linear','', 'Path for saved linear model')
flags.DEFINE_string('model_kw','fmnist', 'Keyword for dataset')
flags.DEFINE_integer('train_len', 60000, 'Length of train set')
flags.DEFINE_string('train_data_path','./data/fashion-mnist_train_n_s.csv', 'Path for saved linear model')
flags.DEFINE_integer('test_len', 10000, 'Length of test set')
flags.DEFINE_string('test_data_path','./data/fashion-mnist_test_n_s.csv', 'Path for saved linear model')
flags.DEFINE_string(
    'master',
    'local',
    'The BNS address of the first TPU worker.')

rng = np.random.default_rng()


def _get_distribution_strategy_scope(
    platform: str, tpu_address: str,
    ) -> ContextManager[Any]:
  """Returns the device-specific TensorFlow strategy scope used for training."""
  if platform == 'tpu':
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu=tpu_address)
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    return tf.distribute.TPUStrategy(cluster_resolver).scope()
  elif platform == 'gpu':
    return tf.distribute.MirroredStrategy().scope()
  elif platform == 'cpu':
    # Return a dummy distribution strategy for 'cpu' case.
    return contextlib.nullcontext()
  else:
    raise ValueError(f'Invalid "platform": {platform!r}. Only "tpu",'
                     ' "gpu", and "cpu" are supported.')


def mask_and_ind(arr, mask_pct=0.15):
    """Mask a given array unformly and randomly and return non-masked part of array, non-masked indices, masked indices"""
    r, c = arr.shape
    new_c = ((100-mask_pct)*c)//100
    rem_c = c - new_c
    shuff_idx = np.array([rng.permutation(c) for _ in range(r)])
    rem_idx = shuff_idx[:, :rem_c]
    new_idx = shuff_idx[:, rem_c:]
    new_idx.sort(axis=1)
    rem_idx.sort(axis=1)
    new_arr = np.zeros((r, new_c))
    for i in range(r):
        new_arr[i] = arr[i][new_idx[i]]
    return new_arr, new_idx, rem_idx


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, inp_path, indexes, mask_pct = 15, batch_size=128 , shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.inp_path = inp_path
        self.orig_df = pd.read_csv(inp_path)
        self.columns = list(self.orig_df.columns)
        self.maxlen = len(self.columns)-1
        self.mask_pct= mask_pct
        self.small_maxlen = ((100-mask_pct)*self.maxlen)//100
        self.indexes = indexes
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        num_batches = int(np.floor(len(self.indexes) / self.batch_size))
        return num_batches

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        data = self.orig_df.loc[indexes].iloc[:,1:]
        arr = data.to_numpy()
        new_arr, new_idx, rem_idx = mask_and_ind(arr, self.mask_pct)
        X_train = [[],[],[],[]]
        Y_train = []

        for i in range(self.batch_size):
          X_train[0].append(new_arr[i])
          X_train[1].append(new_idx[i])
          X_train[2].append(rem_idx[i])
          X_train[3].append(np.ones(self.small_maxlen))
          Y_train.append(arr[i][list(new_idx[i])+list(rem_idx[i])])

        X_train[0]=np.array(X_train[0])
        X_train[1]=np.array(X_train[1])
        X_train[2]=np.array(X_train[2])
        X_train[3]=np.array(X_train[3])
        Y_train=np.array(Y_train)
        return X_train, Y_train

class DataGeneratorEmbed(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, inp_path, indexes, model, batch_size=128 , shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.inp_path = inp_path
        self.model = model
        self.orig_df = pd.read_csv(inp_path)
        self.labels = np.array(self.orig_df.iloc[:,0])
        self.one_hot = np.eye(FLAGS.num_classes)[self.labels]
        self.columns = list(self.orig_df.columns)
        self.indexes = indexes
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        num_batches = int(np.floor(len(self.indexes) / self.batch_size))
        return num_batches

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        classes_one_hot = self.one_hot[indexes]
        data = self.orig_df.loc[indexes].iloc[:,1:].to_numpy()
        pred_embed = embed_pipeline(self.model, data)
        return pred_embed, classes_one_hot


class TransformerBlock(tf.keras.layers.Layer):
  """Transformer Block Attributes:

    att:
    ffn:
    layernorm1:
    layernorm2:
    dropout1:
    dropout2:
  """

  def __init__(self, embed_dim, num_heads, ff_dim, name, rate=0.1):
    super(TransformerBlock, self).__init__()
    self._name = name
    self.att = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim)
    self.ffn = tf.keras.Sequential([
        tf.keras.layers.Dense(ff_dim, activation='relu'),
        tf.keras.layers.Dense(embed_dim),
    ])
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, inputs, training):
    attn_output = self.att(inputs, inputs)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(inputs + attn_output)
    ffn_output = self.ffn(out1)
    ffn_output = self.dropout2(ffn_output, training=training)
    return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(tf.keras.layers.Layer):
  def __init__(self, maxlen, embed_dim, name='EmbLayer'):
    super(TokenAndPositionEmbedding, self).__init__()
    self._name = name
    self.embed_dim = embed_dim
    self.pos_emb = tf.keras.layers.Embedding(
        input_dim=maxlen, output_dim=embed_dim)
    self.concat = tf.keras.layers.Concatenate(axis=2)

  def call(self, x, positions_unmask, positions_mask):
    positions_unmask = self.pos_emb(positions_unmask)
    if positions_mask.shape[1]>=2:
      positions_mask = self.pos_emb(positions_mask)
    else:
      positions_mask = []
    x = tf.expand_dims(x, -1)
    x = tf.cast(x, tf.float32)
    x = self.concat([x,positions_unmask])
    return x, positions_mask

def train_METModel(
    embed_dim=128,
    num_heads=2,
    ff_dim=128,
    model_depth_enc=6,
    model_depth_dec=1,
    mask_pct=15,
    batch_size=128,
    save_path='./saved_models/'):
    indexes_trn = [i for i in range(FLAGS.train_len)]
    indexes_tst = [i for i in range(FLAGS.test_len)]

    batch_size = min(batch_size, len(indexes_trn), len(indexes_tst))

    trn_dg = DataGenerator(
      FLAGS.train_data_path, indexes_trn, mask_pct=mask_pct, batch_size=batch_size)
    tst_dg = DataGenerator(FLAGS.test_data_path,
                         indexes_tst, mask_pct=mask_pct, batch_size=batch_size)

    maxlen = trn_dg.maxlen
    small_maxlen = ((100 - mask_pct) * maxlen) // 100
    fixed_input = tf.keras.layers.Input(
        shape=(small_maxlen,), batch_size=batch_size)

    inputs = tf.keras.layers.Input(shape=(small_maxlen,), batch_size=batch_size)
    inputs_unmask_idx = tf.keras.layers.Input(
        shape=(small_maxlen,), dtype='int32', batch_size=batch_size)
    inputs_mask_idx = tf.keras.layers.Input(
        shape=(maxlen - small_maxlen,), dtype='int32', batch_size=batch_size)

    embedding_layer = TokenAndPositionEmbedding(
        maxlen, embed_dim, name='EmbeddingLayer')
    non_mask_embed, mask_pos = embedding_layer(inputs, inputs_unmask_idx,
                                        inputs_mask_idx)

    for i in range(model_depth_enc):
        transformer_block = TransformerBlock(
            embed_dim+1, num_heads, ff_dim, name='TransformerLayerEncoder_' + str(i))
        non_mask_embed = transformer_block(non_mask_embed)

    mask_embed = tf.keras.layers.Dense(1)(
        fixed_input)  # Layer to learn mask layer -- constant tensor
    mask_embed = tf.expand_dims(mask_embed, axis=1)
    mask_embed = tf.repeat(mask_embed, repeats=maxlen - small_maxlen, axis=1)
    mask_embed = tf.keras.layers.Concatenate(axis=2)([mask_embed, mask_pos])

    all_embed = tf.keras.layers.Concatenate(axis=1)([non_mask_embed, mask_embed])

    for i in range(model_depth_dec):
        transformer_block = TransformerBlock(
            embed_dim+1, num_heads, ff_dim, name='TransformerLayerDecoder_' + str(i))
        all_embed = transformer_block(all_embed)

    pred = tf.keras.layers.Dense(1)(all_embed)
    model = tf.keras.Model(
        inputs=[inputs, inputs_unmask_idx, inputs_mask_idx, fixed_input],
        outputs=pred)
    opt = tf.keras.optimizers.Adam(learning_rate=FLAGS.lr)
    model.compile(opt, 'mse', metrics=['accuracy'])
    if FLAGS.model_path == '':
        checkpoint_filepath = save_path + FLAGS.model_kw +'_' + str(
            embed_dim) + '_' + str(num_heads) + '_' + str(ff_dim) + '_' + str(
                model_depth_enc) + '_' + str(model_depth_dec) + '_' + str(mask_pct) + '_' + str(FLAGS.lr)
    else:
        checkpoint_filepath = FLAGS.model_path
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    model.summary()
    try:
        model.load_weights(checkpoint_filepath)
        print('Pre-Trained Model loaded')
    except:
        print("No model found at given path")
        exit()
    return model, maxlen

def embed_pipeline(model, inp):
    lay_stack = []
    for lay in model.layers:
        if 'EmbeddingLayer' in lay._name or 'TransformerLayerEncoder_' in lay._name:
            lay_stack.append(lay)
    r, c = inp.shape
    unmask_pos = np.array([np.arange(0, c, 1) for _ in range(r)])
    mask_pos = np.array([[] for _ in range(r)])
    embed, _ = lay_stack[0].call(inp, unmask_pos, mask_pos)
    for lay in lay_stack[1:]:
        embed = lay.call(embed, training=False)
    return embed


def get_met_embed(model, arr, idx, new_len):
    pred_embed = embed_pipeline(model, arr)
    pred_emb_train = []
    pred_emb_test = []
    for i in idx[:new_len]:
        pred_emb_train.append(pred_embed[i])
    for i in idx[new_len:]:
        pred_emb_test.append(pred_embed[i])

    pred_emb_train = np.array(pred_emb_train)
    pred_emb_test = np.array(pred_emb_test)
    return pred_emb_train, pred_emb_test


def train_and_eval_met(model, maxlen,
                       embed_dim=128, batch_size=40, save_path='./saved_models/'):
    mlp_inp = tf.keras.layers.Input(shape=(maxlen, embed_dim+1))
    mlp_layer = tf.keras.layers.Flatten()(mlp_inp)
    mlp_layer = tf.keras.layers.Dense(5040, activation='relu')(mlp_layer)
    mlp_layer = tf.keras.layers.Dense(256, activation='relu')(mlp_layer)
    mlp_pred = tf.keras.layers.Dense(FLAGS.num_classes, activation='softmax')(mlp_layer)
    mlp_model = tf.keras.Model(inputs=mlp_inp, outputs=mlp_pred)
    opt = tf.keras.optimizers.Adam(learning_rate=FLAGS.lr)
    if FLAGS.model_path_linear == '':
        if 'adv' in FLAGS.model_path:
            checkpoint_filepath = save_path + FLAGS.model_kw +'_linear_adv_' + str(
                FLAGS.embed_dim) + '_' + str(FLAGS.num_heads) + '_' + str(FLAGS.ff_dim) + '_' + str(
                    FLAGS.model_depth_enc) + '_' + str(FLAGS.model_depth_dec) + '_' + str(FLAGS.mask_pct) + '_' + str(FLAGS.lr)
        else:
            checkpoint_filepath = save_path + FLAGS.model_kw +'_linear_' + str(
                FLAGS.embed_dim) + '_' + str(FLAGS.num_heads) + '_' + str(FLAGS.ff_dim) + '_' + str(
                    FLAGS.model_depth_enc) + '_' + str(FLAGS.model_depth_dec) + '_' + str(FLAGS.mask_pct) + '_' + str(FLAGS.lr)
    else:
        checkpoint_filepath = FLAGS.model_path_linear
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    mlp_model.compile(opt, 'categorical_crossentropy', metrics=['accuracy'])

    indexes_trn = [i for i in range(int(FLAGS.train_len))]
    indexes_tst = [i for i in range(int(FLAGS.test_len))]
    batch_size = min(batch_size, len(indexes_trn), len(indexes_tst))

    trn_dg = DataGeneratorEmbed(
        FLAGS.train_data_path,
        indexes_trn,
        model,
        batch_size=batch_size)
    tst_dg = DataGeneratorEmbed(
        FLAGS.test_data_path,
        indexes_tst,
        model,
        batch_size=batch_size)
    try:
        mlp_model.load_weights(checkpoint_filepath)
        print("Pre-Trained MLP model loaded from", checkpoint_filepath)
    except:
        mlp_model.fit(trn_dg, epochs=100, validation_data=tst_dg, workers=1, callbacks=[model_checkpoint_callback])
        mlp_model.load_weights(checkpoint_filepath)

    acc = mlp_model.evaluate(tst_dg)
    return acc


def main(argv: Sequence[str]) -> None:
    scope = _get_distribution_strategy_scope(FLAGS.platform, FLAGS.master)
    logging.info(
        'Train-Parms :- embed_dim %d, num_heads %d, ff_dim %d, model_depth_enc %d, model_depth_dec %d, mask_pct %d',
        FLAGS.embed_dim, FLAGS.num_heads, FLAGS.ff_dim, FLAGS.model_depth_enc,
        FLAGS.model_depth_dec, FLAGS.mask_pct)
    with scope:
        met_model, maxlen = train_METModel(
            embed_dim=FLAGS.embed_dim,
            num_heads=FLAGS.num_heads,
            ff_dim=FLAGS.ff_dim,
            model_depth_enc=FLAGS.model_depth_enc,
            model_depth_dec=FLAGS.model_depth_dec,
            mask_pct=FLAGS.mask_pct,
            batch_size=126)
        met_acc = train_and_eval_met(met_model,maxlen,FLAGS.embed_dim,batch_size=126)
        with open('./met.csv','a+') as f:
            f.write(str(FLAGS.embed_dim)+','+str(FLAGS.ff_dim)+','+str(FLAGS.num_heads)+','+str(FLAGS.model_depth_enc)+','+str(FLAGS.model_depth_dec)+','+str(FLAGS.lr)+','+str(FLAGS.mask_pct)+','+','+str(met_acc[1])+'\n')


if __name__ == '__main__':
  app.run(main)
