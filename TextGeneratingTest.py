'''Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
import time
import tensorflow as tf
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import CuDNNLSTM, Embedding, CuDNNGRU
import numpy as np
from tensorflow import keras
import random
import sys
import io
import re
sess = tf.keras.backend.get_session()


def _parse_function(example_proto):
    sequence_features = {
        'data': tf.FixedLenSequenceFeature([], tf.int64, default_value=None, allow_missing=True),
        'label': tf.FixedLenSequenceFeature([], tf.int64, default_value=None, allow_missing=True)
    }
    features = {}
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        example_proto, context_features=features, sequence_features=sequence_features)

    data = sequence_parsed["data"]
    labels = sequence_parsed["label"]
    return data, labels


def on_epoch_end(epoch):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(word_text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = []
        sentence = word_text[start_index: start_index + maxlen]
        generated.append(sentence)
        print('----- Generating with seed: "' + ' '.join(sentence) + '"')
        sys.stdout.write(' '.join(sentence))

        for i in range(30):
            x_pred = np.zeros((1, maxlen), dtype=np.int)
            for t, word in enumerate(sentence):
                x_pred[0, t] = words_dict.index(word)

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_word = words_dict[next_index]

            generated += next_word
            sentence = sentence[1:] + [next_word]

            sys.stdout.write(' '+next_word)
            sys.stdout.flush()
        print()


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# cut the text in semi-redundant sequences of maxlen characters
maxlen = 20
batch_size = 64

dict = eval(open("dictionary.txt").read())
words_dict = list(dict.keys())

filenames = ['train1.tfrecords']
dataset = tf.data.TFRecordDataset(filenames).apply(tf.contrib.data.map_and_batch(_parse_function, 256, num_parallel_calls=12)).shuffle(4096).prefetch(1)

model = Sequential()
model.add(Embedding(len(words_dict), 50, input_length=maxlen))
# model.add(LSTM(256, return_sequences=True))
# model.add(Dropout(0.1))
model.add(CuDNNGRU(1000, return_sequences=False))
model.add(Dropout(rate=0.20))
model.add(Dense(100, activation='relu'))
model.add(Dense(len(words_dict), activation='softmax'))

optimizer = tf.train.AdamOptimizer(0.001)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer, metrics=['accuracy'])
iter = dataset.make_one_shot_iterator()

for i in range(10000):
    print(i)
    t=time.time()
    print(model.train_on_batch(iter))
    print(time.time()-t)

model.save('model.h5')
# on_epoch_end(0)
