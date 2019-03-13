'''Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, Embedding
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.utils import to_categorical
import numpy as np
import keras
import random
import sys
import io
import os
import re
import random
import tensorflow as tf


def convert_to(data, labels, name, writer):
    """Converts a dataset to tfrecords."""
    num_examples = len(labels)
    for index in range(num_examples):
        example=tf.train.SequenceExample()
        feature_labels=example.feature_lists.feature_list['label']
        feature_labels.feature.add().int64_list.value.append(labels[index])
        feature_data=example.feature_lists.feature_list['data']
        for input_ in data[index]:
            feature_data.feature.add().int64_list.value.append(input_)
        writer.write(example.SerializeToString())
    return num_examples


# cut the text in semi-redundant sequences of maxlen characters
maxlen = 20
batch_size = 64
val_chance = 0.2
dict = eval(open("dictionary.txt").read())
words_dict = list(dict.keys())

q = 0
c = 0

trainWriter = tf.python_io.TFRecordWriter('train1.tfrecords')
validateWriter = tf.python_io.TFRecordWriter('validate1.tfrecords')
with io.open('document.txt', 'r', encoding='utf-8') as f:
    for line in f:
        try:
            print(c)
            c += 1
            if c <= -1:
                continue
            word_text = re.split(r'([\W])+', line)
            word_text = [x for x in word_text if x not in [' ', '\n']]
            sentences = []
            next_words = []
            for i in range(0, len(word_text) - maxlen, 1):
                sentences.append(word_text[i: i + maxlen])
                next_words.append(word_text[i + maxlen])
            print('nb sequences:', len(sentences))
            x = np.zeros((len(sentences), maxlen), dtype=np.int)
            y = np.zeros((len(sentences)), dtype=np.int)
            for i, sentence in enumerate(sentences):
                for t, word in enumerate(sentence):
                    x[i, t] = words_dict.index(word)
                y[i] = words_dict.index(next_words[i])
            if random.random() > val_chance:
                q += convert_to(x, y, 'train', trainWriter)
            else:
                q += convert_to(x, y, 'validation', validateWriter)
            print(q)
        except Exception as e:
            pass
validateWriter.close()
trainWriter.close()
