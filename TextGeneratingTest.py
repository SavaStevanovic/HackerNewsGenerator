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
import re

path = get_file(
    'nietzsche.txt',
    origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
word_text = re.split(r'([\W])+', text)
word_text = [x for x in word_text if x not in [' ', '\n']]
words = sorted(list(set(word_text)))
print('total chars:', len(chars))
print("total number of unique words", len(words))
word_indices = dict((w, i) for i, w in enumerate(words))
indices_word = dict((i, w) for i, w in enumerate(words))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 50
step = 3
sentences = []
next_words = []
for i in range(0, len(word_text) - maxlen, 1):
    sentences.append(word_text[i: i + maxlen])
    next_words.append(word_text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen), dtype=np.int)
y = np.zeros((len(sentences)), dtype=np.int)
for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        x[i, t] = word_indices[word]
    y[i] = word_indices[next_words[i]]

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(Embedding(len(words), 12, input_length=maxlen))
# model.add(LSTM(256, return_sequences=True))
# model.add(Dropout(0.1))
model.add(LSTM(1000))
model.add(Dropout(0.25))
model.add(Dense(len(words), activation='softmax'))

optimizer = keras.optimizers.adam(0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, _):
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
                x_pred[0, t] = word_indices[word]

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_word = indices_word[next_index]

            generated += next_word
            sentence = sentence[1:] + [next_word]

            sys.stdout.write(' '+next_word)
            sys.stdout.flush()
        print()


print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y,
          batch_size=128,
          validation_split=0.2,
          epochs=1000,
          callbacks=[print_callback])
model.save('model.h5')
